# -*- coding: utf-8 -*-
"""
普洱景迈山植被覆盖度（FVC）估算与古茶林长势分级工具 —— 核心算法模块
=====================================================================

本模块只负责“算”和“画”，不依赖 Streamlit，可单独在控制台运行（见文件末尾 __main__）。

主要流程：
    读取双波段影像(B4 红 / B8 近红外)
      -> NDVI = (NIR - Red) / (NIR + Red)
      -> 直方图 5% / 95% 分位数确定 NDVI_soil / NDVI_veg
      -> 像元二分模型 FVC = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)
      -> 按阈值重分类为 4 级长势
      -> 按像元真实面积统计各级面积与占比
      -> 输出 FVC 栅格 / 分级栅格 / 统计表 / 专题图

坐标与面积说明：
    * NDVI、FVC 均为比值，对反射率整体缩放不敏感，因此 S2 L2A 的 DN 值
      （反射率 x 10000，或含 +1000 偏移）不换算也能正确计算；
    * 面积 = 像元个数 x 像元真实面积，投影坐标系下由仿射变换直接求得，
      地理坐标系（经纬度）下按中心纬度近似换算为米。

作者：景迈山古茶林监测小组
"""

from __future__ import annotations

import json
import math
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 常量配置
# ----------------------------------------------------------------------------

#: 长势等级名称（与分级阈值一一对应：4 级 = 3 个阈值 + 1）
CLASS_NAMES: tuple[str, ...] = (
    "裸地/低覆盖",
    "较差长势古茶林",
    "中等长势古茶林",
    "优良长势古茶林",
)

#: 默认配色：浅黄 -> 浅橙 -> 浅绿 -> 深绿
CLASS_COLORS: tuple[str, ...] = ("#F6E7A8", "#F2B36B", "#9CD08C", "#1E7A45")

#: 默认分级阈值（FVC）
DEFAULT_BREAKS: tuple[float, ...] = (0.20, 0.40, 0.70)

#: 输出浮点栅格时使用的 NoData 值（GeoTIFF 中 -9999 比 NaN 兼容性更好）
FLOAT_NODATA: float = -9999.0

#: 默认分位数：裸土取 5%，纯植被取 95%
DEFAULT_SOIL_PERCENTILE: float = 5.0
DEFAULT_VEG_PERCENTILE: float = 95.0

_PLOT_READY = False


# ----------------------------------------------------------------------------
# 数据结构
# ----------------------------------------------------------------------------


@dataclass
class BandStack:
    """一幅已读取到内存的双波段影像（含地理参考信息）。"""

    red: np.ndarray
    nir: np.ndarray
    transform: object  # affine.Affine
    crs: str | None
    nodata: float | None = None
    descriptions: tuple[str | None, ...] = ()
    source_name: str = ""
    tags: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.red.shape[0])

    @property
    def width(self) -> int:
        return int(self.red.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def pixel_size_m(self) -> float:
        """像元边长（米）的近似值。"""
        return float(math.sqrt(pixel_area_m2(self.transform, self.crs, self.height, self.width)))


@dataclass
class FvcResult:
    """一次完整反演的结果集合。"""

    ndvi: np.ndarray
    fvc: np.ndarray
    classes: np.ndarray
    stats: pd.DataFrame
    breaks: tuple[float, ...]
    ndvi_soil: float
    ndvi_veg: float
    auto_soil: float
    auto_veg: float
    valid_pixels: int
    total_pixels: int
    mean_fvc: float
    pixel_area: float
    class_names: tuple[str, ...] = CLASS_NAMES
    crs: str | None = None
    transform: object = None

    @property
    def area_km2(self) -> float:
        return float(self.valid_pixels * self.pixel_area / 1e6)


# ----------------------------------------------------------------------------
# 读取影像
# ----------------------------------------------------------------------------


@contextmanager
def open_raster(source):
    """打开栅格。source 可以是路径，也可以是内存中的字节流（Streamlit 上传）。"""
    import rasterio

    if isinstance(source, (bytes, bytearray, memoryview)):
        with rasterio.MemoryFile(bytes(source)) as memfile:
            with memfile.open() as ds:
                yield ds
    else:
        with rasterio.open(str(source)) as ds:
            yield ds


def raster_info(source) -> dict:
    """只读取头文件信息，用于界面上的波段选择。"""
    with open_raster(source) as ds:
        try:
            bounds = tuple(float(v) for v in ds.bounds)
        except Exception:  # pragma: no cover - 极端异常的影像
            bounds = None
        return {
            "count": int(ds.count),
            "width": int(ds.width),
            "height": int(ds.height),
            "dtype": str(ds.dtypes[0]) if ds.count else "",
            "crs": ds.crs.to_string() if ds.crs else None,
            "descriptions": tuple(ds.descriptions),
            "nodata": None if ds.nodata is None else float(ds.nodata),
            "bounds": bounds,
            "tags": dict(ds.tags()),
        }


def read_bands(source, red_band: int = 1, nir_band: int = 2) -> BandStack:
    """
    读取红波段和近红外波段。

    参数
    ----
    source : 路径或 bytes
    red_band, nir_band : 波段序号（从 1 开始，与 GDAL/QGIS 一致）

    说明
    ----
    * 使用 masked 读取，自动把 NoData 转为 NaN；
    * 不做反射率换算，因为 NDVI 是比值，缩放不影响结果。
    """
    with open_raster(source) as ds:
        if ds.count < 2:
            raise ValueError(
                f"影像只有 {ds.count} 个波段，本工具需要至少 2 个波段（红、近红外）。"
            )
        for idx in (red_band, nir_band):
            if not 1 <= idx <= ds.count:
                raise ValueError(f"波段序号 {idx} 超出范围（1~{ds.count}）。")
        if red_band == nir_band:
            raise ValueError("红波段与近红外波段不能是同一个波段。")

        red = ds.read(red_band, masked=True).astype("float32")
        nir = ds.read(nir_band, masked=True).astype("float32")
        red = np.ma.filled(red, np.nan)
        nir = np.ma.filled(nir, np.nan)

        stack = BandStack(
            red=red,
            nir=nir,
            transform=ds.transform,
            crs=ds.crs.to_string() if ds.crs else None,
            nodata=None if ds.nodata is None else float(ds.nodata),
            descriptions=tuple(ds.descriptions),
            source_name=getattr(source, "name", None) or str(source)[:80],
            tags=dict(ds.tags()),
        )
    return stack


# ----------------------------------------------------------------------------
# 核心算法
# ----------------------------------------------------------------------------


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    计算 NDVI，无效像元（NoData、0、负值、非有限值）设为 NaN。

    NDVI = (NIR - Red) / (NIR + Red)
    """
    red = np.asarray(red, dtype="float32")
    nir = np.asarray(nir, dtype="float32")
    if red.shape != nir.shape:
        raise ValueError("红波段与近红外波段的尺寸不一致。")

    valid = np.isfinite(red) & np.isfinite(nir) & (red > 0) & (nir > 0)
    ndvi = np.full(red.shape, np.nan, dtype="float32")
    np.divide(nir - red, nir + red, out=ndvi, where=valid)
    return np.clip(ndvi, -1.0, 1.0)


def mask_water(ndvi: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """把水体/阴影（NDVI < 阈值）设为 NaN，避免拉低 NDVI_soil 分位数。"""
    out = ndvi.copy()
    out[out < threshold] = np.nan
    return out


def ndvi_endmembers(
    ndvi: np.ndarray,
    p_soil: float = DEFAULT_SOIL_PERCENTILE,
    p_veg: float = DEFAULT_VEG_PERCENTILE,
) -> tuple[float, float]:
    """
    用有效 NDVI 直方图的分位数确定纯裸土与纯植被端元。

    5% 分位数约等于“最像裸土”的像元，95% 分位数约等于“最像纯植被”的像元，
    可以自动剔除云、水、阴影等极端值。
    """
    values = ndvi[np.isfinite(ndvi)]
    if values.size < 100:
        raise ValueError(
            f"有效像元只有 {values.size} 个，无法稳定统计分位数，请检查影像或掩膜设置。"
        )
    soil, veg = np.percentile(values, [p_soil, p_veg]).astype(float)
    return float(soil), float(veg)


def compute_fvc(ndvi: np.ndarray, ndvi_soil: float, ndvi_veg: float) -> np.ndarray:
    """像元二分模型：FVC = (NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)，并截断到 0~1。"""
    if not np.isfinite(ndvi_soil) or not np.isfinite(ndvi_veg):
        raise ValueError("NDVI_soil / NDVI_veg 必须是有效数字。")
    if ndvi_veg - ndvi_soil < 1e-6:
        raise ValueError(
            f"NDVI_veg ({ndvi_veg:.3f}) 必须大于 NDVI_soil ({ndvi_soil:.3f})，"
            "请调整分位数或手动输入端元值。"
        )

    fvc = (ndvi.astype("float32") - np.float32(ndvi_soil)) / np.float32(ndvi_veg - ndvi_soil)
    fvc = np.clip(fvc, 0.0, 1.0)
    fvc[~np.isfinite(ndvi)] = np.nan
    return fvc


def classify_fvc(
    fvc: np.ndarray,
    breaks: Sequence[float] = DEFAULT_BREAKS,
    names: Sequence[str] = CLASS_NAMES,
) -> np.ndarray:
    """
    FVC 重分类为长势等级：0 表示无效，1~4 表示长势等级。

    规则（与作业说明书一致，左闭右开）：
        1: 0 <= FVC < b1
        2: b1 <= FVC < b2
        3: b2 <= FVC < b3
        4: b3 <= FVC <= 1
    """
    breaks = tuple(float(b) for b in breaks)
    if len(breaks) != len(names) - 1:
        raise ValueError("分级阈值个数必须比等级数少 1。")
    if any(b2 <= b1 for b1, b2 in zip(breaks[:-1], breaks[1:])):
        raise ValueError("分级阈值必须严格递增。")
    if breaks[0] <= 0.0 or breaks[-1] >= 1.0:
        raise ValueError("分级阈值应位于 0~1 之间。")

    classes = np.zeros(np.shape(fvc), dtype="uint8")
    ok = np.isfinite(fvc)
    # 用 float64 比较，避免 float32 的 0.2 变成 0.20000000298 导致边界像元错级
    classes[ok] = (
        np.digitize(fvc[ok].astype("float64"), np.asarray(breaks, dtype="float64"), right=False) + 1
    )
    return classes


def class_range_labels(breaks: Sequence[float] = DEFAULT_BREAKS) -> list[str]:
    """生成 '0.00 ≤ FVC < 0.20' 这样的区间文字。"""
    edges = [0.0] + [float(b) for b in breaks] + [1.0]
    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        left = "≤" if i > 0 else "≤"
        right = "≤" if i == len(edges) - 2 else "<"
        labels.append(f"{lo:.2f} {left} FVC {right} {hi:.2f}")
    return labels


def pixel_area_m2(transform, crs: str | None, height: int, width: int) -> float:
    """
    计算单个像元的真实面积（平方米）。

    * 投影坐标系：|a*e - b*d|；
    * 地理坐标系：按影像中心纬度把经纬度差换算成米。
    """
    if transform is None:
        return 100.0
    det = abs(float(transform.a) * float(transform.e) - float(transform.b) * float(transform.d))
    if det <= 0:
        return 100.0

    s = (crs or "").upper()
    is_geographic = any(k in s for k in ("4326", "CRS84", "LATLONG", "GEOGCS"))
    if not is_geographic:
        return det

    lat = float(transform.f) + float(transform.e) * (height / 2.0)
    dlon = abs(float(transform.a))
    dlat = abs(float(transform.e))
    meters_per_deg_lat = 110574.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
    return dlon * meters_per_deg_lon * dlat * meters_per_deg_lat


def class_table(
    classes: np.ndarray,
    transform,
    crs: str | None,
    breaks: Sequence[float] = DEFAULT_BREAKS,
    names: Sequence[str] = CLASS_NAMES,
) -> tuple[pd.DataFrame, dict]:
    """统计每一长势等级的像元数、真实面积与占比，返回 (表格, 汇总信息)。"""
    breaks = tuple(float(b) for b in breaks)
    names = tuple(names)
    classes = np.asarray(classes)
    height, width = classes.shape
    counts_all = np.bincount(classes.ravel(), minlength=len(names) + 1)
    counts = counts_all[1 : len(names) + 1].astype(np.int64)

    px_area = pixel_area_m2(transform, crs, height, width)
    valid_px = int(counts.sum())
    total_area_km2 = valid_px * px_area / 1e6

    labels = class_range_labels(breaks)
    rows = []
    for i, name in enumerate(names):
        area_km2 = counts[i] * px_area / 1e6
        rows.append(
            {
                "等级": i + 1,
                "长势类型": name,
                "FVC 区间": labels[i],
                "像元数": int(counts[i]),
                "面积(km²)": round(area_km2, 4),
                "面积(公顷)": round(area_km2 * 100.0, 2),
                "面积(亩)": round(area_km2 * 1500.0, 1),
                "占比(%)": round(counts[i] / valid_px * 100.0, 2) if valid_px else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    meta = {
        "pixel_area_m2": px_area,
        "valid_pixels": valid_px,
        "total_pixels": int(classes.size),
        "valid_ratio": valid_px / classes.size if classes.size else 0.0,
        "total_area_km2": total_area_km2,
    }
    return df, meta


def table_with_total(df: pd.DataFrame) -> pd.DataFrame:
    """在统计表末尾追加“合计”行，便于打印与导出。"""
    if df.empty:
        return df
    total = {
        "等级": "合计",
        "长势类型": "合计",
        "FVC 区间": "",
        "像元数": int(df["像元数"].sum()),
        "面积(km²)": round(float(df["面积(km²)"].sum()), 4),
        "面积(公顷)": round(float(df["面积(公顷)"].sum()), 2),
        "面积(亩)": round(float(df["面积(亩)"].sum()), 1),
        "占比(%)": round(float(df["占比(%)"].sum()), 2),
    }
    out = df.copy()
    out["等级"] = out["等级"].astype(str)
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """把表格转成 Markdown（不依赖 tabulate）。"""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines) + "\n"


def run_pipeline(
    stack: BandStack,
    p_soil: float = DEFAULT_SOIL_PERCENTILE,
    p_veg: float = DEFAULT_VEG_PERCENTILE,
    ndvi_soil: float | None = None,
    ndvi_veg: float | None = None,
    breaks: Sequence[float] = DEFAULT_BREAKS,
    remove_water: bool = True,
    names: Sequence[str] = CLASS_NAMES,
) -> FvcResult:
    """
    一键反演：NDVI -> 端元 -> FVC -> 分级 -> 统计。

    ndvi_soil / ndvi_veg 为 None 时使用分位数自动值，否则使用手动输入值。
    """
    ndvi = compute_ndvi(stack.red, stack.nir)
    if remove_water:
        ndvi = mask_water(ndvi, 0.0)

    auto_soil, auto_veg = ndvi_endmembers(ndvi, p_soil, p_veg)
    soil = auto_soil if ndvi_soil is None else float(ndvi_soil)
    veg = auto_veg if ndvi_veg is None else float(ndvi_veg)

    fvc = compute_fvc(ndvi, soil, veg)
    classes = classify_fvc(fvc, breaks, names)
    stats, meta = class_table(classes, stack.transform, stack.crs, breaks, names)

    finite = np.isfinite(fvc)
    mean_fvc = float(np.nanmean(fvc)) if finite.any() else float("nan")

    return FvcResult(
        ndvi=ndvi,
        fvc=fvc,
        classes=classes,
        stats=stats,
        breaks=tuple(float(b) for b in breaks),
        ndvi_soil=soil,
        ndvi_veg=veg,
        auto_soil=auto_soil,
        auto_veg=auto_veg,
        valid_pixels=int(meta["valid_pixels"]),
        total_pixels=int(meta["total_pixels"]),
        mean_fvc=mean_fvc,
        pixel_area=float(meta["pixel_area_m2"]),
        class_names=tuple(names),
        crs=stack.crs,
        transform=stack.transform,
    )


# ----------------------------------------------------------------------------
# 显示辅助
# ----------------------------------------------------------------------------


def stretch(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """百分位拉伸到 0~1，用于灰度图/假彩色显示。"""
    arr = np.asarray(arr, dtype="float32")
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return np.zeros(arr.shape, dtype="float32")
    lo, hi = np.percentile(values, [p_low, p_high])
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def decimate(arr: np.ndarray, max_dim: int = 1200) -> tuple[np.ndarray, int]:
    """按步长抽样，返回 (缩小后的数组, 抽样步长)，仅用于显示，不影响统计。"""
    arr = np.asarray(arr)
    step = max(1, int(math.ceil(max(arr.shape) / float(max_dim))))
    if step == 1:
        return arr, 1
    return arr[::step, ::step], step


# ----------------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------------


def _setup_mpl() -> None:
    global _PLOT_READY
    if _PLOT_READY:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    _PLOT_READY = True


def _add_scalebar(ax, pixel_size_m: float | None, width_px: int, height_px: int) -> None:
    """在图上叠加比例尺（根据像元大小自动取整）。"""
    if not pixel_size_m or pixel_size_m <= 0:
        return
    target = width_px * pixel_size_m * 0.22
    if target <= 0:
        return
    exp = math.floor(math.log10(target))
    base = target / (10 ** exp)
    for nice in (1, 2, 5, 10):
        if base <= nice:
            bar_m = nice * (10 ** exp)
            break
    else:
        bar_m = 10 ** (exp + 1)
    bar_px = bar_m / pixel_size_m
    x0 = width_px * 0.05
    y0 = height_px * 0.95
    ax.plot([x0, x0 + bar_px], [y0, y0], color="black", lw=5, solid_capstyle="butt")
    ax.plot(
        [x0, x0, x0 + bar_px, x0 + bar_px],
        [y0 - height_px * 0.012, y0 + height_px * 0.012, y0 + height_px * 0.012, y0 - height_px * 0.012],
        color="black",
        lw=1.2,
    )
    label = f"{bar_m/1000:g} km" if bar_m >= 1000 else f"{bar_m:g} m"
    ax.text(x0 + bar_px / 2, y0 - height_px * 0.03, label, ha="center", va="bottom", fontsize=9)


def _add_north_arrow(ax) -> None:
    ax.annotate(
        "N",
        xy=(0.94, 0.90),
        xytext=(0.94, 0.80),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        arrowprops=dict(facecolor="black", width=1.5, headwidth=8, headlength=10),
    )


def plot_preview(red: np.ndarray, nir: np.ndarray, source_name: str = ""):
    """原始影像预览：红波段、近红外波段、假彩色合成。"""
    _setup_mpl()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    axes[0].imshow(stretch(red), cmap="gray")
    axes[0].set_title("红波段 B4（2%~98% 拉伸）", fontsize=11)
    axes[1].imshow(stretch(nir), cmap="gray")
    axes[1].set_title("近红外波段 B8（2%~98% 拉伸）", fontsize=11)

    false_color = np.dstack([stretch(nir), stretch(red), stretch(red)])
    axes[2].imshow(false_color)
    axes[2].set_title("假彩色合成（R=NIR, G=Red, B=Red）", fontsize=11)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if source_name:
        fig.suptitle(f"原始影像预览  |  {source_name}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_ndvi_map(ndvi: np.ndarray, pixel_size_m: float | None = None):
    """NDVI 分布图。"""
    _setup_mpl()
    import matplotlib.pyplot as plt

    disp = np.ma.masked_invalid(ndvi)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(disp, cmap="RdYlGn", vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("NDVI")
    ax.set_title("归一化植被指数 NDVI", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    _add_scalebar(ax, pixel_size_m, ndvi.shape[1], ndvi.shape[0])
    _add_north_arrow(ax)
    fig.tight_layout()
    return fig


def plot_fvc_map(fvc: np.ndarray, pixel_size_m: float | None = None):
    """FVC 分布图。"""
    _setup_mpl()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "fvc", ["#FFF8E7", "#FDE3AE", "#C9E6B0", "#63B37A", "#14532D"]
    )
    disp = np.ma.masked_invalid(fvc)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("FVC")
    ax.set_title("植被覆盖度 FVC（像元二分模型）", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    _add_scalebar(ax, pixel_size_m, fvc.shape[1], fvc.shape[0])
    _add_north_arrow(ax)
    fig.tight_layout()
    return fig


def plot_class_map(
    classes: np.ndarray,
    breaks: Sequence[float] = DEFAULT_BREAKS,
    colors: Sequence[str] = CLASS_COLORS,
    names: Sequence[str] = CLASS_NAMES,
    pixel_size_m: float | None = None,
    title: str = "古茶林长势分级专题图",
):
    """长势等级专题图（含图例、比例尺、指北针）。"""
    _setup_mpl()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(list(colors))
    cmap.set_bad("#FFFFFF")
    disp = np.ma.masked_where(np.asarray(classes) <= 0, np.asarray(classes, dtype="float32"))

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.imshow(disp, cmap=cmap, vmin=0.5, vmax=len(names) + 0.5, interpolation="nearest")

    labels = class_range_labels(breaks)
    handles = [
        Patch(facecolor=colors[i], edgecolor="#555555", label=f"{i+1}  {names[i]}（{labels[i]}）")
        for i in range(len(names))
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.92,
        title="长势等级",
        title_fontsize=9,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    _add_scalebar(ax, pixel_size_m, classes.shape[1], classes.shape[0])
    _add_north_arrow(ax)
    fig.tight_layout()
    return fig


def plot_stats_bar(
    stats: pd.DataFrame,
    colors: Sequence[str] = CLASS_COLORS,
    value_col: str = "面积(km²)",
):
    """各级面积柱状图，柱顶标注占比。"""
    _setup_mpl()
    import matplotlib.pyplot as plt

    df = stats
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(df))
    bars = ax.bar(x, df[value_col].to_numpy(dtype=float), color=list(colors), edgecolor="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['等级']}\n{r['长势类型']}" for _, r in df.iterrows()], fontsize=9)
    ax.set_ylabel("面积（km²）")
    ax.set_title("各长势等级面积统计", fontsize=13)
    ymax = float(df[value_col].max()) if len(df) else 0
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{row['面积(km²)']:.2f} km²\n{row['占比(%)']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_ndvi_hist(
    ndvi: np.ndarray,
    ndvi_soil: float,
    ndvi_veg: float,
    p_soil: float = DEFAULT_SOIL_PERCENTILE,
    p_veg: float = DEFAULT_VEG_PERCENTILE,
):
    """NDVI 直方图与端元取值位置。"""
    _setup_mpl()
    import matplotlib.pyplot as plt

    values = ndvi[np.isfinite(ndvi)]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(values, bins=120, range=(-1, 1), color="#7FB77E", edgecolor="white", linewidth=0.3)
    ax.axvline(ndvi_soil, color="#C0392B", linestyle="--", lw=1.6,
               label=f"NDVI_soil = {ndvi_soil:.3f}（{p_soil:g}% 分位）")
    ax.axvline(ndvi_veg, color="#1E6B3A", linestyle="--", lw=1.6,
               label=f"NDVI_veg = {ndvi_veg:.3f}（{p_veg:g}% 分位）")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("像元数")
    ax.set_title("NDVI 直方图与端元自动取值", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_summary(stack: BandStack, result: FvcResult, pixel_size_m: float | None = None):
    """四联图：假彩色、NDVI、FVC、长势分级（用于报告与答辩 PPT）。"""
    _setup_mpl()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    dec = min(stack.height, stack.width)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 11.6))

    false_color = np.dstack([stretch(stack.nir), stretch(stack.red), stretch(stack.red)])
    axes[0, 0].imshow(false_color)
    axes[0, 0].set_title("(a) Sentinel-2 假彩色合成", fontsize=11)

    im1 = axes[0, 1].imshow(np.ma.masked_invalid(result.ndvi), cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0, 1].set_title("(b) NDVI", fontsize=11)
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.75, pad=0.02)

    cmap_fvc = LinearSegmentedColormap.from_list(
        "fvc", ["#FFF8E7", "#FDE3AE", "#C9E6B0", "#63B37A", "#14532D"]
    )
    im2 = axes[1, 0].imshow(np.ma.masked_invalid(result.fvc), cmap=cmap_fvc, vmin=0, vmax=1)
    axes[1, 0].set_title("(c) FVC（像元二分模型）", fontsize=11)
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.75, pad=0.02)

    cmap_cls = ListedColormap(list(CLASS_COLORS))
    cmap_cls.set_bad("#FFFFFF")
    disp = np.ma.masked_where(result.classes <= 0, result.classes.astype("float32"))
    axes[1, 1].imshow(disp, cmap=cmap_cls, vmin=0.5, vmax=len(CLASS_NAMES) + 0.5, interpolation="nearest")
    labels = class_range_labels(result.breaks)
    handles = [
        Patch(facecolor=CLASS_COLORS[i], edgecolor="#555555",
              label=f"{i+1} {CLASS_NAMES[i]}（{labels[i]}）")
        for i in range(len(CLASS_NAMES))
    ]
    axes[1, 1].legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.9)
    axes[1, 1].set_title("(d) 古茶林长势分级", fontsize=11)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"景迈山古茶林 FVC 与长势分级结果　|　平均 FVC={result.mean_fvc:.2f}，"
        f"优良长势占比={result.stats.loc[3, '占比(%)']:.1f}%",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ----------------------------------------------------------------------------
# 成果输出
# ----------------------------------------------------------------------------


def _profile_for(data: np.ndarray, transform, crs, dtype: str, nodata) -> dict:
    height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    if min(height, width) >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    return profile


def array_to_geotiff_bytes(
    data: np.ndarray,
    transform,
    crs: str | None,
    dtype: str = "float32",
    nodata=None,
    description: str | None = None,
) -> bytes:
    """把数组写成内存中的 GeoTIFF 字节流，供 Streamlit 下载。"""
    import rasterio
    from rasterio.io import MemoryFile

    profile = _profile_for(data, transform, crs, dtype, nodata)
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(np.asarray(data, dtype=dtype), 1)
            if description:
                dst.set_band_description(1, description)
        return memfile.read()


def write_geotiff(
    path: str | Path,
    data: np.ndarray,
    transform,
    crs: str | None,
    dtype: str = "float32",
    nodata=None,
    description: str | None = None,
) -> str:
    """把数组写成 GeoTIFF 文件（保持原影像的坐标与仿射变换）。"""
    import rasterio

    profile = _profile_for(data, transform, crs, dtype, nodata)
    path = str(path)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.asarray(data, dtype=dtype), 1)
        if description:
            dst.set_band_description(1, description)
    return path


def save_figure(fig, path: str | Path, dpi: int = 200) -> str:
    """保存图片（白底、紧凑边距）。"""
    path = str(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def export_all(
    stack: BandStack,
    result: FvcResult,
    outdir: str | Path,
    dpi: int = 200,
    make_plots: bool = True,
) -> dict:
    """
    导出全部成果：FVC 栅格、分级栅格、统计 CSV、四类专题图、汇总 JSON。
    返回 {类别: 文件路径}。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    fvc_tif = np.where(np.isfinite(result.fvc), result.fvc, FLOAT_NODATA).astype("float32")
    outputs["fvc_tif"] = write_geotiff(
        outdir / "FVC.tif", fvc_tif, stack.transform, stack.crs,
        dtype="float32", nodata=FLOAT_NODATA, description="FVC (0-1)",
    )
    outputs["class_tif"] = write_geotiff(
        outdir / "GrowthClass.tif", result.classes, stack.transform, stack.crs,
        dtype="uint8", nodata=0, description="Growth class 1-4",
    )
    outputs["stats_csv"] = str(outdir / "FVC_statistics.csv")
    table = table_with_total(result.stats)
    table.to_csv(outputs["stats_csv"], index=False, encoding="utf-8-sig")

    outputs["stats_table"] = str(outdir / "FVC_statistics.md")
    with open(outputs["stats_table"], "w", encoding="utf-8") as f:
        f.write(_df_to_markdown(table))

    if make_plots:
        import matplotlib.pyplot as plt

        fig_specs = {
            "preview_png": (plot_preview(stack.red, stack.nir), "原始影像预览.png"),
            "ndvi_png": (plot_ndvi_map(result.ndvi, stack.pixel_size_m), "NDVI.png"),
            "fvc_png": (plot_fvc_map(result.fvc, stack.pixel_size_m), "FVC.png"),
            "class_png": (plot_class_map(result.classes, result.breaks,
                                         pixel_size_m=stack.pixel_size_m), "长势分级专题图.png"),
            "stats_png": (plot_stats_bar(result.stats), "面积统计.png"),
            "summary_png": (plot_summary(stack, result), "四联图.png"),
        }
        for key, (fig, filename) in fig_specs.items():
            outputs[key] = save_figure(fig, outdir / filename, dpi=dpi)
            plt.close(fig)

    summary = {
        "source": stack.source_name,
        "ndvi_soil": round(result.ndvi_soil, 4),
        "ndvi_veg": round(result.ndvi_veg, 4),
        "auto_ndvi_soil": round(result.auto_soil, 4),
        "auto_ndvi_veg": round(result.auto_veg, 4),
        "mean_fvc": round(result.mean_fvc, 4),
        "breaks": list(result.breaks),
        "valid_pixels": result.valid_pixels,
        "total_pixels": result.total_pixels,
        "pixel_area_m2": round(result.pixel_area, 3),
        "total_area_km2": round(result.area_km2, 4),
        "crs": result.crs,
        "stats": result.stats.to_dict(orient="records"),
    }
    outputs["summary_json"] = str(outdir / "summary.json")
    with open(outputs["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return outputs


# ----------------------------------------------------------------------------
# 控制台入口（阶段2：先跑通命令行版本）
# ----------------------------------------------------------------------------


def _parse_args(argv: Iterable[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="景迈山 FVC 估算与古茶林长势分级（命令行版）",
    )
    parser.add_argument("input", help="双波段 GeoTIFF（B4 红、B8 近红外）")
    parser.add_argument("-o", "--outdir", default="outputs", help="输出目录")
    parser.add_argument("--red-band", type=int, default=1, help="红波段序号（默认 1）")
    parser.add_argument("--nir-band", type=int, default=2, help="近红外波段序号（默认 2）")
    parser.add_argument("--p-soil", type=float, default=DEFAULT_SOIL_PERCENTILE)
    parser.add_argument("--p-veg", type=float, default=DEFAULT_VEG_PERCENTILE)
    parser.add_argument("--ndvi-soil", type=float, default=None)
    parser.add_argument("--ndvi-veg", type=float, default=None)
    parser.add_argument("--breaks", type=float, nargs=3, default=list(DEFAULT_BREAKS),
                        help="三个分级阈值，默认 0.2 0.4 0.7")
    parser.add_argument("--keep-water", action="store_true", help="保留水体（默认掩膜去掉 NDVI<0）")
    parser.add_argument("--no-plots", action="store_true", help="不输出图片")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    print(f"[1/4] 读取影像：{args.input}")
    stack = read_bands(args.input, args.red_band, args.nir_band)
    print(f"      尺寸 {stack.width} x {stack.height} 像元，像元面积约 {stack.pixel_size_m:.1f} m")

    print("[2/4] NDVI / FVC / 分级计算 ...")
    result = run_pipeline(
        stack,
        p_soil=args.p_soil,
        p_veg=args.p_veg,
        ndvi_soil=args.ndvi_soil,
        ndvi_veg=args.ndvi_veg,
        breaks=tuple(args.breaks),
        remove_water=not args.keep_water,
    )
    print(f"      NDVI_soil={result.ndvi_soil:.4f}  NDVI_veg={result.ndvi_veg:.4f}  "
          f"平均 FVC={result.mean_fvc:.3f}")

    print("[3/4] 面积统计 ...")
    print(table_with_total(result.stats).to_string(index=False))

    print(f"[4/4] 导出成果到：{args.outdir}")
    outputs = export_all(stack, result, args.outdir, make_plots=not args.no_plots)
    for key, path in outputs.items():
        print(f"      {key:<12} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
