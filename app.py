# -*- coding: utf-8 -*-
"""
普洱景迈山植被覆盖度（FVC）估算与古茶林长势分级工具 —— Streamlit 网页版
======================================================================

启动方式：
    streamlit run app.py

界面布局：
    侧边栏：数据上传 / 参数设置 / 分级阈值 / 运行按钮
    主页面：原始影像、NDVI、FVC、长势分级专题图、面积统计与成果下载
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import fvc_core as fc  # noqa: E402

REAL_TIF = BASE_DIR / "data" / "jingmai_s2_20250217_B4B8.tif"
DEMO_TIF = BASE_DIR / "demo_data" / "jingmai_demo_s2_B4B8.tif"
# 优先使用真实的景迈山 Sentinel-2 场景，没有时退回合成演示数据
SAMPLE_TIF = REAL_TIF if REAL_TIF.exists() else DEMO_TIF
SAMPLE_BYTES = SAMPLE_TIF.read_bytes() if SAMPLE_TIF.exists() else None
SAMPLE_IS_REAL = SAMPLE_TIF == REAL_TIF

TS_DIR = BASE_DIR / "outputs" / "timeseries"
TS_CSV = TS_DIR / "annual_stats.csv"
WET_CSV = BASE_DIR / "outputs" / "wetland" / "wetland_stats.csv"
HERITAGE_IMG = BASE_DIR / "docs" / "figures" / "heritage_landscape.png"

MAX_DISPLAY_PIXELS = 1200  # 显示时最长边最多 1200 像素，保证页面流畅

st.set_page_config(
    page_title="景迈山古茶林长势分级工具",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
      h1 {font-size: 1.85rem !important; margin-bottom: .2rem;}
      div[data-testid="stMetricValue"] {font-size: 1.45rem;}
      div[data-testid="stMetricLabel"] {font-size: .85rem;}
      .stTabs [data-baseweb="tab"] {font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 缓存函数：避免每次交互都重新读影像 / 重算
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_info(data: bytes) -> dict:
    return fc.raster_info(data)


@st.cache_data(show_spinner=False)
def cached_stack(data: bytes, red_band: int, nir_band: int) -> fc.BandStack:
    return fc.read_bands(data, red_band, nir_band)


@st.cache_data(show_spinner="正在计算 NDVI、FVC 并统计面积 ...")
def cached_result(
    data: bytes,
    red_band: int,
    nir_band: int,
    p_soil: float,
    p_veg: float,
    ndvi_soil: float | None,
    ndvi_veg: float | None,
    breaks: tuple[float, ...],
    remove_water: bool,
) -> fc.FvcResult:
    stack = fc.read_bands(data, red_band, nir_band)
    return fc.run_pipeline(
        stack,
        p_soil=p_soil,
        p_veg=p_veg,
        ndvi_soil=ndvi_soil,
        ndvi_veg=ndvi_veg,
        breaks=breaks,
        remove_water=remove_water,
    )


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------


def show_figure(fig) -> None:
    st.pyplot(fig, width="stretch")
    import matplotlib.pyplot as plt

    plt.close(fig)


def figure_png(fig, dpi: int = 180) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.clear()
    return buf.getvalue()


def download_tif_bytes(result: fc.FvcResult, stack: fc.BandStack, kind: str) -> bytes:
    if kind == "fvc":
        arr = np.where(np.isfinite(result.fvc), result.fvc, fc.FLOAT_NODATA).astype("float32")
        return fc.array_to_geotiff_bytes(
            arr, stack.transform, stack.crs, dtype="float32",
            nodata=fc.FLOAT_NODATA, description="FVC (0-1)",
        )
    return fc.array_to_geotiff_bytes(
        result.classes, stack.transform, stack.crs, dtype="uint8",
        nodata=0, description="Growth class 1-4",
    )


def guess_bands(info: dict) -> tuple[int, int]:
    """根据波段描述猜 B4 / B8，猜不到就用 1 / 2。"""
    desc = [str(d).upper() if d else "" for d in info.get("descriptions", ())]
    red_idx, nir_idx = 1, 2

    def find(keywords, default):
        for i, d in enumerate(desc):
            if any(k in d for k in keywords):
                return i + 1
        return default

    red_idx = find(("B4", "B04", "RED", "红"), 1)
    nir_idx = find(("B8", "B08", "NIR", "近红外"), 2)
    if red_idx == nir_idx:
        red_idx, nir_idx = 1, 2
    return red_idx, nir_idx


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def render_timeseries() -> None:
    """十年变化标签页：读取 build_timeseries.py 生成的年度统计与图件。"""
    st.subheader("近十年旱季 FVC 与长势变化")
    if not TS_CSV.exists():
        st.info(
            "尚未生成年际数据。请在项目目录运行：\n\n"
            "`python build_timeseries.py`\n\n"
            "脚本会自动下载 2017—2026 年旱季影像并生成统计表与图件。"
        )
        return

    df = pd.read_csv(TS_CSV)
    if df.empty:
        st.warning("年际统计表为空。")
        return

    years = df["年份"].to_numpy()
    fixed = df["平均FVC(固定端元)"].to_numpy(dtype=float)
    n = len(df)
    slope10 = float(np.polyfit(years, fixed, 1)[0] * 10) if n >= 2 else float("nan")
    best = df.loc[df["平均FVC(固定端元)"].idxmax()]
    worst = df.loc[df["平均FVC(固定端元)"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("统计年份", f"{int(years.min())}—{int(years.max())}（{n} 年）")
    c2.metric("十年平均 FVC", f"{fixed.mean():.3f}")
    c3.metric("变化趋势", f"{slope10:+.3f} / 10 年")
    c4.metric("最高 / 最低", f"{int(best['年份'])} / {int(worst['年份'])}")

    st.caption(
        "固定端元（NDVI_soil=0.10，NDVI_veg=0.90）用于保证年际可比；"
        "逐年 5%/95% 端元结果作为对照。云、云影、雪已用 SCL 掩膜剔除。"
    )

    trend_png = TS_DIR / "trend.png"
    if trend_png.exists():
        st.image(str(trend_png), width="stretch")
    st.caption("图：平均 FVC 年际变化（实线为逐年端元，虚线为固定端元）")

    class_png = TS_DIR / "class_trend.png"
    if class_png.exists():
        st.image(str(class_png), width="stretch")
    st.caption("图：四级长势面积占比的年际结构变化")

    st.markdown("#### 年度统计表")
    show_cols = [
        "年份", "成像日期", "云量(%)", "有效像元比例(%)",
        "平均FVC(自动端元)", "平均FVC(固定端元)",
        "占比1级(%)", "占比2级(%)", "占比3级(%)", "占比4级(%)",
    ]
    show = df[[c for c in show_cols if c in df.columns]]
    st.dataframe(show, hide_index=True, width="stretch")

    st.download_button(
        "⬇️ 下载年度统计表 (.csv)",
        TS_CSV.read_bytes(),
        "jingmai_FVC_annual_stats.csv",
        "text/csv",
        width="stretch",
    )

    maps_png = TS_DIR / "annual_class_maps.png"
    if maps_png.exists():
        st.markdown("#### 年度长势分级专题图")
        st.image(str(maps_png), width="stretch")
        st.caption("图：2017—2026 年旱季长势分级对比（白色为云掩膜或无效像元）")

    with st.expander("数据说明与引用", expanded=False):
        st.markdown(
            "- 数据源：Sentinel-2 L2A（10 m），每年 1—4 月云量最低的一景，优先 47QPE；\n"
            "- 云掩膜：SCL 类别 0、1、2、3、7、8、9、10、11 视为无效；\n"
            "- 固定端元：NDVI_soil=0.10、NDVI_veg=0.90，用于消除端元年际漂移；\n"
            "- 逐年端元：每年有效 NDVI 的 5%/95% 分位数，反映当年影像特征；\n"
            "- 每景的景号、日期、云量见上表，可用于论文数据说明。"
        )


def render_heritage() -> None:
    """遗产地概况：景迈山古茶林文化景观的价值、保护与产业本底。"""
    st.subheader("景迈山古茶林文化景观：林茶共生，人地和谐")
    st.caption(
        "世界文化遗产（2023）· 全球首个茶主题遗产地 · "
        "“绿水青山就是金山银山”实践创新基地（2023）"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("遗产地面积", "19 095.74 ha")
    c2.metric("遗产区面积", "7 167.89 ha")
    c3.metric("古茶林", "5 片")
    c4.metric("传统村落", "9 个")
    c5.metric("人口", "6 830 人")

    st.caption(
        "遗产地约 190.96 km²，遗产区约 71.68 km²；分布傣族、布朗族、哈尼族、佤族、汉族 5 个民族，"
        "共 1 699 户，另有 3 片分隔防护林。"
    )

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("#### 空间格局与生态本底")
        st.markdown(
            "- “神山—森林—茶林—村寨—耕地—水田—河流”垂直布局，"
            "“茶在森林中、村在茶林中、耕地在茶林外”；\n"
            "- 记录原生种子植物 134 科 524 属约 1 500 种，陆生脊椎动物 187 种，"
            "保护野生动物 150 种以上；\n"
            "- 森林、茶林、村寨、农田与水系相互穿插，构成完整的山地生态屏障。"
        )
    with right:
        if HERITAGE_IMG.exists():
            st.image(str(HERITAGE_IMG), width="stretch")
            st.caption("图：景迈山垂直景观格局示意（据公开资料绘制）")

    st.markdown("#### 世界遗产价值")
    st.markdown(
        "2023 年 9 月，景迈山古茶林文化景观在联合国教科文组织第 45 届世界遗产大会上列入《世界遗产名录》，"
        "成为全球首个茶主题世界文化遗产。遗产地由 **5 片古茶林、9 个传统村落、3 片分隔防护林** 构成，"
        "遗产区 7 167.89 ha，缓冲区 11 927.85 ha，合计 19 095.74 ha。\n\n"
        "其突出普遍价值在于：布朗族、傣族等世居民族在长期实践中创造了“林下茶种植”体系，"
        "保留乔木层—茶树层—草本层的立体群落，形成茶林、村寨、森林、耕地与水系相互依存的文化景观，"
        "是“人与自然和谐共生”的活态范例。"
    )
    st.markdown(
        "- **标准 (iii)**：为仍然存续的林下茶种植传统与茶文化提供独特见证；\n"
        "- **标准 (v)**：是山地茶林—村寨—森林土地利用和人类聚居的杰出范例。"
    )

    st.markdown("#### 历史脉络")
    st.markdown(
        """
<div style="border-left:3px solid #C9A227; margin:8px 0 6px 8px; padding-left:20px;">
  <div style="margin-bottom:16px;">
    <div style="font-weight:700;color:#1E5B3A;font-size:1.02rem;">约公元 10 世纪</div>
    <div style="color:#444;line-height:1.65;">布朗族先民迁居景迈山，传说茶祖帕哎冷带领族人种下第一片茶园，开创“林下茶”传统。</div>
  </div>
  <div style="margin-bottom:16px;">
    <div style="font-weight:700;color:#1E5B3A;font-size:1.02rem;">明 清</div>
    <div style="color:#444;line-height:1.65;">茶山与村落格局逐步定型，普洱茶贸易兴盛，景迈山茶叶经茶马古道外销。</div>
  </div>
  <div style="margin-bottom:16px;">
    <div style="font-weight:700;color:#1E5B3A;font-size:1.02rem;">2012 年</div>
    <div style="color:#444;line-height:1.65;">景迈山古茶林列入《中国世界文化遗产预备名单》，保护与申遗工作系统推进。</div>
  </div>
  <div style="margin-bottom:16px;">
    <div style="font-weight:700;color:#1E5B3A;font-size:1.02rem;">2023 年 9 月</div>
    <div style="color:#444;line-height:1.65;">列入《世界遗产名录》，适用标准 (iii)(v)，成为全球首个茶主题世界文化遗产。</div>
  </div>
  <div>
    <div style="font-weight:700;color:#1E5B3A;font-size:1.02rem;">2023 年 10 月</div>
    <div style="color:#444;line-height:1.65;">被生态环境部命名为“绿水青山就是金山银山”实践创新基地。</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 民族文化和传统知识")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("**民族与村落**")
        st.markdown(
            "傣族、布朗族、哈尼族、佤族、汉族 5 个民族共居，9 个传统村落分属景迈、芒景两个行政村。\n\n"
            "翁基（布朗族）、糯岗（傣族）等村寨保留干栏式民居、缅寺、寨心等传统要素，"
            "村寨与茶林、森林相互穿插。"
        )
    with m2:
        st.markdown("**茶祖与神山**")
        st.markdown(
            "布朗族尊奉茶祖 **帕哎冷**，每年祭茶祖、敬神山，形成社区自发的保护规范。\n\n"
            "村寨后山为“神山/竜林”，禁止砍伐，既是信仰空间，也是水源涵养地和生物多样性庇护所。"
        )
    with m3:
        st.markdown("**节庆与技艺**")
        st.markdown(
            "傣历新年泼水节、布朗族山康节（茶祖节）、新米节、关门节/开门节等延续至今。\n\n"
            "制茶沿用杀青、揉捻、晒青等手工技艺；林下茶不施化肥农药、保留遮阴树，"
            "以森林生态系统防治病虫害。"
        )

    with st.expander("📖 文化小词典", expanded=False):
        st.markdown(
            "- **林下茶**：在天然林冠下种植茶树，保留乔木层遮阴，形成“乔木—茶树—草本”立体结构。\n"
            "- **神山**：村寨后山，禁止砍伐与开垦，用于祭祀和水源保护。\n"
            "- **竜林**：傣族、布朗族等民族的神圣林地，通常保存较完整的原生植被。\n"
            "- **茶祖**：布朗族传说中的茶祖帕哎冷，被视为景迈山种茶始祖。\n"
            "- **山康节**：布朗族祭祀茶祖、祈求茶叶丰收的传统节日。\n"
            "- **干栏式建筑**：底层架空、上层住人的传统民居，适应湿热山地环境。\n"
            "- **茶马古道**：历史上普洱茶外运的贸易通道，景迈山茶叶由此走向各地。\n"
            "- **寨心**：村寨中心的祭祀与公共活动空间，是村落格局的核心。"
        )

    st.markdown("#### 文化景观的监测启示")
    monitor = pd.DataFrame(
        [
            ["5 片古茶林", "林冠覆盖、林下植被、茶树长势", "FVC、NDVI、分级面积"],
            ["9 个传统村落", "村落缓冲带植被、建设扩张", "缓冲带 FVC / NDVI"],
            ["3 片分隔防护林", "森林连通性、边界稳定", "FVC、NDVI"],
            ["神山 / 竜林", "郁闭度、人为干扰", "NDVI / FVC"],
            ["茶林—村寨—耕地格局", "地类边界、景观完整性", "需结合土地覆盖分类"],
        ],
        columns=["文化景观要素", "监测重点", "本工具可提供"],
    )
    st.dataframe(monitor, hide_index=True, width="stretch")
    st.info(
        "FVC 只反映植被覆盖维度。文化景观监测还应结合村落风貌、古树健康、"
        "节庆活态传承等指标；本工具可作为生态维度的快速补充。"
    )

    st.markdown("#### 小微湿地监测（补充）")
    if WET_CSV.exists():
        wet = pd.read_csv(WET_CSV)
        w25 = wet.loc[wet["year"] == 2025]
        if not w25.empty:
            w25 = w25.iloc[0]
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("2025 水体/湿地", f"{w25['water_ha']:.2f} ha")
            w2.metric("斑块数", f"{int(w25['patches'])} 个")
            w3.metric("缓冲带 FVC", f"{w25['buffer_fvc']:.3f}")
            w4.metric("十年范围", f"{wet['water_ha'].min():.2f}—{wet['water_ha'].max():.2f} ha")
            st.caption(
                "小微湿地按 NDVI < 0 提取，统计 ≥5 像元的斑块，缓冲带取水体周边 30 m；"
                "2017—2026 年面积与斑块数波动明显，水陆交错带 FVC 偏低，是管护重点。"
            )
    else:
        st.caption("运行 `python build_wetland.py` 可生成小微湿地统计与分布图。")

    st.markdown("#### 生态家底、保护管理与产业民生")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**生态家底**")
        st.markdown(
            "- 负氧离子月均 **7 000 个/cm³**\n"
            "- 南朗河水质优于 **Ⅲ类**\n"
            "- 澜沧县“中国天然氧吧”\n"
            "- 2 个环境空气质量手工监测点"
        )
    with b:
        st.markdown("**保护管理**")
        st.markdown(
            "- 1 600 株古树“一树一档”\n"
            "- 5 片古茶林消防全覆盖\n"
            "- 1 500 亩古茶树群环境修复\n"
            "- 800 亩有害生物防治\n"
            "- 166 个“同心互助组”"
        )
    with c:
        st.markdown("**产业民生**")
        st.markdown(
            "- 古树茶 **2.8 万亩**、生态茶 **6.5 万亩**\n"
            "- 茶企与合作社 **298 家**\n"
            "- 2025 年游客 **69.43 万人次**、营收 **5.63 亿元**\n"
            "- 酒店民宿 340 家、餐饮 145 家"
        )

    st.caption(
        "资料来源：UNESCO World Heritage Centre, No. 1665（2023）；"
        "普洱市人民政府《普洱市国家生态产品价值实现机制试点案例之十八｜"
        "林茶共生 人地和谐——澜沧县景迈山“两山”实践创新基地的探索与实践》，2026-09-18。"
    )


# ----------------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------------

st.sidebar.title("🍵 参数面板")

st.sidebar.subheader("① 数据输入")
options = ["上传影像", "示例数据"]
default_index = 1 if SAMPLE_BYTES is not None else 0
source = st.sidebar.radio(
    "数据来源",
    options,
    index=default_index,
    help=(
        "示例数据为景迈山 2025-02-17 的 Sentinel-2 L2A 真实影像（6 km × 6 km），"
        "用于快速体验流程。"
        if SAMPLE_IS_REAL
        else "示例数据为程序合成的 6 km × 6 km 演示影像，用于快速体验流程。"
    ),
)

data: bytes | None = None
file_name = ""
if source == "上传影像":
    uploaded = st.sidebar.file_uploader(
        "上传双波段 GeoTIFF（B4 红 / B8 近红外）",
        type=["tif", "tiff", "img", "vrt"],
        help="推荐使用 prep_sentinel2.py 裁剪好的研究区影像，文件小、计算快。",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        file_name = uploaded.name
else:
    data = SAMPLE_BYTES
    file_name = SAMPLE_TIF.name if SAMPLE_BYTES else ""

if data is None:
    st.sidebar.warning("请先上传一幅双波段 GeoTIFF 影像。")
    st.title("🍵 普洱景迈山植被覆盖度（FVC）估算与古茶林长势分级工具")
    st.info(
        "在左侧上传研究区影像，或在数据来源中选择「示例数据」快速体验。\n\n"
        "**输入要求**：Sentinel-2 L2A 裁剪后的双波段 GeoTIFF，第 1 波段 B4（红），第 2 波段 B8（近红外）。"
    )
    st.stop()

try:
    info = cached_info(data)
except Exception as exc:  # 文件损坏 / 不是栅格
    st.sidebar.error(f"影像读取失败：{exc}")
    st.stop()

st.sidebar.caption(
    f"{file_name} ｜ {human_bytes(len(data))} ｜ {info['width']}×{info['height']} 像元 ｜ {info['count']} 波段"
)
if info.get("crs"):
    st.sidebar.caption(f"坐标系：{info['crs']}")

desc = info.get("descriptions") or ()
band_labels = {
    i: (f"波段 {i + 1} · {desc[i]}" if i < len(desc) and desc[i] else f"波段 {i + 1}")
    for i in range(info["count"])
}
guess_red, guess_nir = guess_bands(info)
red_band = st.sidebar.selectbox(
    "红波段 B4", list(range(1, info["count"] + 1)),
    index=guess_red - 1, format_func=lambda i: band_labels[i - 1],
)
nir_band = st.sidebar.selectbox(
    "近红外波段 B8", list(range(1, info["count"] + 1)),
    index=guess_nir - 1, format_func=lambda i: band_labels[i - 1],
)
if red_band == nir_band:
    st.sidebar.error("红波段与近红外波段不能相同，请重新选择。")
    st.stop()
if info["count"] > 2:
    st.sidebar.caption("提示：多波段影像请确认 B4 / B8 的波段序号，选错会得到完全错误的 FVC。")

st.sidebar.divider()
st.sidebar.subheader("② 端元设置")
auto_endmembers = st.sidebar.checkbox(
    "自动取分位数（推荐）", value=True,
    help="5% 分位数作为 NDVI_soil，95% 分位数作为 NDVI_veg，可剔除极端噪声像元。",
)
p_soil = st.sidebar.slider("NDVI_soil 分位数 (%)", 1, 30, 5, 1, disabled=not auto_endmembers)
p_veg = st.sidebar.slider("NDVI_veg 分位数 (%)", 70, 99, 95, 1, disabled=not auto_endmembers)

ndvi_soil = ndvi_veg = None
if not auto_endmembers:
    ndvi_soil = st.sidebar.number_input(
        "手动 NDVI_soil", min_value=-1.0, max_value=1.0, value=0.10, step=0.01, format="%.3f"
    )
    ndvi_veg = st.sidebar.number_input(
        "手动 NDVI_veg", min_value=-1.0, max_value=1.0, value=0.85, step=0.01, format="%.3f"
    )
    st.sidebar.caption("手动值会覆盖分位数结果；建议先用自动值观察直方图再调整。")

remove_water = st.sidebar.checkbox(
    "掩膜水体（NDVI < 0）", value=True, help="把水塘、阴影等负值像元设为无效，不参与 FVC 统计。"
)

st.sidebar.divider()
st.sidebar.subheader("③ 长势分级阈值")
with st.sidebar.expander("自定义 FVC 分级阈值", expanded=False):
    t1 = st.number_input("阈值 1（裸地 / 较差）", 0.01, 0.98, float(fc.DEFAULT_BREAKS[0]), 0.01, format="%.2f")
    t2 = st.number_input("阈值 2（较差 / 中等）", 0.01, 0.98, float(fc.DEFAULT_BREAKS[1]), 0.01, format="%.2f")
    t3 = st.number_input("阈值 3（中等 / 优良）", 0.01, 0.98, float(fc.DEFAULT_BREAKS[2]), 0.01, format="%.2f")
breaks = (t1, t2, t3)

st.sidebar.caption("默认：0.20 / 0.40 / 0.70（可在答辩时做敏感性分析）")

with st.sidebar.expander("🎨 分级配色", expanded=False):
    colors = []
    for i, name in enumerate(fc.CLASS_NAMES):
        colors.append(
            st.color_picker(f"{i + 1} {name}", fc.CLASS_COLORS[i], key=f"color_{i}")
        )

st.sidebar.divider()
run_clicked = st.sidebar.button("🚀 开始计算", type="primary", width="stretch")

# 参数合法性检查
if not (t1 < t2 < t3):
    st.sidebar.error("三个阈值必须严格递增，例如 0.20 < 0.40 < 0.70。")
    st.stop()
if not auto_endmembers and ndvi_veg is not None and ndvi_soil is not None and ndvi_veg <= ndvi_soil:
    st.sidebar.error("NDVI_veg 必须大于 NDVI_soil。")
    st.stop()

# 当前参数签名：参数变化后旧结果自动失效
sig = (
    source, file_name, len(data), red_band, nir_band,
    p_soil, p_veg, ndvi_soil, ndvi_veg, breaks, remove_water, tuple(colors),
)
if st.session_state.get("result") is not None and st.session_state.get("sig") != sig:
    st.session_state.pop("result", None)
    st.session_state.pop("stack", None)

auto_demo = source == "示例数据" and SAMPLE_BYTES is not None
if run_clicked or (auto_demo and st.session_state.get("result") is None):
    try:
        stack = cached_stack(data, red_band, nir_band)
        result = cached_result(
            data, red_band, nir_band, p_soil, p_veg, ndvi_soil, ndvi_veg, breaks, remove_water
        )
    except Exception as exc:
        st.error(f"计算失败：{exc}")
        with st.expander("查看详细错误信息"):
            st.exception(exc)
        st.stop()
    st.session_state["result"] = result
    st.session_state["stack"] = stack
    st.session_state["sig"] = sig
    st.session_state["file_name"] = file_name

result: fc.FvcResult | None = st.session_state.get("result")
stack: fc.BandStack | None = st.session_state.get("stack")


# ----------------------------------------------------------------------------
# 主页面
# ----------------------------------------------------------------------------

st.title("🍵 普洱景迈山植被覆盖度（FVC）估算与古茶林长势分级工具")
st.caption(
    "Sentinel-2 L2A 双波段影像 · NDVI + 像元二分模型 · 10 m 分辨率 · "
    "自动统计各级面积并导出成果"
)

if result is None or stack is None:
    st.info("在左侧设置参数后点击 **🚀 开始计算**。")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "### 📥 数据输入\n"
            "1. Sentinel-2 L2A（已大气校正）\n"
            "2. 按景迈山古茶林边界裁剪\n"
            "3. 只保留 B4（红）、B8（近红外）\n"
            "4. 保存为双波段 GeoTIFF"
        )
    with c2:
        st.markdown(
            "### ⚙️ 处理流程\n"
            "1. NDVI = (NIR − Red)/(NIR + Red)\n"
            "2. 5% / 95% 分位数确定端元\n"
            "3. 像元二分模型反演 FVC\n"
            "4. 按阈值分为 4 级长势"
        )
    with c3:
        st.markdown(
            "### 📤 成果输出\n"
            "1. FVC 栅格（GeoTIFF）\n"
            "2. 长势分级栅格\n"
            "3. 面积统计表（CSV）\n"
            "4. 长势专题图（PNG）"
        )
    st.divider()
    st.markdown("#### 核心公式")
    st.latex(r"NDVI = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red}}")
    st.latex(
        r"FVC = \frac{NDVI - NDVI_{soil}}{NDVI_{veg} - NDVI_{soil}}, \qquad "
        r"NDVI_{soil}=P_{5\%},\quad NDVI_{veg}=P_{95\%}"
    )
    st.stop()

# --- 关键指标 --------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("研究区面积", f"{result.area_km2:.2f} km²")
m2.metric("有效像元", f"{result.valid_pixels:,}")
m3.metric("平均 FVC", f"{result.mean_fvc:.3f}")
m4.metric("优良长势占比", f"{result.stats.loc[3, '占比(%)']:.1f}%")
m5.metric("NDVI_soil / veg", f"{result.ndvi_soil:.2f} / {result.ndvi_veg:.2f}")

st.caption(
    f"像元面积 {result.pixel_area:.0f} m² ｜ 有效像元占 {result.valid_pixels / result.total_pixels * 100:.1f}% ｜ "
    f"端元：自动 {result.auto_soil:.3f} / {result.auto_veg:.3f}"
    + ("（已手动覆盖）" if ndvi_soil is not None else "")
)

tabs = st.tabs(["① 原始影像", "② NDVI", "③ FVC", "④ 长势分级", "⑤ 统计与导出", "⑥ 十年变化", "⑦ 遗产地概况"])

# --- ① 原始影像 ------------------------------------------------------------
with tabs[0]:
    st.subheader("原始影像预览")
    st.markdown(
        f"**文件**：`{st.session_state.get('file_name', file_name)}` ｜ "
        f"**尺寸**：{stack.width} × {stack.height} 像元 ｜ "
        f"**坐标系**：{stack.crs or '未知'} ｜ **像元**：{stack.pixel_size_m:.1f} m"
    )
    red_d, _ = fc.decimate(stack.red, MAX_DISPLAY_PIXELS)
    nir_d, _ = fc.decimate(stack.nir, MAX_DISPLAY_PIXELS)
    fig = fc.plot_preview(red_d, nir_d, st.session_state.get("file_name", file_name))
    show_figure(fig)
    st.caption("仅用于预览，显示时做了 2%~98% 百分位拉伸；统计与导出仍使用原始分辨率。")
    st.caption(
        "研究区位于遗产区核心；遗产地总面积 19 095.74 ha，含 5 片古茶林、9 个传统村落、3 片分隔防护林。"
    )

# --- ② NDVI ----------------------------------------------------------------
with tabs[1]:
    st.subheader("NDVI 计算与端元取值")
    ndvi_d, step = fc.decimate(result.ndvi, MAX_DISPLAY_PIXELS)
    fig = fc.plot_ndvi_map(ndvi_d, stack.pixel_size_m * step)
    show_figure(fig)

    st.markdown("#### NDVI 直方图")
    hist_ndvi, _ = fc.decimate(result.ndvi, 800)
    fig = fc.plot_ndvi_hist(hist_ndvi, result.ndvi_soil, result.ndvi_veg, p_soil, p_veg)
    show_figure(fig)
    st.caption(
        "红色虚线为 NDVI_soil，绿色虚线为 NDVI_veg。默认取 5% 与 95% 分位数，"
        "可有效剔除云、水、阴影等极端像元。"
    )

# --- ③ FVC -----------------------------------------------------------------
with tabs[2]:
    st.subheader("植被覆盖度 FVC")
    fvc_d, step = fc.decimate(result.fvc, MAX_DISPLAY_PIXELS)
    fig = fc.plot_fvc_map(fvc_d, stack.pixel_size_m * step)
    show_figure(fig)
    st.markdown(
        f"FVC 取值范围 0~1，均值 **{result.mean_fvc:.3f}**，"
        f"最大 **{np.nanmax(result.fvc):.3f}**，最小 **{np.nanmin(result.fvc):.3f}**。"
    )
    st.caption("FVC 越高表示植被覆盖越好，古茶林长势越佳；水体等无效像元显示为白色。")

# --- ④ 长势分级 ------------------------------------------------------------
with tabs[3]:
    st.subheader("古茶林长势分级专题图")
    cls_d, step = fc.decimate(result.classes, MAX_DISPLAY_PIXELS)
    fig = fc.plot_class_map(
        cls_d, result.breaks, colors=colors, names=result.class_names,
        pixel_size_m=stack.pixel_size_m * step,
    )
    show_figure(fig)

    st.markdown("#### 分级标准")
    legend = result.stats[["等级", "长势类型", "FVC 区间", "占比(%)"]].copy()
    st.dataframe(legend, hide_index=True, width="stretch")
    st.caption(
        "分级阈值可在左侧「自定义 FVC 分级阈值」中修改；修改后重新计算即可看到专题图与统计变化。"
    )

# --- ⑤ 统计与导出 ----------------------------------------------------------
with tabs[4]:
    st.subheader("面积统计")
    table = fc.table_with_total(result.stats)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "面积(km²)": st.column_config.NumberColumn(format="%.4f"),
            "面积(公顷)": st.column_config.NumberColumn(format="%.2f"),
            "面积(亩)": st.column_config.NumberColumn(format="%.1f"),
            "占比(%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    cbar, cnote = st.columns([2, 1])
    with cbar:
        fig = fc.plot_stats_bar(result.stats, colors=colors)
        show_figure(fig)
    with cnote:
        st.markdown("#### 说明")
        st.markdown(
            "- 面积 = 像元数 × 像元真实面积\n"
            f"- 单像元面积约 **{result.pixel_area:.0f} m²**\n"
            f"- 有效统计面积 **{result.area_km2:.3f} km²**\n"
            "- 占比以有效像元总数（去掉水体、NoData）为分母"
        )

    st.divider()
    st.subheader("成果导出")
    st.caption("所有成果保持原影像的坐标系与仿射变换，可直接在 QGIS / ArcGIS 中叠加。")

    fvc_bytes = download_tif_bytes(result, stack, "fvc")
    class_bytes = download_tif_bytes(result, stack, "class")
    csv_bytes = fc.table_with_total(result.stats).to_csv(index=False).encode("utf-8-sig")
    map_fig = fc.plot_class_map(
        result.classes, result.breaks, colors=colors, names=result.class_names,
        pixel_size_m=stack.pixel_size_m,
    )
    png_bytes = figure_png(map_fig, dpi=200)

    # 一键打包
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("FVC.tif", fvc_bytes)
        zf.writestr("GrowthClass.tif", class_bytes)
        zf.writestr("FVC_statistics.csv", csv_bytes)
        zf.writestr("长势分级专题图.png", png_bytes)
        zf.writestr(
            "README.txt",
            "景迈山 FVC 与古茶林长势分级成果\n"
            f"NDVI_soil={result.ndvi_soil:.4f}, NDVI_veg={result.ndvi_veg:.4f}\n"
            f"分级阈值={result.breaks}\n"
            "FVC.tif: 0~1 浮点栅格, NoData=-9999\n"
            "GrowthClass.tif: 1~4 分级, 0 为无效\n",
        )
    zip_bytes = zip_buf.getvalue()

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("⬇️ FVC 栅格 (.tif)", fvc_bytes, "FVC.tif", "image/tiff", width="stretch")
    with d2:
        st.download_button("⬇️ 分级栅格 (.tif)", class_bytes, "GrowthClass.tif", "image/tiff", width="stretch")
    with d3:
        st.download_button("⬇️ 统计表 (.csv)", csv_bytes, "FVC_statistics.csv", "text/csv", width="stretch")
    with d4:
        st.download_button("⬇️ 专题图 (.png)", png_bytes, "长势分级专题图.png", "image/png", width="stretch")

    st.download_button(
        "📦 一键下载全部成果（.zip）",
        zip_bytes,
        "景迈山FVC_长势分级成果.zip",
        "application/zip",
        type="primary",
        width="stretch",
    )

with tabs[5]:
    render_timeseries()

with tabs[6]:
    render_heritage()

st.divider()
with st.expander("ℹ️ 方法说明与使用建议", expanded=False):
    st.markdown(
        """
**计算流程**

1. 读取双波段影像，NDVI = (B8 − B4) / (B8 + B4)，无效值与 NDVI < 0 的水体像元设为 NaN；
2. 对有效 NDVI 取 5% 分位数为 NDVI_soil、95% 分位数为 NDVI_veg（也可手动指定）；
3. 像元二分模型计算 FVC，并截断到 0~1；
4. 按阈值 0.20 / 0.40 / 0.70 将 FVC 分为 4 级，统计面积与占比；
5. 以原影像的坐标系和仿射变换输出 GeoTIFF，保证成果与底图严格对齐。

**使用建议**

- 输入影像应为 **Sentinel-2 L2A**（已做大气校正），不要用 L1C 原始 DN；
- 务必确认波段顺序：本工具默认第 1 波段为 B4、第 2 波段为 B8，选错会得到相反的结论；
- 研究区尽量裁剪到古茶林边界内（约 5~10 km 范围），整景影像不仅慢，也会把大量非古茶林像元混入统计；
- 更换季节影像（旱季 / 雨季）或端元参数后，建议对比 FVC 均值与各级占比，做敏感性分析。

**已知局限**

- 像元二分模型假设像元内只有“裸土 + 植被”两种组分，对混合像元、林下多层结构只能近似；
- 10 m 分辨率下，古茶林与周边天然林的光谱差异有限，本工具依赖**矢量边界裁剪**来限定分析对象；
- 水体、云影、建筑物阴影会干扰端元分位数，建议在预处理阶段先做云掩膜。
"""
    )

st.caption("景迈山古茶林长势分级工具 · 课程项目版 · 示例数据为景迈山 2025-02-17 真实影像；合成演示数据见 demo_data/")
