# 景迈山监测 · 实时分析服务部署包

这是一个可实时上传影像、计算 FVC、分级、统计并下载成果的 Streamlit 服务，
不是静态网页。

## 目录

```
deploy/
├── app.py                 # Streamlit 主程序
├── fvc_core.py            # FVC/NDVI/分级/制图算法
├── requirements.txt       # 固定版本依赖
├── .streamlit/config.toml # 端口、上传大小等
├── Dockerfile
├── data/                  # 示例 Sentinel-2 影像
├── demo_data/             # 合成演示数据
├── outputs/timeseries/    # 十年统计与图件
├── outputs/wetland/       # 小微湿地统计与图件
└── docs/figures/          # 遗产地示意图
```

## 方式一：本机 / 局域网（最快）

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

同一 Wi-Fi/校园网的用户访问：

```
http://<你的局域网IP>:8502
```

Windows 首次启动如果弹防火墙提示，选择“允许访问（专用网络）”。

## 方式二：Streamlit Community Cloud（永久公网 URL，免费）

1. 把 `deploy/` 上传到一个 GitHub 仓库；
2. 打开 https://share.streamlit.io/ ，用 GitHub 登录；
3. New app → 选择仓库 → Main file path 填 `app.py`；
4. Python 版本选 3.12，Deploy；
5. 得到 `https://<app>.streamlit.app` 公网地址，任何用户浏览器可访问。

注意：仓库不要包含 `.venv`、`serverdeps` 等大目录；`data/` 约 20 MB，可以直接上传。

## 方式三：Docker / Render / Railway / Hugging Face

```bash
docker build -t jingmai-monitor .
docker run -p 8501:8501 jingmai-monitor
```

- Render：New Web Service → Docker → 选仓库；
- Railway：Deploy from repo；
- Hugging Face Spaces：SDK 选 Docker，上传本目录。

平台会自动提供公网 HTTPS 地址。

## 方式四：任意云服务器

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

用 Nginx 反向代理到 8501，并加 HTTPS。

## 公网部署前建议

- 加访问口令或反向代理 Basic Auth，避免陌生人占用算力；
- 限制上传大小（默认 50 MB）；
- 不要把 `outputs/` 的临时成果长期暴露；
- 若并发用户多，增加 CPU/内存，或把重计算放到队列。

## 数据与算法

- Sentinel-2 L2A，10 m，B4/B8，SCL 云掩膜；
- NDVI + 像元二分模型，阈值 0.20/0.40/0.70；
- 2022 年 +1000 偏移自动扣除；
- 小微湿地 NDVI<0，斑块 ≥5 像元，缓冲 30 m。

## 已验证

- 本机 Streamlit 服务：健康检查 `ok`，首页 HTTP 200；
- 7 个标签页，示例影像可完成 FVC/分级/统计；
- 上传双波段 GeoTIFF 后走同一套 `fvc_core` 流程。
