# 危險人物即時識別

透過 webcam 即時偵測**手部持有危險物品**的人（刀、槍、手榴彈等），並提供 **Web UI** 即時顯示影像。

## 偵測原理

1. **手部偵測**：MediaPipe Hand Landmarker 計算手部 bounding box
2. **危險物品偵測**：公開 YOLO 模型（Hugging Face）偵測刀、槍等
3. **空間關聯**：危險物品 bbox 與手部 bbox **IoU 重疊** → 判定為危險

## Web UI 啟動

```bash
uv sync
uv run python web_app.py
```

瀏覽器開啟：**http://localhost:8080**

- 左側：即時 MJPEG 影像串流（含偵測框）
- 右側：FPS、手部/危險物品數量、警報紀錄

## 本機 OpenCV 視窗模式

```bash
uv run python main.py
```

## Docker Compose

```bash
docker compose up -d --build
```

開啟 **http://localhost:8080**

## 常用參數

```bash
uv run python web_app.py --source 0 --port 8080 --conf 0.4
uv run python web_app.py --snapshot-dir data/alerts
```

## 環境變數（Docker `.env`）

| 變數 | 說明 | 預設 |
|------|------|------|
| `WEB_PORT` | Web UI 連接埠 | `8080` |
| `VIDEO_DEVICE` | 主機 webcam | `/dev/video0` |
| `VIDEO_SOURCE` | 容器內視訊來源 | `/dev/video0` |
| `DETECTION_CONF` | 偵測信心度 | `0.4` |

## 公開模型

| 用途 | 來源 |
|------|------|
| 危險物品 | [SyncRobotic/weapon-detection-yolov8n-v2](https://huggingface.co/SyncRobotic/weapon-detection-yolov8n-v2) |
| 手部 | [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |

首次執行自動下載至 `models/`。
