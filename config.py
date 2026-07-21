"""系統設定"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent

# 模型路徑（首次執行自動從公開來源下載）
WEAPON_MODEL = ROOT_DIR / "models" / "weapon.pt"
HAND_MODEL = ROOT_DIR / "models" / "hand_landmarker.task"

# 視訊來源：0 = 預設 webcam，或填入影片/RTSP 路徑
VIDEO_SOURCE = 0

# 擷取解析度
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

# Linux webcam 建議 MJPG，YUYV 在 720p 常只有 ~5 FPS
CAPTURE_FOURCC = "MJPG"
CAPTURE_FPS = 30

# 推論用影像最大寬度（縮小後再偵測，bbox 會映射回原始尺寸）
DETECTION_MAX_WIDTH = 320

# 兩次偵測之間最短間隔（毫秒）
DETECTION_MIN_INTERVAL_MS = 120

# 偵測信心度閾值
DETECTION_CONF = 0.6

# 危險物品模型推論尺寸（需與訓練 imgsz 一致，640）
WEAPON_IMGSZ = 640

# 手部與危險物品 segmentation mask 重疊比例閾值（intersection / min area）
HAND_WEAPON_MASK_OVERLAP_THRESHOLD = 0.05

# 手部 convex hull mask 膨脹像素（讓邊緣更容易與物品重疊）
HAND_MASK_DILATE_PX = 20

# 手部 bbox 由 landmark 計算時的邊界擴展比例
HAND_BOX_PADDING_RATIO = 0.1

# 不視為手持的類別（如爆炸效果）
EXCLUDED_WEAPON_LABELS = {"explosion"}

# Web UI
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
WEB_DISPLAY_WIDTH = 1280
WEB_VIDEO_REFRESH_MS = 33
WEB_STATUS_REFRESH_MS = 500
ALERT_DIR = ROOT_DIR / "data" / "alerts"

# 顯示視窗名稱
WINDOW_NAME = "危險人物即時識別"

# 邊框顏色 (BGR)
COLOR_DANGER = (0, 0, 255)
COLOR_HAND = (255, 200, 0)
COLOR_WEAPON = (0, 165, 255)


def capture_backend() -> int:
    """選擇低延遲的視訊後端。"""
    import cv2

    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform == "linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY
