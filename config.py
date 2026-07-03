"""系統設定"""

from pathlib import Path

ROOT_DIR = Path(__file__).parent

# 模型路徑（首次執行自動從公開來源下載）
WEAPON_MODEL = ROOT_DIR / "models" / "weapon.pt"
HAND_MODEL = ROOT_DIR / "models" / "hand_landmarker.task"

# 視訊來源：0 = 預設 webcam，或填入影片/RTSP 路徑
VIDEO_SOURCE = 0

# 每 N 幀執行一次偵測
DETECT_INTERVAL = 2

# 偵測信心度閾值
DETECTION_CONF = 0.4

# 危險物品模型推論尺寸（SyncRobotic 模型建議 960）
WEAPON_IMGSZ = 960

# 手部與危險物品 bounding box 的 IoU 閾值
HAND_WEAPON_IOU_THRESHOLD = 0.01

# 手部 bbox 由 landmark 計算時的邊界擴展比例
HAND_BOX_PADDING_RATIO = 0.1

# 不視為手持的類別（如爆炸效果）
EXCLUDED_WEAPON_LABELS = {"explosion"}

# Web UI
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
ALERT_DIR = ROOT_DIR / "data" / "alerts"

# 顯示視窗名稱
WINDOW_NAME = "危險人物即時識別"

# 邊框顏色 (BGR)
COLOR_DANGER = (0, 0, 255)
COLOR_HAND = (255, 200, 0)
COLOR_WEAPON = (0, 165, 255)
