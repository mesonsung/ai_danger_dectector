"""下載公開偵測模型"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# 公開模型：knife / gun / grenade 等危險物品
WEAPON_HF_REPO = "SyncRobotic/weapon-detection-yolov8n-v2"
WEAPON_HF_FILE = "weights/best.pt"
WEAPON_HF_FALLBACK_REPO = "Subh775/Firearm_Detection_Yolov8n"
WEAPON_HF_FALLBACK_FILE = "weights/best.pt"


def ensure_hand_model(dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("下載 MediaPipe 手部模型...")
    urllib.request.urlretrieve(HAND_MODEL_URL, dest)
    logger.info("手部模型就緒: %s", dest)
    return dest


def ensure_weapon_model(dest: Path) -> Path:
    if dest.exists():
        return dest

    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("從 Hugging Face 下載危險物品模型...")

    try:
        cached = hf_hub_download(repo_id=WEAPON_HF_REPO, filename=WEAPON_HF_FILE)
        logger.info("使用模型: %s", WEAPON_HF_REPO)
    except Exception as exc:
        logger.warning("主模型下載失敗 (%s)，改用備用模型", exc)
        cached = hf_hub_download(
            repo_id=WEAPON_HF_FALLBACK_REPO,
            filename=WEAPON_HF_FALLBACK_FILE,
        )
        logger.info("使用模型: %s", WEAPON_HF_FALLBACK_REPO)

    shutil.copy(cached, dest)
    logger.info("危險物品模型就緒: %s", dest)
    return dest
