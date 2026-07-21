"""即時 webcam 危險人物識別（手部持有危險物品）"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

import config
from camera_service import CameraService
from object_detector import ObjectDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="即時 webcam 危險人物識別（手部持有危險物品）")
    parser.add_argument("--source", type=str, default=str(config.VIDEO_SOURCE))
    parser.add_argument("--weapon-model", type=str, default=str(config.WEAPON_MODEL))
    parser.add_argument("--hand-model", type=str, default=str(config.HAND_MODEL))
    parser.add_argument("--conf", type=float, default=config.DETECTION_CONF)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--snapshot-dir", type=str, default=None)
    parser.add_argument("--no-download", action="store_true", help="不自動下載公開模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    snapshot_dir: Path | None = None
    if args.snapshot_dir:
        snapshot_dir = Path(args.snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("初始化偵測模型（首次執行會自動下載公開模型）...")
    detector = ObjectDetector(
        weapon_model_path=args.weapon_model,
        hand_model_path=args.hand_model,
        conf=args.conf,
        auto_download=not args.no_download,
    )

    service = CameraService(args.source, detector, snapshot_dir)
    service.start()

    logger.info("按 Q 退出" if not args.headless else "無 GUI 模式（Ctrl+C 退出）")

    last_frame_seq = -1
    try:
        while True:
            frame, frame_seq = service.get_frame_bgr()
            if frame is not None and frame_seq != last_frame_seq:
                last_frame_seq = frame_seq
                if not args.headless:
                    cv2.imshow(config.WINDOW_NAME, frame)

            if args.headless:
                time.sleep(0.01)
                continue

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
