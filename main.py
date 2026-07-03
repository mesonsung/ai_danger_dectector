"""即時 webcam 危險人物識別（手部持有危險物品）"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

import config
from camera_service import open_capture
from object_detector import ObjectDetector
from overlay import render_frame

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

    cap = open_capture(args.source)
    frame_idx = fps_count = 0
    last_hands, last_weapons, last_dangerous = [], [], []
    fps_timer, fps = time.time(), 0.0
    timestamp_ms = 0
    last_detect_at = 0.0
    detect_interval = config.DETECTION_MIN_INTERVAL_MS / 1000

    logger.info("按 Q 退出" if not args.headless else "無 GUI 模式（Ctrl+C 退出）")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fps_count += 1
        danger_this_frame = False

        now = time.time()
        if now - last_detect_at >= detect_interval:
            last_detect_at = now
            timestamp_ms += 33
            last_hands, last_weapons, last_dangerous = detector.detect(frame, timestamp_ms)
            for dp in last_dangerous:
                danger_this_frame = True
                logger.warning(
                    "危險: %s手持有 %s at hand=%s",
                    dp.hand.label, dp.weapon_labels, dp.hand.bbox,
                )

        danger_this_frame = len(last_dangerous) > 0

        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        status = f"FPS: {fps:.1f} | 危險: {len(last_dangerous)} | {datetime.now():%H:%M:%S}"
        render_frame(frame, last_hands, last_weapons, last_dangerous, status, danger_this_frame)

        if danger_this_frame and snapshot_dir:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(str(snapshot_dir / f"alert_{ts}.jpg"), frame)

        if args.headless:
            time.sleep(0.01)
            continue

        cv2.imshow(config.WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
