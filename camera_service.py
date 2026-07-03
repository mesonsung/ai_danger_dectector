"""攝影機擷取與即時偵測服務"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

import config
from hand_detector import HandDetection
from object_detector import DangerousPerson, ObjectDetector, WeaponDetection
from overlay import render_frame

logger = logging.getLogger(__name__)


@dataclass
class DetectionStatus:
    fps: float = 0.0
    danger_count: int = 0
    danger_active: bool = False
    hands: int = 0
    weapons: int = 0
    alerts: list[str] = field(default_factory=list)
    updated_at: str = ""


def open_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟視訊來源: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


class CameraService:
    """背景執行緒持續擷取影像、執行偵測並提供最新畫面"""

    def __init__(
        self,
        source: str,
        detector: ObjectDetector,
        snapshot_dir: Path | None = None,
    ):
        self.source = source
        self.detector = detector
        self.snapshot_dir = snapshot_dir
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._status = DetectionStatus()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("攝影機服務已啟動")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_status(self) -> DetectionStatus:
        with self._lock:
            return DetectionStatus(
                fps=self._status.fps,
                danger_count=self._status.danger_count,
                danger_active=self._status.danger_active,
                hands=self._status.hands,
                weapons=self._status.weapons,
                alerts=list(self._status.alerts),
                updated_at=self._status.updated_at,
            )

    def _loop(self) -> None:
        cap = open_capture(self.source)
        frame_idx = fps_count = 0
        last_hands: list[HandDetection] = []
        last_weapons: list[WeaponDetection] = []
        last_dangerous: list[DangerousPerson] = []
        fps_timer, fps = time.time(), 0.0
        recent_alerts: list[str] = []

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("無法讀取畫面，重試中...")
                    time.sleep(0.5)
                    continue

                frame_idx += 1
                fps_count += 1
                danger_this_frame = False

                if frame_idx % config.DETECT_INTERVAL == 0:
                    last_hands, last_weapons, last_dangerous = self.detector.detect(frame)
                    for dp in last_dangerous:
                        danger_this_frame = True
                        msg = f"{dp.hand.label}手持有 {dp.weapon_labels}"
                        logger.warning("危險: %s at %s", msg, dp.hand.bbox)
                        recent_alerts.insert(0, f"{datetime.now():%H:%M:%S} {msg}")
                        recent_alerts = recent_alerts[:20]

                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps = fps_count / elapsed
                    fps_count = 0
                    fps_timer = time.time()

                status_text = (
                    f"FPS: {fps:.1f} | 危險: {len(last_dangerous)} | "
                    f"{datetime.now():%H:%M:%S}"
                )
                render_frame(frame, last_hands, last_weapons, last_dangerous, status_text, danger_this_frame)

                if danger_this_frame and self.snapshot_dir:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = self.snapshot_dir / f"alert_{ts}.jpg"
                    cv2.imwrite(str(path), frame)

                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._latest_jpeg = jpeg.tobytes()
                        self._status = DetectionStatus(
                            fps=round(fps, 1),
                            danger_count=len(last_dangerous),
                            danger_active=danger_this_frame,
                            hands=len(last_hands),
                            weapons=len(last_weapons),
                            alerts=recent_alerts,
                            updated_at=datetime.now().isoformat(timespec="seconds"),
                        )

                time.sleep(0.001)
        finally:
            cap.release()
