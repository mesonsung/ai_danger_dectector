"""攝影機擷取與即時偵測服務"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config
from hand_detector import HandDetection
from object_detector import DangerousPerson, ObjectDetector, WeaponDetection
from overlay import render_frame

logger = logging.getLogger(__name__)


@dataclass
class DetectionStatus:
    fps: float = 0.0
    detect_fps: float = 0.0
    danger_count: int = 0
    danger_active: bool = False
    hands: int = 0
    weapons: int = 0
    infer_device: str = "cpu"
    alerts: list[str] = field(default_factory=list)
    updated_at: str = ""


def open_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    backend = config.capture_backend() if isinstance(src, int) else 0
    cap = cv2.VideoCapture(src, backend) if backend else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟視訊來源: {source}")

    if isinstance(src, int) and config.CAPTURE_FOURCC:
        fourcc = cv2.VideoWriter_fourcc(*config.CAPTURE_FOURCC)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
    if config.CAPTURE_FPS:
        cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
    logger.info(
        "視訊來源: %sx%s fourcc=%s fps=%s",
        actual_w,
        actual_h,
        fourcc_str,
        cap.get(cv2.CAP_PROP_FPS),
    )
    return cap


class CameraService:
    """擷取與偵測分離：擷取執行緒維持高 FPS，偵測執行緒非同步處理最新幀。"""

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
        self._latest_frame_rgb: np.ndarray | None = None
        self._latest_frame_bgr: np.ndarray | None = None
        self._frame_seq = 0
        self._status = DetectionStatus()
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None

        self._pending_frame: np.ndarray | None = None
        self._frame_event = threading.Event()
        self._results_lock = threading.Lock()
        self._last_hands: list[HandDetection] = []
        self._last_weapons: list[WeaponDetection] = []
        self._last_dangerous: list[DangerousPerson] = []
        self._prev_danger_active = False
        self._recent_alerts: list[str] = []
        self._timestamp_ms = 0
        self._infer_device = detector.infer_device_label

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._detect_thread.start()
        self._capture_thread.start()
        logger.info("攝影機服務已啟動（擷取/偵測分離）")

    def stop(self) -> None:
        self._running = False
        self._frame_event.set()
        for thread in (self._capture_thread, self._detect_thread):
            if thread:
                thread.join(timeout=3)

    def get_frame_rgb(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest_frame_rgb is None:
                return None, self._frame_seq
            return self._latest_frame_rgb.copy(), self._frame_seq

    def get_frame_bgr(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest_frame_bgr is None:
                return None, self._frame_seq
            return self._latest_frame_bgr.copy(), self._frame_seq

    def get_status(self) -> DetectionStatus:
        with self._lock:
            return DetectionStatus(
                fps=self._status.fps,
                detect_fps=self._status.detect_fps,
                danger_count=self._status.danger_count,
                danger_active=self._status.danger_active,
                hands=self._status.hands,
                weapons=self._status.weapons,
                infer_device=self._infer_device,
                alerts=list(self._status.alerts),
                updated_at=self._status.updated_at,
            )

    def _submit_frame(self, frame: np.ndarray) -> None:
        with self._results_lock:
            self._pending_frame = frame
        self._frame_event.set()

    def _take_pending_frame(self) -> np.ndarray | None:
        with self._results_lock:
            frame = self._pending_frame
            self._pending_frame = None
        return frame

    def _detect_loop(self) -> None:
        detect_count = 0
        detect_timer = time.time()
        detect_fps = 0.0
        min_interval = config.DETECTION_MIN_INTERVAL_MS / 1000

        while self._running:
            if not self._frame_event.wait(timeout=0.05):
                continue
            self._frame_event.clear()

            frame = self._take_pending_frame()
            while True:
                with self._results_lock:
                    newer = self._pending_frame
                    if newer is not None:
                        frame = newer
                        self._pending_frame = None
                    else:
                        break

            if frame is None:
                continue

            started = time.time()
            self._timestamp_ms += 33
            hands, weapons, dangerous = self.detector.detect(frame, self._timestamp_ms)

            detect_count += 1
            elapsed = time.time() - detect_timer
            if elapsed >= 1.0:
                detect_fps = detect_count / elapsed
                detect_count = 0
                detect_timer = time.time()

            danger_active = len(dangerous) > 0
            if danger_active and not self._prev_danger_active:
                for dp in dangerous:
                    msg = f"{dp.hand.label}手持有 {dp.weapon_labels}"
                    logger.warning("危險: %s at %s", msg, dp.hand.bbox)
                    self._recent_alerts.insert(0, f"{datetime.now():%H:%M:%S} {msg}")
                    self._recent_alerts = self._recent_alerts[:20]
            self._prev_danger_active = danger_active

            with self._results_lock:
                self._last_hands = hands
                self._last_weapons = weapons
                self._last_dangerous = dangerous

            with self._lock:
                self._status.detect_fps = round(detect_fps, 1)
                self._status.danger_count = len(dangerous)
                self._status.hands = len(hands)
                self._status.weapons = len(weapons)
                self._status.alerts = list(self._recent_alerts)

            spent = time.time() - started
            if spent < min_interval:
                time.sleep(min_interval - spent)

    def _capture_loop(self) -> None:
        cap = open_capture(self.source)
        fps_count = 0
        fps_timer, fps = time.time(), 0.0
        last_snapshot_at = 0.0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("無法讀取畫面，重試中...")
                    time.sleep(0.5)
                    continue

                fps_count += 1
                self._submit_frame(frame)

                with self._results_lock:
                    hands = list(self._last_hands)
                    weapons = list(self._last_weapons)
                    dangerous = list(self._last_dangerous)

                danger_active = len(dangerous) > 0

                elapsed = time.time() - fps_timer
                if elapsed >= 1.0:
                    fps = fps_count / elapsed
                    fps_count = 0
                    fps_timer = time.time()

                status_text = (
                    f"FPS: {fps:.1f} | 偵測: {self._status.detect_fps:.1f} | "
                    f"GPU: {self._infer_device} | 危險: {len(dangerous)} | "
                    f"{datetime.now():%H:%M:%S}"
                )
                render_frame(frame, hands, weapons, dangerous, status_text, danger_active)

                if danger_active and self.snapshot_dir:
                    now = time.time()
                    if now - last_snapshot_at >= 1.0:
                        last_snapshot_at = now
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        path = self.snapshot_dir / f"alert_{ts}.jpg"
                        cv2.imwrite(str(path), frame)

                h, w = frame.shape[:2]
                display_w = config.WEB_DISPLAY_WIDTH
                if w > display_w:
                    scale = display_w / w
                    display = cv2.resize(
                        frame,
                        (display_w, int(h * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    display = frame
                rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                with self._lock:
                    self._latest_frame_bgr = display.copy()
                    self._latest_frame_rgb = rgb
                    self._frame_seq += 1
                    self._status.fps = round(fps, 1)
                    self._status.danger_active = danger_active
                    self._status.updated_at = datetime.now().isoformat(timespec="seconds")
        finally:
            cap.release()
