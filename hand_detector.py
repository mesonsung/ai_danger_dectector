"""MediaPipe 手部 segmentation（landmark convex hull）"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import HAND_BOX_PADDING_RATIO, HAND_MASK_DILATE_PX


@dataclass
class HandDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray | None = None  # bool H×W，與推論影像同尺寸


def _landmarks_to_mask(
    landmarks,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    pad_x = int((max(xs) - min(xs)) * HAND_BOX_PADDING_RATIO)
    pad_y = int((max(ys) - min(ys)) * HAND_BOX_PADDING_RATIO)
    x1 = max(0, int(min(xs)) - pad_x)
    y1 = max(0, int(min(ys)) - pad_y)
    x2 = min(width, int(max(xs)) + pad_x)
    y2 = min(height, int(max(ys)) + pad_y)

    points = np.array(
        [(int(lm.x * width), int(lm.y * height)) for lm in landmarks],
        dtype=np.int32,
    )
    hull = cv2.convexHull(points)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)

    if HAND_MASK_DILATE_PX > 0:
        k = HAND_MASK_DILATE_PX * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    return (x1, y1, x2, y2), mask.astype(bool)


class HandDetector:
    def __init__(self, model_path: str, min_confidence: float = 0.5):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hands: list[HandDetection] = []
        if not result.hand_landmarks:
            return hands

        for idx, landmarks in enumerate(result.hand_landmarks):
            bbox, mask = _landmarks_to_mask(landmarks, w, h)

            label = "hand"
            confidence = 1.0
            if result.handedness and idx < len(result.handedness):
                handed = result.handedness[idx][0]
                label = handed.category_name.lower()
                confidence = handed.score

            hands.append(
                HandDetection(label=label, confidence=confidence, bbox=bbox, mask=mask),
            )

        return hands
