"""危險物品偵測與手部空間關聯"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from config import (
    EXCLUDED_WEAPON_LABELS,
    HAND_WEAPON_IOU_THRESHOLD,
    WEAPON_IMGSZ,
)
from hand_detector import HandDetection, HandDetector
from model_loader import ensure_hand_model, ensure_weapon_model

logger = logging.getLogger(__name__)


@dataclass
class WeaponDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class DangerousPerson:
    """手部持有危險物品"""

    hand: HandDetection
    weapons: list[WeaponDetection] = field(default_factory=list)

    @property
    def weapon_labels(self) -> str:
        return ", ".join(w.label for w in self.weapons)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class ObjectDetector:
    """偵測手部與危險物品，以 bounding box 重疊判斷是否手持"""

    def __init__(
        self,
        weapon_model_path: str | Path,
        hand_model_path: str | Path,
        conf: float = 0.5,
        auto_download: bool = True,
    ):
        weapon_path = Path(weapon_model_path)
        hand_path = Path(hand_model_path)

        if auto_download:
            ensure_weapon_model(weapon_path)
            ensure_hand_model(hand_path)

        self.weapon_model = YOLO(str(weapon_path))
        self.hand_detector = HandDetector(str(hand_path), min_confidence=conf)
        self.conf = conf

    def _detect_weapons(self, frame: np.ndarray) -> list[WeaponDetection]:
        results = self.weapon_model(frame, conf=self.conf, imgsz=WEAPON_IMGSZ, verbose=False)
        weapons: list[WeaponDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                if label.lower() in EXCLUDED_WEAPON_LABELS:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                weapons.append(WeaponDetection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return weapons

    def detect(
        self, frame: np.ndarray,
    ) -> tuple[list[HandDetection], list[WeaponDetection], list[DangerousPerson]]:
        hands = self.hand_detector.detect(frame)
        weapons = self._detect_weapons(frame)
        dangerous = self._match_hand_weapon(hands, weapons)
        return hands, weapons, dangerous

    def _match_hand_weapon(
        self,
        hands: list[HandDetection],
        weapons: list[WeaponDetection],
    ) -> list[DangerousPerson]:
        if not hands or not weapons:
            return []

        dangerous: list[DangerousPerson] = []
        for hand in hands:
            matched = [
                weapon for weapon in weapons
                if _iou(hand.bbox, weapon.bbox) >= HAND_WEAPON_IOU_THRESHOLD
            ]
            if matched:
                dangerous.append(DangerousPerson(hand=hand, weapons=matched))
        return dangerous
