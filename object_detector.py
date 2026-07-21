"""危險物品偵測與手部 segmentation 空間關聯"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    DETECTION_MAX_WIDTH,
    EXCLUDED_WEAPON_LABELS,
    HAND_WEAPON_MASK_OVERLAP_THRESHOLD,
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
    mask: np.ndarray | None = None  # bool H×W，與推論影像同尺寸


@dataclass
class DangerousPerson:
    """手部持有危險物品"""

    hand: HandDetection
    weapons: list[WeaponDetection] = field(default_factory=list)

    @property
    def weapon_labels(self) -> str:
        return ", ".join(w.label for w in self.weapons)


def _mask_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """intersection / min(area)，衡量 mask 是否實質重疊。"""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    area_a = int(a.sum())
    area_b = int(b.sum())
    return inter / min(area_a, area_b)


def _bbox_to_mask(bbox: tuple[int, int, int, int], height: int, width: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _scale_bbox(bbox: tuple[int, int, int, int], inv_scale: float) -> tuple[int, int, int, int]:
    if inv_scale == 1.0:
        return bbox
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * inv_scale),
        int(y1 * inv_scale),
        int(x2 * inv_scale),
        int(y2 * inv_scale),
    )


class ObjectDetector:
    """偵測手部與危險物品，以 segmentation mask 重疊判斷是否手持"""

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
        self._weapon_has_masks = self.weapon_model.task == "segment"
        self._device = self._resolve_device()
        self.infer_device_label = self._device_label(self._device)
        self.weapon_model.to(self._device)
        self._infer_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="detect")
        self._warmup()
        logger.info("YOLO 推論裝置: %s", self.infer_device_label)
        if self._weapon_has_masks:
            logger.info("危險物品模型支援 instance segmentation")
        else:
            logger.info("危險物品模型為 detect 模式，武器 mask 以 bbox 近似")

    @staticmethod
    def _resolve_device() -> str | int:
        forced = os.environ.get("INFER_DEVICE", "").strip()
        if forced:
            return forced

        try:
            import torch

            if torch.cuda.is_available():
                return 0
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _device_label(device: str | int) -> str:
        if device == "cpu":
            return "CPU"
        try:
            import torch

            if isinstance(device, int) and torch.cuda.is_available():
                return torch.cuda.get_device_name(device)
        except ImportError:
            pass
        return str(device)

    def _warmup(self) -> None:
        dummy = np.zeros((WEAPON_IMGSZ, WEAPON_IMGSZ, 3), dtype=np.uint8)
        self.weapon_model(
            dummy,
            conf=self.conf,
            imgsz=WEAPON_IMGSZ,
            device=self._device,
            verbose=False,
        )
        if self._device != "cpu":
            import torch

            torch.cuda.synchronize()

    def _prepare_infer_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        max_w = DETECTION_MAX_WIDTH
        if max_w and w > max_w:
            scale = max_w / w
            infer = cv2.resize(
                frame,
                (max_w, int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            return infer, scale
        return frame, 1.0

    def _extract_weapon_mask(
        self,
        result,
        box_index: int,
        bbox: tuple[int, int, int, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        if self._weapon_has_masks and result.masks is not None:
            mask_data = result.masks.data[box_index]
            if hasattr(mask_data, "cpu"):
                mask_data = mask_data.cpu().numpy()
            mask = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
            return mask > 0.5
        return _bbox_to_mask(bbox, height, width)

    def _detect_weapons(self, frame: np.ndarray) -> list[WeaponDetection]:
        height, width = frame.shape[:2]
        results = self.weapon_model(
            frame,
            conf=self.conf,
            imgsz=WEAPON_IMGSZ,
            device=self._device,
            verbose=False,
        )
        weapons: list[WeaponDetection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box_index, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id))
                if label.lower() in EXCLUDED_WEAPON_LABELS:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)
                mask = self._extract_weapon_mask(result, box_index, bbox, height, width)
                weapons.append(
                    WeaponDetection(label=label, confidence=conf, bbox=bbox, mask=mask),
                )
        return weapons

    def detect(
        self,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> tuple[list[HandDetection], list[WeaponDetection], list[DangerousPerson]]:
        infer_frame, scale = self._prepare_infer_frame(frame)
        inv_scale = 1.0 / scale

        hand_future = self._infer_pool.submit(
            self.hand_detector.detect, infer_frame, timestamp_ms,
        )
        weapon_future = self._infer_pool.submit(self._detect_weapons, infer_frame)
        hands = hand_future.result()
        weapons = weapon_future.result()

        dangerous = self._match_hand_weapon(hands, weapons)

        if inv_scale != 1.0:
            hands = [
                HandDetection(
                    label=hand.label,
                    confidence=hand.confidence,
                    bbox=_scale_bbox(hand.bbox, inv_scale),
                )
                for hand in hands
            ]
            weapons = [
                WeaponDetection(
                    label=weapon.label,
                    confidence=weapon.confidence,
                    bbox=_scale_bbox(weapon.bbox, inv_scale),
                )
                for weapon in weapons
            ]
            dangerous = [
                DangerousPerson(
                    hand=HandDetection(
                        label=dp.hand.label,
                        confidence=dp.hand.confidence,
                        bbox=_scale_bbox(dp.hand.bbox, inv_scale),
                    ),
                    weapons=[
                        WeaponDetection(
                            label=weapon.label,
                            confidence=weapon.confidence,
                            bbox=_scale_bbox(weapon.bbox, inv_scale),
                        )
                        for weapon in dp.weapons
                    ],
                )
                for dp in dangerous
            ]

        return hands, weapons, dangerous

    def _match_hand_weapon(
        self,
        hands: list[HandDetection],
        weapons: list[WeaponDetection],
    ) -> list[DangerousPerson]:
        if not hands or not weapons:
            return []

        threshold = HAND_WEAPON_MASK_OVERLAP_THRESHOLD
        dangerous: list[DangerousPerson] = []
        for hand in hands:
            if hand.mask is None:
                continue
            matched = [
                weapon for weapon in weapons
                if weapon.mask is not None
                and _mask_overlap(hand.mask, weapon.mask) >= threshold
            ]
            if matched:
                dangerous.append(DangerousPerson(hand=hand, weapons=matched))
        return dangerous
