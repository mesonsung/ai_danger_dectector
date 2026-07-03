"""畫面疊加繪製"""

from __future__ import annotations

import cv2

import config
from hand_detector import HandDetection
from object_detector import DangerousPerson, WeaponDetection


def draw_label(
    frame, text: str, origin: tuple[int, int],
    color: tuple[int, int, int], bg_color: tuple[int, int, int] | None = None,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    ty = max(y, th + 8)
    if bg_color:
        cv2.rectangle(frame, (x, ty - th - 8), (x + tw + 8, ty + baseline), bg_color, -1)
    cv2.putText(frame, text, (x + 4, ty - 4), font, scale, color, thickness, cv2.LINE_AA)


def draw_hands(frame, hands: list[HandDetection]) -> None:
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_HAND, 2)
        draw_label(frame, f"{hand.label} {hand.confidence:.0%}", (x1, y1), (0, 0, 0), config.COLOR_HAND)


def draw_weapons(frame, weapons: list[WeaponDetection]) -> None:
    for weapon in weapons:
        x1, y1, x2, y2 = weapon.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_WEAPON, 2)
        draw_label(frame, f"{weapon.label} {weapon.confidence:.0%}", (x1, y1), (255, 255, 255), config.COLOR_WEAPON)


def draw_dangerous(frame, persons: list[DangerousPerson]) -> None:
    for dp in persons:
        x1, y1, x2, y2 = dp.hand.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_DANGER, 3)
        label = f"危險: {dp.hand.label}手持有 {dp.weapon_labels}"
        draw_label(frame, label, (x1, y1 - 10), (255, 255, 255), config.COLOR_DANGER)


def draw_status_bar(frame, status: str, danger: bool) -> None:
    h, w = frame.shape[:2]
    color = config.COLOR_DANGER if danger else (50, 50, 50)
    cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_alert_banner(frame) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), config.COLOR_DANGER, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(
        frame, "!!! 偵測到手持危險物品 !!!", (w // 2 - 240, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )


def render_frame(
    frame,
    hands,
    weapons,
    dangerous,
    status: str,
    danger: bool,
) -> None:
    draw_hands(frame, hands)
    draw_weapons(frame, weapons)
    if dangerous:
        draw_dangerous(frame, dangerous)
    draw_status_bar(frame, status, danger)
    if danger:
        draw_alert_banner(frame)
