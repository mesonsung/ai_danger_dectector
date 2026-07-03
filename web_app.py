"""Web UI：即時影像串流與偵測狀態（Streamlit）"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

import config
from camera_service import CameraService
from object_detector import ObjectDetector

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web UI 即時危險人物識別")
    parser.add_argument("--host", default=config.WEB_HOST)
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument(
        "--source",
        type=str,
        default=os.environ.get("VIDEO_SOURCE", str(config.VIDEO_SOURCE)),
    )
    parser.add_argument("--weapon-model", type=str, default=str(config.WEAPON_MODEL))
    parser.add_argument("--hand-model", type=str, default=str(config.HAND_MODEL))
    parser.add_argument(
        "--conf",
        type=float,
        default=float(os.environ.get("DETECTION_CONF", config.DETECTION_CONF)),
    )
    parser.add_argument("--snapshot-dir", type=str, default=str(config.ALERT_DIR))
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args(argv)


def get_cli_args() -> argparse.Namespace:
    if "--" in sys.argv:
        return parse_args(sys.argv[sys.argv.index("--") + 1 :])
    return parse_args()


@st.cache_resource
def load_detector(
    weapon_model: str,
    hand_model: str,
    conf: float,
    auto_download: bool,
) -> ObjectDetector:
    logger.info("初始化偵測模型...")
    return ObjectDetector(
        weapon_model_path=weapon_model,
        hand_model_path=hand_model,
        conf=conf,
        auto_download=auto_download,
    )


def get_camera_service(args: argparse.Namespace) -> CameraService:
    if "camera_service" not in st.session_state:
        snapshot_dir = Path(args.snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        detector = load_detector(
            args.weapon_model,
            args.hand_model,
            args.conf,
            auto_download=not args.no_download,
        )
        service = CameraService(args.source, detector, snapshot_dir)
        service.start()
        st.session_state.camera_service = service
        atexit.register(service.stop)
    return st.session_state.camera_service


def run_ui(args: argparse.Namespace) -> None:
    st.set_page_config(
        page_title="危險人物即時識別",
        page_icon="⚠️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    service = get_camera_service(args)

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.title("危險人物即時識別")
    badge_slot = header_right.empty()

    col_video, col_panel = st.columns([2, 1], gap="large")
    video_slot = col_video.empty()

    with col_panel:
        st.subheader("即時狀態")
        fps_metric = st.empty()
        detect_fps_label = st.empty()
        stat_cols = st.columns(3)
        danger_metric = stat_cols[0].empty()
        hands_metric = stat_cols[1].empty()
        weapons_metric = stat_cols[2].empty()
        st.divider()
        st.caption("圖例")
        st.markdown(
            "- 🟡 手部\n- 🟠 危險物品\n- 🔴 手持危險物品（警報）",
        )
        st.divider()
        st.subheader("警報紀錄")
        alerts_slot = st.empty()

    last_frame_seq = -1
    last_status_at = 0.0
    last_danger_active: bool | None = None
    last_alerts_key: tuple[str, ...] | None = None
    video_interval = config.WEB_VIDEO_REFRESH_MS / 1000
    status_interval = config.WEB_STATUS_REFRESH_MS / 1000

    while True:
        frame_rgb, frame_seq = service.get_frame_rgb()
        if frame_rgb is not None and frame_seq != last_frame_seq:
            last_frame_seq = frame_seq
            video_slot.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True,
                output_format="JPEG",
            )

        now = time.time()
        if now - last_status_at >= status_interval:
            last_status_at = now
            status = service.get_status()

            if status.danger_active != last_danger_active:
                last_danger_active = status.danger_active
                if status.danger_active:
                    badge_slot.error("⚠ 危險警報", icon="🚨")
                else:
                    badge_slot.success("監控中", icon="✅")

            fps_metric.metric("FPS", f"{status.fps:.1f}")
            detect_fps_label.caption(f"偵測 {status.detect_fps:.1f} FPS")
            danger_metric.metric("危險事件", status.danger_count)
            hands_metric.metric("手部", status.hands)
            weapons_metric.metric("危險物品", status.weapons)

            alerts_key = tuple(status.alerts)
            if alerts_key != last_alerts_key:
                last_alerts_key = alerts_key
                if status.alerts:
                    alerts_slot.markdown(
                        "\n".join(f"- 🔴 `{alert}`" for alert in status.alerts),
                    )
                else:
                    alerts_slot.caption("尚無警報")

        time.sleep(video_interval)


def launch_streamlit(args: argparse.Namespace) -> None:
    passthrough = []
    if "--" in sys.argv:
        passthrough = sys.argv[sys.argv.index("--") + 1 :]
    elif len(sys.argv) > 1:
        passthrough = sys.argv[1:]

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        __file__,
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        *passthrough,
    ]
    logger.info("Web UI: http://%s:%d", args.host, args.port)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cli_args = get_cli_args()

    if get_script_run_ctx() is None:
        launch_streamlit(cli_args)
    else:
        run_ui(cli_args)
