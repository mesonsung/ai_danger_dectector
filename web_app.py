"""Web UI：即時影像串流與偵測狀態"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

import config
from camera_service import CameraService
from object_detector import ObjectDetector

logger = logging.getLogger(__name__)

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>危險人物即時識別</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #0f1117;
      color: #e8eaed;
      min-height: 100vh;
    }
    header {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid #2a2f3a;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    h1 { font-size: 1.25rem; font-weight: 600; }
    .badge {
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      background: #1e3a2f;
      color: #4ade80;
    }
    .badge.danger { background: #3f1d1d; color: #f87171; animation: pulse 1s infinite; }
    @keyframes pulse { 50% { opacity: 0.7; } }
    main {
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 1rem;
      padding: 1rem;
      max-width: 1400px;
      margin: 0 auto;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
    .video-wrap {
      background: #1a1d26;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid #2a2f3a;
    }
    .video-wrap img {
      width: 100%;
      display: block;
      background: #000;
      min-height: 360px;
      object-fit: contain;
    }
    .panel {
      background: #1a1d26;
      border-radius: 12px;
      border: 1px solid #2a2f3a;
      padding: 1rem;
    }
    .panel h2 {
      font-size: 0.9rem;
      color: #9aa0a6;
      margin-bottom: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem; }
    .stat {
      background: #0f1117;
      border-radius: 8px;
      padding: 0.75rem;
      text-align: center;
    }
    .stat .val { font-size: 1.5rem; font-weight: 700; }
    .stat .lbl { font-size: 0.75rem; color: #9aa0a6; margin-top: 0.25rem; }
    .legend { font-size: 0.85rem; line-height: 1.8; color: #9aa0a6; margin-bottom: 1rem; }
    .legend span { display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
    .alerts { max-height: 280px; overflow-y: auto; }
    .alert-item {
      padding: 0.5rem 0.6rem;
      border-radius: 6px;
      background: #2a1515;
      color: #fca5a5;
      font-size: 0.85rem;
      margin-bottom: 0.4rem;
      border-left: 3px solid #ef4444;
    }
    .empty { color: #6b7280; font-size: 0.85rem; }
  </style>
</head>
<body>
  <header>
    <h1>危險人物即時識別</h1>
    <span id="status-badge" class="badge">監控中</span>
  </header>
  <main>
    <div class="video-wrap">
      <img src="/video_feed" alt="即時影像" />
    </div>
    <aside class="panel">
      <h2>即時狀態</h2>
      <div class="stats">
        <div class="stat"><div class="val" id="fps">-</div><div class="lbl">FPS</div></div>
        <div class="stat"><div class="val" id="danger">0</div><div class="lbl">危險事件</div></div>
        <div class="stat"><div class="val" id="hands">0</div><div class="lbl">手部</div></div>
        <div class="stat"><div class="val" id="weapons">0</div><div class="lbl">危險物品</div></div>
      </div>
      <div class="legend">
        <div><span style="background:#ffc800"></span>手部</div>
        <div><span style="background:#ffa500"></span>危險物品</div>
        <div><span style="background:#ff0000"></span>手持危險物品（警報）</div>
      </div>
      <h2>警報紀錄</h2>
      <div class="alerts" id="alerts"><p class="empty">尚無警報</p></div>
    </aside>
  </main>
  <script>
    async function poll() {
      try {
        const r = await fetch('/api/status');
        const s = await r.json();
        document.getElementById('fps').textContent = s.fps;
        document.getElementById('danger').textContent = s.danger_count;
        document.getElementById('hands').textContent = s.hands;
        document.getElementById('weapons').textContent = s.weapons;
        const badge = document.getElementById('status-badge');
        if (s.danger_active) {
          badge.textContent = '⚠ 危險警報';
          badge.className = 'badge danger';
        } else {
          badge.textContent = '監控中';
          badge.className = 'badge';
        }
        const box = document.getElementById('alerts');
        if (s.alerts.length === 0) {
          box.innerHTML = '<p class="empty">尚無警報</p>';
        } else {
          box.innerHTML = s.alerts.map(a => `<div class="alert-item">${a}</div>`).join('');
        }
      } catch (e) { /* ignore */ }
    }
    setInterval(poll, 1000);
    poll();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web UI 即時危險人物識別")
    parser.add_argument("--host", default=config.WEB_HOST)
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument("--source", type=str, default=str(config.VIDEO_SOURCE))
    parser.add_argument("--weapon-model", type=str, default=str(config.WEAPON_MODEL))
    parser.add_argument("--hand-model", type=str, default=str(config.HAND_MODEL))
    parser.add_argument("--conf", type=float, default=config.DETECTION_CONF)
    parser.add_argument("--snapshot-dir", type=str, default=str(config.ALERT_DIR))
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def create_app(service: CameraService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        yield
        service.stop()

    app = FastAPI(title="危險人物即時識別", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return INDEX_HTML

    @app.get("/api/status")
    async def api_status():
        s = service.get_status()
        return {
            "fps": s.fps,
            "danger_count": s.danger_count,
            "danger_active": s.danger_active,
            "hands": s.hands,
            "weapons": s.weapons,
            "alerts": s.alerts,
            "updated_at": s.updated_at,
        }

    @app.get("/video_feed")
    async def video_feed():
        async def stream():
            while True:
                jpeg = service.get_jpeg()
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                await asyncio.sleep(0.033)

        return StreamingResponse(
            stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("初始化偵測模型...")
    detector = ObjectDetector(
        weapon_model_path=args.weapon_model,
        hand_model_path=args.hand_model,
        conf=args.conf,
        auto_download=not args.no_download,
    )

    camera = CameraService(args.source, detector, snapshot_dir)
    app = create_app(camera)

    logger.info("Web UI: http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
