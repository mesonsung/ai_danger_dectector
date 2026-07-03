#!/bin/bash
set -euo pipefail

echo "[entrypoint] 檢查並下載公開模型..."
python - <<'PY'
from pathlib import Path
from model_loader import ensure_hand_model, ensure_weapon_model

ensure_hand_model(Path("/app/models/hand_landmarker.task"))
ensure_weapon_model(Path("/app/models/weapon.pt"))
print("所有模型就緒")
PY

exec python /app/web_app.py --no-download "$@"
