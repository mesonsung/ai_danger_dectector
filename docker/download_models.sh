#!/bin/bash
# 下載公開偵測模型

set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from pathlib import Path
from model_loader import ensure_hand_model, ensure_weapon_model

ensure_hand_model(Path("models/hand_landmarker.task"))
ensure_weapon_model(Path("models/weapon.pt"))
print("所有模型就緒")
PY
