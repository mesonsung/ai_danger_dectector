# 武器 Segmentation 模型訓練

使用 Roboflow 公開資料集 [Weapon Detection (seg)](https://universe.roboflow.com/training-3hezf/weapon-detection-q9trv)（knife、Long gun、short gun）fine-tune `yolov8n-seg.pt`。

## 目錄結構

```
model-training/
├── train_weapon_seg.py   # 訓練腳本
├── data/                 # Roboflow 資料集（gitignore）
├── runs/                 # 訓練輸出（gitignore）
└── yolov8n-seg.pt        # 預訓練 base 模型（首次訓練自動下載）
```

## 設定

1. 至 [Roboflow API 設定](https://app.roboflow.com/settings/api) 取得免費 API Key
2. 複製環境變數範本：

```bash
cp model-training/.env.example model-training/.env
```

## 執行

```bash
uv sync --group dev
uv run python model-training/train_weapon_seg.py --epochs 50 --batch 8
```

訓練完成後會自動備份舊的 `models/weapon.pt`，並部署新的 segmentation 模型。若資料集已下載，可加 `--skip-download` 跳過下載。

常用參數：

```bash
uv run python model-training/train_weapon_seg.py --epochs 100 --imgsz 640 --batch 8 --device 0
uv run python model-training/train_weapon_seg.py --skip-download --no-deploy   # 只訓練，不覆寫模型
uv run python model-training/train_weapon_seg.py --resume --device 0             # 從 last.pt 接續
```
