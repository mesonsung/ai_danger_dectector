"""從 Roboflow 下載武器 instance segmentation 資料集，並 fine-tune YOLOv8n-seg。"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parent
ROOT = TRAINING_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(TRAINING_ROOT / ".env")
except ImportError:
    pass

import config

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "training-3hezf"
DEFAULT_PROJECT = "weapon-detection-q9trv"
DEFAULT_DATASET_DIR = TRAINING_ROOT / "data" / "weapon-seg"
DEFAULT_RUNS_DIR = TRAINING_ROOT / "runs" / "segment"
DEFAULT_BASE_MODEL = TRAINING_ROOT / "yolov8n-seg.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下載 Roboflow 武器 seg 資料集並 fine-tune yolov8n-seg",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API Key（或設定環境變數 ROBOFLOW_API_KEY）",
    )
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=0, help="資料集版本，0 表示最新版")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="推論裝置，預設自動選 GPU")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-name", default="weapon-seg")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="從上次中斷處接續訓練（預設讀取 runs/.../weights/last.pt）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="接續訓練用的 checkpoint（預設為 run 目錄下的 weights/last.pt）",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=config.WEAPON_MODEL,
        help="訓練完成後複製 best.pt 到此路徑",
    )
    parser.add_argument("--no-deploy", action="store_true", help="訓練後不覆寫 models/weapon.pt")
    return parser.parse_args()


def dataset_search_roots(args: argparse.Namespace) -> tuple[Path, ...]:
    return (
        args.dataset_dir,
        args.dataset_dir.parent,
        TRAINING_ROOT / "data",
        TRAINING_ROOT,
        ROOT,
    )


def find_data_yaml(dataset_dir: Path) -> Path:
    direct = dataset_dir / "data.yaml"
    if direct.exists():
        return direct
    matches = sorted(dataset_dir.glob("**/data.yaml"))
    if not matches:
        raise FileNotFoundError(f"在 {dataset_dir} 找不到 data.yaml，請先下載資料集")
    return matches[0]


def try_find_data_yaml(dataset_dir: Path) -> Path | None:
    try:
        return find_data_yaml(dataset_dir)
    except FileNotFoundError:
        return None


def resolve_dataset_yaml(args: argparse.Namespace) -> Path:
    for base in dataset_search_roots(args):
        found = try_find_data_yaml(base)
        if found is not None:
            return found
    raise FileNotFoundError(
        f"找不到 data.yaml，請先下載資料集至 {args.dataset_dir}",
    )


def download_dataset(args: argparse.Namespace) -> Path:
    if not args.api_key:
        raise RuntimeError(
            "缺少 Roboflow API Key。請至 https://app.roboflow.com/settings/api 取得，"
            "並設定 ROBOFLOW_API_KEY 環境變數或傳入 --api-key",
        )

    for base in dataset_search_roots(args):
        existing = try_find_data_yaml(base)
        if existing is not None:
            logger.info("使用既有資料集: %s", existing)
            return existing

    from roboflow import Roboflow

    logger.info(
        "下載 Roboflow 資料集 %s/%s ...",
        args.workspace,
        args.project,
    )

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)

    if args.version:
        version = project.version(args.version)
    else:
        versions = project.versions()
        if not versions:
            raise RuntimeError("找不到任何資料集版本")
        version = max(versions, key=lambda item: int(item.version))
        logger.info("使用最新版本: v%s", version.version)

    download_root = TRAINING_ROOT / "data"
    download_root.mkdir(parents=True, exist_ok=True)
    dataset = version.download("yolov8", location=str(download_root))
    download_path = Path(dataset.location)
    data_yaml = find_data_yaml(download_path)
    logger.info("資料集就緒: %s", data_yaml)
    return data_yaml


def resolve_device(requested: str) -> str | int:
    if requested:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return "cpu"


def run_dir(args: argparse.Namespace) -> Path:
    return args.runs_dir / args.run_name


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint or (run_dir(args) / "weights" / "last.pt")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint: {checkpoint}，請確認先前訓練已儲存或不要加 --resume",
        )
    return checkpoint


def train(args: argparse.Namespace, data_yaml: Path | None) -> Path:
    from ultralytics import YOLO

    device = resolve_device(args.device)

    if args.resume:
        checkpoint = resolve_checkpoint(args)
        logger.info("接續訓練 checkpoint=%s device=%s", checkpoint, device)
        model = YOLO(str(checkpoint))
        results = model.train(resume=True, device=device, verbose=True)
        save_dir = Path(results.save_dir)
    else:
        if data_yaml is None:
            raise ValueError("新訓練需要 data.yaml")
        logger.info(
            "開始訓練 base=%s device=%s epochs=%d",
            args.base_model,
            device,
            args.epochs,
        )
        model = YOLO(args.base_model)
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(args.runs_dir),
            name=args.run_name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )
        save_dir = Path(results.save_dir)

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"找不到訓練輸出: {best_pt}")

    logger.info("訓練完成: %s", best_pt)
    return best_pt


def deploy_model(best_pt: Path, output_model: Path) -> None:
    output_model.parent.mkdir(parents=True, exist_ok=True)
    if output_model.exists():
        backup = output_model.with_suffix(
            f".detect-backup-{datetime.now():%Y%m%d_%H%M%S}.pt",
        )
        shutil.copy2(output_model, backup)
        logger.info("已備份舊模型: %s", backup)

    shutil.copy2(best_pt, output_model)
    logger.info("已部署 segmentation 模型: %s", output_model)

    from ultralytics import YOLO

    model = YOLO(str(output_model))
    logger.info("模型 task=%s classes=%s", model.task, model.names)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()

    if args.resume:
        logger.info("接續訓練模式，略過資料集下載")
        data_yaml = None
    elif args.skip_download:
        data_yaml = resolve_dataset_yaml(args)
        logger.info("略過下載，使用既有資料集: %s", data_yaml)
    else:
        data_yaml = download_dataset(args)

    best_pt = train(args, data_yaml)

    if not args.no_deploy:
        deploy_model(best_pt, args.output_model)


if __name__ == "__main__":
    main()
