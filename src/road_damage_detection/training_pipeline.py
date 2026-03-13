from __future__ import annotations

import argparse
from pathlib import Path

import yaml

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def create_data_config(data_dir: Path, output_path: Path) -> Path:
    config = {
        "train": str(data_dir / "train"),
        "val": str(data_dir / "valid"),
        "test": str(data_dir / "test"),
        "nc": 1,
        "names": ["damage"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output_path


def train_model(data_dir: Path, output_dir: Path, model_name: str, epochs: int, imgsz: int) -> None:
    if YOLO is None:
        raise RuntimeError("ultralytics is required to train the road damage model.")

    data_config = create_data_config(data_dir, output_dir / "data.yaml")
    model = YOLO(model_name)
    model.train(
        data=str(data_config),
        epochs=epochs,
        imgsz=imgsz,
        project=str(output_dir),
        name="road-damage-yolo",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a YOLO model for road-damage detection.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root containing train/valid/test.")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Directory for checkpoints and config.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
