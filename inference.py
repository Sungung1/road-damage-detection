from __future__ import annotations

import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference for a single road image.")
    parser.add_argument("image", type=Path, help="Path to an input road image.")
    parser.add_argument("--output", type=Path, default=Path("data/results/output.jpg"), help="Path to save the rendered result.")
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"), help="Trained checkpoint path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from src.road_damage_detection.predict import predict_image

    model_path = args.model if args.model.exists() else None
    predict_image(args.image, args.output, model_path=model_path)


if __name__ == "__main__":
    main()
