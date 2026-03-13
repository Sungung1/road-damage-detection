from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def predict_image(image_path: Path, output_path: Path, model_path: Path | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path is None or YOLO is None:
        shutil.copy2(image_path, output_path)
        return output_path

    model = YOLO(str(model_path))
    results = model.predict(source=str(image_path), save=False, verbose=False)

    if not results:
        shutil.copy2(image_path, output_path)
        return output_path

    plotted = results[0].plot()
    Image.fromarray(plotted[..., ::-1]).save(output_path)
    return output_path
