# Road Damage Detection

## Overview

Road Damage Detection is a computer vision project for identifying cracks and surface defects from uploaded road images. The repository is organized around a dedicated `src/` package, a standalone training entrypoint, and a standalone inference entrypoint so the project can evolve beyond the original demo app.

## Tech Stack

- Python
- Flask
- PyTorch
- Ultralytics YOLO
- OpenCV
- Pillow

## Architecture

Input image -> YOLO model -> Defect localization -> Rendered output image

## Usage

```bash
pip install -r requirements.txt
python train.py
python inference.py sample.jpg
```

## Results

The current demo saves the latest inference image to `data/results/output.jpg`. Once a trained checkpoint is available at `models/best.pt`, uploads are rendered with predicted damage boxes.
