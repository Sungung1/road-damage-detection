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

Road image -> YOLO detector -> Bounding box rendering -> Result image

## Project Structure

```text
road-damage-detection/
├── src/road_damage_detection/
├── data/
├── models/
├── notebooks/
├── templates/
├── train.py
├── inference.py
├── app.py
├── requirements.txt
└── README.md
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python train.py --help
```

Run inference:

```bash
python inference.py --help
```

Launch the web demo:

```bash
python app.py
```

## Results

The current demo saves the latest inference image to `data/results/output.jpg`. Once a trained checkpoint is available at `models/best.pt`, uploads are rendered with predicted damage boxes.

## Demo

- Web upload flow via Flask
- Original project report: `miniporject_YOLO_Road_Damage_Detect.pdf`
