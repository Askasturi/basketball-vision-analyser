
# Basketball Vision Analyser

Basketball Vision Analyser is a Python computer-vision project for analyzing basketball video. It detects objects, tracks players and the ball, classifies teams, estimates possession, detects passes and interceptions, maps movement to court coordinates, calculates speed and distance, and renders annotated videos.

## Features

- Video loading, frame iteration, and video writing
- Detection core with bounding boxes, detections, results, and detector factory
- Mock detector for reliable local testing
- Local YOLO detector backend
- Roboflow API detector backend
- IoU-based multi-object tracking
- Ball interpolation utilities
- Color-based team classification
- Possession estimation
- Pass and interception detection
- Court keypoints and homography utilities
- Tactical top-down court rendering
- Speed and distance analytics
- Frame annotation and full per-frame pipeline
- Synthetic demo video generator
- Mock full-pipeline demo with annotated MP4 and JSON report

## Project structure

```text
src/basketball_vision_analyser/
├── analytics/
├── classification/
├── court/
├── detection/
├── events/
├── pipeline/
├── possession/
├── tracking/
├── video/
└── visualization/

tests/
docs/
examples/
scripts/
data/
outputs/
models/
```

## Setup

```bash
git clone https://github.com/Askasturi/basketball-vision-analyser.git
cd basketball-vision-analyser

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For optional AI model integrations:

```bash
python -m pip install -e ".[ai]"
```

## Verify

```bash
python -m ruff check .
python -m pytest
```

Current verified status:

```text
221 passed
```

## Run a demo

Create a synthetic input clip:

```bash
python scripts/create_demo_video.py \
  --output data/raw/demo_synthetic.mp4 \
  --frames 60 \
  --width 640 \
  --height 360 \
  --fps 30
```

Run the full mock pipeline:

```bash
PYTHONPATH=src python examples/run_mock_pipeline_demo.py \
  --input data/raw/demo_synthetic.mp4 \
  --output-video outputs/videos/mock_pipeline_demo.mp4 \
  --output-report outputs/reports/mock_pipeline_demo.json \
  --max-frames 60
```

Outputs:

```text
outputs/videos/mock_pipeline_demo.mp4
outputs/reports/mock_pipeline_demo.json
```

## Local YOLO detection

Install optional AI dependencies:

```bash
python -m pip install -e ".[ai]"
```

Then run:

```bash
PYTHONPATH=src python examples/run_local_yolo_detection.py \
  --input data/raw/demo_synthetic.mp4 \
  --output outputs/reports/local_yolo_detections.json \
  --model yolo11n.pt \
  --max-frames 60
```

## Roboflow API backend

Copy `.env.example` to `.env` and set:

```bash
ROBOFLOW_API_KEY=your_key_here
ROBOFLOW_MODEL_ID=basketball-players-fy4c2-vfsuv/13
```

The Roboflow backend is implemented and unit-tested with a fake client. Real hosted inference requires a valid Roboflow API key.

## Documentation

- [Usage](docs/USAGE.md)
- [Milestones](docs/MILESTONES.md)

## Status

The project has a complete tested core pipeline and is ready for demo clips, real model integration, and portfolio polishing.
