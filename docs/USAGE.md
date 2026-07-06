
# Basketball Vision Analyser Usage

## Verify the project

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest
```

## Create a synthetic demo clip

```bash
python scripts/create_demo_video.py \
  --output data/raw/demo_synthetic.mp4 \
  --frames 60 \
  --width 640 \
  --height 360 \
  --fps 30
```

## Run the full mock pipeline

```bash
python examples/run_mock_pipeline_demo.py \
  --input data/raw/demo_synthetic.mp4 \
  --output-video outputs/videos/mock_pipeline_demo.mp4 \
  --output-report outputs/reports/mock_pipeline_demo.json \
  --max-frames 60
```

If Python cannot find the package, run:

```bash
PYTHONPATH=src python examples/run_mock_pipeline_demo.py \
  --input data/raw/demo_synthetic.mp4 \
  --output-video outputs/videos/mock_pipeline_demo.mp4 \
  --output-report outputs/reports/mock_pipeline_demo.json \
  --max-frames 60
```

This runs:

1. Mock detection
2. IoU tracking
3. Color team classification
4. Possession estimation
5. Frame annotation
6. JSON report export

## Expected outputs

```text
outputs/videos/mock_pipeline_demo.mp4
outputs/reports/mock_pipeline_demo.json
```

## Run local YOLO detection

Install AI extras first:

```bash
python -m pip install -e ".[ai]"
```

Then run:

```bash
python examples/run_local_yolo_detection.py \
  --input data/raw/demo_synthetic.mp4 \
  --output outputs/reports/local_yolo_detections.json \
  --model yolo11n.pt \
  --max-frames 60
```

If needed:

```bash
PYTHONPATH=src python examples/run_local_yolo_detection.py \
  --input data/raw/demo_synthetic.mp4 \
  --output outputs/reports/local_yolo_detections.json \
  --model yolo11n.pt \
  --max-frames 60
```

## Roboflow backend

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Then set:

```bash
ROBOFLOW_API_KEY=your_key_here
ROBOFLOW_MODEL_ID=basketball-players-fy4c2-vfsuv/13
```

The Roboflow backend is implemented and unit-tested with a fake client. Real Roboflow inference requires a valid API key and the optional AI dependencies.

## Useful commands

Run all tests:

```bash
python -m pytest
```

Run one test folder:

```bash
python -m pytest tests/pipeline
```

Run Ruff:

```bash
python -m ruff check .
```

Auto-fix Ruff issues:

```bash
python -m ruff check . --fix
```

Check Git status:

```bash
git status
```

## Current pipeline

```text
VideoLoader
    ↓
Detector
    ↓
SimpleTracker
    ↓
ColorTeamClassifier
    ↓
PossessionEstimator
    ↓
FrameAnnotator
    ↓
Annotated video + JSON report
```

## Main demo workflow

```bash
python scripts/create_demo_video.py \
  --output data/raw/demo_synthetic.mp4 \
  --frames 60 \
  --width 640 \
  --height 360 \
  --fps 30

python examples/run_mock_pipeline_demo.py \
  --input data/raw/demo_synthetic.mp4 \
  --output-video outputs/videos/mock_pipeline_demo.mp4 \
  --output-report outputs/reports/mock_pipeline_demo.json \
  --max-frames 60
```
