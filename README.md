
# Basketball Vision Analyser

Basketball Vision Analyser is a Python computer-vision project for analyzing basketball game footage.

The goal is to detect players, the ball, hoop, referees, scoreboard, and clock; track players and ball across frames; assign teams; estimate possession; detect passes and interceptions; map player movement to a tactical court view; calculate speed and distance; and export annotated videos plus JSON/CSV analytics.

## Planned Features

- Video loading and output rendering with OpenCV
- Local YOLO detection backend
- Optional Roboflow API detection backend
- Player and ball tracking
- Team classification using jersey colors and optional FashionCLIP
- Ball possession estimation
- Pass and interception detection
- Court keypoint detection and tactical top-down mapping
- Speed and distance metrics
- Annotated video export
- Structured analytics export

## Project Structure

```text
basketball-vision-analyser/
├── src/
│   └── basketball_vision_analyser/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_package_import.py
├── docs/
├── examples/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── weights/
│   └── checkpoints/
├── outputs/
│   ├── videos/
│   └── reports/
├── scripts/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore 
