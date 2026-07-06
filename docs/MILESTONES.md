
# Basketball Vision Analyser Milestones

## Completed milestones

- Milestone 0: Project setup
- Milestone 1: Video infrastructure
- Milestone 2: Detection core
- Milestone 3: Local YOLO backend
- Milestone 4: Roboflow API backend
- Milestone 5: Tracking
- Milestone 6: Team classification
- Milestone 7: Possession estimation
- Milestone 8: Pass and interception detection
- Milestone 9: Court homography and tactical view
- Milestone 10: Speed and distance analytics
- Milestone 11: Visualization and frame pipeline
- Milestone 12: Demo, docs, and GitHub polish

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

## What works now

- Synthetic demo video generation
- Mock full-pipeline demo
- Annotated video export
- JSON report export
- Detection data models
- Local YOLO detector backend
- Roboflow detector backend
- Tracking
- Team classification
- Possession estimation
- Pass and interception detection
- Court homography utilities
- Tactical view rendering
- Speed and distance analytics
- Full tested frame pipeline

## Optional advanced upgrades

- Train a basketball-specific detection model
- Add real Roboflow inference demo
- Add jersey number recognition
- Add shot detection and make/miss classification
- Add automatic court keypoint detection
- Add Streamlit or FastAPI demo app
- Add multi-video batch processing
- Add CSV export
