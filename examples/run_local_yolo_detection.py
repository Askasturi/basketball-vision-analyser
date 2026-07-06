
"""Run local YOLO detection on a video and export JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from basketball_vision_analyser.detection import (
    Detection,
    LocalYOLODetector,
    YOLODetectorConfig,
)
from basketball_vision_analyser.video import VideoLoader


def detection_to_dict(detection: Detection) -> dict[str, object]:
    """Convert a detection to a JSON-safe dictionary."""

    return {
        "class_name": detection.class_name.value,
        "confidence": detection.confidence,
        "box_xyxy": detection.box.to_xyxy(),
        "center": detection.center,
        "track_id": detection.track_id,
        "metadata": detection.metadata,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run local YOLO detection on a basketball video."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/local_yolo_detections.json"),
    )
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    """Run detection and write a JSON report."""

    args = parse_args()

    config = YOLODetectorConfig(
        model_path=args.model,
        confidence_threshold=args.confidence,
    )
    detector = LocalYOLODetector(config=config)
    loader = VideoLoader(args.input)

    records: list[dict[str, object]] = []

    for video_frame in loader.frames(max_frames=args.max_frames):
        result = detector.predict_frame(
            video_frame.image,
            frame_index=video_frame.index,
        )
        records.append(
            {
                "frame_index": result.frame_index,
                "detections": [
                    detection_to_dict(detection)
                    for detection in result.detections
                ],
                "metadata": result.metadata,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))

    print(f"Wrote detection report to {args.output}")


if __name__ == "__main__":
    main()
