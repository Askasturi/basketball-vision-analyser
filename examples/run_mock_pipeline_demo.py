
"""Run the full pipeline with the deterministic mock detector."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2

from basketball_vision_analyser.detection import MockDetector
from basketball_vision_analyser.pipeline import (
    BasketballFrameAnalyzer,
    PipelineFrameResult,
)
from basketball_vision_analyser.video import VideoLoader


def result_to_dict(result: PipelineFrameResult) -> dict[str, Any]:
    """Convert a pipeline frame result to JSON-safe data."""

    return {
        "frame_index": result.frame_index,
        "detections": [
            {
                "class_name": detection.class_name.value,
                "confidence": detection.confidence,
                "box_xyxy": detection.box.to_xyxy(),
                "track_id": detection.track_id,
                "metadata": detection.metadata,
            }
            for detection in result.detection_result.detections
        ],
        "tracks": [
            {
                "track_id": tracked_object.track_id,
                "class_name": tracked_object.class_name.value,
                "confidence": tracked_object.confidence,
                "box_xyxy": tracked_object.box.to_xyxy(),
                "center": tracked_object.center,
                "age": tracked_object.age,
                "hits": tracked_object.hits,
                "lost_frames": tracked_object.lost_frames,
            }
            for tracked_object in result.tracking_result.objects
        ],
        "teams": [
            {
                "track_id": assignment.track_id,
                "team": assignment.team.value,
                "confidence": assignment.confidence,
                "dominant_color_rgb": assignment.dominant_color_rgb,
            }
            for assignment in result.team_result.assignments
        ],
        "possession": {
            "status": result.possession_result.assignment.status.value,
            "player_track_id": result.possession_result.player_track_id,
            "ball_track_id": result.possession_result.ball_track_id,
            "team": result.possession_result.team.value,
            "confidence": result.possession_result.assignment.confidence,
            "distance_px": result.possession_result.assignment.distance_px,
        },
    }


def run_demo(
    *,
    input_path: Path,
    output_video_path: Path,
    output_report_path: Path,
    max_frames: int,
) -> list[dict[str, Any]]:
    """Run mock pipeline demo and write annotated video plus JSON report."""

    if max_frames <= 0:
        msg = "max_frames must be greater than 0."
        raise ValueError(msg)

    loader = VideoLoader(input_path)
    metadata = loader.get_metadata()
    analyzer = BasketballFrameAnalyzer(detector=MockDetector())

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )

    if not writer.isOpened():
        msg = f"Could not open video writer for {output_video_path}."
        raise OSError(msg)

    records: list[dict[str, Any]] = []

    try:
        for video_frame in loader.frames(max_frames=max_frames):
            result = analyzer.analyze_frame(
                video_frame.image,
                frame_index=video_frame.index,
            )
            annotated_frame = analyzer.render_frame(video_frame.image, result)
            writer.write(annotated_frame)
            records.append(result_to_dict(result))
    finally:
        writer.release()

    output_report_path.write_text(json.dumps(records, indent=2))

    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the mock Basketball Vision Analyser pipeline."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("outputs/videos/mock_pipeline_demo.mp4"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("outputs/reports/mock_pipeline_demo.json"),
    )
    parser.add_argument("--max-frames", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    records = run_demo(
        input_path=args.input,
        output_video_path=args.output_video,
        output_report_path=args.output_report,
        max_frames=args.max_frames,
    )

    print(f"Analyzed {len(records)} frames")
    print(f"Wrote annotated video to {args.output_video}")
    print(f"Wrote JSON report to {args.output_report}")


if __name__ == "__main__":
    main()
