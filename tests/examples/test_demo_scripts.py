
import json
from pathlib import Path

import cv2
import pytest

from examples.run_mock_pipeline_demo import run_demo
from scripts.create_demo_video import create_demo_video


def test_create_demo_video_writes_video(tmp_path: Path) -> None:
    output_path = tmp_path / "demo.mp4"

    create_demo_video(
        output_path,
        frame_count=5,
        width=160,
        height=120,
        fps=10,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    capture = cv2.VideoCapture(str(output_path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    finally:
        capture.release()


def test_create_demo_video_rejects_invalid_frame_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame_count"):
        create_demo_video(tmp_path / "bad.mp4", frame_count=0)


def test_run_mock_pipeline_demo_writes_video_and_report(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_video_path = tmp_path / "output.mp4"
    output_report_path = tmp_path / "report.json"

    create_demo_video(
        input_path,
        frame_count=6,
        width=160,
        height=120,
        fps=10,
    )

    records = run_demo(
        input_path=input_path,
        output_video_path=output_video_path,
        output_report_path=output_report_path,
        max_frames=4,
    )

    assert len(records) == 4
    assert output_video_path.exists()
    assert output_video_path.stat().st_size > 0
    assert output_report_path.exists()

    loaded_records = json.loads(output_report_path.read_text())

    assert len(loaded_records) == 4
    assert "detections" in loaded_records[0]
    assert "tracks" in loaded_records[0]
    assert "teams" in loaded_records[0]
    assert "possession" in loaded_records[0]


def test_run_mock_pipeline_demo_rejects_invalid_max_frames(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    create_demo_video(input_path, frame_count=2, width=160, height=120, fps=10)

    with pytest.raises(ValueError, match="max_frames"):
        run_demo(
            input_path=input_path,
            output_video_path=tmp_path / "output.mp4",
            output_report_path=tmp_path / "report.json",
            max_frames=0,
        )
