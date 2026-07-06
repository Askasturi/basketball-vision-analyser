from pathlib import Path

import pytest

from basketball_vision_analyser.video import VideoMetadata


def test_video_metadata_properties() -> None:
    metadata = VideoMetadata(
        path=Path("sample.mp4"),
        fps=10.0,
        frame_count=25,
        width=640,
        height=360,
    )

    assert metadata.duration_seconds == 2.5
    assert metadata.resolution == (640, 360)
    assert metadata.is_valid is True


def test_video_metadata_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError, match="fps"):
        VideoMetadata(
            path=Path("sample.mp4"),
            fps=0.0,
            frame_count=25,
            width=640,
            height=360,
        )


def test_video_metadata_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="width"):
        VideoMetadata(
            path=Path("sample.mp4"),
            fps=10.0,
            frame_count=25,
            width=0,
            height=360,
        )
