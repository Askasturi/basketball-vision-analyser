
from pathlib import Path

import pytest

from basketball_vision_analyser.video import VideoLoader, VideoReadError


def test_video_loader_reads_metadata(sample_video_path: Path) -> None:
    loader = VideoLoader(sample_video_path)
    metadata = loader.get_metadata()

    assert metadata.path == sample_video_path
    assert metadata.fps > 0
    assert metadata.frame_count > 0
    assert metadata.width == 64
    assert metadata.height == 48


def test_video_loader_reads_single_frame(sample_video_path: Path) -> None:
    loader = VideoLoader(sample_video_path)
    frame = loader.read_frame(0)

    assert frame.index == 0
    assert frame.timestamp_seconds == 0
    assert frame.image.shape == (48, 64, 3)


def test_video_loader_rejects_negative_frame_index(sample_video_path: Path) -> None:
    loader = VideoLoader(sample_video_path)

    with pytest.raises(ValueError, match="index"):
        loader.read_frame(-1)


def test_video_loader_raises_for_missing_video(tmp_path: Path) -> None:
    loader = VideoLoader(tmp_path / "missing.mp4")

    with pytest.raises(FileNotFoundError):
        loader.get_metadata()


def test_video_loader_raises_for_unreadable_frame(sample_video_path: Path) -> None:
    loader = VideoLoader(sample_video_path)

    with pytest.raises(VideoReadError):
        loader.read_frame(10_000)
