
from pathlib import Path

import pytest

from basketball_vision_analyser.video import FrameIterator, VideoLoader


def test_frame_iterator_reads_all_frames(sample_video_path: Path) -> None:
    frames = list(VideoLoader(sample_video_path).frames())

    assert len(frames) == 12
    assert frames[0].index == 0
    assert frames[1].index == 1
    assert frames[0].image.shape == (48, 64, 3)


def test_frame_iterator_supports_max_frames(sample_video_path: Path) -> None:
    frames = list(VideoLoader(sample_video_path).frames(max_frames=3))

    assert [frame.index for frame in frames] == [0, 1, 2]


def test_frame_iterator_supports_stride(sample_video_path: Path) -> None:
    frames = list(VideoLoader(sample_video_path).frames(stride=2, max_frames=4))

    assert [frame.index for frame in frames] == [0, 2, 4, 6]


def test_frame_iterator_supports_frame_range(sample_video_path: Path) -> None:
    frames = list(
        VideoLoader(sample_video_path).frames(
            start_frame=2,
            end_frame=5,
        )
    )

    assert [frame.index for frame in frames] == [2, 3, 4]


def test_frame_iterator_rejects_invalid_stride(sample_video_path: Path) -> None:
    with pytest.raises(ValueError, match="stride"):
        FrameIterator(sample_video_path, stride=0)
