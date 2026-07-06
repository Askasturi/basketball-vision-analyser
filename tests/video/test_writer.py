
from pathlib import Path

import numpy as np
import pytest

from basketball_vision_analyser.video import InvalidFrameError, VideoLoader, VideoWriter


def test_video_writer_writes_video(tmp_path: Path) -> None:
    output_path = tmp_path / "output.mp4"
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    with VideoWriter(output_path, fps=10.0, frame_size=(64, 48)) as writer:
        writer.write_frame(frame)
        writer.write_frame(frame)

    assert output_path.exists()

    metadata = VideoLoader(output_path).get_metadata()
    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.frame_count > 0


def test_video_writer_rejects_wrong_frame_size(tmp_path: Path) -> None:
    output_path = tmp_path / "output.mp4"
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    with VideoWriter(output_path, fps=10.0, frame_size=(64, 48)) as writer:
        with pytest.raises(InvalidFrameError, match="frame shape"):
            writer.write_frame(frame)


def test_video_writer_rejects_invalid_frame_dimensions(tmp_path: Path) -> None:
    output_path = tmp_path / "output.mp4"
    frame = np.zeros((48, 64), dtype=np.uint8)

    with VideoWriter(output_path, fps=10.0, frame_size=(64, 48)) as writer:
        with pytest.raises(InvalidFrameError, match="height x width x 3"):
            writer.write_frame(frame)
