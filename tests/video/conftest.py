
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def sample_video_path(tmp_path: Path) -> Path:
    output_path = tmp_path / "sample.mp4"
    width = 64
    height = 48
    fps = 10.0
    frame_count = 12

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        pytest.fail("Could not create sample test video.")

    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = index * 10
        frame[:, :, 1] = 100
        frame[:, :, 2] = 200
        cv2.putText(
            frame,
            str(index),
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        writer.write(frame)

    writer.release()
    return output_path
