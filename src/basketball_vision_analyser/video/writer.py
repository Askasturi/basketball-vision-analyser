
"""Video writing utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from basketball_vision_analyser.video.exceptions import (
    InvalidFrameError,
    VideoWriteError,
)


class VideoWriter:
    """Write frames to a video file."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
        create_parent: bool = True,
    ) -> None:
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.create_parent = create_parent
        self._writer: cv2.VideoWriter | None = None

        self._validate_config()

    def _validate_config(self) -> None:
        width, height = self.frame_size

        if self.fps <= 0:
            msg = "fps must be greater than 0."
            raise ValueError(msg)

        if width <= 0 or height <= 0:
            msg = "frame_size values must be greater than 0."
            raise ValueError(msg)

        if len(self.codec) != 4:
            msg = "codec must be a 4-character string."
            raise ValueError(msg)

    def open(self) -> None:
        """Open the video writer."""

        if self.create_parent:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            self.frame_size,
        )

        if not self._writer.isOpened():
            msg = f"Could not open video writer for: {self.output_path}"
            raise VideoWriteError(msg)

    def is_open(self) -> bool:
        """Return whether the video writer is open."""

        return self._writer is not None and self._writer.isOpened()

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one frame to the video."""

        if self._writer is None:
            self.open()

        self._validate_frame(frame)

        if self._writer is None:
            msg = "Video writer is not open."
            raise VideoWriteError(msg)

        self._writer.write(frame)

    def write_frames(self, frames: Iterable[np.ndarray]) -> int:
        """Write multiple frames and return the count written."""

        count = 0
        for frame in frames:
            self.write_frame(frame)
            count += 1
        return count

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy array."
            raise InvalidFrameError(msg)

        if frame.ndim != 3 or frame.shape[2] != 3:
            msg = "frame must have shape height x width x 3."
            raise InvalidFrameError(msg)

        expected_width, expected_height = self.frame_size
        actual_height, actual_width = frame.shape[:2]

        if (actual_width, actual_height) != (expected_width, expected_height):
            msg = (
                "frame shape does not match writer frame_size. "
                f"Expected {(expected_width, expected_height)}, "
                f"got {(actual_width, actual_height)}."
            )
            raise InvalidFrameError(msg)

    def close(self) -> None:
        """Release the video writer."""

        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
