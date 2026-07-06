
"""Frame iteration utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from basketball_vision_analyser.video.exceptions import VideoOpenError


@dataclass(frozen=True)
class VideoFrame:
    """A single decoded video frame."""

    index: int
    image: np.ndarray
    timestamp_seconds: float


class FrameIterator(Iterator[VideoFrame]):
    """Iterate through frames from a video file."""

    def __init__(
        self,
        path: str | Path,
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.stride = stride
        self.max_frames = max_frames

        self._capture: cv2.VideoCapture | None = None
        self._current_frame = start_frame
        self._frames_yielded = 0
        self._fps = 0.0

        self._validate()

    def _validate(self) -> None:
        if self.start_frame < 0:
            msg = "start_frame must be greater than or equal to 0."
            raise ValueError(msg)

        if self.end_frame is not None and self.end_frame < self.start_frame:
            msg = "end_frame must be greater than or equal to start_frame."
            raise ValueError(msg)

        if self.stride <= 0:
            msg = "stride must be greater than 0."
            raise ValueError(msg)

        if self.max_frames is not None and self.max_frames < 0:
            msg = "max_frames must be greater than or equal to 0."
            raise ValueError(msg)

    def __iter__(self) -> FrameIterator:
        self.close()

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            msg = f"Could not open video file: {self.path}"
            raise VideoOpenError(msg)

        capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 1.0

        self._capture = capture
        self._current_frame = self.start_frame
        self._frames_yielded = 0
        self._fps = fps

        return self

    def __next__(self) -> VideoFrame:
        if self._capture is None:
            self.__iter__()

        if self.max_frames is not None and self._frames_yielded >= self.max_frames:
            self.close()
            raise StopIteration

        if self.end_frame is not None and self._current_frame >= self.end_frame:
            self.close()
            raise StopIteration

        if self._capture is None:
            raise StopIteration

        ok, image = self._capture.read()
        if not ok:
            self.close()
            raise StopIteration

        frame = VideoFrame(
            index=self._current_frame,
            image=image,
            timestamp_seconds=self._current_frame / self._fps,
        )

        self._frames_yielded += 1
        self._current_frame += self.stride

        if self.stride > 1 and self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)

        return frame

    def close(self) -> None:
        """Release the OpenCV capture object."""

        if self._capture is not None:
            self._capture.release()
            self._capture = None
