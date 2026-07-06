
"""Video loading utilities."""

from __future__ import annotations

from pathlib import Path

import cv2

from basketball_vision_analyser.video.exceptions import VideoOpenError, VideoReadError
from basketball_vision_analyser.video.frame_iterator import FrameIterator, VideoFrame
from basketball_vision_analyser.video.metadata import VideoMetadata


class VideoLoader:
    """Load video metadata and frames from disk."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _open_capture(self) -> cv2.VideoCapture:
        if not self.path.exists():
            msg = f"Video file does not exist: {self.path}"
            raise FileNotFoundError(msg)

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            msg = f"Could not open video file: {self.path}"
            raise VideoOpenError(msg)

        return capture

    def get_metadata(self) -> VideoMetadata:
        """Return metadata for the video."""

        capture = self._open_capture()

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        return VideoMetadata(
            path=self.path,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )

    def read_frame(self, index: int) -> VideoFrame:
        """Read a single frame by index."""

        if index < 0:
            msg = "index must be greater than or equal to 0."
            raise ValueError(msg)

        capture = self._open_capture()

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 1.0

            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
        finally:
            capture.release()

        if not ok:
            msg = f"Could not read frame {index} from video: {self.path}"
            raise VideoReadError(msg)

        return VideoFrame(
            index=index,
            image=image,
            timestamp_seconds=index / fps,
        )

    def frames(
        self,
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> FrameIterator:
        """Return an iterator over video frames."""

        return FrameIterator(
            self.path,
            start_frame=start_frame,
            end_frame=end_frame,
            stride=stride,
            max_frames=max_frames,
        )
