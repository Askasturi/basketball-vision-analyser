
"""Video metadata models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing a video file."""

    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fps <= 0:
            msg = "fps must be greater than 0."
            raise ValueError(msg)

        if self.frame_count < 0:
            msg = "frame_count must be greater than or equal to 0."
            raise ValueError(msg)

        if self.width <= 0:
            msg = "width must be greater than 0."
            raise ValueError(msg)

        if self.height <= 0:
            msg = "height must be greater than 0."
            raise ValueError(msg)

        object.__setattr__(self, "path", Path(self.path))

    @property
    def duration_seconds(self) -> float:
        """Return the video duration in seconds."""

        return self.frame_count / self.fps

    @property
    def resolution(self) -> tuple[int, int]:
        """Return the video resolution as width, height."""

        return self.width, self.height

    @property
    def is_valid(self) -> bool:
        """Return whether the metadata describes a readable video."""

        return (
            self.fps > 0
            and self.frame_count >= 0
            and self.width > 0
            and self.height > 0
        )
