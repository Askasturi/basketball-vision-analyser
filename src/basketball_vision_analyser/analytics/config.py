
"""Movement analytics configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MovementAnalysisConfig:
    """Configuration for speed and distance analytics."""

    frame_rate_fps: float = 30.0
    max_frame_gap: int = 30
    max_speed_ft_per_second: float | None = 45.0

    def __post_init__(self) -> None:
        if self.frame_rate_fps <= 0:
            msg = "frame_rate_fps must be greater than 0."
            raise ValueError(msg)

        if self.max_frame_gap <= 0:
            msg = "max_frame_gap must be greater than 0."
            raise ValueError(msg)

        if (
            self.max_speed_ft_per_second is not None
            and self.max_speed_ft_per_second <= 0
        ):
            msg = "max_speed_ft_per_second must be greater than 0 when provided."
            raise ValueError(msg)
