
"""Event detection configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventDetectionConfig:
    """Configuration for pass and interception detection."""

    max_gap_frames: int = 30
    require_known_teams: bool = True

    def __post_init__(self) -> None:
        if self.max_gap_frames < 0:
            msg = "max_gap_frames must be greater than or equal to 0."
            raise ValueError(msg)
