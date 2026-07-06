
"""Movement analytics models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.court import Point2D


@dataclass(frozen=True)
class TrackPosition:
    """A tracked object's court position for one frame."""

    track_id: int
    frame_index: int
    court_point: Point2D | None
    team: PlayerTeam = PlayerTeam.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.track_id < 0:
            msg = "track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)


@dataclass(frozen=True)
class MovementSample:
    """Frame-to-frame movement for one track."""

    track_id: int
    start_frame_index: int
    end_frame_index: int
    start_point: Point2D
    end_point: Point2D
    distance_ft: float
    elapsed_seconds: float
    speed_ft_per_second: float
    team: PlayerTeam = PlayerTeam.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.track_id < 0:
            msg = "track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.start_frame_index < 0:
            msg = "start_frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        if self.end_frame_index <= self.start_frame_index:
            msg = "end_frame_index must be greater than start_frame_index."
            raise ValueError(msg)

        if self.distance_ft < 0:
            msg = "distance_ft must be greater than or equal to 0."
            raise ValueError(msg)

        if self.elapsed_seconds <= 0:
            msg = "elapsed_seconds must be greater than 0."
            raise ValueError(msg)

        if self.speed_ft_per_second < 0:
            msg = "speed_ft_per_second must be greater than or equal to 0."
            raise ValueError(msg)


@dataclass(frozen=True)
class MovementSummary:
    """Movement summary for one track."""

    track_id: int
    total_distance_ft: float
    total_elapsed_seconds: float
    average_speed_ft_per_second: float
    max_speed_ft_per_second: float
    sample_count: int
    team: PlayerTeam = PlayerTeam.UNKNOWN

    def __post_init__(self) -> None:
        if self.track_id < 0:
            msg = "track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.total_distance_ft < 0:
            msg = "total_distance_ft must be greater than or equal to 0."
            raise ValueError(msg)

        if self.total_elapsed_seconds < 0:
            msg = "total_elapsed_seconds must be greater than or equal to 0."
            raise ValueError(msg)

        if self.average_speed_ft_per_second < 0:
            msg = "average_speed_ft_per_second must be greater than or equal to 0."
            raise ValueError(msg)

        if self.max_speed_ft_per_second < 0:
            msg = "max_speed_ft_per_second must be greater than or equal to 0."
            raise ValueError(msg)

        if self.sample_count < 0:
            msg = "sample_count must be greater than or equal to 0."
            raise ValueError(msg)
