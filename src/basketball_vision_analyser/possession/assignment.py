
"""Possession assignment model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.possession.types import PossessionStatus


@dataclass(frozen=True)
class PossessionAssignment:
    """Possession estimate for one frame."""

    frame_index: int
    status: PossessionStatus
    player_track_id: int | None = None
    ball_track_id: int | None = None
    team: PlayerTeam = PlayerTeam.UNKNOWN
    distance_px: float | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        if self.player_track_id is not None and self.player_track_id < 0:
            msg = "player_track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.ball_track_id is not None and self.ball_track_id < 0:
            msg = "ball_track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.distance_px is not None and self.distance_px < 0:
            msg = "distance_px must be greater than or equal to 0."
            raise ValueError(msg)

        if not 0 <= self.confidence <= 1:
            msg = "confidence must be between 0 and 1."
            raise ValueError(msg)
