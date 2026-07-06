
"""Possession result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.possession.assignment import PossessionAssignment
from basketball_vision_analyser.possession.types import PossessionStatus


@dataclass(frozen=True)
class PossessionResult:
    """Possession result for one frame."""

    frame_index: int
    assignment: PossessionAssignment
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        if self.assignment.frame_index != self.frame_index:
            msg = "assignment frame_index must match result frame_index."
            raise ValueError(msg)

    @property
    def has_player_control(self) -> bool:
        """Return whether a player controls the ball."""

        return self.assignment.status == PossessionStatus.PLAYER_CONTROL

    @property
    def player_track_id(self) -> int | None:
        """Return possessing player track ID."""

        return self.assignment.player_track_id

    @property
    def ball_track_id(self) -> int | None:
        """Return ball track ID."""

        return self.assignment.ball_track_id

    @property
    def team(self) -> PlayerTeam:
        """Return possessing team."""

        return self.assignment.team
