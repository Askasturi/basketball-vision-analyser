
"""Basketball event model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.events.types import EventType


@dataclass(frozen=True)
class PlayEvent:
    """A detected basketball event."""

    event_type: EventType
    start_frame_index: int
    end_frame_index: int
    from_player_track_id: int
    to_player_track_id: int
    from_team: PlayerTeam
    to_team: PlayerTeam
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_frame_index < 0:
            msg = "start_frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        if self.end_frame_index < self.start_frame_index:
            msg = "end_frame_index must be greater than or equal to start_frame_index."
            raise ValueError(msg)

        if self.from_player_track_id < 0:
            msg = "from_player_track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.to_player_track_id < 0:
            msg = "to_player_track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.from_player_track_id == self.to_player_track_id:
            msg = "from_player_track_id and to_player_track_id must differ."
            raise ValueError(msg)

        if not 0 <= self.confidence <= 1:
            msg = "confidence must be between 0 and 1."
            raise ValueError(msg)
