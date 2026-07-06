
"""Team assignment model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.classification.config import RGBColor
from basketball_vision_analyser.classification.types import PlayerTeam


@dataclass(frozen=True)
class TeamAssignment:
    """Team classification for one tracked object."""

    track_id: int
    team: PlayerTeam
    confidence: float
    frame_index: int
    dominant_color_rgb: RGBColor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.track_id < 0:
            msg = "track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if not 0 <= self.confidence <= 1:
            msg = "confidence must be between 0 and 1."
            raise ValueError(msg)

        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)
