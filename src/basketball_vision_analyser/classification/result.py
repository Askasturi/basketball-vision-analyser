
"""Team classification result model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.classification.assignment import TeamAssignment
from basketball_vision_analyser.classification.types import PlayerTeam


@dataclass(frozen=True)
class TeamClassificationResult:
    """Team assignments for one frame."""

    frame_index: int
    assignments: tuple[TeamAssignment, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        object.__setattr__(self, "assignments", tuple(self.assignments))

    def __len__(self) -> int:
        return len(self.assignments)

    def __iter__(self) -> Iterator[TeamAssignment]:
        return iter(self.assignments)

    def for_track_id(self, track_id: int) -> TeamAssignment | None:
        """Return an assignment by track ID."""

        for assignment in self.assignments:
            if assignment.track_id == track_id:
                return assignment

        return None

    def for_team(self, team: PlayerTeam) -> tuple[TeamAssignment, ...]:
        """Return all assignments for a team."""

        return tuple(
            assignment for assignment in self.assignments if assignment.team == team
        )

    def count(self, team: PlayerTeam | None = None) -> int:
        """Count assignments, optionally filtered by team."""

        if team is None:
            return len(self.assignments)

        return sum(assignment.team == team for assignment in self.assignments)
