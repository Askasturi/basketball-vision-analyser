
"""Movement analytics result model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.analytics.movement import (
    MovementSample,
    MovementSummary,
)


@dataclass(frozen=True)
class MovementAnalysisResult:
    """Speed and distance analytics result."""

    samples: tuple[MovementSample, ...] = field(default_factory=tuple)
    summaries: tuple[MovementSummary, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "summaries", tuple(self.summaries))

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[MovementSample]:
        return iter(self.samples)

    @property
    def track_ids(self) -> set[int]:
        """Return track IDs with movement samples."""

        return {sample.track_id for sample in self.samples}

    @property
    def total_distance_ft(self) -> float:
        """Return total distance across all track summaries."""

        return sum(summary.total_distance_ft for summary in self.summaries)

    def samples_for_track(self, track_id: int) -> tuple[MovementSample, ...]:
        """Return samples for one track."""

        return tuple(sample for sample in self.samples if sample.track_id == track_id)

    def summary_for_track(self, track_id: int) -> MovementSummary | None:
        """Return summary for one track."""

        for summary in self.summaries:
            if summary.track_id == track_id:
                return summary

        return None
