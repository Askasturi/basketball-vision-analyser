
"""Event detection result model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.events.event import PlayEvent
from basketball_vision_analyser.events.types import EventType


@dataclass(frozen=True)
class EventDetectionResult:
    """Detected basketball events for a sequence."""

    events: tuple[PlayEvent, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[PlayEvent]:
        return iter(self.events)

    def count(self, event_type: EventType | None = None) -> int:
        """Count events, optionally filtered by type."""

        if event_type is None:
            return len(self.events)

        return sum(event.event_type == event_type for event in self.events)

    def for_type(self, event_type: EventType) -> tuple[PlayEvent, ...]:
        """Return events matching one type."""

        return tuple(event for event in self.events if event.event_type == event_type)
