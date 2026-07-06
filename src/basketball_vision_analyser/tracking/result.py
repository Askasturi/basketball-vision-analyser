
"""Tracking result model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.detection import DetectionClass
from basketball_vision_analyser.tracking.track import TrackedObject


@dataclass(frozen=True)
class TrackingResult:
    """Tracked objects for one frame."""

    frame_index: int
    objects: tuple[TrackedObject, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        object.__setattr__(self, "objects", tuple(self.objects))

    def __len__(self) -> int:
        return len(self.objects)

    def __iter__(self) -> Iterator[TrackedObject]:
        return iter(self.objects)

    @property
    def track_ids(self) -> set[int]:
        """Return all track IDs in this result."""

        return {tracked_object.track_id for tracked_object in self.objects}

    def count(self, class_name: DetectionClass | str | None = None) -> int:
        """Count tracked objects, optionally filtered by class."""

        if class_name is None:
            return len(self.objects)

        target = (
            DetectionClass.from_label(class_name)
            if isinstance(class_name, str)
            else class_name
        )

        return sum(obj.class_name == target for obj in self.objects)

    def for_class(self, class_name: DetectionClass | str) -> tuple[TrackedObject, ...]:
        """Return tracked objects matching a class."""

        target = (
            DetectionClass.from_label(class_name)
            if isinstance(class_name, str)
            else class_name
        )

        return tuple(obj for obj in self.objects if obj.class_name == target)

    def by_track_id(self, track_id: int) -> TrackedObject | None:
        """Return one tracked object by track ID."""

        for tracked_object in self.objects:
            if tracked_object.track_id == track_id:
                return tracked_object

        return None
