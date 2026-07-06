
"""Single detection model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.detection.bounding_box import BoundingBox
from basketball_vision_analyser.detection.types import DetectionClass


@dataclass(frozen=True)
class Detection:
    """A single object detection."""

    box: BoundingBox
    class_name: DetectionClass | str
    confidence: float
    track_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.class_name, str):
            normalized_class = DetectionClass.from_label(self.class_name)
            object.__setattr__(self, "class_name", normalized_class)

        if not 0 <= self.confidence <= 1:
            msg = "confidence must be between 0 and 1."
            raise ValueError(msg)

        if self.track_id is not None and self.track_id < 0:
            msg = "track_id must be greater than or equal to 0."
            raise ValueError(msg)

    @property
    def center(self) -> tuple[float, float]:
        """Return detection center as x, y."""

        return self.box.center

    @property
    def area(self) -> float:
        """Return detection bounding-box area."""

        return self.box.area

    def with_track_id(self, track_id: int) -> Detection:
        """Return a copy with a track ID assigned."""

        return Detection(
            box=self.box,
            class_name=self.class_name,
            confidence=self.confidence,
            track_id=track_id,
            metadata=dict(self.metadata),
        )
