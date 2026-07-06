
"""Detection result model."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.detection.detection import Detection
from basketball_vision_analyser.detection.types import DetectionClass


@dataclass(frozen=True)
class DetectionResult:
    """Detections produced for one frame."""

    frame_index: int
    detections: tuple[Detection, ...] = field(default_factory=tuple)
    image_shape: tuple[int, ...] | None = None
    inference_time_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        object.__setattr__(self, "detections", tuple(self.detections))

        if self.image_shape is not None and len(self.image_shape) not in {2, 3}:
            msg = "image_shape must have length 2 or 3."
            raise ValueError(msg)

        if self.inference_time_ms is not None and self.inference_time_ms < 0:
            msg = "inference_time_ms must be greater than or equal to 0."
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.detections)

    def __iter__(self) -> Iterator[Detection]:
        return iter(self.detections)

    def count(self, class_name: DetectionClass | str | None = None) -> int:
        """Count detections, optionally filtered by class."""

        if class_name is None:
            return len(self.detections)

        target = (
            DetectionClass.from_label(class_name)
            if isinstance(class_name, str)
            else class_name
        )
        return sum(detection.class_name == target for detection in self.detections)

    def for_class(self, class_name: DetectionClass | str) -> tuple[Detection, ...]:
        """Return detections matching one class."""

        target = (
            DetectionClass.from_label(class_name)
            if isinstance(class_name, str)
            else class_name
        )
        return tuple(
            detection
            for detection in self.detections
            if detection.class_name == target
        )

    def by_min_confidence(self, threshold: float) -> DetectionResult:
        """Return a filtered result containing detections above a threshold."""

        if not 0 <= threshold <= 1:
            msg = "threshold must be between 0 and 1."
            raise ValueError(msg)

        return DetectionResult(
            frame_index=self.frame_index,
            detections=tuple(
                detection
                for detection in self.detections
                if detection.confidence >= threshold
            ),
            image_shape=self.image_shape,
            inference_time_ms=self.inference_time_ms,
            metadata=dict(self.metadata),
        )

    @property
    def classes(self) -> set[DetectionClass]:
        """Return all classes present in this result."""

        return {detection.class_name for detection in self.detections}
