
"""Tracking configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from basketball_vision_analyser.detection import DetectionClass


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration for object tracking."""

    iou_threshold: float = 0.3
    max_lost_frames: int = 5
    min_confidence: float = 0.0
    track_classes: tuple[DetectionClass | str, ...] = field(
        default_factory=lambda: (
            DetectionClass.PLAYER,
            DetectionClass.BALL,
            DetectionClass.REFEREE,
        )
    )

    def __post_init__(self) -> None:
        if not 0 <= self.iou_threshold <= 1:
            msg = "iou_threshold must be between 0 and 1."
            raise ValueError(msg)

        if self.max_lost_frames < 0:
            msg = "max_lost_frames must be greater than or equal to 0."
            raise ValueError(msg)

        if not 0 <= self.min_confidence <= 1:
            msg = "min_confidence must be between 0 and 1."
            raise ValueError(msg)

        normalized_classes: list[DetectionClass] = []

        for class_name in self.track_classes:
            if isinstance(class_name, str):
                normalized_classes.append(DetectionClass.from_label(class_name))
            else:
                normalized_classes.append(class_name)

        object.__setattr__(self, "track_classes", tuple(normalized_classes))
