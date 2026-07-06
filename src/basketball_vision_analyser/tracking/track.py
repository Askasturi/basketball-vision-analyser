
"""Tracked object models."""

from __future__ import annotations

from dataclasses import dataclass

from basketball_vision_analyser.detection import BoundingBox, Detection, DetectionClass


@dataclass(frozen=True)
class TrackedObject:
    """A detection assigned to a persistent track."""

    track_id: int
    detection: Detection
    frame_index: int
    age: int
    hits: int
    lost_frames: int = 0

    def __post_init__(self) -> None:
        if self.track_id < 0:
            msg = "track_id must be greater than or equal to 0."
            raise ValueError(msg)

        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        if self.age <= 0:
            msg = "age must be greater than 0."
            raise ValueError(msg)

        if self.hits <= 0:
            msg = "hits must be greater than 0."
            raise ValueError(msg)

        if self.lost_frames < 0:
            msg = "lost_frames must be greater than or equal to 0."
            raise ValueError(msg)

    @property
    def class_name(self) -> DetectionClass:
        """Return tracked object class."""

        return self.detection.class_name

    @property
    def box(self) -> BoundingBox:
        """Return tracked object bounding box."""

        return self.detection.box

    @property
    def center(self) -> tuple[float, float]:
        """Return tracked object center."""

        return self.detection.center

    @property
    def confidence(self) -> float:
        """Return detection confidence."""

        return self.detection.confidence
