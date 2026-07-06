
"""Detector configuration."""

from __future__ import annotations

from dataclasses import dataclass

from basketball_vision_analyser.detection.types import DetectorBackend


@dataclass(frozen=True)
class DetectorConfig:
    """Shared detector configuration."""

    backend: DetectorBackend | str = DetectorBackend.MOCK
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "cpu"
    max_detections: int = 300

    def __post_init__(self) -> None:
        if isinstance(self.backend, str):
            object.__setattr__(self, "backend", DetectorBackend(self.backend))

        if not 0 <= self.confidence_threshold <= 1:
            msg = "confidence_threshold must be between 0 and 1."
            raise ValueError(msg)

        if not 0 <= self.iou_threshold <= 1:
            msg = "iou_threshold must be between 0 and 1."
            raise ValueError(msg)

        if self.image_size <= 0:
            msg = "image_size must be greater than 0."
            raise ValueError(msg)

        if self.max_detections <= 0:
            msg = "max_detections must be greater than 0."
            raise ValueError(msg)

        if not self.device.strip():
            msg = "device must not be empty."
            raise ValueError(msg)
