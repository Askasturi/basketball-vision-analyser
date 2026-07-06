
"""Detector factory."""

from __future__ import annotations

from basketball_vision_analyser.detection.base_detector import BaseDetector
from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.local_yolo_detector import LocalYOLODetector
from basketball_vision_analyser.detection.mock_detector import MockDetector
from basketball_vision_analyser.detection.types import DetectorBackend


class DetectorFactory:
    """Create detector instances from configuration."""

    _registry: dict[DetectorBackend, type[BaseDetector]] = {
        DetectorBackend.MOCK: MockDetector,
        DetectorBackend.LOCAL_YOLO: LocalYOLODetector,
    }

    @classmethod
    def register(
        cls,
        backend: DetectorBackend,
        detector_cls: type[BaseDetector],
    ) -> None:
        """Register a detector class for a backend."""

        cls._registry[backend] = detector_cls

    @classmethod
    def create(
        cls,
        config: DetectorConfig | None = None,
    ) -> BaseDetector:
        """Create a detector for the requested backend."""

        config = config or DetectorConfig()
        detector_cls = cls._registry.get(config.backend)

        if detector_cls is None:
            msg = f"No detector registered for backend: {config.backend}"
            raise ValueError(msg)

        return detector_cls(config=config)
