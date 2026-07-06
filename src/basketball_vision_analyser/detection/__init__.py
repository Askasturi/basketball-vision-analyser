
"""Detection core for Basketball Vision Analyser."""

from basketball_vision_analyser.detection.base_detector import BaseDetector
from basketball_vision_analyser.detection.bounding_box import BoundingBox
from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.detection import Detection
from basketball_vision_analyser.detection.factory import DetectorFactory
from basketball_vision_analyser.detection.mock_detector import MockDetector
from basketball_vision_analyser.detection.result import DetectionResult
from basketball_vision_analyser.detection.types import DetectionClass, DetectorBackend

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionClass",
    "DetectionResult",
    "DetectorBackend",
    "DetectorConfig",
    "DetectorFactory",
    "MockDetector",
]
