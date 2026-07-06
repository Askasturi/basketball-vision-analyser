
"""Detection core for Basketball Vision Analyser."""

from basketball_vision_analyser.detection.base_detector import BaseDetector
from basketball_vision_analyser.detection.bounding_box import BoundingBox
from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.detection import Detection
from basketball_vision_analyser.detection.factory import DetectorFactory
from basketball_vision_analyser.detection.local_yolo_detector import LocalYOLODetector
from basketball_vision_analyser.detection.mock_detector import MockDetector
from basketball_vision_analyser.detection.result import DetectionResult
from basketball_vision_analyser.detection.roboflow_config import RoboflowDetectorConfig
from basketball_vision_analyser.detection.roboflow_detector import RoboflowAPIDetector
from basketball_vision_analyser.detection.types import DetectionClass, DetectorBackend
from basketball_vision_analyser.detection.yolo_config import YOLODetectorConfig

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionClass",
    "DetectionResult",
    "DetectorBackend",
    "DetectorConfig",
    "DetectorFactory",
    "LocalYOLODetector",
    "MockDetector",
    "RoboflowAPIDetector",
    "RoboflowDetectorConfig",
    "YOLODetectorConfig",
]
