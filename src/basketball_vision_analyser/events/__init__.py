
"""Event detection for Basketball Vision Analyser."""

from basketball_vision_analyser.events.config import EventDetectionConfig
from basketball_vision_analyser.events.detector import PossessionEventDetector
from basketball_vision_analyser.events.event import PlayEvent
from basketball_vision_analyser.events.result import EventDetectionResult
from basketball_vision_analyser.events.types import EventType

__all__ = [
    "EventDetectionConfig",
    "EventDetectionResult",
    "EventType",
    "PlayEvent",
    "PossessionEventDetector",
]
