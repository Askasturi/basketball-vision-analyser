
"""Mock detector used for tests and development."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from basketball_vision_analyser.detection.base_detector import BaseDetector
from basketball_vision_analyser.detection.bounding_box import BoundingBox
from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.detection import Detection
from basketball_vision_analyser.detection.result import DetectionResult
from basketball_vision_analyser.detection.types import DetectionClass, DetectorBackend


class MockDetector(BaseDetector):
    """Deterministic detector that does not require model files."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        detections: Sequence[Detection] | None = None,
    ) -> None:
        config = config or DetectorConfig(backend=DetectorBackend.MOCK)
        super().__init__(config=config)
        self._detections = tuple(detections) if detections is not None else None

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Return deterministic fake detections for one frame."""

        self.validate_frame(frame)

        if frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        if self._detections is not None:
            detections = self._detections
        else:
            detections = self._generate_default_detections(frame)

        return DetectionResult(
            frame_index=frame_index,
            detections=detections,
            image_shape=frame.shape,
            inference_time_ms=0.0,
            metadata={"backend": DetectorBackend.MOCK.value},
        )

    def _generate_default_detections(
        self,
        frame: np.ndarray,
    ) -> tuple[Detection, ...]:
        height, width = frame.shape[:2]

        player_box = BoundingBox(
            x1=width * 0.20,
            y1=height * 0.20,
            x2=width * 0.45,
            y2=height * 0.90,
        )
        ball_box = BoundingBox(
            x1=width * 0.50,
            y1=height * 0.35,
            x2=width * 0.57,
            y2=height * 0.45,
        )
        hoop_box = BoundingBox(
            x1=width * 0.75,
            y1=height * 0.10,
            x2=width * 0.92,
            y2=height * 0.30,
        )

        return (
            Detection(
                box=player_box,
                class_name=DetectionClass.PLAYER,
                confidence=0.90,
            ),
            Detection(
                box=ball_box,
                class_name=DetectionClass.BALL,
                confidence=0.85,
            ),
            Detection(
                box=hoop_box,
                class_name=DetectionClass.HOOP,
                confidence=0.80,
            ),
        )
