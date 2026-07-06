
"""Base detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.result import DetectionResult


class BaseDetector(ABC):
    """Base class for all detector backends."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()

    @abstractmethod
    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Run detection on one frame."""

    def predict_frames(
        self,
        frames: Iterable[np.ndarray],
        *,
        start_index: int = 0,
    ) -> list[DetectionResult]:
        """Run detection on multiple frames."""

        results: list[DetectionResult] = []

        for offset, frame in enumerate(frames):
            results.append(
                self.predict_frame(
                    frame,
                    frame_index=start_index + offset,
                )
            )

        return results

    @staticmethod
    def validate_frame(frame: np.ndarray) -> None:
        """Validate that a frame is a color image."""

        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy array."
            raise TypeError(msg)

        if frame.ndim != 3 or frame.shape[2] != 3:
            msg = "frame must have shape height x width x 3."
            raise ValueError(msg)
