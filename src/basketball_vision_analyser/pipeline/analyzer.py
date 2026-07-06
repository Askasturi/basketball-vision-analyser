
"""Full per-frame basketball analysis pipeline."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from basketball_vision_analyser.classification import ColorTeamClassifier
from basketball_vision_analyser.detection import DetectionResult
from basketball_vision_analyser.pipeline.frame_result import PipelineFrameResult
from basketball_vision_analyser.possession import PossessionEstimator
from basketball_vision_analyser.tracking import SimpleTracker
from basketball_vision_analyser.visualization import FrameAnnotator


class FrameDetector(Protocol):
    """Protocol for frame detectors used by the pipeline."""

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Predict detections for one frame."""


class BasketballFrameAnalyzer:
    """Run detection, tracking, classification, and possession per frame."""

    def __init__(
        self,
        detector: FrameDetector,
        *,
        tracker: SimpleTracker | None = None,
        team_classifier: ColorTeamClassifier | None = None,
        possession_estimator: PossessionEstimator | None = None,
        annotator: FrameAnnotator | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker or SimpleTracker()
        self.team_classifier = team_classifier or ColorTeamClassifier()
        self.possession_estimator = possession_estimator or PossessionEstimator()
        self.annotator = annotator or FrameAnnotator()

    def analyze_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
    ) -> PipelineFrameResult:
        """Run the full analysis stack on one frame."""

        self._validate_frame(frame)

        detection_result = self.detector.predict_frame(
            frame,
            frame_index=frame_index,
        )
        tracking_result = self.tracker.update(detection_result)
        team_result = self.team_classifier.classify_frame(frame, tracking_result)
        possession_result = self.possession_estimator.estimate_frame(
            tracking_result,
            team_result,
        )

        return PipelineFrameResult(
            frame_index=frame_index,
            detection_result=detection_result,
            tracking_result=tracking_result,
            team_result=team_result,
            possession_result=possession_result,
            metadata={"pipeline": "basketball_frame_analyzer"},
        )

    def render_frame(
        self,
        frame: np.ndarray,
        result: PipelineFrameResult,
    ) -> np.ndarray:
        """Render annotations for one pipeline result."""

        self._validate_frame(frame)

        return self.annotator.annotate_frame(
            frame,
            detection_result=result.detection_result,
            tracking_result=result.tracking_result,
            team_result=result.team_result,
            possession_result=result.possession_result,
        )

    def reset(self) -> None:
        """Reset stateful pipeline components."""

        self.tracker.reset()

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy array."
            raise TypeError(msg)

        if frame.ndim != 3 or frame.shape[2] != 3:
            msg = "frame must have shape height x width x 3."
            raise ValueError(msg)
