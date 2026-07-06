
"""Frame analysis pipeline result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basketball_vision_analyser.classification import TeamClassificationResult
from basketball_vision_analyser.detection import DetectionResult
from basketball_vision_analyser.possession import PossessionResult
from basketball_vision_analyser.tracking import TrackingResult


@dataclass(frozen=True)
class PipelineFrameResult:
    """Combined analysis result for one frame."""

    frame_index: int
    detection_result: DetectionResult
    tracking_result: TrackingResult
    team_result: TeamClassificationResult
    possession_result: PossessionResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        frame_indexes = {
            self.detection_result.frame_index,
            self.tracking_result.frame_index,
            self.team_result.frame_index,
            self.possession_result.frame_index,
        }

        if frame_indexes != {self.frame_index}:
            msg = "all result frame indexes must match frame_index."
            raise ValueError(msg)
