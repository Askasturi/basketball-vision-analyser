
import pytest

from basketball_vision_analyser.classification import TeamClassificationResult
from basketball_vision_analyser.detection import DetectionResult
from basketball_vision_analyser.pipeline import PipelineFrameResult
from basketball_vision_analyser.possession import (
    PossessionAssignment,
    PossessionResult,
    PossessionStatus,
)
from basketball_vision_analyser.tracking import TrackingResult


def make_pipeline_frame_result(frame_index: int = 0) -> PipelineFrameResult:
    return PipelineFrameResult(
        frame_index=frame_index,
        detection_result=DetectionResult(frame_index=frame_index),
        tracking_result=TrackingResult(frame_index=frame_index),
        team_result=TeamClassificationResult(frame_index=frame_index),
        possession_result=PossessionResult(
            frame_index=frame_index,
            assignment=PossessionAssignment(
                frame_index=frame_index,
                status=PossessionStatus.NO_BALL,
            ),
        ),
    )


def test_pipeline_frame_result_properties() -> None:
    result = make_pipeline_frame_result(frame_index=3)

    assert result.frame_index == 3
    assert result.detection_result.frame_index == 3
    assert result.tracking_result.frame_index == 3
    assert result.team_result.frame_index == 3
    assert result.possession_result.frame_index == 3


def test_pipeline_frame_result_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        make_pipeline_frame_result(frame_index=-1)


def test_pipeline_frame_result_requires_matching_frame_indexes() -> None:
    with pytest.raises(ValueError, match="all result frame indexes"):
        PipelineFrameResult(
            frame_index=0,
            detection_result=DetectionResult(frame_index=1),
            tracking_result=TrackingResult(frame_index=0),
            team_result=TeamClassificationResult(frame_index=0),
            possession_result=PossessionResult(
                frame_index=0,
                assignment=PossessionAssignment(
                    frame_index=0,
                    status=PossessionStatus.NO_BALL,
                ),
            ),
        )
