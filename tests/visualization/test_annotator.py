
import numpy as np
import pytest

from basketball_vision_analyser.classification import (
    PlayerTeam,
    TeamAssignment,
    TeamClassificationResult,
)
from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionResult,
)
from basketball_vision_analyser.possession import (
    PossessionAssignment,
    PossessionResult,
    PossessionStatus,
)
from basketball_vision_analyser.tracking import TrackedObject, TrackingResult
from basketball_vision_analyser.visualization import FrameAnnotator


def blank_frame() -> np.ndarray:
    return np.zeros((100, 160, 3), dtype=np.uint8)


def make_detection(class_name: DetectionClass) -> Detection:
    return Detection(
        box=BoundingBox(x1=10, y1=10, x2=60, y2=60),
        class_name=class_name,
        confidence=0.9,
    )


def make_tracked_player(track_id: int = 1) -> TrackedObject:
    detection = make_detection(DetectionClass.PLAYER).with_track_id(track_id)

    return TrackedObject(
        track_id=track_id,
        detection=detection,
        frame_index=0,
        age=1,
        hits=1,
    )


def test_frame_annotator_draws_detection_result() -> None:
    frame = blank_frame()
    detection_result = DetectionResult(
        frame_index=0,
        detections=(make_detection(DetectionClass.PLAYER),),
        image_shape=frame.shape,
    )

    output = FrameAnnotator().annotate_frame(
        frame,
        detection_result=detection_result,
    )

    assert output is not frame
    assert np.any(output != frame)


def test_frame_annotator_draws_tracking_and_team_result() -> None:
    frame = blank_frame()
    tracking_result = TrackingResult(
        frame_index=0,
        objects=(make_tracked_player(7),),
    )
    team_result = TeamClassificationResult(
        frame_index=0,
        assignments=(
            TeamAssignment(
                track_id=7,
                team=PlayerTeam.TEAM_A,
                confidence=0.9,
                frame_index=0,
            ),
        ),
    )

    output = FrameAnnotator().annotate_frame(
        frame,
        tracking_result=tracking_result,
        team_result=team_result,
    )

    assert np.any(output != frame)


def test_frame_annotator_draws_possession_banner() -> None:
    frame = blank_frame()
    possession = PossessionResult(
        frame_index=0,
        assignment=PossessionAssignment(
            frame_index=0,
            status=PossessionStatus.PLAYER_CONTROL,
            player_track_id=1,
            ball_track_id=2,
            team=PlayerTeam.TEAM_A,
            confidence=0.8,
        ),
    )

    output = FrameAnnotator().annotate_frame(
        frame,
        possession_result=possession,
    )

    assert np.any(output != frame)


def test_frame_annotator_rejects_invalid_frame_shape() -> None:
    frame = np.zeros((100, 160), dtype=np.uint8)

    with pytest.raises(ValueError, match="height x width x 3"):
        FrameAnnotator().annotate_frame(frame)


def test_frame_annotator_rejects_non_array_frame() -> None:
    with pytest.raises(TypeError, match="numpy array"):
        FrameAnnotator().annotate_frame("bad")  # type: ignore[arg-type]
