
import numpy as np
import pytest

from basketball_vision_analyser.classification import (
    ColorTeamClassifier,
    PlayerTeam,
    TeamClassificationConfig,
)
from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
)
from basketball_vision_analyser.tracking import TrackedObject, TrackingResult


def make_tracked_object(
    track_id: int,
    class_name: DetectionClass,
    box: BoundingBox,
) -> TrackedObject:
    detection = Detection(
        box=box,
        class_name=class_name,
        confidence=0.9,
        track_id=track_id,
    )
    return TrackedObject(
        track_id=track_id,
        detection=detection,
        frame_index=0,
        age=1,
        hits=1,
    )


def make_frame() -> np.ndarray:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    frame[10:60, 10:60] = (0, 0, 255)
    frame[10:60, 100:150] = (255, 0, 0)

    return frame


def test_color_team_classifier_assigns_teams_from_jersey_color() -> None:
    frame = make_frame()
    tracking_result = TrackingResult(
        frame_index=0,
        objects=(
            make_tracked_object(
                1,
                DetectionClass.PLAYER,
                BoundingBox(x1=10, y1=10, x2=60, y2=60),
            ),
            make_tracked_object(
                2,
                DetectionClass.PLAYER,
                BoundingBox(x1=100, y1=10, x2=150, y2=60),
            ),
        ),
    )
    classifier = ColorTeamClassifier(
        TeamClassificationConfig(
            team_a_rgb=(255, 0, 0),
            team_b_rgb=(0, 0, 255),
        )
    )

    result = classifier.classify_frame(frame, tracking_result)

    assert result.for_track_id(1).team == PlayerTeam.TEAM_A
    assert result.for_track_id(2).team == PlayerTeam.TEAM_B
    assert result.count(PlayerTeam.TEAM_A) == 1
    assert result.count(PlayerTeam.TEAM_B) == 1


def test_color_team_classifier_marks_referee_from_detection_class() -> None:
    frame = make_frame()
    tracking_result = TrackingResult(
        frame_index=0,
        objects=(
            make_tracked_object(
                3,
                DetectionClass.REFEREE,
                BoundingBox(x1=10, y1=10, x2=60, y2=60),
            ),
        ),
    )

    classifier = ColorTeamClassifier()
    result = classifier.classify_frame(frame, tracking_result)

    assignment = result.for_track_id(3)

    assert assignment.team == PlayerTeam.REFEREE
    assert assignment.confidence == 1.0


def test_color_team_classifier_ignores_ball_tracks() -> None:
    frame = make_frame()
    tracking_result = TrackingResult(
        frame_index=0,
        objects=(
            make_tracked_object(
                4,
                DetectionClass.BALL,
                BoundingBox(x1=10, y1=10, x2=60, y2=60),
            ),
        ),
    )

    classifier = ColorTeamClassifier()
    result = classifier.classify_frame(frame, tracking_result)

    assert len(result) == 0


def test_color_team_classifier_marks_small_crop_unknown() -> None:
    frame = make_frame()
    tracking_result = TrackingResult(
        frame_index=0,
        objects=(
            make_tracked_object(
                5,
                DetectionClass.PLAYER,
                BoundingBox(x1=0, y1=0, x2=1, y2=1),
            ),
        ),
    )
    classifier = ColorTeamClassifier(
        TeamClassificationConfig(min_crop_area=25)
    )

    result = classifier.classify_frame(frame, tracking_result)
    assignment = result.for_track_id(5)

    assert assignment.team == PlayerTeam.UNKNOWN
    assert assignment.confidence == 0.0


def test_color_team_classifier_rejects_invalid_frame_shape() -> None:
    frame = np.zeros((100, 200), dtype=np.uint8)
    classifier = ColorTeamClassifier()
    tracking_result = TrackingResult(frame_index=0)

    with pytest.raises(ValueError, match="height x width x 3"):
        classifier.classify_frame(frame, tracking_result)
