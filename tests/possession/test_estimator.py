
from basketball_vision_analyser.classification import (
    PlayerTeam,
    TeamAssignment,
    TeamClassificationResult,
)
from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
)
from basketball_vision_analyser.possession import (
    PossessionConfig,
    PossessionEstimator,
    PossessionStatus,
)
from basketball_vision_analyser.tracking import TrackedObject, TrackingResult


def make_tracked_object(
    track_id: int,
    class_name: DetectionClass,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
) -> TrackedObject:
    detection = Detection(
        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_name=class_name,
        confidence=confidence,
        track_id=track_id,
    )

    return TrackedObject(
        track_id=track_id,
        detection=detection,
        frame_index=0,
        age=1,
        hits=1,
    )


def test_possession_estimator_assigns_nearest_player() -> None:
    ball = make_tracked_object(99, DetectionClass.BALL, 45, 45, 55, 55)
    player_near = make_tracked_object(
        1,
        DetectionClass.PLAYER,
        30,
        30,
        70,
        70,
    )
    player_far = make_tracked_object(
        2,
        DetectionClass.PLAYER,
        150,
        150,
        190,
        190,
    )
    tracking_result = TrackingResult(
        frame_index=0,
        objects=(ball, player_near, player_far),
    )

    estimator = PossessionEstimator()
    result = estimator.estimate_frame(tracking_result)

    assert result.assignment.status == PossessionStatus.PLAYER_CONTROL
    assert result.player_track_id == 1
    assert result.ball_track_id == 99
    assert result.has_player_control is True
    assert result.assignment.confidence == 1.0


def test_possession_estimator_uses_team_classification() -> None:
    ball = make_tracked_object(99, DetectionClass.BALL, 45, 45, 55, 55)
    player = make_tracked_object(7, DetectionClass.PLAYER, 30, 30, 70, 70)
    tracking_result = TrackingResult(frame_index=4, objects=(ball, player))
    team_result = TeamClassificationResult(
        frame_index=4,
        assignments=(
            TeamAssignment(
                track_id=7,
                team=PlayerTeam.TEAM_A,
                confidence=0.9,
                frame_index=4,
            ),
        ),
    )

    estimator = PossessionEstimator()
    result = estimator.estimate_frame(tracking_result, team_result)

    assert result.team == PlayerTeam.TEAM_A
    assert result.player_track_id == 7


def test_possession_estimator_marks_loose_ball_when_player_too_far() -> None:
    ball = make_tracked_object(99, DetectionClass.BALL, 0, 0, 10, 10)
    player = make_tracked_object(1, DetectionClass.PLAYER, 200, 200, 240, 240)
    tracking_result = TrackingResult(frame_index=0, objects=(ball, player))

    estimator = PossessionEstimator(
        PossessionConfig(max_control_distance_px=25),
    )
    result = estimator.estimate_frame(tracking_result)

    assert result.assignment.status == PossessionStatus.LOOSE_BALL
    assert result.player_track_id is None
    assert result.ball_track_id == 99
    assert result.assignment.confidence == 0.0


def test_possession_estimator_marks_no_ball() -> None:
    player = make_tracked_object(1, DetectionClass.PLAYER, 30, 30, 70, 70)
    tracking_result = TrackingResult(frame_index=0, objects=(player,))

    estimator = PossessionEstimator()
    result = estimator.estimate_frame(tracking_result)

    assert result.assignment.status == PossessionStatus.NO_BALL
    assert result.ball_track_id is None
    assert result.player_track_id is None


def test_possession_estimator_marks_no_player() -> None:
    ball = make_tracked_object(99, DetectionClass.BALL, 45, 45, 55, 55)
    tracking_result = TrackingResult(frame_index=0, objects=(ball,))

    estimator = PossessionEstimator()
    result = estimator.estimate_frame(tracking_result)

    assert result.assignment.status == PossessionStatus.NO_PLAYER
    assert result.ball_track_id == 99
    assert result.player_track_id is None


def test_possession_estimator_respects_min_control_confidence() -> None:
    ball = make_tracked_object(99, DetectionClass.BALL, 0, 0, 10, 10)
    player = make_tracked_object(1, DetectionClass.PLAYER, 40, 0, 80, 40)
    tracking_result = TrackingResult(frame_index=0, objects=(ball, player))

    estimator = PossessionEstimator(
        PossessionConfig(
            max_control_distance_px=80,
            min_control_confidence=0.8,
        ),
    )
    result = estimator.estimate_frame(tracking_result)

    assert result.assignment.status == PossessionStatus.LOOSE_BALL
    assert result.player_track_id == 1
    assert result.assignment.confidence < 0.8


def test_possession_estimator_selects_highest_confidence_ball() -> None:
    weak_ball = make_tracked_object(
        91,
        DetectionClass.BALL,
        200,
        200,
        210,
        210,
        confidence=0.2,
    )
    strong_ball = make_tracked_object(
        99,
        DetectionClass.BALL,
        45,
        45,
        55,
        55,
        confidence=0.95,
    )
    player = make_tracked_object(1, DetectionClass.PLAYER, 30, 30, 70, 70)
    tracking_result = TrackingResult(
        frame_index=0,
        objects=(weak_ball, strong_ball, player),
    )

    estimator = PossessionEstimator()
    result = estimator.estimate_frame(tracking_result)

    assert result.ball_track_id == 99
    assert result.player_track_id == 1
