
import pytest

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.possession import (
    PossessionAssignment,
    PossessionStatus,
)


def test_possession_assignment_properties() -> None:
    assignment = PossessionAssignment(
        frame_index=2,
        status=PossessionStatus.PLAYER_CONTROL,
        player_track_id=10,
        ball_track_id=99,
        team=PlayerTeam.TEAM_A,
        distance_px=12.5,
        confidence=0.8,
    )

    assert assignment.frame_index == 2
    assert assignment.status == PossessionStatus.PLAYER_CONTROL
    assert assignment.player_track_id == 10
    assert assignment.ball_track_id == 99
    assert assignment.team == PlayerTeam.TEAM_A
    assert assignment.distance_px == 12.5
    assert assignment.confidence == 0.8


def test_possession_assignment_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        PossessionAssignment(
            frame_index=-1,
            status=PossessionStatus.NO_BALL,
        )


def test_possession_assignment_rejects_negative_player_track_id() -> None:
    with pytest.raises(ValueError, match="player_track_id"):
        PossessionAssignment(
            frame_index=0,
            status=PossessionStatus.PLAYER_CONTROL,
            player_track_id=-1,
        )


def test_possession_assignment_rejects_negative_ball_track_id() -> None:
    with pytest.raises(ValueError, match="ball_track_id"):
        PossessionAssignment(
            frame_index=0,
            status=PossessionStatus.PLAYER_CONTROL,
            ball_track_id=-1,
        )


def test_possession_assignment_rejects_negative_distance() -> None:
    with pytest.raises(ValueError, match="distance_px"):
        PossessionAssignment(
            frame_index=0,
            status=PossessionStatus.PLAYER_CONTROL,
            distance_px=-1,
        )


def test_possession_assignment_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PossessionAssignment(
            frame_index=0,
            status=PossessionStatus.PLAYER_CONTROL,
            confidence=2,
        )
