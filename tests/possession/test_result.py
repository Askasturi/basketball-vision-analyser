
import pytest

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.possession import (
    PossessionAssignment,
    PossessionResult,
    PossessionStatus,
)


def test_possession_result_properties() -> None:
    assignment = PossessionAssignment(
        frame_index=5,
        status=PossessionStatus.PLAYER_CONTROL,
        player_track_id=1,
        ball_track_id=2,
        team=PlayerTeam.TEAM_B,
        confidence=0.7,
    )
    result = PossessionResult(frame_index=5, assignment=assignment)

    assert result.has_player_control is True
    assert result.player_track_id == 1
    assert result.ball_track_id == 2
    assert result.team == PlayerTeam.TEAM_B


def test_possession_result_rejects_negative_frame_index() -> None:
    assignment = PossessionAssignment(
        frame_index=0,
        status=PossessionStatus.NO_BALL,
    )

    with pytest.raises(ValueError, match="frame_index"):
        PossessionResult(frame_index=-1, assignment=assignment)


def test_possession_result_requires_matching_frame_index() -> None:
    assignment = PossessionAssignment(
        frame_index=1,
        status=PossessionStatus.NO_BALL,
    )

    with pytest.raises(ValueError, match="assignment frame_index"):
        PossessionResult(frame_index=2, assignment=assignment)
