
import pytest

from basketball_vision_analyser.classification import PlayerTeam, TeamAssignment


def test_team_assignment_properties() -> None:
    assignment = TeamAssignment(
        track_id=1,
        team=PlayerTeam.TEAM_A,
        confidence=0.9,
        frame_index=3,
        dominant_color_rgb=(255, 0, 0),
    )

    assert assignment.track_id == 1
    assert assignment.team == PlayerTeam.TEAM_A
    assert assignment.confidence == 0.9
    assert assignment.frame_index == 3
    assert assignment.dominant_color_rgb == (255, 0, 0)


def test_team_assignment_rejects_negative_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        TeamAssignment(
            track_id=-1,
            team=PlayerTeam.TEAM_A,
            confidence=0.9,
            frame_index=0,
        )


def test_team_assignment_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        TeamAssignment(
            track_id=1,
            team=PlayerTeam.TEAM_A,
            confidence=2,
            frame_index=0,
        )


def test_team_assignment_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        TeamAssignment(
            track_id=1,
            team=PlayerTeam.TEAM_A,
            confidence=0.9,
            frame_index=-1,
        )
