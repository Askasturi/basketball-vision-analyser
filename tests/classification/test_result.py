
from basketball_vision_analyser.classification import (
    PlayerTeam,
    TeamAssignment,
    TeamClassificationResult,
)


def test_team_classification_result_filters_and_counts() -> None:
    team_a = TeamAssignment(
        track_id=1,
        team=PlayerTeam.TEAM_A,
        confidence=0.9,
        frame_index=0,
    )
    team_b = TeamAssignment(
        track_id=2,
        team=PlayerTeam.TEAM_B,
        confidence=0.8,
        frame_index=0,
    )

    result = TeamClassificationResult(
        frame_index=0,
        assignments=(team_a, team_b),
    )

    assert len(result) == 2
    assert result.count() == 2
    assert result.count(PlayerTeam.TEAM_A) == 1
    assert result.for_team(PlayerTeam.TEAM_B) == (team_b,)
    assert result.for_track_id(1) == team_a
    assert result.for_track_id(999) is None
