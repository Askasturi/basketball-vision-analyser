
import pytest

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.events import EventType, PlayEvent


def test_play_event_properties() -> None:
    event = PlayEvent(
        event_type=EventType.PASS,
        start_frame_index=10,
        end_frame_index=14,
        from_player_track_id=1,
        to_player_track_id=2,
        from_team=PlayerTeam.TEAM_A,
        to_team=PlayerTeam.TEAM_A,
        confidence=0.8,
    )

    assert event.event_type == EventType.PASS
    assert event.start_frame_index == 10
    assert event.end_frame_index == 14
    assert event.from_player_track_id == 1
    assert event.to_player_track_id == 2
    assert event.confidence == 0.8


def test_play_event_rejects_negative_start_frame() -> None:
    with pytest.raises(ValueError, match="start_frame_index"):
        PlayEvent(
            event_type=EventType.PASS,
            start_frame_index=-1,
            end_frame_index=0,
            from_player_track_id=1,
            to_player_track_id=2,
            from_team=PlayerTeam.TEAM_A,
            to_team=PlayerTeam.TEAM_A,
            confidence=0.8,
        )


def test_play_event_rejects_end_before_start() -> None:
    with pytest.raises(ValueError, match="end_frame_index"):
        PlayEvent(
            event_type=EventType.PASS,
            start_frame_index=10,
            end_frame_index=9,
            from_player_track_id=1,
            to_player_track_id=2,
            from_team=PlayerTeam.TEAM_A,
            to_team=PlayerTeam.TEAM_A,
            confidence=0.8,
        )


def test_play_event_rejects_same_player_ids() -> None:
    with pytest.raises(ValueError, match="must differ"):
        PlayEvent(
            event_type=EventType.PASS,
            start_frame_index=10,
            end_frame_index=11,
            from_player_track_id=1,
            to_player_track_id=1,
            from_team=PlayerTeam.TEAM_A,
            to_team=PlayerTeam.TEAM_A,
            confidence=0.8,
        )


def test_play_event_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PlayEvent(
            event_type=EventType.PASS,
            start_frame_index=10,
            end_frame_index=11,
            from_player_track_id=1,
            to_player_track_id=2,
            from_team=PlayerTeam.TEAM_A,
            to_team=PlayerTeam.TEAM_A,
            confidence=2,
        )
