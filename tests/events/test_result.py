
from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.events import (
    EventDetectionResult,
    EventType,
    PlayEvent,
)


def make_event(event_type: EventType) -> PlayEvent:
    return PlayEvent(
        event_type=event_type,
        start_frame_index=0,
        end_frame_index=1,
        from_player_track_id=1,
        to_player_track_id=2,
        from_team=PlayerTeam.TEAM_A,
        to_team=PlayerTeam.TEAM_A,
        confidence=0.8,
    )


def test_event_detection_result_counts_and_filters() -> None:
    pass_event = make_event(EventType.PASS)
    interception = make_event(EventType.INTERCEPTION)

    result = EventDetectionResult(events=(pass_event, interception))

    assert len(result) == 2
    assert result.count() == 2
    assert result.count(EventType.PASS) == 1
    assert result.count(EventType.INTERCEPTION) == 1
    assert result.for_type(EventType.PASS) == (pass_event,)
