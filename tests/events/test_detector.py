
from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.events import (
    EventDetectionConfig,
    EventType,
    PossessionEventDetector,
)
from basketball_vision_analyser.possession import (
    PossessionAssignment,
    PossessionResult,
    PossessionStatus,
)


def make_control(
    frame_index: int,
    player_track_id: int,
    team: PlayerTeam,
    confidence: float = 0.9,
) -> PossessionResult:
    assignment = PossessionAssignment(
        frame_index=frame_index,
        status=PossessionStatus.PLAYER_CONTROL,
        player_track_id=player_track_id,
        ball_track_id=99,
        team=team,
        distance_px=10,
        confidence=confidence,
    )
    return PossessionResult(frame_index=frame_index, assignment=assignment)


def make_loose_ball(frame_index: int) -> PossessionResult:
    assignment = PossessionAssignment(
        frame_index=frame_index,
        status=PossessionStatus.LOOSE_BALL,
        ball_track_id=99,
    )
    return PossessionResult(frame_index=frame_index, assignment=assignment)


def test_possession_event_detector_detects_pass_between_same_team() -> None:
    detector = PossessionEventDetector()
    result = detector.detect(
        (
            make_control(0, 1, PlayerTeam.TEAM_A, confidence=0.8),
            make_control(5, 2, PlayerTeam.TEAM_A, confidence=0.7),
        )
    )

    assert len(result) == 1

    event = result.events[0]

    assert event.event_type == EventType.PASS
    assert event.start_frame_index == 0
    assert event.end_frame_index == 5
    assert event.from_player_track_id == 1
    assert event.to_player_track_id == 2
    assert event.from_team == PlayerTeam.TEAM_A
    assert event.to_team == PlayerTeam.TEAM_A
    assert event.confidence == 0.7


def test_possession_event_detector_detects_interception_between_teams() -> None:
    detector = PossessionEventDetector()
    result = detector.detect(
        (
            make_control(0, 1, PlayerTeam.TEAM_A),
            make_control(3, 8, PlayerTeam.TEAM_B),
        )
    )

    assert result.count(EventType.INTERCEPTION) == 1

    event = result.events[0]

    assert event.event_type == EventType.INTERCEPTION
    assert event.from_player_track_id == 1
    assert event.to_player_track_id == 8
    assert event.from_team == PlayerTeam.TEAM_A
    assert event.to_team == PlayerTeam.TEAM_B


def test_possession_event_detector_ignores_same_player_control() -> None:
    detector = PossessionEventDetector()
    result = detector.detect(
        (
            make_control(0, 1, PlayerTeam.TEAM_A),
            make_control(1, 1, PlayerTeam.TEAM_A),
        )
    )

    assert len(result) == 0


def test_possession_event_detector_allows_loose_ball_between_controls() -> None:
    detector = PossessionEventDetector()
    result = detector.detect(
        (
            make_control(0, 1, PlayerTeam.TEAM_A),
            make_loose_ball(2),
            make_control(4, 2, PlayerTeam.TEAM_A),
        )
    )

    assert result.count(EventType.PASS) == 1


def test_possession_event_detector_respects_max_gap_frames() -> None:
    detector = PossessionEventDetector(
        EventDetectionConfig(max_gap_frames=2),
    )
    result = detector.detect(
        (
            make_control(0, 1, PlayerTeam.TEAM_A),
            make_control(5, 2, PlayerTeam.TEAM_A),
        )
    )

    assert len(result) == 0


def test_possession_event_detector_skips_unknown_teams_by_default() -> None:
    detector = PossessionEventDetector()
    result = detector.detect(
        (
            make_control(0, 1, PlayerTeam.UNKNOWN),
            make_control(5, 2, PlayerTeam.UNKNOWN),
        )
    )

    assert len(result) == 0


def test_possession_event_detector_can_allow_unknown_teams() -> None:
    detector = PossessionEventDetector(
        EventDetectionConfig(require_known_teams=False),
    )
    result = detector.detect(
        (
            make_control(0, 1, PlayerTeam.UNKNOWN),
            make_control(5, 2, PlayerTeam.UNKNOWN),
        )
    )

    assert result.count(EventType.PASS) == 1


def test_possession_event_detector_sorts_input_by_frame_index() -> None:
    detector = PossessionEventDetector()
    result = detector.detect(
        (
            make_control(5, 2, PlayerTeam.TEAM_A),
            make_control(0, 1, PlayerTeam.TEAM_A),
        )
    )

    assert result.count(EventType.PASS) == 1
    assert result.events[0].start_frame_index == 0
    assert result.events[0].end_frame_index == 5


def test_possession_event_detector_handles_empty_input() -> None:
    detector = PossessionEventDetector()
    result = detector.detect(())

    assert len(result) == 0
