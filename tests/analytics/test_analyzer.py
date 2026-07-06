

from basketball_vision_analyser.analytics import (
    MovementAnalysisConfig,
    MovementAnalyzer,
    TrackPosition,
)
from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.court import Point2D


def position(
    track_id: int,
    frame_index: int,
    x: float | None,
    y: float | None,
    team: PlayerTeam = PlayerTeam.UNKNOWN,
) -> TrackPosition:
    point = None if x is None or y is None else Point2D(x, y)

    return TrackPosition(
        track_id=track_id,
        frame_index=frame_index,
        court_point=point,
        team=team,
    )


def test_movement_analyzer_computes_distance_and_speed() -> None:
    analyzer = MovementAnalyzer(
        MovementAnalysisConfig(
            frame_rate_fps=1,
            max_speed_ft_per_second=None,
        )
    )

    result = analyzer.analyze_positions(
        (
            position(1, 0, 0, 0, PlayerTeam.TEAM_A),
            position(1, 1, 3, 4, PlayerTeam.TEAM_A),
        )
    )

    assert len(result) == 1

    sample = result.samples[0]
    summary = result.summary_for_track(1)

    assert sample.distance_ft == 5
    assert sample.elapsed_seconds == 1
    assert sample.speed_ft_per_second == 5
    assert summary.total_distance_ft == 5
    assert summary.average_speed_ft_per_second == 5
    assert summary.max_speed_ft_per_second == 5
    assert summary.sample_count == 1
    assert summary.team == PlayerTeam.TEAM_A


def test_movement_analyzer_sorts_positions_by_frame() -> None:
    analyzer = MovementAnalyzer(
        MovementAnalysisConfig(
            frame_rate_fps=1,
            max_speed_ft_per_second=None,
        )
    )

    result = analyzer.analyze_positions(
        (
            position(1, 2, 6, 8),
            position(1, 0, 0, 0),
            position(1, 1, 3, 4),
        )
    )

    assert len(result.samples) == 2
    assert result.total_distance_ft == 10


def test_movement_analyzer_handles_multiple_tracks() -> None:
    analyzer = MovementAnalyzer(
        MovementAnalysisConfig(
            frame_rate_fps=1,
            max_speed_ft_per_second=None,
        )
    )

    result = analyzer.analyze_positions(
        (
            position(1, 0, 0, 0),
            position(1, 1, 3, 4),
            position(2, 0, 0, 0),
            position(2, 1, 0, 10),
        )
    )

    assert result.track_ids == {1, 2}
    assert result.summary_for_track(1).total_distance_ft == 5
    assert result.summary_for_track(2).total_distance_ft == 10


def test_movement_analyzer_skips_missing_positions() -> None:
    analyzer = MovementAnalyzer(
        MovementAnalysisConfig(
            frame_rate_fps=1,
            max_speed_ft_per_second=None,
        )
    )

    result = analyzer.analyze_positions(
        (
            position(1, 0, 0, 0),
            position(1, 1, None, None),
            position(1, 2, 6, 8),
        )
    )

    assert len(result.samples) == 1
    assert result.samples[0].distance_ft == 10


def test_movement_analyzer_respects_max_frame_gap() -> None:
    analyzer = MovementAnalyzer(
        MovementAnalysisConfig(
            frame_rate_fps=1,
            max_frame_gap=2,
            max_speed_ft_per_second=None,
        )
    )

    result = analyzer.analyze_positions(
        (
            position(1, 0, 0, 0),
            position(1, 3, 3, 4),
        )
    )

    assert len(result.samples) == 0
    assert result.summary_for_track(1) is None


def test_movement_analyzer_filters_unrealistic_speed() -> None:
    analyzer = MovementAnalyzer(
        MovementAnalysisConfig(
            frame_rate_fps=30,
            max_speed_ft_per_second=20,
        )
    )

    result = analyzer.analyze_positions(
        (
            position(1, 0, 0, 0),
            position(1, 1, 100, 0),
        )
    )

    assert len(result.samples) == 0


def test_movement_analyzer_handles_empty_input() -> None:
    analyzer = MovementAnalyzer()

    result = analyzer.analyze_positions(())

    assert len(result) == 0
    assert result.summaries == ()
