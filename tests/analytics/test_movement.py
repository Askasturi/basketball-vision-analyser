
import pytest

from basketball_vision_analyser.analytics import (
    MovementSample,
    MovementSummary,
    TrackPosition,
)
from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.court import Point2D


def test_track_position_properties() -> None:
    position = TrackPosition(
        track_id=1,
        frame_index=2,
        court_point=Point2D(10, 20),
        team=PlayerTeam.TEAM_A,
    )

    assert position.track_id == 1
    assert position.frame_index == 2
    assert position.court_point == Point2D(10, 20)
    assert position.team == PlayerTeam.TEAM_A


def test_track_position_rejects_negative_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        TrackPosition(
            track_id=-1,
            frame_index=0,
            court_point=None,
        )


def test_track_position_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        TrackPosition(
            track_id=1,
            frame_index=-1,
            court_point=None,
        )


def test_movement_sample_properties() -> None:
    sample = MovementSample(
        track_id=1,
        start_frame_index=0,
        end_frame_index=30,
        start_point=Point2D(0, 0),
        end_point=Point2D(3, 4),
        distance_ft=5,
        elapsed_seconds=1,
        speed_ft_per_second=5,
        team=PlayerTeam.TEAM_B,
    )

    assert sample.track_id == 1
    assert sample.distance_ft == 5
    assert sample.elapsed_seconds == 1
    assert sample.speed_ft_per_second == 5
    assert sample.team == PlayerTeam.TEAM_B


def test_movement_sample_rejects_invalid_end_frame() -> None:
    with pytest.raises(ValueError, match="end_frame_index"):
        MovementSample(
            track_id=1,
            start_frame_index=5,
            end_frame_index=5,
            start_point=Point2D(0, 0),
            end_point=Point2D(1, 1),
            distance_ft=1,
            elapsed_seconds=1,
            speed_ft_per_second=1,
        )


def test_movement_summary_properties() -> None:
    summary = MovementSummary(
        track_id=1,
        total_distance_ft=15,
        total_elapsed_seconds=3,
        average_speed_ft_per_second=5,
        max_speed_ft_per_second=7,
        sample_count=2,
        team=PlayerTeam.TEAM_A,
    )

    assert summary.track_id == 1
    assert summary.total_distance_ft == 15
    assert summary.total_elapsed_seconds == 3
    assert summary.average_speed_ft_per_second == 5
    assert summary.max_speed_ft_per_second == 7
    assert summary.sample_count == 2
    assert summary.team == PlayerTeam.TEAM_A


def test_movement_summary_rejects_negative_distance() -> None:
    with pytest.raises(ValueError, match="total_distance_ft"):
        MovementSummary(
            track_id=1,
            total_distance_ft=-1,
            total_elapsed_seconds=3,
            average_speed_ft_per_second=5,
            max_speed_ft_per_second=7,
            sample_count=2,
        )
