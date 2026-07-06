
from basketball_vision_analyser.analytics import (
    MovementAnalysisResult,
    MovementSample,
    MovementSummary,
)
from basketball_vision_analyser.court import Point2D


def make_sample(track_id: int, distance: float) -> MovementSample:
    return MovementSample(
        track_id=track_id,
        start_frame_index=0,
        end_frame_index=1,
        start_point=Point2D(0, 0),
        end_point=Point2D(distance, 0),
        distance_ft=distance,
        elapsed_seconds=1,
        speed_ft_per_second=distance,
    )


def make_summary(track_id: int, distance: float) -> MovementSummary:
    return MovementSummary(
        track_id=track_id,
        total_distance_ft=distance,
        total_elapsed_seconds=1,
        average_speed_ft_per_second=distance,
        max_speed_ft_per_second=distance,
        sample_count=1,
    )


def test_movement_analysis_result_filters_by_track() -> None:
    first_sample = make_sample(1, 5)
    second_sample = make_sample(2, 10)
    first_summary = make_summary(1, 5)
    second_summary = make_summary(2, 10)

    result = MovementAnalysisResult(
        samples=(first_sample, second_sample),
        summaries=(first_summary, second_summary),
    )

    assert len(result) == 2
    assert result.track_ids == {1, 2}
    assert result.total_distance_ft == 15
    assert result.samples_for_track(1) == (first_sample,)
    assert result.summary_for_track(2) == second_summary
    assert result.summary_for_track(99) is None
