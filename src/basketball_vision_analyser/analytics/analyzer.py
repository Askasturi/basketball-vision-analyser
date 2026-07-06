
"""Speed and distance movement analyzer."""

from __future__ import annotations

from collections import defaultdict

from basketball_vision_analyser.analytics.config import MovementAnalysisConfig
from basketball_vision_analyser.analytics.movement import (
    MovementSample,
    MovementSummary,
    TrackPosition,
)
from basketball_vision_analyser.analytics.result import MovementAnalysisResult


class MovementAnalyzer:
    """Analyze speed and distance from court positions."""

    def __init__(self, config: MovementAnalysisConfig | None = None) -> None:
        self.config = config or MovementAnalysisConfig()

    def analyze_positions(
        self,
        positions: tuple[TrackPosition, ...] | list[TrackPosition],
    ) -> MovementAnalysisResult:
        """Analyze movement from track positions."""

        if not positions:
            return MovementAnalysisResult(
                metadata={"analyzer": "movement_analyzer"},
            )

        positions_by_track = self._group_positions_by_track(positions)
        samples: list[MovementSample] = []

        for track_id, track_positions in positions_by_track.items():
            samples.extend(self._samples_for_track(track_id, track_positions))

        samples.sort(key=lambda sample: (sample.track_id, sample.start_frame_index))
        summaries = self._summaries_from_samples(tuple(samples))

        return MovementAnalysisResult(
            samples=tuple(samples),
            summaries=summaries,
            metadata={
                "analyzer": "movement_analyzer",
                "track_count": len(positions_by_track),
            },
        )

    @staticmethod
    def _group_positions_by_track(
        positions: tuple[TrackPosition, ...] | list[TrackPosition],
    ) -> dict[int, tuple[TrackPosition, ...]]:
        grouped: dict[int, list[TrackPosition]] = defaultdict(list)

        for position in positions:
            grouped[position.track_id].append(position)

        return {
            track_id: tuple(
                sorted(track_positions, key=lambda item: item.frame_index)
            )
            for track_id, track_positions in grouped.items()
        }

    def _samples_for_track(
        self,
        track_id: int,
        positions: tuple[TrackPosition, ...],
    ) -> list[MovementSample]:
        samples: list[MovementSample] = []
        previous_position: TrackPosition | None = None

        for position in positions:
            if position.court_point is None:
                continue

            if previous_position is None:
                previous_position = position
                continue

            sample = self._sample_between(previous_position, position)

            if sample is not None:
                samples.append(sample)

            previous_position = position

        return samples

    def _sample_between(
        self,
        previous_position: TrackPosition,
        current_position: TrackPosition,
    ) -> MovementSample | None:
        if (
            previous_position.court_point is None
            or current_position.court_point is None
        ):
            return None

        frame_gap = current_position.frame_index - previous_position.frame_index

        if frame_gap <= 0:
            return None

        if frame_gap > self.config.max_frame_gap:
            return None

        distance_ft = previous_position.court_point.distance_to(
            current_position.court_point
        )
        elapsed_seconds = frame_gap / self.config.frame_rate_fps
        speed_ft_per_second = distance_ft / elapsed_seconds

        if (
            self.config.max_speed_ft_per_second is not None
            and speed_ft_per_second > self.config.max_speed_ft_per_second
        ):
            return None

        return MovementSample(
            track_id=current_position.track_id,
            start_frame_index=previous_position.frame_index,
            end_frame_index=current_position.frame_index,
            start_point=previous_position.court_point,
            end_point=current_position.court_point,
            distance_ft=distance_ft,
            elapsed_seconds=elapsed_seconds,
            speed_ft_per_second=speed_ft_per_second,
            team=current_position.team,
            metadata={"frame_gap": frame_gap},
        )

    @staticmethod
    def _summaries_from_samples(
        samples: tuple[MovementSample, ...],
    ) -> tuple[MovementSummary, ...]:
        samples_by_track: dict[int, list[MovementSample]] = defaultdict(list)

        for sample in samples:
            samples_by_track[sample.track_id].append(sample)

        summaries: list[MovementSummary] = []

        for track_id, track_samples in samples_by_track.items():
            total_distance = sum(sample.distance_ft for sample in track_samples)
            total_elapsed = sum(sample.elapsed_seconds for sample in track_samples)
            average_speed = total_distance / total_elapsed if total_elapsed else 0.0
            max_speed = max(sample.speed_ft_per_second for sample in track_samples)
            team = track_samples[-1].team

            summaries.append(
                MovementSummary(
                    track_id=track_id,
                    total_distance_ft=total_distance,
                    total_elapsed_seconds=total_elapsed,
                    average_speed_ft_per_second=average_speed,
                    max_speed_ft_per_second=max_speed,
                    sample_count=len(track_samples),
                    team=team,
                )
            )

        return tuple(sorted(summaries, key=lambda summary: summary.track_id))
