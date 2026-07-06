
"""Frame-level possession estimator."""

from __future__ import annotations

import math

from basketball_vision_analyser.classification import (
    PlayerTeam,
    TeamClassificationResult,
)
from basketball_vision_analyser.detection import DetectionClass
from basketball_vision_analyser.possession.assignment import PossessionAssignment
from basketball_vision_analyser.possession.config import PossessionConfig
from basketball_vision_analyser.possession.result import PossessionResult
from basketball_vision_analyser.possession.types import PossessionStatus
from basketball_vision_analyser.tracking import TrackedObject, TrackingResult


class PossessionEstimator:
    """Estimate ball possession from tracked players and ball."""

    def __init__(self, config: PossessionConfig | None = None) -> None:
        self.config = config or PossessionConfig()

    def estimate_frame(
        self,
        tracking_result: TrackingResult,
        team_result: TeamClassificationResult | None = None,
    ) -> PossessionResult:
        """Estimate possession for one frame."""

        ball = self._select_ball(tracking_result)
        players = tracking_result.for_class(DetectionClass.PLAYER)

        if ball is None:
            return self._result(
                frame_index=tracking_result.frame_index,
                status=PossessionStatus.NO_BALL,
                metadata={"reason": "no_ball_track"},
            )

        if not players:
            return self._result(
                frame_index=tracking_result.frame_index,
                status=PossessionStatus.NO_PLAYER,
                ball_track_id=ball.track_id,
                metadata={"reason": "no_player_tracks"},
            )

        player, distance = self._nearest_player(ball, players)

        if distance > self.config.max_control_distance_px:
            return self._result(
                frame_index=tracking_result.frame_index,
                status=PossessionStatus.LOOSE_BALL,
                ball_track_id=ball.track_id,
                distance_px=distance,
                metadata={"reason": "nearest_player_too_far"},
            )

        confidence = self._confidence_from_distance(distance)

        if confidence < self.config.min_control_confidence:
            return self._result(
                frame_index=tracking_result.frame_index,
                status=PossessionStatus.LOOSE_BALL,
                ball_track_id=ball.track_id,
                player_track_id=player.track_id,
                distance_px=distance,
                confidence=confidence,
                metadata={"reason": "confidence_below_threshold"},
            )

        team = self._team_for_player(player.track_id, team_result)

        return self._result(
            frame_index=tracking_result.frame_index,
            status=PossessionStatus.PLAYER_CONTROL,
            ball_track_id=ball.track_id,
            player_track_id=player.track_id,
            team=team,
            distance_px=distance,
            confidence=confidence,
            metadata={"reason": "nearest_player_within_control_distance"},
        )

    @staticmethod
    def _select_ball(tracking_result: TrackingResult) -> TrackedObject | None:
        balls = tracking_result.for_class(DetectionClass.BALL)

        if not balls:
            return None

        return max(balls, key=lambda ball: ball.confidence)

    @staticmethod
    def _nearest_player(
        ball: TrackedObject,
        players: tuple[TrackedObject, ...],
    ) -> tuple[TrackedObject, float]:
        ball_x, ball_y = ball.center

        nearest_player = players[0]
        nearest_distance = math.inf

        for player in players:
            player_x, player_y = player.center
            distance = math.hypot(ball_x - player_x, ball_y - player_y)

            if distance < nearest_distance:
                nearest_player = player
                nearest_distance = distance

        return nearest_player, nearest_distance

    def _confidence_from_distance(self, distance_px: float) -> float:
        confidence = 1 - distance_px / self.config.max_control_distance_px
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _team_for_player(
        player_track_id: int,
        team_result: TeamClassificationResult | None,
    ) -> PlayerTeam:
        if team_result is None:
            return PlayerTeam.UNKNOWN

        assignment = team_result.for_track_id(player_track_id)

        if assignment is None:
            return PlayerTeam.UNKNOWN

        return assignment.team

    @staticmethod
    def _result(
        *,
        frame_index: int,
        status: PossessionStatus,
        player_track_id: int | None = None,
        ball_track_id: int | None = None,
        team: PlayerTeam = PlayerTeam.UNKNOWN,
        distance_px: float | None = None,
        confidence: float = 0.0,
        metadata: dict[str, str] | None = None,
    ) -> PossessionResult:
        assignment = PossessionAssignment(
            frame_index=frame_index,
            status=status,
            player_track_id=player_track_id,
            ball_track_id=ball_track_id,
            team=team,
            distance_px=distance_px,
            confidence=confidence,
            metadata=metadata or {},
        )

        return PossessionResult(
            frame_index=frame_index,
            assignment=assignment,
            metadata={"estimator": "nearest_player_distance"},
        )
