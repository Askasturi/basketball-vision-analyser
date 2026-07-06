
"""Pass and interception detector."""

from __future__ import annotations

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.events.config import EventDetectionConfig
from basketball_vision_analyser.events.event import PlayEvent
from basketball_vision_analyser.events.result import EventDetectionResult
from basketball_vision_analyser.events.types import EventType
from basketball_vision_analyser.possession import (
    PossessionAssignment,
    PossessionResult,
    PossessionStatus,
)


class PossessionEventDetector:
    """Detect passes and interceptions from possession results."""

    def __init__(self, config: EventDetectionConfig | None = None) -> None:
        self.config = config or EventDetectionConfig()

    def detect(
        self,
        possession_results: tuple[PossessionResult, ...] | list[PossessionResult],
    ) -> EventDetectionResult:
        """Detect events from a sequence of possession results."""

        if not possession_results:
            return EventDetectionResult(
                metadata={"detector": "possession_event_detector"},
            )

        ordered_results = tuple(
            sorted(possession_results, key=lambda result: result.frame_index)
        )

        events: list[PlayEvent] = []
        previous_control: PossessionAssignment | None = None

        for result in ordered_results:
            assignment = result.assignment

            if not self._is_player_control(assignment):
                continue

            if previous_control is None:
                previous_control = assignment
                continue

            if assignment.player_track_id == previous_control.player_track_id:
                previous_control = assignment
                continue

            event = self._event_from_transition(
                previous_control=previous_control,
                current_control=assignment,
            )

            if event is not None:
                events.append(event)

            previous_control = assignment

        return EventDetectionResult(
            events=tuple(events),
            metadata={
                "detector": "possession_event_detector",
                "frames_seen": len(ordered_results),
            },
        )

    @staticmethod
    def _is_player_control(assignment: PossessionAssignment) -> bool:
        return (
            assignment.status == PossessionStatus.PLAYER_CONTROL
            and assignment.player_track_id is not None
            and assignment.ball_track_id is not None
        )

    def _event_from_transition(
        self,
        *,
        previous_control: PossessionAssignment,
        current_control: PossessionAssignment,
    ) -> PlayEvent | None:
        frame_gap = current_control.frame_index - previous_control.frame_index

        if frame_gap > self.config.max_gap_frames:
            return None

        if not self._teams_are_usable(previous_control.team, current_control.team):
            return None

        event_type = self._event_type_for_transition(
            previous_control.team,
            current_control.team,
        )

        if event_type is None:
            return None

        if (
            previous_control.player_track_id is None
            or current_control.player_track_id is None
        ):
            return None

        confidence = min(previous_control.confidence, current_control.confidence)

        return PlayEvent(
            event_type=event_type,
            start_frame_index=previous_control.frame_index,
            end_frame_index=current_control.frame_index,
            from_player_track_id=previous_control.player_track_id,
            to_player_track_id=current_control.player_track_id,
            from_team=previous_control.team,
            to_team=current_control.team,
            confidence=confidence,
            metadata={
                "frame_gap": frame_gap,
                "previous_ball_track_id": previous_control.ball_track_id,
                "current_ball_track_id": current_control.ball_track_id,
            },
        )

    def _teams_are_usable(
        self,
        previous_team: PlayerTeam,
        current_team: PlayerTeam,
    ) -> bool:
        if not self.config.require_known_teams:
            return True

        return (
            previous_team != PlayerTeam.UNKNOWN
            and current_team != PlayerTeam.UNKNOWN
        )

    @staticmethod
    def _event_type_for_transition(
        previous_team: PlayerTeam,
        current_team: PlayerTeam,
    ) -> EventType | None:
        if previous_team == current_team:
            return EventType.PASS

        return EventType.INTERCEPTION
