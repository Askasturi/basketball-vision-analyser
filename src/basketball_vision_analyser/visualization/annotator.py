"""Frame annotation utilities."""

from __future__ import annotations

import cv2
import numpy as np

from basketball_vision_analyser.classification import (
    PlayerTeam,
    TeamClassificationResult,
)
from basketball_vision_analyser.detection import DetectionClass, DetectionResult
from basketball_vision_analyser.possession import PossessionResult
from basketball_vision_analyser.tracking import TrackingResult
from basketball_vision_analyser.visualization.config import (
    BGRColor,
    VisualizationConfig,
)


class FrameAnnotator:
    """Draw detection, tracking, team, and possession overlays."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        self.config = config or VisualizationConfig()

    def annotate_frame(
        self,
        frame: np.ndarray,
        *,
        detection_result: DetectionResult | None = None,
        tracking_result: TrackingResult | None = None,
        team_result: TeamClassificationResult | None = None,
        possession_result: PossessionResult | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of a frame."""

        self._validate_frame(frame)
        output = frame.copy()

        if tracking_result is not None:
            self._draw_tracking_result(output, tracking_result, team_result)
        elif detection_result is not None:
            self._draw_detection_result(output, detection_result)

        if possession_result is not None and self.config.draw_possession_banner:
            self._draw_possession_banner(output, possession_result)

        return output

    def _draw_detection_result(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult,
    ) -> None:
        for detection in detection_result.detections:
            color = self._color_for_class(detection.class_name)
            x1, y1, x2, y2 = self._box_to_ints(detection.box.to_xyxy())

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                self.config.box_thickness,
            )

            if self.config.draw_labels:
                label = detection.class_name.value

                if self.config.draw_confidence:
                    label = f"{label} {detection.confidence:.2f}"

                self._draw_label(frame, label, (x1, y1), color)

    def _draw_tracking_result(
        self,
        frame: np.ndarray,
        tracking_result: TrackingResult,
        team_result: TeamClassificationResult | None,
    ) -> None:
        for tracked_object in tracking_result.objects:
            team = self._team_for_track(tracked_object.track_id, team_result)
            color = self._color_for_team_or_class(team, tracked_object.class_name)
            x1, y1, x2, y2 = self._box_to_ints(tracked_object.box.to_xyxy())

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                self.config.box_thickness,
            )

            if not self.config.draw_labels:
                continue

            label_parts = [
                f"#{tracked_object.track_id}",
                tracked_object.class_name.value,
            ]

            if team != PlayerTeam.UNKNOWN:
                label_parts.append(team.value)

            if self.config.draw_confidence:
                label_parts.append(f"{tracked_object.confidence:.2f}")

            self._draw_label(frame, " ".join(label_parts), (x1, y1), color)

    def _draw_possession_banner(
        self,
        frame: np.ndarray,
        possession_result: PossessionResult,
    ) -> None:
        assignment = possession_result.assignment
        text = f"Possession: {assignment.status.value}"

        if assignment.player_track_id is not None:
            text += f" | player #{assignment.player_track_id}"

        if assignment.team != PlayerTeam.UNKNOWN:
            text += f" | {assignment.team.value}"

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (30, 30, 30), -1)
        cv2.putText(
            frame,
            text,
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            (255, 255, 255),
            self.config.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_label(
        self,
        frame: np.ndarray,
        label: str,
        origin: tuple[int, int],
        color: BGRColor,
    ) -> None:
        x, y = origin
        text_y = max(14, y - 6)

        cv2.putText(
            frame,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            color,
            self.config.font_thickness,
            cv2.LINE_AA,
        )

    def _color_for_team_or_class(
        self,
        team: PlayerTeam,
        class_name: DetectionClass,
    ) -> BGRColor:
        if team == PlayerTeam.TEAM_A:
            return self.config.team_a_color_bgr

        if team == PlayerTeam.TEAM_B:
            return self.config.team_b_color_bgr

        if team == PlayerTeam.REFEREE:
            return self.config.referee_color_bgr

        return self._color_for_class(class_name)

    def _color_for_class(self, class_name: DetectionClass) -> BGRColor:
        if class_name == DetectionClass.PLAYER:
            return self.config.player_color_bgr

        if class_name == DetectionClass.BALL:
            return self.config.ball_color_bgr

        if class_name == DetectionClass.HOOP:
            return self.config.hoop_color_bgr

        if class_name == DetectionClass.REFEREE:
            return self.config.referee_color_bgr

        return self.config.unknown_color_bgr

    @staticmethod
    def _team_for_track(
        track_id: int,
        team_result: TeamClassificationResult | None,
    ) -> PlayerTeam:
        if team_result is None:
            return PlayerTeam.UNKNOWN

        assignment = team_result.for_track_id(track_id)

        if assignment is None:
            return PlayerTeam.UNKNOWN

        return assignment.team

    @staticmethod
    def _box_to_ints(
        box_xyxy: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box_xyxy

        return (
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy array."
            raise TypeError(msg)

        if frame.ndim != 3 or frame.shape[2] != 3:
            msg = "frame must have shape height x width x 3."
            raise ValueError(msg)
