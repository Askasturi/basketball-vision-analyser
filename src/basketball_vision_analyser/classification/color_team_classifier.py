
"""Color-based team classifier."""

from __future__ import annotations

import math

import numpy as np

from basketball_vision_analyser.classification.assignment import TeamAssignment
from basketball_vision_analyser.classification.config import (
    RGBColor,
    TeamClassificationConfig,
)
from basketball_vision_analyser.classification.result import TeamClassificationResult
from basketball_vision_analyser.classification.types import PlayerTeam
from basketball_vision_analyser.detection import DetectionClass
from basketball_vision_analyser.tracking import TrackingResult


class ColorTeamClassifier:
    """Classify tracked players into teams using jersey color."""

    def __init__(self, config: TeamClassificationConfig | None = None) -> None:
        self.config = config or TeamClassificationConfig()

    def classify_frame(
        self,
        frame: np.ndarray,
        tracking_result: TrackingResult,
    ) -> TeamClassificationResult:
        """Classify tracked objects in one frame."""

        self._validate_frame(frame)

        assignments: list[TeamAssignment] = []

        for tracked_object in tracking_result.objects:
            if tracked_object.class_name == DetectionClass.REFEREE:
                assignments.append(
                    TeamAssignment(
                        track_id=tracked_object.track_id,
                        team=PlayerTeam.REFEREE,
                        confidence=1.0,
                        frame_index=tracking_result.frame_index,
                        metadata={"reason": "tracked_object_is_referee"},
                    )
                )
                continue

            if tracked_object.class_name != DetectionClass.PLAYER:
                continue

            crop = self._crop_tracked_object(frame, tracked_object.box.to_xyxy())

            if crop is None:
                assignments.append(
                    TeamAssignment(
                        track_id=tracked_object.track_id,
                        team=PlayerTeam.UNKNOWN,
                        confidence=0.0,
                        frame_index=tracking_result.frame_index,
                        metadata={"reason": "invalid_or_small_crop"},
                    )
                )
                continue

            dominant_color = self._mean_rgb(crop)
            team, confidence = self._nearest_team(dominant_color)

            if confidence < self.config.unknown_confidence_threshold:
                team = PlayerTeam.UNKNOWN

            assignments.append(
                TeamAssignment(
                    track_id=tracked_object.track_id,
                    team=team,
                    confidence=confidence,
                    frame_index=tracking_result.frame_index,
                    dominant_color_rgb=dominant_color,
                    metadata={"method": "mean_rgb_nearest_color"},
                )
            )

        return TeamClassificationResult(
            frame_index=tracking_result.frame_index,
            assignments=tuple(assignments),
            metadata={"classifier": "color_team_classifier"},
        )

    def _crop_tracked_object(
        self,
        frame: np.ndarray,
        box_xyxy: tuple[float, float, float, float],
    ) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box_xyxy

        left = max(0, min(width, int(math.floor(x1))))
        top = max(0, min(height, int(math.floor(y1))))
        right = max(0, min(width, int(math.ceil(x2))))
        bottom = max(0, min(height, int(math.ceil(y2))))

        if right <= left or bottom <= top:
            return None

        crop = frame[top:bottom, left:right]

        if crop.size == 0:
            return None

        if crop.shape[0] * crop.shape[1] < self.config.min_crop_area:
            return None

        return crop

    @staticmethod
    def _mean_rgb(crop_bgr: np.ndarray) -> RGBColor:
        mean_bgr = crop_bgr.mean(axis=(0, 1))
        blue, green, red = mean_bgr

        return (
            int(round(red)),
            int(round(green)),
            int(round(blue)),
        )

    def _nearest_team(self, color_rgb: RGBColor) -> tuple[PlayerTeam, float]:
        team_a_confidence = self._color_similarity(color_rgb, self.config.team_a_rgb)
        team_b_confidence = self._color_similarity(color_rgb, self.config.team_b_rgb)

        if team_a_confidence >= team_b_confidence:
            return PlayerTeam.TEAM_A, team_a_confidence

        return PlayerTeam.TEAM_B, team_b_confidence

    @staticmethod
    def _color_similarity(color_a: RGBColor, color_b: RGBColor) -> float:
        max_distance = math.sqrt(3 * 255**2)
        distance = math.sqrt(
            sum((channel_a - channel_b) ** 2 for channel_a, channel_b in zip(
                color_a,
                color_b,
                strict=True,
            ))
        )

        return max(0.0, min(1.0, 1 - distance / max_distance))

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy array."
            raise TypeError(msg)

        if frame.ndim != 3 or frame.shape[2] != 3:
            msg = "frame must have shape height x width x 3."
            raise ValueError(msg)
