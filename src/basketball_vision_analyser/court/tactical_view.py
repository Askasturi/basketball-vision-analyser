
"""Top-down tactical court rendering."""

from __future__ import annotations

import cv2
import numpy as np

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.court.config import BasketballCourtConfig
from basketball_vision_analyser.court.geometry import Point2D


class TacticalViewRenderer:
    """Render simple top-down tactical court views."""

    def __init__(self, config: BasketballCourtConfig | None = None) -> None:
        self.config = config or BasketballCourtConfig()

    def create_blank_view(self) -> np.ndarray:
        """Create a blank top-down court image."""

        view = np.full(
            (
                self.config.tactical_view_height_px,
                self.config.tactical_view_width_px,
                3,
            ),
            fill_value=245,
            dtype=np.uint8,
        )
        self._draw_court_lines(view)

        return view

    def court_to_view(self, point: Point2D) -> Point2D:
        """Convert court feet coordinates to tactical view pixels."""

        x_ratio = point.x / self.config.court_width_ft
        y_ratio = point.y / self.config.court_length_ft

        x_px = self.config.margin_px + x_ratio * self.config.playable_width_px
        y_px = self.config.margin_px + y_ratio * self.config.playable_height_px

        return Point2D(x_px, y_px)

    def draw_marker(
        self,
        view: np.ndarray,
        point: Point2D,
        *,
        label: str = "",
        team: PlayerTeam = PlayerTeam.UNKNOWN,
    ) -> np.ndarray:
        """Draw one marker on a tactical view."""

        self._validate_view(view)

        view_point = self.court_to_view(point)
        center = (int(round(view_point.x)), int(round(view_point.y)))
        color = self._team_color_bgr(team)

        cv2.circle(view, center, 8, color, thickness=-1)
        cv2.circle(view, center, 8, (30, 30, 30), thickness=1)

        if label:
            cv2.putText(
                view,
                label,
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

        return view

    def _draw_court_lines(self, view: np.ndarray) -> None:
        left = self.config.margin_px
        top = self.config.margin_px
        right = self.config.tactical_view_width_px - self.config.margin_px
        bottom = self.config.tactical_view_height_px - self.config.margin_px
        center_y = (top + bottom) // 2

        line_color = (40, 40, 40)

        cv2.rectangle(view, (left, top), (right, bottom), line_color, thickness=2)
        cv2.line(view, (left, center_y), (right, center_y), line_color, thickness=1)
        cv2.circle(
            view,
            ((left + right) // 2, center_y),
            45,
            line_color,
            thickness=1,
        )

    @staticmethod
    def _team_color_bgr(team: PlayerTeam) -> tuple[int, int, int]:
        if team == PlayerTeam.TEAM_A:
            return (40, 40, 220)

        if team == PlayerTeam.TEAM_B:
            return (220, 80, 40)

        if team == PlayerTeam.REFEREE:
            return (30, 30, 30)

        return (120, 120, 120)

    @staticmethod
    def _validate_view(view: np.ndarray) -> None:
        if not isinstance(view, np.ndarray):
            msg = "view must be a numpy array."
            raise TypeError(msg)

        if view.ndim != 3 or view.shape[2] != 3:
            msg = "view must have shape height x width x 3."
            raise ValueError(msg)
