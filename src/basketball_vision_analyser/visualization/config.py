
"""Visualization configuration."""

from __future__ import annotations

from dataclasses import dataclass

BGRColor = tuple[int, int, int]


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for frame annotation."""

    player_color_bgr: BGRColor = (40, 220, 40)
    ball_color_bgr: BGRColor = (40, 160, 255)
    hoop_color_bgr: BGRColor = (0, 165, 255)
    referee_color_bgr: BGRColor = (40, 40, 40)
    unknown_color_bgr: BGRColor = (180, 180, 180)
    team_a_color_bgr: BGRColor = (40, 40, 220)
    team_b_color_bgr: BGRColor = (220, 80, 40)
    box_thickness: int = 2
    font_scale: float = 0.45
    font_thickness: int = 1
    draw_labels: bool = True
    draw_confidence: bool = True
    draw_possession_banner: bool = True

    def __post_init__(self) -> None:
        for name in (
            "player_color_bgr",
            "ball_color_bgr",
            "hoop_color_bgr",
            "referee_color_bgr",
            "unknown_color_bgr",
            "team_a_color_bgr",
            "team_b_color_bgr",
        ):
            self._validate_color(getattr(self, name), name)

        if self.box_thickness <= 0:
            msg = "box_thickness must be greater than 0."
            raise ValueError(msg)

        if self.font_scale <= 0:
            msg = "font_scale must be greater than 0."
            raise ValueError(msg)

        if self.font_thickness <= 0:
            msg = "font_thickness must be greater than 0."
            raise ValueError(msg)

    @staticmethod
    def _validate_color(color: BGRColor, name: str) -> None:
        if len(color) != 3:
            msg = f"{name} must contain exactly 3 values."
            raise ValueError(msg)

        for value in color:
            if not isinstance(value, int):
                msg = f"{name} values must be integers."
                raise TypeError(msg)

            if not 0 <= value <= 255:
                msg = f"{name} values must be between 0 and 255."
                raise ValueError(msg)
