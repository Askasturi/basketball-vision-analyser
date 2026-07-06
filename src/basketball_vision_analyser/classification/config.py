
"""Team classification configuration."""

from __future__ import annotations

from dataclasses import dataclass

RGBColor = tuple[int, int, int]


@dataclass(frozen=True)
class TeamClassificationConfig:
    """Configuration for color-based team classification."""

    team_a_rgb: RGBColor = (220, 40, 40)
    team_b_rgb: RGBColor = (40, 80, 220)
    min_crop_area: int = 25
    unknown_confidence_threshold: float = 0.2

    def __post_init__(self) -> None:
        self._validate_rgb_color(self.team_a_rgb, "team_a_rgb")
        self._validate_rgb_color(self.team_b_rgb, "team_b_rgb")

        if self.min_crop_area <= 0:
            msg = "min_crop_area must be greater than 0."
            raise ValueError(msg)

        if not 0 <= self.unknown_confidence_threshold <= 1:
            msg = "unknown_confidence_threshold must be between 0 and 1."
            raise ValueError(msg)

    @staticmethod
    def _validate_rgb_color(color: RGBColor, name: str) -> None:
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
