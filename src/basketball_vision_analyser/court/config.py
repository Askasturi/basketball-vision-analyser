
"""Basketball court configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BasketballCourtConfig:
    """Configuration for a basketball court and tactical view."""

    court_length_ft: float = 94.0
    court_width_ft: float = 50.0
    tactical_view_width_px: int = 500
    tactical_view_height_px: int = 940
    margin_px: int = 20
    keypoint_confidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.court_length_ft <= 0:
            msg = "court_length_ft must be greater than 0."
            raise ValueError(msg)

        if self.court_width_ft <= 0:
            msg = "court_width_ft must be greater than 0."
            raise ValueError(msg)

        if self.tactical_view_width_px <= 0:
            msg = "tactical_view_width_px must be greater than 0."
            raise ValueError(msg)

        if self.tactical_view_height_px <= 0:
            msg = "tactical_view_height_px must be greater than 0."
            raise ValueError(msg)

        if self.margin_px < 0:
            msg = "margin_px must be greater than or equal to 0."
            raise ValueError(msg)

        if not 0 <= self.keypoint_confidence_threshold <= 1:
            msg = "keypoint_confidence_threshold must be between 0 and 1."
            raise ValueError(msg)

    @property
    def playable_width_px(self) -> int:
        """Return tactical view width excluding margins."""

        return self.tactical_view_width_px - 2 * self.margin_px

    @property
    def playable_height_px(self) -> int:
        """Return tactical view height excluding margins."""

        return self.tactical_view_height_px - 2 * self.margin_px
