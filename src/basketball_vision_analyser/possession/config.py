
"""Possession estimation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PossessionConfig:
    """Configuration for possession estimation."""

    max_control_distance_px: float = 80.0
    min_control_confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.max_control_distance_px <= 0:
            msg = "max_control_distance_px must be greater than 0."
            raise ValueError(msg)

        if not 0 <= self.min_control_confidence <= 1:
            msg = "min_control_confidence must be between 0 and 1."
            raise ValueError(msg)
