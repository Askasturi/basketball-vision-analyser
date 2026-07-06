
"""Possession type definitions."""

from __future__ import annotations

from enum import StrEnum


class PossessionStatus(StrEnum):
    """Possible possession states for one frame."""

    PLAYER_CONTROL = "player_control"
    LOOSE_BALL = "loose_ball"
    NO_BALL = "no_ball"
    NO_PLAYER = "no_player"
