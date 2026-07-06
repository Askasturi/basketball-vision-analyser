
"""Team classification type definitions."""

from __future__ import annotations

from enum import StrEnum


class PlayerTeam(StrEnum):
    """Supported team labels."""

    TEAM_A = "team_a"
    TEAM_B = "team_b"
    REFEREE = "referee"
    UNKNOWN = "unknown"
