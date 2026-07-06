
"""Detection type definitions."""

from __future__ import annotations

from enum import StrEnum


class DetectionClass(StrEnum):
    """Supported basketball detection classes."""

    PLAYER = "player"
    BALL = "ball"
    HOOP = "hoop"
    REFEREE = "referee"
    CLOCK = "clock"
    SCOREBOARD = "scoreboard"
    OVERLAY = "overlay"
    UNKNOWN = "unknown"

    @classmethod
    def from_label(cls, label: str) -> DetectionClass:
        """Normalize a raw model label into a supported detection class."""

        cleaned = label.strip().lower().replace("-", "_").replace(" ", "_")

        aliases = {
            "person": cls.PLAYER,
            "player": cls.PLAYER,
            "basketball_player": cls.PLAYER,
            "ball": cls.BALL,
            "basketball": cls.BALL,
            "hoop": cls.HOOP,
            "rim": cls.HOOP,
            "basket": cls.HOOP,
            "ref": cls.REFEREE,
            "referee": cls.REFEREE,
            "official": cls.REFEREE,
            "clock": cls.CLOCK,
            "game_clock": cls.CLOCK,
            "shot_clock": cls.CLOCK,
            "scoreboard": cls.SCOREBOARD,
            "score_board": cls.SCOREBOARD,
            "overlay": cls.OVERLAY,
            "broadcast_overlay": cls.OVERLAY,
        }

        return aliases.get(cleaned, cls.UNKNOWN)


class DetectorBackend(StrEnum):
    """Available detector backend types."""

    MOCK = "mock"
    LOCAL_YOLO = "local_yolo"
    ROBOFLOW_API = "roboflow_api"
