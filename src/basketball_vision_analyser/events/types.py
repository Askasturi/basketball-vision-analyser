
"""Basketball event type definitions."""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Supported possession-derived event types."""

    PASS = "pass"
    INTERCEPTION = "interception"
