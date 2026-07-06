
"""Court geometry primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point2D:
    """A 2D point."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x):
            msg = "x must be finite."
            raise ValueError(msg)

        if not math.isfinite(self.y):
            msg = "y must be finite."
            raise ValueError(msg)

        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))

    def to_tuple(self) -> tuple[float, float]:
        """Return point as a tuple."""

        return (self.x, self.y)

    def distance_to(self, other: Point2D) -> float:
        """Return Euclidean distance to another point."""

        return math.hypot(self.x - other.x, self.y - other.y)
