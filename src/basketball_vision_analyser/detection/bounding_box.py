
"""Bounding box model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box using xyxy coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x1 < 0 or self.y1 < 0:
            msg = "Bounding box coordinates must be non-negative."
            raise ValueError(msg)

        if self.x2 <= self.x1:
            msg = "x2 must be greater than x1."
            raise ValueError(msg)

        if self.y2 <= self.y1:
            msg = "y2 must be greater than y1."
            raise ValueError(msg)

    @property
    def width(self) -> float:
        """Return box width."""

        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Return box height."""

        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Return box area."""

        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Return box center as x, y."""

        return (self.x1 + self.width / 2, self.y1 + self.height / 2)

    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return box as x1, y1, x2, y2."""

        return self.x1, self.y1, self.x2, self.y2

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return box as x, y, width, height."""

        return self.x1, self.y1, self.width, self.height

    def iou(self, other: BoundingBox) -> float:
        """Return intersection-over-union with another box."""

        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        union_area = self.area + other.area - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area
