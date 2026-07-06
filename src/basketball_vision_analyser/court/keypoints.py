
"""Court keypoint models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from basketball_vision_analyser.court.geometry import Point2D


@dataclass(frozen=True)
class CourtKeypoint:
    """A named keypoint connecting image and court coordinates."""

    name: str
    image_point: Point2D
    court_point: Point2D
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            msg = "name must not be empty."
            raise ValueError(msg)

        if not 0 <= self.confidence <= 1:
            msg = "confidence must be between 0 and 1."
            raise ValueError(msg)


@dataclass(frozen=True)
class CourtKeypointSet:
    """A collection of court keypoints."""

    keypoints: tuple[CourtKeypoint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        keypoints = tuple(self.keypoints)
        names = [keypoint.name for keypoint in keypoints]

        if len(names) != len(set(names)):
            msg = "keypoint names must be unique."
            raise ValueError(msg)

        object.__setattr__(self, "keypoints", keypoints)

    def __len__(self) -> int:
        return len(self.keypoints)

    def __iter__(self) -> Iterator[CourtKeypoint]:
        return iter(self.keypoints)

    @property
    def names(self) -> tuple[str, ...]:
        """Return keypoint names."""

        return tuple(keypoint.name for keypoint in self.keypoints)

    def get(self, name: str) -> CourtKeypoint | None:
        """Return a keypoint by name."""

        for keypoint in self.keypoints:
            if keypoint.name == name:
                return keypoint

        return None

    def by_min_confidence(self, threshold: float) -> CourtKeypointSet:
        """Return keypoints with confidence greater than or equal to threshold."""

        if not 0 <= threshold <= 1:
            msg = "threshold must be between 0 and 1."
            raise ValueError(msg)

        return CourtKeypointSet(
            tuple(
                keypoint
                for keypoint in self.keypoints
                if keypoint.confidence >= threshold
            )
        )

    def require_minimum(self, minimum_count: int) -> None:
        """Raise if there are fewer than the requested number of keypoints."""

        if minimum_count <= 0:
            msg = "minimum_count must be greater than 0."
            raise ValueError(msg)

        if len(self.keypoints) < minimum_count:
            msg = f"At least {minimum_count} keypoints are required."
            raise ValueError(msg)

    def image_points(self) -> tuple[tuple[float, float], ...]:
        """Return image points as tuples."""

        return tuple(keypoint.image_point.to_tuple() for keypoint in self.keypoints)

    def court_points(self) -> tuple[tuple[float, float], ...]:
        """Return court points as tuples."""

        return tuple(keypoint.court_point.to_tuple() for keypoint in self.keypoints)
