
import math

import pytest

from basketball_vision_analyser.court import Point2D


def test_point_2d_properties() -> None:
    point = Point2D(1, 2)

    assert point.x == 1.0
    assert point.y == 2.0
    assert point.to_tuple() == (1.0, 2.0)


def test_point_2d_distance_to_other_point() -> None:
    first = Point2D(0, 0)
    second = Point2D(3, 4)

    assert first.distance_to(second) == 5


def test_point_2d_rejects_non_finite_x() -> None:
    with pytest.raises(ValueError, match="x"):
        Point2D(math.inf, 0)


def test_point_2d_rejects_non_finite_y() -> None:
    with pytest.raises(ValueError, match="y"):
        Point2D(0, math.nan)
