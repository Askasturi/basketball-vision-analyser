
import numpy as np
import pytest

from basketball_vision_analyser.court import (
    CourtKeypoint,
    CourtKeypointSet,
    HomographyTransformer,
    Point2D,
)


def make_square_keypoints() -> CourtKeypointSet:
    return CourtKeypointSet(
        (
            CourtKeypoint("top_left", Point2D(0, 0), Point2D(0, 0)),
            CourtKeypoint("top_right", Point2D(100, 0), Point2D(50, 0)),
            CourtKeypoint("bottom_right", Point2D(100, 200), Point2D(50, 94)),
            CourtKeypoint("bottom_left", Point2D(0, 200), Point2D(0, 94)),
        )
    )


def test_homography_transforms_image_to_court() -> None:
    transformer = HomographyTransformer.from_keypoints(make_square_keypoints())

    court_point = transformer.image_to_court(Point2D(50, 100))

    assert court_point.x == pytest.approx(25)
    assert court_point.y == pytest.approx(47)


def test_homography_transforms_court_to_image() -> None:
    transformer = HomographyTransformer.from_keypoints(make_square_keypoints())

    image_point = transformer.court_to_image(Point2D(25, 47))

    assert image_point.x == pytest.approx(50)
    assert image_point.y == pytest.approx(100)


def test_homography_filters_low_confidence_keypoints() -> None:
    keypoints = CourtKeypointSet(
        (
            CourtKeypoint("a", Point2D(0, 0), Point2D(0, 0), confidence=0.9),
            CourtKeypoint("b", Point2D(1, 0), Point2D(1, 0), confidence=0.9),
            CourtKeypoint("c", Point2D(1, 1), Point2D(1, 1), confidence=0.9),
            CourtKeypoint("d", Point2D(0, 1), Point2D(0, 1), confidence=0.1),
        )
    )

    with pytest.raises(ValueError, match="At least 4"):
        HomographyTransformer.from_keypoints(keypoints, min_confidence=0.5)


def test_homography_requires_four_keypoints() -> None:
    keypoints = CourtKeypointSet(
        (
            CourtKeypoint("a", Point2D(0, 0), Point2D(0, 0)),
            CourtKeypoint("b", Point2D(1, 0), Point2D(1, 0)),
            CourtKeypoint("c", Point2D(1, 1), Point2D(1, 1)),
        )
    )

    with pytest.raises(ValueError, match="At least 4"):
        HomographyTransformer.from_keypoints(keypoints)


def test_homography_rejects_invalid_matrix_shape() -> None:
    with pytest.raises(ValueError, match="3 x 3"):
        HomographyTransformer(np.eye(2))
