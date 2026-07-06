
import pytest

from basketball_vision_analyser.court import (
    CourtKeypoint,
    CourtKeypointSet,
    Point2D,
)


def make_keypoint(name: str, confidence: float = 1.0) -> CourtKeypoint:
    return CourtKeypoint(
        name=name,
        image_point=Point2D(10, 20),
        court_point=Point2D(1, 2),
        confidence=confidence,
    )


def test_court_keypoint_properties() -> None:
    keypoint = make_keypoint("corner")

    assert keypoint.name == "corner"
    assert keypoint.image_point == Point2D(10, 20)
    assert keypoint.court_point == Point2D(1, 2)
    assert keypoint.confidence == 1.0


def test_court_keypoint_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        CourtKeypoint(
            name="",
            image_point=Point2D(0, 0),
            court_point=Point2D(0, 0),
        )


def test_court_keypoint_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        make_keypoint("bad", confidence=2)


def test_court_keypoint_set_filters_and_gets_keypoints() -> None:
    low = make_keypoint("low", confidence=0.2)
    high = make_keypoint("high", confidence=0.9)
    keypoint_set = CourtKeypointSet((low, high))

    filtered = keypoint_set.by_min_confidence(0.5)

    assert len(keypoint_set) == 2
    assert keypoint_set.names == ("low", "high")
    assert keypoint_set.get("high") == high
    assert keypoint_set.get("missing") is None
    assert len(filtered) == 1
    assert filtered.get("high") == high


def test_court_keypoint_set_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        CourtKeypointSet((make_keypoint("same"), make_keypoint("same")))


def test_court_keypoint_set_require_minimum() -> None:
    keypoint_set = CourtKeypointSet((make_keypoint("one"),))

    with pytest.raises(ValueError, match="At least 2"):
        keypoint_set.require_minimum(2)
