
import pytest

from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
)


def test_detection_normalizes_string_class_name() -> None:
    detection = Detection(
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        class_name="person",
        confidence=0.9,
    )

    assert detection.class_name == DetectionClass.PLAYER
    assert detection.center == (5, 5)
    assert detection.area == 100


def test_detection_accepts_enum_class_name() -> None:
    detection = Detection(
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        class_name=DetectionClass.BALL,
        confidence=0.8,
    )

    assert detection.class_name == DetectionClass.BALL


def test_detection_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        Detection(
            box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            class_name=DetectionClass.PLAYER,
            confidence=1.5,
        )


def test_detection_rejects_negative_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        Detection(
            box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            class_name=DetectionClass.PLAYER,
            confidence=0.9,
            track_id=-1,
        )


def test_detection_with_track_id_returns_copy() -> None:
    detection = Detection(
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        class_name=DetectionClass.PLAYER,
        confidence=0.9,
    )

    tracked = detection.with_track_id(7)

    assert detection.track_id is None
    assert tracked.track_id == 7
    assert tracked.box == detection.box
