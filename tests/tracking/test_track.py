
import pytest

from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
)
from basketball_vision_analyser.tracking import TrackedObject


def make_detection() -> Detection:
    return Detection(
        box=BoundingBox(x1=0, y1=0, x2=10, y2=20),
        class_name=DetectionClass.PLAYER,
        confidence=0.9,
        track_id=3,
    )


def test_tracked_object_properties() -> None:
    tracked = TrackedObject(
        track_id=3,
        detection=make_detection(),
        frame_index=5,
        age=2,
        hits=2,
    )

    assert tracked.class_name == DetectionClass.PLAYER
    assert tracked.box.to_xyxy() == (0, 0, 10, 20)
    assert tracked.center == (5, 10)
    assert tracked.confidence == 0.9


def test_tracked_object_rejects_negative_track_id() -> None:
    with pytest.raises(ValueError, match="track_id"):
        TrackedObject(
            track_id=-1,
            detection=make_detection(),
            frame_index=0,
            age=1,
            hits=1,
        )


def test_tracked_object_rejects_invalid_age() -> None:
    with pytest.raises(ValueError, match="age"):
        TrackedObject(
            track_id=1,
            detection=make_detection(),
            frame_index=0,
            age=0,
            hits=1,
        )
