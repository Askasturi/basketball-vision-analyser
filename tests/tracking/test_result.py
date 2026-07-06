
from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
)
from basketball_vision_analyser.tracking import TrackedObject, TrackingResult


def make_tracked_object(
    track_id: int,
    class_name: DetectionClass,
) -> TrackedObject:
    detection = Detection(
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        class_name=class_name,
        confidence=0.9,
        track_id=track_id,
    )
    return TrackedObject(
        track_id=track_id,
        detection=detection,
        frame_index=0,
        age=1,
        hits=1,
    )


def test_tracking_result_counts_and_filters() -> None:
    player = make_tracked_object(1, DetectionClass.PLAYER)
    ball = make_tracked_object(2, DetectionClass.BALL)

    result = TrackingResult(frame_index=0, objects=(player, ball))

    assert len(result) == 2
    assert result.track_ids == {1, 2}
    assert result.count() == 2
    assert result.count(DetectionClass.PLAYER) == 1
    assert result.count("basketball") == 1
    assert result.for_class(DetectionClass.BALL) == (ball,)
    assert result.by_track_id(1) == player
    assert result.by_track_id(999) is None
