
from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionResult,
)
from basketball_vision_analyser.tracking import SimpleTracker, TrackingConfig


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_name: DetectionClass = DetectionClass.PLAYER,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        class_name=class_name,
        confidence=confidence,
    )


def make_result(
    frame_index: int,
    detections: tuple[Detection, ...],
) -> DetectionResult:
    return DetectionResult(
        frame_index=frame_index,
        detections=detections,
        image_shape=(100, 100, 3),
    )


def test_simple_tracker_assigns_new_track_ids() -> None:
    tracker = SimpleTracker()
    result = tracker.update(
        make_result(
            0,
            (
                make_detection(0, 0, 10, 10),
                make_detection(50, 50, 60, 60, DetectionClass.BALL),
            ),
        )
    )

    assert result.track_ids == {0, 1}
    assert result.count(DetectionClass.PLAYER) == 1
    assert result.count(DetectionClass.BALL) == 1


def test_simple_tracker_keeps_track_id_for_overlapping_detection() -> None:
    tracker = SimpleTracker()

    first = tracker.update(
        make_result(0, (make_detection(0, 0, 20, 20),))
    )
    second = tracker.update(
        make_result(1, (make_detection(2, 2, 22, 22),))
    )

    assert first.objects[0].track_id == second.objects[0].track_id
    assert second.objects[0].age == 2
    assert second.objects[0].hits == 2


def test_simple_tracker_creates_new_id_for_low_iou_detection() -> None:
    tracker = SimpleTracker(config=TrackingConfig(iou_threshold=0.5))

    first = tracker.update(
        make_result(0, (make_detection(0, 0, 10, 10),))
    )
    second = tracker.update(
        make_result(1, (make_detection(80, 80, 90, 90),))
    )

    assert first.objects[0].track_id == 0
    assert second.objects[0].track_id == 1


def test_simple_tracker_does_not_match_different_classes() -> None:
    tracker = SimpleTracker()

    first = tracker.update(
        make_result(0, (make_detection(0, 0, 20, 20),))
    )
    second = tracker.update(
        make_result(
            1,
            (make_detection(0, 0, 20, 20, DetectionClass.BALL),),
        )
    )

    assert first.objects[0].track_id == 0
    assert second.objects[0].track_id == 1


def test_simple_tracker_filters_low_confidence_detections() -> None:
    tracker = SimpleTracker(config=TrackingConfig(min_confidence=0.5))

    result = tracker.update(
        make_result(0, (make_detection(0, 0, 10, 10, confidence=0.2),))
    )

    assert len(result) == 0
    assert tracker.active_track_count == 0


def test_simple_tracker_expires_lost_tracks() -> None:
    tracker = SimpleTracker(config=TrackingConfig(max_lost_frames=1))

    tracker.update(make_result(0, (make_detection(0, 0, 20, 20),)))
    assert tracker.active_track_count == 1

    tracker.update(make_result(1, ()))
    assert tracker.active_track_count == 1

    tracker.update(make_result(2, ()))
    assert tracker.active_track_count == 0


def test_simple_tracker_reset_clears_tracks() -> None:
    tracker = SimpleTracker()

    tracker.update(make_result(0, (make_detection(0, 0, 20, 20),)))
    assert tracker.active_track_count == 1

    tracker.reset()

    assert tracker.active_track_count == 0
    assert tracker.next_track_id == 0
