
import pytest

from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionResult,
)


def make_detection(
    class_name: DetectionClass,
    confidence: float,
) -> Detection:
    return Detection(
        box=BoundingBox(x1=0, y1=0, x2=10, y2=10),
        class_name=class_name,
        confidence=confidence,
    )


def test_detection_result_counts_and_filters() -> None:
    player = make_detection(DetectionClass.PLAYER, 0.9)
    ball = make_detection(DetectionClass.BALL, 0.8)

    result = DetectionResult(
        frame_index=2,
        detections=(player, ball),
        image_shape=(48, 64, 3),
    )

    assert len(result) == 2
    assert result.count() == 2
    assert result.count(DetectionClass.PLAYER) == 1
    assert result.count("basketball") == 1
    assert result.for_class(DetectionClass.BALL) == (ball,)
    assert result.classes == {DetectionClass.PLAYER, DetectionClass.BALL}


def test_detection_result_confidence_filter() -> None:
    low = make_detection(DetectionClass.PLAYER, 0.2)
    high = make_detection(DetectionClass.BALL, 0.9)

    result = DetectionResult(frame_index=0, detections=(low, high))
    filtered = result.by_min_confidence(0.5)

    assert len(filtered) == 1
    assert filtered.detections[0] == high


def test_detection_result_rejects_invalid_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        DetectionResult(frame_index=-1)


def test_detection_result_rejects_invalid_image_shape() -> None:
    with pytest.raises(ValueError, match="image_shape"):
        DetectionResult(frame_index=0, image_shape=(1, 2, 3, 4))


def test_detection_result_rejects_invalid_inference_time() -> None:
    with pytest.raises(ValueError, match="inference_time"):
        DetectionResult(frame_index=0, inference_time_ms=-1)


def test_detection_result_rejects_invalid_confidence_threshold() -> None:
    result = DetectionResult(frame_index=0)

    with pytest.raises(ValueError, match="threshold"):
        result.by_min_confidence(2)
