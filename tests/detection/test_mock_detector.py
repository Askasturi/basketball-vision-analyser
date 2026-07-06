
import numpy as np
import pytest

from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
    MockDetector,
)


def test_mock_detector_predicts_default_detections() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detector = MockDetector()

    result = detector.predict_frame(frame, frame_index=5)

    assert result.frame_index == 5
    assert result.image_shape == (100, 200, 3)
    assert result.count(DetectionClass.PLAYER) == 1
    assert result.count(DetectionClass.BALL) == 1
    assert result.count(DetectionClass.HOOP) == 1


def test_mock_detector_predicts_custom_detections() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    custom = Detection(
        box=BoundingBox(x1=1, y1=2, x2=10, y2=20),
        class_name=DetectionClass.REFEREE,
        confidence=0.7,
    )

    detector = MockDetector(detections=(custom,))
    result = detector.predict_frame(frame)

    assert result.detections == (custom,)
    assert result.count(DetectionClass.REFEREE) == 1


def test_mock_detector_predicts_multiple_frames() -> None:
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(3)]
    detector = MockDetector()

    results = detector.predict_frames(frames, start_index=10)

    assert [result.frame_index for result in results] == [10, 11, 12]


def test_mock_detector_rejects_invalid_frame_shape() -> None:
    frame = np.zeros((100, 200), dtype=np.uint8)
    detector = MockDetector()

    with pytest.raises(ValueError, match="height x width x 3"):
        detector.predict_frame(frame)


def test_mock_detector_rejects_negative_frame_index() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detector = MockDetector()

    with pytest.raises(ValueError, match="frame_index"):
        detector.predict_frame(frame, frame_index=-1)
