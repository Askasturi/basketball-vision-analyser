
from __future__ import annotations

import numpy as np
import pytest

from basketball_vision_analyser.detection import (
    DetectionClass,
    LocalYOLODetector,
    YOLODetectorConfig,
)


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = np.array(
            [
                [10, 20, 30, 60],
                [70, 10, 90, 30],
                [1, 1, 2, 2],
            ],
            dtype=float,
        )
        self.conf = np.array([0.90, 0.80, 0.10], dtype=float)
        self.cls = np.array([0, 32, 99], dtype=float)


class FakeResult:
    def __init__(self) -> None:
        self.boxes = FakeBoxes()
        self.names = {
            0: "person",
            32: "sports ball",
            99: "mystery object",
        }


class FakeEmptyResult:
    boxes = None
    names: dict[int, str] = {}


class FakeModel:
    def __init__(self, result: object | None = None) -> None:
        self.result = result or FakeResult()
        self.names = {
            0: "person",
            32: "sports ball",
        }
        self.last_kwargs: dict[str, object] = {}

    def predict(self, source: np.ndarray, **kwargs: object) -> list[object]:
        self.last_kwargs = kwargs
        assert source.shape == (100, 200, 3)
        return [self.result]


def test_local_yolo_detector_converts_yolo_predictions() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    model = FakeModel()
    config = YOLODetectorConfig(
        confidence_threshold=0.25,
        iou_threshold=0.5,
        image_size=320,
        device="cpu",
        max_detections=10,
    )
    detector = LocalYOLODetector(config=config, model=model)

    result = detector.predict_frame(frame, frame_index=4)

    assert result.frame_index == 4
    assert result.image_shape == (100, 200, 3)
    assert result.count(DetectionClass.PLAYER) == 1
    assert result.count(DetectionClass.BALL) == 1
    assert result.count(DetectionClass.UNKNOWN) == 0
    assert result.metadata["backend"] == "local_yolo"

    assert model.last_kwargs["conf"] == 0.25
    assert model.last_kwargs["iou"] == 0.5
    assert model.last_kwargs["imgsz"] == 320
    assert model.last_kwargs["device"] == "cpu"


def test_local_yolo_detector_uses_custom_class_names() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    model = FakeModel()
    config = YOLODetectorConfig(
        confidence_threshold=0.25,
        class_names={0: "referee", 32: "basketball"},
    )
    detector = LocalYOLODetector(config=config, model=model)

    result = detector.predict_frame(frame)

    assert result.count(DetectionClass.REFEREE) == 1
    assert result.count(DetectionClass.BALL) == 1


def test_local_yolo_detector_handles_empty_results() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    model = FakeModel(result=FakeEmptyResult())
    detector = LocalYOLODetector(model=model)

    result = detector.predict_frame(frame)

    assert len(result) == 0


def test_local_yolo_detector_respects_max_detections() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    model = FakeModel()
    config = YOLODetectorConfig(max_detections=1)
    detector = LocalYOLODetector(config=config, model=model)

    result = detector.predict_frame(frame)

    assert len(result) == 1


def test_local_yolo_detector_rejects_invalid_frame_shape() -> None:
    frame = np.zeros((100, 200), dtype=np.uint8)
    detector = LocalYOLODetector(model=FakeModel())

    with pytest.raises(ValueError, match="height x width x 3"):
        detector.predict_frame(frame)


def test_local_yolo_detector_rejects_negative_frame_index() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detector = LocalYOLODetector(model=FakeModel())

    with pytest.raises(ValueError, match="frame_index"):
        detector.predict_frame(frame, frame_index=-1)
