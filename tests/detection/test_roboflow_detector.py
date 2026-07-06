from __future__ import annotations

import numpy as np
import pytest

from basketball_vision_analyser.detection import (
    DetectionClass,
    RoboflowAPIDetector,
    RoboflowDetectorConfig,
)


class FakeRoboflowClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.last_model_id: str | None = None
        self.last_image_shape: tuple[int, ...] | None = None

    def infer(self, image: np.ndarray, *, model_id: str) -> dict[str, object]:
        self.last_model_id = model_id
        self.last_image_shape = image.shape
        return self.response


def test_roboflow_detector_converts_predictions() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    response = {
        "predictions": [
            {
                "x": 50,
                "y": 60,
                "width": 20,
                "height": 40,
                "confidence": 0.91,
                "class": "Player",
            },
            {
                "x": 120,
                "y": 30,
                "width": 10,
                "height": 10,
                "confidence": 0.82,
                "class": "Ball",
            },
            {
                "x": 1,
                "y": 1,
                "width": 1,
                "height": 1,
                "confidence": 0.01,
                "class": "Overlay",
            },
        ]
    }
    client = FakeRoboflowClient(response)
    config = RoboflowDetectorConfig(
        api_key="abc123",
        confidence_threshold=0.25,
        model_id="basketball-players-fy4c2-vfsuv/13",
    )

    detector = RoboflowAPIDetector(config=config, client=client)
    result = detector.predict_frame(frame, frame_index=3)

    assert result.frame_index == 3
    assert result.image_shape == (100, 200, 3)
    assert result.count(DetectionClass.PLAYER) == 1
    assert result.count(DetectionClass.BALL) == 1
    assert result.count(DetectionClass.OVERLAY) == 0
    assert result.metadata["backend"] == "roboflow_api"
    assert client.last_model_id == "basketball-players-fy4c2-vfsuv/13"
    assert client.last_image_shape == (100, 200, 3)


def test_roboflow_detector_handles_empty_predictions() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    client = FakeRoboflowClient({"predictions": []})
    detector = RoboflowAPIDetector(
        config=RoboflowDetectorConfig(api_key="abc123"),
        client=client,
    )

    result = detector.predict_frame(frame)

    assert len(result) == 0


def test_roboflow_detector_skips_invalid_prediction_boxes() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    client = FakeRoboflowClient(
        {
            "predictions": [
                {
                    "x": 50,
                    "y": 50,
                    "width": 0,
                    "height": 20,
                    "confidence": 0.9,
                    "class": "Player",
                },
                {
                    "confidence": 0.9,
                    "class": "Ball",
                },
            ]
        }
    )
    detector = RoboflowAPIDetector(
        config=RoboflowDetectorConfig(api_key="abc123"),
        client=client,
    )

    result = detector.predict_frame(frame)

    assert len(result) == 0


def test_roboflow_detector_respects_max_detections() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    client = FakeRoboflowClient(
        {
            "predictions": [
                {
                    "x": 10,
                    "y": 10,
                    "width": 5,
                    "height": 5,
                    "confidence": 0.9,
                    "class": "Player",
                },
                {
                    "x": 30,
                    "y": 30,
                    "width": 5,
                    "height": 5,
                    "confidence": 0.9,
                    "class": "Ball",
                },
            ]
        }
    )
    detector = RoboflowAPIDetector(
        config=RoboflowDetectorConfig(
            api_key="abc123",
            max_detections=1,
        ),
        client=client,
    )

    result = detector.predict_frame(frame)

    assert len(result) == 1


def test_roboflow_detector_rejects_invalid_frame_shape() -> None:
    frame = np.zeros((100, 200), dtype=np.uint8)
    detector = RoboflowAPIDetector(
        config=RoboflowDetectorConfig(api_key="abc123"),
        client=FakeRoboflowClient({"predictions": []}),
    )

    with pytest.raises(ValueError, match="height x width x 3"):
        detector.predict_frame(frame)


def test_roboflow_detector_rejects_negative_frame_index() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    detector = RoboflowAPIDetector(
        config=RoboflowDetectorConfig(api_key="abc123"),
        client=FakeRoboflowClient({"predictions": []}),
    )

    with pytest.raises(ValueError, match="frame_index"):
        detector.predict_frame(frame, frame_index=-1)


def test_roboflow_detector_requires_api_key_without_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Roboflow API key"):
        RoboflowAPIDetector(config=RoboflowDetectorConfig(api_key=None))
