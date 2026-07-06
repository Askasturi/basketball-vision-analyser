from pathlib import Path

import pytest

from basketball_vision_analyser.detection import DetectorBackend, YOLODetectorConfig


def test_yolo_detector_config_defaults() -> None:
    config = YOLODetectorConfig()

    assert config.backend == DetectorBackend.LOCAL_YOLO
    assert config.model_path == Path("yolo11n.pt")
    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.image_size == 640
    assert config.device == "cpu"
    assert config.max_detections == 300
    assert config.class_names == {}


def test_yolo_detector_config_accepts_custom_values() -> None:
    config = YOLODetectorConfig(
        model_path="models/weights/basketball.pt",
        confidence_threshold=0.5,
        iou_threshold=0.6,
        image_size=1280,
        device="mps",
        max_detections=50,
        class_names={0: "player", 1: "basketball"},
    )

    assert config.model_path == Path("models/weights/basketball.pt")
    assert config.confidence_threshold == 0.5
    assert config.iou_threshold == 0.6
    assert config.image_size == 1280
    assert config.device == "mps"
    assert config.max_detections == 50
    assert config.class_names[1] == "basketball"


def test_yolo_detector_config_rejects_empty_model_path() -> None:
    with pytest.raises(ValueError, match="model_path"):
        YOLODetectorConfig(model_path="")


def test_yolo_detector_config_rejects_non_int_class_id() -> None:
    with pytest.raises(TypeError, match="class_names keys"):
        YOLODetectorConfig(class_names={"0": "player"})  # type: ignore[dict-item]


def test_yolo_detector_config_rejects_negative_class_id() -> None:
    with pytest.raises(ValueError, match="class_names keys"):
        YOLODetectorConfig(class_names={-1: "player"})


def test_yolo_detector_config_rejects_empty_class_name() -> None:
    with pytest.raises(ValueError, match="class_names values"):
        YOLODetectorConfig(class_names={0: ""})
