
import pytest

from basketball_vision_analyser.detection import DetectorBackend, DetectorConfig


def test_detector_config_defaults() -> None:
    config = DetectorConfig()

    assert config.backend == DetectorBackend.MOCK
    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.image_size == 640
    assert config.device == "cpu"
    assert config.max_detections == 300


def test_detector_config_accepts_string_backend() -> None:
    config = DetectorConfig(backend="mock")

    assert config.backend == DetectorBackend.MOCK


def test_detector_config_rejects_invalid_confidence_threshold() -> None:
    with pytest.raises(ValueError, match="confidence_threshold"):
        DetectorConfig(confidence_threshold=-0.1)


def test_detector_config_rejects_invalid_iou_threshold() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        DetectorConfig(iou_threshold=1.2)


def test_detector_config_rejects_invalid_image_size() -> None:
    with pytest.raises(ValueError, match="image_size"):
        DetectorConfig(image_size=0)


def test_detector_config_rejects_empty_device() -> None:
    with pytest.raises(ValueError, match="device"):
        DetectorConfig(device="")


def test_detector_config_rejects_invalid_max_detections() -> None:
    with pytest.raises(ValueError, match="max_detections"):
        DetectorConfig(max_detections=0)
