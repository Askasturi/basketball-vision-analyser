
import pytest

from basketball_vision_analyser.detection import DetectionClass
from basketball_vision_analyser.tracking import TrackingConfig


def test_tracking_config_defaults() -> None:
    config = TrackingConfig()

    assert config.iou_threshold == 0.3
    assert config.max_lost_frames == 5
    assert config.min_confidence == 0.0
    assert DetectionClass.PLAYER in config.track_classes
    assert DetectionClass.BALL in config.track_classes


def test_tracking_config_normalizes_string_classes() -> None:
    config = TrackingConfig(track_classes=("person", "basketball"))

    assert config.track_classes == (
        DetectionClass.PLAYER,
        DetectionClass.BALL,
    )


def test_tracking_config_rejects_invalid_iou_threshold() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        TrackingConfig(iou_threshold=2)


def test_tracking_config_rejects_invalid_max_lost_frames() -> None:
    with pytest.raises(ValueError, match="max_lost_frames"):
        TrackingConfig(max_lost_frames=-1)


def test_tracking_config_rejects_invalid_min_confidence() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        TrackingConfig(min_confidence=-0.1)
