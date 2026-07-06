
import pytest

from basketball_vision_analyser.detection import (
    DetectorBackend,
    RoboflowDetectorConfig,
)


def test_roboflow_config_defaults() -> None:
    config = RoboflowDetectorConfig(api_key="abc123")

    assert config.backend == DetectorBackend.ROBOFLOW_API
    assert config.api_key == "abc123"
    assert config.model_id == "basketball-players-fy4c2-vfsuv/13"
    assert config.api_url == "https://serverless.roboflow.com"


def test_roboflow_config_rejects_empty_api_key_when_provided() -> None:
    with pytest.raises(ValueError, match="api_key"):
        RoboflowDetectorConfig(api_key="")


def test_roboflow_config_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        RoboflowDetectorConfig(api_key="abc123", model_id="")


def test_roboflow_config_rejects_invalid_model_id_format() -> None:
    with pytest.raises(ValueError, match="project/version"):
        RoboflowDetectorConfig(api_key="abc123", model_id="bad-model-id")


def test_roboflow_config_rejects_empty_api_url() -> None:
    with pytest.raises(ValueError, match="api_url"):
        RoboflowDetectorConfig(api_key="abc123", api_url="")
