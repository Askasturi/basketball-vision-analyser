
import pytest

from basketball_vision_analyser.possession import PossessionConfig


def test_possession_config_defaults() -> None:
    config = PossessionConfig()

    assert config.max_control_distance_px == 80.0
    assert config.min_control_confidence == 0.0


def test_possession_config_accepts_custom_values() -> None:
    config = PossessionConfig(
        max_control_distance_px=50,
        min_control_confidence=0.4,
    )

    assert config.max_control_distance_px == 50
    assert config.min_control_confidence == 0.4


def test_possession_config_rejects_invalid_distance() -> None:
    with pytest.raises(ValueError, match="max_control_distance_px"):
        PossessionConfig(max_control_distance_px=0)


def test_possession_config_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="min_control_confidence"):
        PossessionConfig(min_control_confidence=2)
