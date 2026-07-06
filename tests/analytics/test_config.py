
import pytest

from basketball_vision_analyser.analytics import MovementAnalysisConfig


def test_movement_analysis_config_defaults() -> None:
    config = MovementAnalysisConfig()

    assert config.frame_rate_fps == 30.0
    assert config.max_frame_gap == 30
    assert config.max_speed_ft_per_second == 45.0


def test_movement_analysis_config_accepts_custom_values() -> None:
    config = MovementAnalysisConfig(
        frame_rate_fps=60,
        max_frame_gap=10,
        max_speed_ft_per_second=None,
    )

    assert config.frame_rate_fps == 60
    assert config.max_frame_gap == 10
    assert config.max_speed_ft_per_second is None


def test_movement_analysis_config_rejects_invalid_fps() -> None:
    with pytest.raises(ValueError, match="frame_rate_fps"):
        MovementAnalysisConfig(frame_rate_fps=0)


def test_movement_analysis_config_rejects_invalid_gap() -> None:
    with pytest.raises(ValueError, match="max_frame_gap"):
        MovementAnalysisConfig(max_frame_gap=0)


def test_movement_analysis_config_rejects_invalid_max_speed() -> None:
    with pytest.raises(ValueError, match="max_speed_ft_per_second"):
        MovementAnalysisConfig(max_speed_ft_per_second=0)
