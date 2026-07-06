
import pytest

from basketball_vision_analyser.visualization import VisualizationConfig


def test_visualization_config_defaults() -> None:
    config = VisualizationConfig()

    assert config.player_color_bgr == (40, 220, 40)
    assert config.ball_color_bgr == (40, 160, 255)
    assert config.box_thickness == 2
    assert config.font_scale == 0.45
    assert config.draw_labels is True


def test_visualization_config_rejects_bad_color_length() -> None:
    with pytest.raises(ValueError, match="player_color_bgr"):
        VisualizationConfig(player_color_bgr=(1, 2))  # type: ignore[arg-type]


def test_visualization_config_rejects_bad_color_value_type() -> None:
    with pytest.raises(TypeError, match="player_color_bgr"):
        VisualizationConfig(
            player_color_bgr=(1, 2, 3.5),  # type: ignore[arg-type]
        )


def test_visualization_config_rejects_bad_color_range() -> None:
    with pytest.raises(ValueError, match="player_color_bgr"):
        VisualizationConfig(player_color_bgr=(1, 2, 999))


def test_visualization_config_rejects_invalid_thickness() -> None:
    with pytest.raises(ValueError, match="box_thickness"):
        VisualizationConfig(box_thickness=0)
