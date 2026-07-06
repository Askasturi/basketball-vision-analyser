
import pytest

from basketball_vision_analyser.court import BasketballCourtConfig


def test_basketball_court_config_defaults() -> None:
    config = BasketballCourtConfig()

    assert config.court_length_ft == 94.0
    assert config.court_width_ft == 50.0
    assert config.tactical_view_width_px == 500
    assert config.tactical_view_height_px == 940
    assert config.margin_px == 20
    assert config.playable_width_px == 460
    assert config.playable_height_px == 900


def test_basketball_court_config_accepts_custom_values() -> None:
    config = BasketballCourtConfig(
        court_length_ft=84,
        court_width_ft=50,
        tactical_view_width_px=250,
        tactical_view_height_px=420,
        margin_px=10,
    )

    assert config.court_length_ft == 84
    assert config.tactical_view_width_px == 250
    assert config.playable_width_px == 230


def test_basketball_court_config_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="court_length_ft"):
        BasketballCourtConfig(court_length_ft=0)


def test_basketball_court_config_rejects_invalid_margin() -> None:
    with pytest.raises(ValueError, match="margin_px"):
        BasketballCourtConfig(margin_px=-1)


def test_basketball_court_config_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="keypoint_confidence_threshold"):
        BasketballCourtConfig(keypoint_confidence_threshold=2)
