
import pytest

from basketball_vision_analyser.classification import TeamClassificationConfig


def test_team_classification_config_defaults() -> None:
    config = TeamClassificationConfig()

    assert config.team_a_rgb == (220, 40, 40)
    assert config.team_b_rgb == (40, 80, 220)
    assert config.min_crop_area == 25
    assert config.unknown_confidence_threshold == 0.2


def test_team_classification_config_accepts_custom_values() -> None:
    config = TeamClassificationConfig(
        team_a_rgb=(255, 0, 0),
        team_b_rgb=(0, 0, 255),
        min_crop_area=10,
        unknown_confidence_threshold=0.5,
    )

    assert config.team_a_rgb == (255, 0, 0)
    assert config.team_b_rgb == (0, 0, 255)
    assert config.min_crop_area == 10
    assert config.unknown_confidence_threshold == 0.5


def test_team_classification_config_rejects_bad_rgb_length() -> None:
    with pytest.raises(ValueError, match="team_a_rgb"):
        TeamClassificationConfig(team_a_rgb=(255, 0))  # type: ignore[arg-type]


def test_team_classification_config_rejects_non_integer_rgb_value() -> None:
    with pytest.raises(TypeError, match="team_a_rgb"):
        TeamClassificationConfig(
            team_a_rgb=(255, 0, 0.5),  # type: ignore[arg-type]
        )


def test_team_classification_config_rejects_rgb_out_of_range() -> None:
    with pytest.raises(ValueError, match="team_a_rgb"):
        TeamClassificationConfig(team_a_rgb=(256, 0, 0))


def test_team_classification_config_rejects_invalid_min_crop_area() -> None:
    with pytest.raises(ValueError, match="min_crop_area"):
        TeamClassificationConfig(min_crop_area=0)


def test_team_classification_config_rejects_invalid_unknown_threshold() -> None:
    with pytest.raises(ValueError, match="unknown_confidence_threshold"):
        TeamClassificationConfig(unknown_confidence_threshold=2)
