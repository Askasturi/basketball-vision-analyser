
import pytest

from basketball_vision_analyser.events import EventDetectionConfig


def test_event_detection_config_defaults() -> None:
    config = EventDetectionConfig()

    assert config.max_gap_frames == 30
    assert config.require_known_teams is True


def test_event_detection_config_accepts_custom_values() -> None:
    config = EventDetectionConfig(
        max_gap_frames=5,
        require_known_teams=False,
    )

    assert config.max_gap_frames == 5
    assert config.require_known_teams is False


def test_event_detection_config_rejects_negative_gap() -> None:
    with pytest.raises(ValueError, match="max_gap_frames"):
        EventDetectionConfig(max_gap_frames=-1)
