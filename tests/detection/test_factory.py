
import pytest

from basketball_vision_analyser.detection import (
    DetectorBackend,
    DetectorConfig,
    DetectorFactory,
    MockDetector,
)


def test_detector_factory_creates_mock_detector_by_default() -> None:
    detector = DetectorFactory.create()

    assert isinstance(detector, MockDetector)


def test_detector_factory_creates_mock_detector_from_config() -> None:
    detector = DetectorFactory.create(
        DetectorConfig(backend=DetectorBackend.MOCK)
    )

    assert isinstance(detector, MockDetector)


def test_detector_factory_raises_for_temporarily_unregistered_backend() -> None:
    original = DetectorFactory._registry.pop(DetectorBackend.ROBOFLOW_API)

    try:
        config = DetectorConfig(backend=DetectorBackend.ROBOFLOW_API)

        with pytest.raises(ValueError, match="No detector registered"):
            DetectorFactory.create(config)
    finally:
        DetectorFactory._registry[DetectorBackend.ROBOFLOW_API] = original
