
from basketball_vision_analyser.detection import DetectionClass, DetectorBackend


def test_detection_class_from_label_handles_aliases() -> None:
    assert DetectionClass.from_label("person") == DetectionClass.PLAYER
    assert DetectionClass.from_label("basketball") == DetectionClass.BALL
    assert DetectionClass.from_label("sports ball") == DetectionClass.BALL
    assert DetectionClass.from_label("rim") == DetectionClass.HOOP
    assert DetectionClass.from_label("ref") == DetectionClass.REFEREE
    assert DetectionClass.from_label("game clock") == DetectionClass.CLOCK


def test_detection_class_from_label_returns_unknown() -> None:
    assert DetectionClass.from_label("random object") == DetectionClass.UNKNOWN


def test_detector_backend_values() -> None:
    assert DetectorBackend.MOCK.value == "mock"
    assert DetectorBackend.LOCAL_YOLO.value == "local_yolo"
    assert DetectorBackend.ROBOFLOW_API.value == "roboflow_api"
