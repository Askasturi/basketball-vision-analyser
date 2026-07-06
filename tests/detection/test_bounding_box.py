
import pytest

from basketball_vision_analyser.detection import BoundingBox


def test_bounding_box_properties() -> None:
    box = BoundingBox(x1=10, y1=20, x2=30, y2=50)

    assert box.width == 20
    assert box.height == 30
    assert box.area == 600
    assert box.center == (20, 35)
    assert box.to_xyxy() == (10, 20, 30, 50)
    assert box.to_xywh() == (10, 20, 20, 30)


def test_bounding_box_iou_for_overlapping_boxes() -> None:
    first = BoundingBox(x1=0, y1=0, x2=20, y2=20)
    second = BoundingBox(x1=10, y1=10, x2=30, y2=30)

    assert first.iou(second) == pytest.approx(100 / 700)


def test_bounding_box_iou_for_non_overlapping_boxes() -> None:
    first = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    second = BoundingBox(x1=20, y1=20, x2=30, y2=30)

    assert first.iou(second) == 0


def test_bounding_box_rejects_negative_coordinates() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BoundingBox(x1=-1, y1=0, x2=10, y2=10)


def test_bounding_box_rejects_invalid_x_order() -> None:
    with pytest.raises(ValueError, match="x2"):
        BoundingBox(x1=10, y1=0, x2=10, y2=20)


def test_bounding_box_rejects_invalid_y_order() -> None:
    with pytest.raises(ValueError, match="y2"):
        BoundingBox(x1=0, y1=20, x2=10, y2=20)
