
import numpy as np
import pytest

from basketball_vision_analyser.classification import PlayerTeam
from basketball_vision_analyser.court import (
    BasketballCourtConfig,
    Point2D,
    TacticalViewRenderer,
)


def test_tactical_view_renderer_creates_blank_view() -> None:
    renderer = TacticalViewRenderer(
        BasketballCourtConfig(
            tactical_view_width_px=100,
            tactical_view_height_px=200,
            margin_px=10,
        )
    )

    view = renderer.create_blank_view()

    assert view.shape == (200, 100, 3)
    assert view.dtype == np.uint8


def test_tactical_view_renderer_maps_court_point_to_view() -> None:
    renderer = TacticalViewRenderer(
        BasketballCourtConfig(
            court_width_ft=50,
            court_length_ft=100,
            tactical_view_width_px=120,
            tactical_view_height_px=220,
            margin_px=10,
        )
    )

    point = renderer.court_to_view(Point2D(25, 50))

    assert point.x == pytest.approx(60)
    assert point.y == pytest.approx(110)


def test_tactical_view_renderer_draws_marker() -> None:
    renderer = TacticalViewRenderer()
    view = renderer.create_blank_view()
    before = view.copy()

    output = renderer.draw_marker(
        view,
        Point2D(25, 47),
        label="1",
        team=PlayerTeam.TEAM_A,
    )

    assert output is view
    assert np.any(view != before)


def test_tactical_view_renderer_rejects_invalid_view_shape() -> None:
    renderer = TacticalViewRenderer()
    bad_view = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError, match="height x width x 3"):
        renderer.draw_marker(bad_view, Point2D(1, 1))


def test_tactical_view_renderer_rejects_non_array_view() -> None:
    renderer = TacticalViewRenderer()

    with pytest.raises(TypeError, match="numpy array"):
        renderer.draw_marker("bad", Point2D(1, 1))  # type: ignore[arg-type]
