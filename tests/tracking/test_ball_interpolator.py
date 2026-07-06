
import pytest

from basketball_vision_analyser.tracking import BallInterpolator, BallPosition


def test_ball_interpolator_fills_missing_middle_positions() -> None:
    positions = (
        BallPosition(frame_index=0, center=(0, 0)),
        BallPosition(frame_index=1, center=None),
        BallPosition(frame_index=2, center=None),
        BallPosition(frame_index=3, center=(30, 60)),
    )

    interpolated = BallInterpolator.interpolate(positions)

    assert interpolated[0].center == (0, 0)
    assert interpolated[1].center == pytest.approx((10, 20))
    assert interpolated[2].center == pytest.approx((20, 40))
    assert interpolated[3].center == (30, 60)


def test_ball_interpolator_keeps_leading_and_trailing_missing_values() -> None:
    positions = (
        BallPosition(frame_index=0, center=None),
        BallPosition(frame_index=1, center=(10, 10)),
        BallPosition(frame_index=2, center=None),
    )

    interpolated = BallInterpolator.interpolate(positions)

    assert interpolated[0].center is None
    assert interpolated[1].center == (10, 10)
    assert interpolated[2].center is None


def test_ball_interpolator_sorts_by_frame_index() -> None:
    positions = (
        BallPosition(frame_index=2, center=(20, 20)),
        BallPosition(frame_index=0, center=(0, 0)),
        BallPosition(frame_index=1, center=None),
    )

    interpolated = BallInterpolator.interpolate(positions)

    assert [position.frame_index for position in interpolated] == [0, 1, 2]
    assert interpolated[1].center == pytest.approx((10, 10))


def test_ball_position_rejects_negative_frame_index() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        BallPosition(frame_index=-1, center=None)
