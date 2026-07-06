import numpy as np
import pytest

from basketball_vision_analyser.detection import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionResult,
)
from basketball_vision_analyser.pipeline import BasketballFrameAnalyzer
from basketball_vision_analyser.possession import PossessionStatus


class FakeDetector:
    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
    ) -> DetectionResult:
        player = Detection(
            box=BoundingBox(x1=10, y1=10, x2=60, y2=60),
            class_name=DetectionClass.PLAYER,
            confidence=0.9,
        )
        ball = Detection(
            box=BoundingBox(x1=35, y1=35, x2=45, y2=45),
            class_name=DetectionClass.BALL,
            confidence=0.9,
        )

        return DetectionResult(
            frame_index=frame_index,
            detections=(player, ball),
            image_shape=frame.shape,
        )


def make_frame() -> np.ndarray:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    frame[10:60, 10:60] = (0, 0, 255)

    return frame


def test_basketball_frame_analyzer_runs_full_frame_pipeline() -> None:
    analyzer = BasketballFrameAnalyzer(detector=FakeDetector())
    result = analyzer.analyze_frame(make_frame(), frame_index=4)

    assert result.frame_index == 4
    assert result.detection_result.count() == 2
    assert result.tracking_result.count(DetectionClass.PLAYER) == 1
    assert result.team_result.count() == 1
    assert result.possession_result.assignment.status == (
        PossessionStatus.PLAYER_CONTROL
    )


def test_basketball_frame_analyzer_render_frame_draws_overlay() -> None:
    analyzer = BasketballFrameAnalyzer(detector=FakeDetector())
    frame = make_frame()
    result = analyzer.analyze_frame(frame, frame_index=0)

    output = analyzer.render_frame(frame, result)

    assert output is not frame
    assert np.any(output != frame)


def test_basketball_frame_analyzer_reset_resets_tracker_ids() -> None:
    analyzer = BasketballFrameAnalyzer(detector=FakeDetector())
    frame = make_frame()

    first = analyzer.analyze_frame(frame, frame_index=0)
    analyzer.reset()
    second = analyzer.analyze_frame(frame, frame_index=1)

    assert first.tracking_result.track_ids == {0, 1}
    assert second.tracking_result.track_ids == {0, 1}


def test_basketball_frame_analyzer_rejects_invalid_frame_shape() -> None:
    analyzer = BasketballFrameAnalyzer(detector=FakeDetector())
    frame = np.zeros((100, 160), dtype=np.uint8)

    with pytest.raises(ValueError, match="height x width x 3"):
        analyzer.analyze_frame(frame)


def test_basketball_frame_analyzer_rejects_non_array_frame() -> None:
    analyzer = BasketballFrameAnalyzer(detector=FakeDetector())

    with pytest.raises(TypeError, match="numpy array"):
        analyzer.analyze_frame("bad")  # type: ignore[arg-type]
