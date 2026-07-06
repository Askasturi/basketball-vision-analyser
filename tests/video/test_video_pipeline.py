
from pathlib import Path

import cv2
import numpy as np

from basketball_vision_analyser.video import VideoLoader, VideoPipeline


def test_video_pipeline_copies_video(
    sample_video_path: Path,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "pipeline_output.mp4"

    pipeline = VideoPipeline()
    result = pipeline.run(sample_video_path, output_path, max_frames=4)

    assert result.input_path == sample_video_path
    assert result.output_path == output_path
    assert result.frames_processed == 4
    assert output_path.exists()

    metadata = VideoLoader(output_path).get_metadata()
    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.frame_count > 0


def test_video_pipeline_uses_processor(
    sample_video_path: Path,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "processed_output.mp4"

    def draw_marker(frame: np.ndarray, frame_index: int) -> np.ndarray:
        cv2.putText(
            frame,
            f"F{frame_index}",
            (5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        return frame

    pipeline = VideoPipeline(processor=draw_marker)
    result = pipeline.run(sample_video_path, output_path, max_frames=3)

    assert result.frames_processed == 3
    assert output_path.exists()
