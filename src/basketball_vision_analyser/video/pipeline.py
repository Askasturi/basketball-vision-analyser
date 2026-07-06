
"""Basic video processing pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from basketball_vision_analyser.video.loader import VideoLoader
from basketball_vision_analyser.video.metadata import VideoMetadata
from basketball_vision_analyser.video.writer import VideoWriter

FrameProcessor = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class VideoPipelineResult:
    """Result returned after processing a video."""

    input_path: Path
    output_path: Path
    metadata: VideoMetadata
    frames_processed: int


class VideoPipeline:
    """Read, process, and write video frames."""

    def __init__(self, processor: FrameProcessor | None = None) -> None:
        self.processor = processor

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        max_frames: int | None = None,
    ) -> VideoPipelineResult:
        """Run the pipeline on a video."""

        loader = VideoLoader(input_path)
        metadata = loader.get_metadata()

        frames_processed = 0

        with VideoWriter(
            output_path,
            fps=metadata.fps,
            frame_size=metadata.resolution,
        ) as writer:
            for video_frame in loader.frames(max_frames=max_frames):
                image = video_frame.image

                if self.processor is not None:
                    image = self.processor(image, video_frame.index)

                writer.write_frame(image)
                frames_processed += 1

        return VideoPipelineResult(
            input_path=Path(input_path),
            output_path=Path(output_path),
            metadata=metadata,
            frames_processed=frames_processed,
        )
