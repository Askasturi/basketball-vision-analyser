
"""Create a small synthetic basketball-style demo video."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


def create_demo_video(
    output_path: Path,
    *,
    frame_count: int = 60,
    width: int = 640,
    height: int = 360,
    fps: float = 30.0,
) -> Path:
    """Create a synthetic demo video for pipeline smoke tests."""

    if frame_count <= 0:
        msg = "frame_count must be greater than 0."
        raise ValueError(msg)

    if width <= 0 or height <= 0:
        msg = "width and height must be greater than 0."
        raise ValueError(msg)

    if fps <= 0:
        msg = "fps must be greater than 0."
        raise ValueError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        msg = f"Could not open video writer for {output_path}."
        raise OSError(msg)

    try:
        for frame_index in range(frame_count):
            frame = _draw_frame(frame_index, frame_count, width, height)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def _draw_frame(
    frame_index: int,
    frame_count: int,
    width: int,
    height: int,
) -> np.ndarray:
    frame = np.full((height, width, 3), 35, dtype=np.uint8)

    court_margin = 30
    court_color = (60, 120, 70)
    line_color = (230, 230, 230)

    cv2.rectangle(
        frame,
        (court_margin, court_margin),
        (width - court_margin, height - court_margin),
        court_color,
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (court_margin, court_margin),
        (width - court_margin, height - court_margin),
        line_color,
        thickness=2,
    )
    cv2.line(
        frame,
        (width // 2, court_margin),
        (width // 2, height - court_margin),
        line_color,
        thickness=1,
    )
    cv2.circle(frame, (width // 2, height // 2), 45, line_color, thickness=1)

    progress = frame_index / max(1, frame_count - 1)

    player_a_x = int(110 + progress * 180)
    player_b_x = int(width - 140 - progress * 140)
    player_y = height // 2 - 45

    ball_x = int(player_a_x + 45 + progress * 120)
    ball_y = player_y + 25

    cv2.rectangle(
        frame,
        (player_a_x, player_y),
        (player_a_x + 42, player_y + 80),
        (0, 0, 255),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (player_b_x, player_y + 15),
        (player_b_x + 42, player_y + 95),
        (255, 0, 0),
        thickness=-1,
    )
    cv2.circle(frame, (ball_x, ball_y), 10, (0, 140, 255), thickness=-1)

    cv2.putText(
        frame,
        f"Demo frame {frame_index}",
        (20, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    return frame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Create a synthetic demo video.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/demo_synthetic.mp4"),
    )
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""

    args = parse_args(argv)
    output_path = create_demo_video(
        args.output,
        frame_count=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(f"Wrote demo video to {output_path}")


if __name__ == "__main__":
    main()
