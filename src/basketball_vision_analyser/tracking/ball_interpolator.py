
"""Ball position interpolation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BallPosition:
    """Ball center position for one frame."""

    frame_index: int
    center: tuple[float, float] | None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)


class BallInterpolator:
    """Interpolate missing ball centers between known positions."""

    @staticmethod
    def interpolate(
        positions: tuple[BallPosition, ...] | list[BallPosition],
    ) -> tuple[BallPosition, ...]:
        """Fill missing ball centers between known neighboring centers."""

        if not positions:
            return ()

        sorted_positions = tuple(sorted(positions, key=lambda item: item.frame_index))
        centers: list[tuple[float, float] | None] = [
            position.center for position in sorted_positions
        ]

        known_indices = [
            index for index, center in enumerate(centers) if center is not None
        ]

        for start_index, end_index in zip(
            known_indices,
            known_indices[1:],
            strict=False,
        ):
            start_center = centers[start_index]
            end_center = centers[end_index]

            if start_center is None or end_center is None:
                continue

            gap = end_index - start_index
            if gap <= 1:
                continue

            start_x, start_y = start_center
            end_x, end_y = end_center

            for index in range(start_index + 1, end_index):
                ratio = (index - start_index) / gap
                centers[index] = (
                    start_x + (end_x - start_x) * ratio,
                    start_y + (end_y - start_y) * ratio,
                )

        return tuple(
            BallPosition(
                frame_index=position.frame_index,
                center=centers[index],
            )
            for index, position in enumerate(sorted_positions)
        )
