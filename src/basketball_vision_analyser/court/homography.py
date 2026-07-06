
"""Homography transformation utilities."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from basketball_vision_analyser.court.geometry import Point2D
from basketball_vision_analyser.court.keypoints import CourtKeypoint, CourtKeypointSet


class HomographyTransformer:
    """Transform points between image and court coordinates."""

    def __init__(
        self,
        image_to_court_matrix: np.ndarray,
        court_to_image_matrix: np.ndarray | None = None,
    ) -> None:
        self.image_to_court_matrix = self._validate_matrix(
            image_to_court_matrix,
            "image_to_court_matrix",
        )

        if court_to_image_matrix is None:
            court_to_image_matrix = np.linalg.inv(self.image_to_court_matrix)

        self.court_to_image_matrix = self._validate_matrix(
            court_to_image_matrix,
            "court_to_image_matrix",
        )

    @classmethod
    def from_keypoints(
        cls,
        keypoints: CourtKeypointSet | Iterable[CourtKeypoint],
        *,
        min_confidence: float = 0.0,
    ) -> HomographyTransformer:
        """Create a transformer from at least four keypoints."""

        keypoint_set = (
            keypoints
            if isinstance(keypoints, CourtKeypointSet)
            else CourtKeypointSet(tuple(keypoints))
        )
        keypoint_set = keypoint_set.by_min_confidence(min_confidence)
        keypoint_set.require_minimum(4)

        image_points = np.array(keypoint_set.image_points(), dtype=np.float32)
        court_points = np.array(keypoint_set.court_points(), dtype=np.float32)

        if len(keypoint_set) == 4:
            matrix = cv2.getPerspectiveTransform(image_points, court_points)
        else:
            matrix, _mask = cv2.findHomography(image_points, court_points)

        if matrix is None:
            msg = "Could not compute homography from keypoints."
            raise ValueError(msg)

        inverse_matrix = np.linalg.inv(matrix)

        return cls(
            image_to_court_matrix=matrix,
            court_to_image_matrix=inverse_matrix,
        )

    def image_to_court(self, point: Point2D) -> Point2D:
        """Transform an image point into court coordinates."""

        return self._transform_point(point, self.image_to_court_matrix)

    def court_to_image(self, point: Point2D) -> Point2D:
        """Transform a court point into image coordinates."""

        return self._transform_point(point, self.court_to_image_matrix)

    @staticmethod
    def _transform_point(point: Point2D, matrix: np.ndarray) -> Point2D:
        raw_point = np.array([[[point.x, point.y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(raw_point, matrix)
        x, y = transformed[0][0]

        return Point2D(float(x), float(y))

    @staticmethod
    def _validate_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)

        if matrix.shape != (3, 3):
            msg = f"{name} must have shape 3 x 3."
            raise ValueError(msg)

        return matrix
