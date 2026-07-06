
"""Roboflow API detector backend."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from basketball_vision_analyser.detection.base_detector import BaseDetector
from basketball_vision_analyser.detection.bounding_box import BoundingBox
from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.detection import Detection
from basketball_vision_analyser.detection.result import DetectionResult
from basketball_vision_analyser.detection.roboflow_config import RoboflowDetectorConfig
from basketball_vision_analyser.detection.types import DetectionClass, DetectorBackend


class RoboflowAPIDetector(BaseDetector):
    """Detector that runs Roboflow hosted inference."""

    def __init__(
        self,
        config: RoboflowDetectorConfig | DetectorConfig | None = None,
        *,
        client: Any | None = None,
    ) -> None:
        roboflow_config = self._normalize_config(config)
        super().__init__(config=roboflow_config)

        self.roboflow_config = roboflow_config
        self.client = client if client is not None else self._create_client()

    def _normalize_config(
        self,
        config: RoboflowDetectorConfig | DetectorConfig | None,
    ) -> RoboflowDetectorConfig:
        if config is None:
            return RoboflowDetectorConfig(
                api_key=os.getenv("ROBOFLOW_API_KEY"),
                model_id=os.getenv(
                    "ROBOFLOW_MODEL_ID",
                    "basketball-players-fy4c2-vfsuv/13",
                ),
            )

        if isinstance(config, RoboflowDetectorConfig):
            return config

        return RoboflowDetectorConfig(
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            image_size=config.image_size,
            device=config.device,
            max_detections=config.max_detections,
            api_key=os.getenv("ROBOFLOW_API_KEY"),
            model_id=os.getenv(
                "ROBOFLOW_MODEL_ID",
                "basketball-players-fy4c2-vfsuv/13",
            ),
        )

    def _create_client(self) -> Any:
        api_key = self.roboflow_config.api_key or os.getenv("ROBOFLOW_API_KEY")

        if not api_key:
            msg = (
                "Roboflow API key is required. Set ROBOFLOW_API_KEY in .env "
                "or pass api_key in RoboflowDetectorConfig."
            )
            raise ValueError(msg)

        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            msg = (
                "inference-sdk is required for RoboflowAPIDetector. "
                'Install it with: python -m pip install -e ".[ai]"'
            )
            raise ImportError(msg) from exc

        return InferenceHTTPClient(
            api_url=self.roboflow_config.api_url,
            api_key=api_key,
        )

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Run Roboflow detection on one frame."""

        self.validate_frame(frame)

        if frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        start_time = perf_counter()
        response = self.client.infer(
            self._frame_to_rgb(frame),
            model_id=self.roboflow_config.model_id,
        )
        inference_time_ms = (perf_counter() - start_time) * 1000

        detections = self._convert_response(response)

        return DetectionResult(
            frame_index=frame_index,
            detections=detections,
            image_shape=frame.shape,
            inference_time_ms=inference_time_ms,
            metadata={
                "backend": DetectorBackend.ROBOFLOW_API.value,
                "model_id": self.roboflow_config.model_id,
            },
        )

    @staticmethod
    def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _convert_response(self, response: dict[str, Any]) -> tuple[Detection, ...]:
        predictions = response.get("predictions", [])
        detections: list[Detection] = []

        for prediction in predictions:
            confidence = float(prediction.get("confidence", 0.0))

            if confidence < self.roboflow_config.confidence_threshold:
                continue

            box = self._box_from_prediction(prediction)
            if box is None:
                continue

            raw_label = str(prediction.get("class", "unknown"))
            detections.append(
                Detection(
                    box=box,
                    class_name=DetectionClass.from_label(raw_label),
                    confidence=confidence,
                    metadata={
                        "raw_label": raw_label,
                        "prediction": dict(prediction),
                    },
                )
            )

            if len(detections) >= self.roboflow_config.max_detections:
                break

        return tuple(detections)

    @staticmethod
    def _box_from_prediction(prediction: dict[str, Any]) -> BoundingBox | None:
        try:
            center_x = float(prediction["x"])
            center_y = float(prediction["y"])
            width = float(prediction["width"])
            height = float(prediction["height"])
        except (KeyError, TypeError, ValueError):
            return None

        if width <= 0 or height <= 0:
            return None

        x1 = max(0.0, center_x - width / 2)
        y1 = max(0.0, center_y - height / 2)
        x2 = max(0.0, center_x + width / 2)
        y2 = max(0.0, center_y + height / 2)

        if x2 <= x1 or y2 <= y1:
            return None

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
