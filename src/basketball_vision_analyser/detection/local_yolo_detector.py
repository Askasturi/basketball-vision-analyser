
"""Local Ultralytics YOLO detector backend."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from basketball_vision_analyser.detection.base_detector import BaseDetector
from basketball_vision_analyser.detection.bounding_box import BoundingBox
from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.detection import Detection
from basketball_vision_analyser.detection.result import DetectionResult
from basketball_vision_analyser.detection.types import DetectionClass, DetectorBackend
from basketball_vision_analyser.detection.yolo_config import YOLODetectorConfig


class LocalYOLODetector(BaseDetector):
    """Detector that runs local Ultralytics YOLO models."""

    def __init__(
        self,
        config: YOLODetectorConfig | DetectorConfig | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        yolo_config = self._normalize_config(config)
        super().__init__(config=yolo_config)

        self.yolo_config = yolo_config
        self.model = model if model is not None else self._load_model()

    def _normalize_config(
        self,
        config: YOLODetectorConfig | DetectorConfig | None,
    ) -> YOLODetectorConfig:
        if config is None:
            return YOLODetectorConfig()

        if isinstance(config, YOLODetectorConfig):
            return config

        return YOLODetectorConfig(
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            image_size=config.image_size,
            device=config.device,
            max_detections=config.max_detections,
        )

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            msg = (
                "Ultralytics is required for LocalYOLODetector. "
                'Install it with: python -m pip install -e ".[ai]"'
            )
            raise ImportError(msg) from exc

        return YOLO(str(self.yolo_config.model_path))

    def predict_frame(
        self,
        frame: np.ndarray,
        *,
        frame_index: int = 0,
    ) -> DetectionResult:
        """Run YOLO detection on one frame."""

        self.validate_frame(frame)

        if frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        start_time = perf_counter()
        raw_results = self.model.predict(
            frame,
            conf=self.yolo_config.confidence_threshold,
            iou=self.yolo_config.iou_threshold,
            imgsz=self.yolo_config.image_size,
            device=self.yolo_config.device,
            max_det=self.yolo_config.max_detections,
            verbose=False,
        )
        inference_time_ms = (perf_counter() - start_time) * 1000

        raw_result = raw_results[0] if isinstance(raw_results, list) else raw_results
        detections = self._convert_yolo_result(raw_result)

        return DetectionResult(
            frame_index=frame_index,
            detections=detections,
            image_shape=frame.shape,
            inference_time_ms=inference_time_ms,
            metadata={
                "backend": DetectorBackend.LOCAL_YOLO.value,
                "model_path": str(self.yolo_config.model_path),
            },
        )

    def _convert_yolo_result(self, raw_result: Any) -> tuple[Detection, ...]:
        boxes = getattr(raw_result, "boxes", None)
        if boxes is None:
            return ()

        xyxy_rows = self._to_list(getattr(boxes, "xyxy", []))
        confidence_values = self._to_list(getattr(boxes, "conf", []))
        class_values = self._to_list(getattr(boxes, "cls", []))

        detections: list[Detection] = []

        for xyxy, confidence, class_id_value in zip(
            xyxy_rows,
            confidence_values,
            class_values,
            strict=False,
        ):
            confidence_float = float(confidence)

            if confidence_float < self.yolo_config.confidence_threshold:
                continue

            box = self._box_from_xyxy(xyxy)
            if box is None:
                continue

            class_id = int(float(class_id_value))
            raw_label = self._class_label_for_id(class_id, raw_result)
            class_name = DetectionClass.from_label(raw_label)

            detections.append(
                Detection(
                    box=box,
                    class_name=class_name,
                    confidence=confidence_float,
                    metadata={
                        "raw_class_id": class_id,
                        "raw_label": raw_label,
                    },
                )
            )

            if len(detections) >= self.yolo_config.max_detections:
                break

        return tuple(detections)

    def _class_label_for_id(self, class_id: int, raw_result: Any) -> str:
        if class_id in self.yolo_config.class_names:
            return self.yolo_config.class_names[class_id]

        result_names = getattr(raw_result, "names", None)
        model_names = getattr(self.model, "names", None)
        names = result_names or model_names or {}

        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if isinstance(names, list | tuple) and 0 <= class_id < len(names):
            return str(names[class_id])

        return str(class_id)

    @staticmethod
    def _box_from_xyxy(xyxy: Any) -> BoundingBox | None:
        values = LocalYOLODetector._to_list(xyxy)
        if len(values) != 4:
            return None

        x1, y1, x2, y2 = (float(value) for value in values)

        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = max(0.0, x2)
        y2 = max(0.0, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if value is None:
            return []

        if hasattr(value, "detach"):
            value = value.detach()

        if hasattr(value, "cpu"):
            value = value.cpu()

        if hasattr(value, "numpy"):
            value = value.numpy()

        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        return [value]
