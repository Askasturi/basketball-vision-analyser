
"""Local YOLO detector configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.types import DetectorBackend


@dataclass(frozen=True)
class YOLODetectorConfig(DetectorConfig):
    """Configuration for the local Ultralytics YOLO detector."""

    backend: DetectorBackend | str = DetectorBackend.LOCAL_YOLO
    model_path: str | Path = "yolo11n.pt"
    class_names: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        DetectorConfig.__post_init__(self)

        if isinstance(self.model_path, str) and not self.model_path.strip():
            msg = "model_path must not be empty."
            raise ValueError(msg)

        model_path = Path(self.model_path)

        if not str(model_path).strip():
            msg = "model_path must not be empty."
            raise ValueError(msg)

        for class_id, class_name in self.class_names.items():
            if not isinstance(class_id, int):
                msg = "class_names keys must be integers."
                raise TypeError(msg)

            if class_id < 0:
                msg = "class_names keys must be greater than or equal to 0."
                raise ValueError(msg)

            if not class_name.strip():
                msg = "class_names values must not be empty."
                raise ValueError(msg)

        object.__setattr__(self, "model_path", model_path)
