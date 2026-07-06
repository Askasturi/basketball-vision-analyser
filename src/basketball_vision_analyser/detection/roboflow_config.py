
"""Roboflow API detector configuration."""

from __future__ import annotations

from dataclasses import dataclass

from basketball_vision_analyser.detection.config import DetectorConfig
from basketball_vision_analyser.detection.types import DetectorBackend


@dataclass(frozen=True)
class RoboflowDetectorConfig(DetectorConfig):
    """Configuration for Roboflow hosted inference."""

    backend: DetectorBackend | str = DetectorBackend.ROBOFLOW_API
    api_key: str | None = None
    model_id: str = "basketball-players-fy4c2-vfsuv/13"
    api_url: str = "https://serverless.roboflow.com"

    def __post_init__(self) -> None:
        DetectorConfig.__post_init__(self)

        if self.api_key is not None and not self.api_key.strip():
            msg = "api_key must not be empty when provided."
            raise ValueError(msg)

        if not self.model_id.strip():
            msg = "model_id must not be empty."
            raise ValueError(msg)

        if "/" not in self.model_id:
            msg = "model_id must use the format 'project/version'."
            raise ValueError(msg)

        if not self.api_url.strip():
            msg = "api_url must not be empty."
            raise ValueError(msg)
