
"""Classification infrastructure for Basketball Vision Analyser."""

from basketball_vision_analyser.classification.assignment import TeamAssignment
from basketball_vision_analyser.classification.color_team_classifier import (
    ColorTeamClassifier,
)
from basketball_vision_analyser.classification.config import TeamClassificationConfig
from basketball_vision_analyser.classification.result import TeamClassificationResult
from basketball_vision_analyser.classification.types import PlayerTeam

__all__ = [
    "ColorTeamClassifier",
    "PlayerTeam",
    "TeamAssignment",
    "TeamClassificationConfig",
    "TeamClassificationResult",
]
