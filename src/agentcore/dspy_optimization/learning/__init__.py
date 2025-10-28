"""
Continuous learning pipeline for DSPy optimization

Provides online learning, model updates, drift detection, and automatic
retraining for continuous improvement of optimization models.
"""

from agentcore.dspy_optimization.learning.drift import (
    DriftDetector,
    DriftStatus,
    DriftConfig,
    DriftResult,
)
from agentcore.dspy_optimization.learning.online import (
    OnlineLearner,
    OnlineLearningConfig,
    LearningUpdate,
)
from agentcore.dspy_optimization.learning.versioning import (
    ModelVersion,
    ModelVersionManager,
    DeploymentStrategy,
)
from agentcore.dspy_optimization.learning.retraining import (
    RetrainingTrigger,
    RetrainingConfig,
    RetrainingManager,
    TriggerCondition,
)
from agentcore.dspy_optimization.learning.pipeline import (
    ContinuousLearningPipeline,
    PipelineConfig,
    PipelineStatus,
)

__all__ = [
    "DriftDetector",
    "DriftStatus",
    "DriftConfig",
    "DriftResult",
    "OnlineLearner",
    "OnlineLearningConfig",
    "LearningUpdate",
    "ModelVersion",
    "ModelVersionManager",
    "DeploymentStrategy",
    "RetrainingTrigger",
    "RetrainingConfig",
    "RetrainingManager",
    "TriggerCondition",
    "ContinuousLearningPipeline",
    "PipelineConfig",
    "PipelineStatus",
]
