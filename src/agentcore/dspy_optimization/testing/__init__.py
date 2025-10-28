"""
A/B Testing Framework

Provides experiment design, traffic splitting, and automated rollout
mechanisms for optimization validation with statistical significance testing.
"""

from agentcore.dspy_optimization.testing.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentGroup,
    ExperimentManager,
    ExperimentResult,
    ExperimentStatus,
)
from agentcore.dspy_optimization.testing.rollout import (
    RolloutConfig,
    RolloutDecision,
    RolloutManager,
    RolloutStrategy,
)
from agentcore.dspy_optimization.testing.traffic import (
    RoutingDecision,
    TrafficSplitConfig,
    TrafficSplitter,
)
from agentcore.dspy_optimization.testing.validation import (
    ExperimentValidator,
    ValidationResult,
)

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "ExperimentGroup",
    "ExperimentManager",
    "ExperimentResult",
    "ExperimentStatus",
    "TrafficSplitter",
    "TrafficSplitConfig",
    "RoutingDecision",
    "ExperimentValidator",
    "ValidationResult",
    "RolloutManager",
    "RolloutConfig",
    "RolloutStrategy",
    "RolloutDecision",
]
