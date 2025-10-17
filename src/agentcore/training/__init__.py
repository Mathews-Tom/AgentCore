"""
AgentCore Training Module.

Implements Flow-Based Optimization Engine (GRPO) for agent performance improvement
through reinforcement learning.

Components:
- models: Pydantic models for training configuration and data validation
- database_models: SQLAlchemy ORM models for persistence
- repositories: Data access layer following repository pattern
- reward_registry: Custom reward function registry with validation
"""

from agentcore.training.reward_registry import (
    RewardRegistry,
    RewardValidationError,
    code_quality_reward,
    get_global_registry,
    reset_global_registry,
    response_accuracy_reward,
    task_efficiency_reward,
)

__all__ = [
    "GRPOConfig",
    "Trajectory",
    "TrainingJob",
    "PolicyCheckpoint",
    "TrainingJobRepository",
    "TrajectoryRepository",
    "CheckpointRepository",
    # Reward system
    "RewardRegistry",
    "RewardValidationError",
    "get_global_registry",
    "reset_global_registry",
    # Example reward functions
    "code_quality_reward",
    "response_accuracy_reward",
    "task_efficiency_reward",
]
