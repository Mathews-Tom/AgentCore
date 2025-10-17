"""
Reward computation system for training infrastructure.

Implements outcome-based and shaped reward functions with group normalization.
"""

from __future__ import annotations

from typing import Any, Callable

import structlog

from agentcore.training.models import Trajectory

logger = structlog.get_logger()


class RewardConfig:
    """Configuration for reward computation."""

    def __init__(
        self,
        tool_usage_reward: float = 0.1,
        verification_reward: float = 0.05,
        length_penalty: float = -0.01,
        enable_shaping: bool = True,
    ) -> None:
        """
        Initialize reward configuration.

        Args:
            tool_usage_reward: Reward per successful tool call
            verification_reward: Reward per verification step
            length_penalty: Penalty per step (encourages efficiency)
            enable_shaping: Whether to apply reward shaping
        """
        self.tool_usage_reward = tool_usage_reward
        self.verification_reward = verification_reward
        self.length_penalty = length_penalty
        self.enable_shaping = enable_shaping


class RewardEngine:
    """
    Computes rewards for trajectories using outcome-based and shaped reward functions.

    Supports custom reward functions and group-based normalization.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """
        Initialize reward engine.

        Args:
            config: Reward configuration (uses defaults if not provided)
        """
        self.config = config or RewardConfig()
        self.custom_functions: dict[str, Callable[[Trajectory], float]] = {}

        logger.info(
            "reward_engine_initialized",
            tool_usage_reward=self.config.tool_usage_reward,
            verification_reward=self.config.verification_reward,
            length_penalty=self.config.length_penalty,
            shaping_enabled=self.config.enable_shaping,
        )

    def register_custom_function(
        self, name: str, func: Callable[[Trajectory], float]
    ) -> None:
        """
        Register custom reward function.

        Args:
            name: Function identifier
            func: Callable that takes Trajectory and returns reward
        """
        self.custom_functions[name] = func
        logger.info("custom_reward_function_registered", name=name)

    def compute_reward(
        self, trajectory: Trajectory, custom_function: str | None = None
    ) -> float:
        """
        Compute total reward for trajectory.

        Args:
            trajectory: Trajectory to compute reward for
            custom_function: Optional custom function name to use

        Returns:
            Total reward (outcome + shaped rewards)
        """
        # Use custom function if specified
        if custom_function:
            if custom_function not in self.custom_functions:
                raise ValueError(f"Unknown custom function: {custom_function}")
            reward = self.custom_functions[custom_function](trajectory)
            logger.debug(
                "reward_computed_custom",
                trajectory_id=str(trajectory.trajectory_id) if trajectory.trajectory_id else "none",
                function=custom_function,
                reward=reward,
            )
            return reward

        # Outcome-based reward
        outcome_reward = self._compute_outcome_reward(trajectory)

        # Shaped rewards
        shaped_reward = 0.0
        if self.config.enable_shaping:
            shaped_reward = self._compute_shaped_rewards(trajectory)

        total_reward = outcome_reward + shaped_reward

        logger.debug(
            "reward_computed",
            trajectory_id=str(trajectory.trajectory_id) if trajectory.trajectory_id else "none",
            outcome_reward=outcome_reward,
            shaped_reward=shaped_reward,
            total_reward=total_reward,
            steps=len(trajectory.steps),
        )

        return total_reward

    def compute_rewards(self, trajectories: list[Trajectory]) -> list[float]:
        """
        Compute rewards for multiple trajectories.

        Args:
            trajectories: List of trajectories

        Returns:
            List of rewards (same order as input)
        """
        rewards = [self.compute_reward(traj) for traj in trajectories]

        logger.info(
            "batch_rewards_computed",
            count=len(rewards),
            mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        )

        return rewards

    def normalize_rewards(self, rewards: list[float]) -> list[float]:
        """
        Normalize rewards using group statistics (mean, std).

        Args:
            rewards: Raw rewards

        Returns:
            Normalized rewards (zero mean, unit variance)
        """
        if not rewards:
            return []

        if len(rewards) == 1:
            return [0.0]  # Single reward normalizes to 0

        # Compute statistics
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance**0.5

        # Handle edge case: std = 0 (all rewards identical)
        if std_reward == 0.0:
            logger.warning(
                "reward_normalization_zero_std",
                mean_reward=mean_reward,
                count=len(rewards),
            )
            return [0.0] * len(rewards)

        # Normalize
        normalized = [(r - mean_reward) / std_reward for r in rewards]

        logger.debug(
            "rewards_normalized",
            count=len(rewards),
            mean_before=mean_reward,
            std_before=std_reward,
            mean_after=sum(normalized) / len(normalized),
        )

        return normalized

    def compute_advantages(
        self, trajectories: list[Trajectory], normalize: bool = True
    ) -> list[float]:
        """
        Compute advantages for trajectories (normalized rewards).

        Args:
            trajectories: List of trajectories
            normalize: Whether to normalize rewards

        Returns:
            List of advantages
        """
        rewards = self.compute_rewards(trajectories)

        if normalize:
            advantages = self.normalize_rewards(rewards)
        else:
            advantages = rewards

        logger.info(
            "advantages_computed",
            count=len(advantages),
            positive_count=sum(1 for a in advantages if a > 0),
            negative_count=sum(1 for a in advantages if a < 0),
            mean_advantage=sum(advantages) / len(advantages) if advantages else 0.0,
        )

        return advantages

    def _compute_outcome_reward(self, trajectory: Trajectory) -> float:
        """
        Compute outcome-based reward (success/failure).

        Args:
            trajectory: Trajectory to evaluate

        Returns:
            Outcome reward (1.0 for success, 0.0 for failure)
        """
        return 1.0 if trajectory.success else 0.0

    def _compute_shaped_rewards(self, trajectory: Trajectory) -> float:
        """
        Compute shaped rewards based on trajectory steps.

        Args:
            trajectory: Trajectory to evaluate

        Returns:
            Sum of shaped rewards
        """
        total_shaped = 0.0
        num_steps = len(trajectory.steps)

        for step in trajectory.steps:
            # Tool usage reward
            if self._is_successful_tool_call(step.action):
                total_shaped += self.config.tool_usage_reward

            # Verification reward
            if self._is_verification_step(step.action):
                total_shaped += self.config.verification_reward

        # Length penalty (applied per step)
        total_shaped += num_steps * self.config.length_penalty

        return total_shaped

    def _is_successful_tool_call(self, action: dict[str, Any]) -> bool:
        """
        Check if action represents successful tool usage.

        Args:
            action: Step action data

        Returns:
            True if successful tool call
        """
        # Check for tool-related step types
        step_type = action.get("step_type", "")
        if "tool" in step_type.lower() or "action" in step_type.lower():
            # Assume successful if present (detailed validation in future)
            return True

        return False

    def _is_verification_step(self, action: dict[str, Any]) -> bool:
        """
        Check if action represents verification.

        Args:
            action: Step action data

        Returns:
            True if verification step
        """
        step_type = action.get("step_type", "")
        return "verify" in step_type.lower() or "check" in step_type.lower()
