"""
GRPO (Group Refined Policy Optimization) trainer implementation.

Implements policy gradient updates using advantage-based trajectory comparison.
"""

from __future__ import annotations

from typing import Any

import structlog

from agentcore.training.credit_assignment import CreditAssignment, CreditAssignmentConfig
from agentcore.training.models import Trajectory
from agentcore.training.rewards import RewardEngine

logger = structlog.get_logger()


class GRPOConfig:
    """Configuration for GRPO training algorithm."""

    def __init__(
        self,
        learning_rate: float = 0.0001,
        gradient_clip_value: float = 1.0,
        advantage_threshold: float = 0.0,
        enable_gradient_clipping: bool = True,
    ) -> None:
        """
        Initialize GRPO configuration.

        Args:
            learning_rate: Policy gradient learning rate
            gradient_clip_value: Maximum gradient magnitude
            advantage_threshold: Minimum advantage for policy update
            enable_gradient_clipping: Whether to clip gradients
        """
        self.learning_rate = learning_rate
        self.gradient_clip_value = gradient_clip_value
        self.advantage_threshold = advantage_threshold
        self.enable_gradient_clipping = enable_gradient_clipping


class TrainingMetrics:
    """Training metrics tracker."""

    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self.losses: list[float] = []
        self.avg_rewards: list[float] = []
        self.std_rewards: list[float] = []
        self.positive_advantages: list[int] = []
        self.iterations: int = 0

    def record(
        self,
        loss: float,
        avg_reward: float,
        std_reward: float,
        positive_count: int,
    ) -> None:
        """
        Record training metrics for iteration.

        Args:
            loss: Training loss value
            avg_reward: Average reward across trajectories
            std_reward: Standard deviation of rewards
            positive_count: Number of positive-advantage trajectories
        """
        self.losses.append(loss)
        self.avg_rewards.append(avg_reward)
        self.std_rewards.append(std_reward)
        self.positive_advantages.append(positive_count)
        self.iterations += 1

    def get_latest(self) -> dict[str, float | int]:
        """
        Get latest metrics.

        Returns:
            Dictionary with latest metric values
        """
        if self.iterations == 0:
            return {}

        return {
            "loss": self.losses[-1],
            "avg_reward": self.avg_rewards[-1],
            "std_reward": self.std_rewards[-1],
            "positive_advantages": self.positive_advantages[-1],
            "iteration": self.iterations,
        }

    def is_converged(self, window: int = 10, threshold: float = 0.01) -> bool:
        """
        Check if training has converged.

        Args:
            window: Number of recent iterations to check
            threshold: Maximum allowed loss variance for convergence

        Returns:
            True if converged
        """
        if self.iterations < window:
            return False

        recent_losses = self.losses[-window:]
        mean_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((loss - mean_loss) ** 2 for loss in recent_losses) / len(
            recent_losses
        )

        return variance < threshold


class GRPOTrainer:
    """
    GRPO (Group Refined Policy Optimization) trainer.

    Implements advantage-based policy gradient updates following GRPO algorithm.
    """

    def __init__(
        self,
        reward_engine: RewardEngine,
        config: GRPOConfig | None = None,
        credit_assignment: CreditAssignment | None = None,
    ) -> None:
        """
        Initialize GRPO trainer.

        Args:
            reward_engine: Reward computation engine
            config: GRPO configuration (uses defaults if not provided)
            credit_assignment: Credit assignment module (uses defaults if not provided)
        """
        self.reward_engine = reward_engine
        self.config = config or GRPOConfig()
        self.credit_assignment = credit_assignment or CreditAssignment()
        self.metrics = TrainingMetrics()

        logger.info(
            "grpo_trainer_initialized",
            learning_rate=self.config.learning_rate,
            gradient_clipping=self.config.enable_gradient_clipping,
            clip_value=self.config.gradient_clip_value,
            td_enabled=self.credit_assignment.config.enable_td_rewards,
            gamma=self.credit_assignment.config.gamma,
        )

    def compute_policy_gradient(
        self,
        trajectories: list[Trajectory],
        log_probs: list[float],
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute policy gradient loss.

        Uses credit assignment for advantage computation when TD rewards are enabled,
        otherwise falls back to standard reward-based advantages.

        Args:
            trajectories: List of execution trajectories
            log_probs: Log-probabilities of trajectories under current policy

        Returns:
            Tuple of (loss, gradient_info)
        """
        if len(trajectories) != len(log_probs):
            raise ValueError("Trajectories and log_probs must have same length")

        # Compute advantages using credit assignment
        if self.credit_assignment.config.enable_td_rewards:
            # Use TD rewards for advantage computation
            # Compute baseline (mean trajectory reward)
            rewards = self.reward_engine.compute_rewards(trajectories)
            baseline = sum(rewards) / len(rewards) if rewards else 0.0

            advantages = []
            for traj in trajectories:
                adv = self.credit_assignment.compute_trajectory_advantage(
                    traj, baseline=baseline
                )
                advantages.append(adv)

            logger.debug(
                "advantages_computed_with_td",
                count=len(advantages),
                baseline=baseline,
                mean_advantage=sum(advantages) / len(advantages) if advantages else 0.0,
            )
        else:
            # Use standard reward-based advantages
            advantages = self.reward_engine.compute_advantages(
                trajectories, normalize=True
            )

        # Filter positive advantages
        positive_indices = [i for i, adv in enumerate(advantages) if adv > self.config.advantage_threshold]

        if not positive_indices:
            logger.warning(
                "no_positive_advantages",
                total_trajectories=len(trajectories),
                threshold=self.config.advantage_threshold,
            )
            return 0.0, {
                "positive_count": 0,
                "total_count": len(trajectories),
                "gradients": [],
            }

        # Compute policy gradient: loss = -log_prob * advantage
        # Only update trajectories with positive advantage
        gradients = []
        for idx in positive_indices:
            log_prob = log_probs[idx]
            advantage = advantages[idx]
            gradient = -log_prob * advantage
            gradients.append(gradient)

        # Apply gradient clipping
        if self.config.enable_gradient_clipping:
            gradients = self._clip_gradients(gradients)

        # Compute total loss
        loss = sum(gradients) / len(gradients)

        gradient_info = {
            "positive_count": len(positive_indices),
            "total_count": len(trajectories),
            "gradients": gradients,
            "mean_advantage": sum(advantages[i] for i in positive_indices)
            / len(positive_indices),
        }

        logger.debug(
            "policy_gradient_computed",
            loss=loss,
            positive_count=len(positive_indices),
            total_count=len(trajectories),
            mean_gradient=sum(gradients) / len(gradients),
        )

        return loss, gradient_info

    def training_step(
        self,
        trajectories: list[Trajectory],
        log_probs: list[float],
    ) -> dict[str, Any]:
        """
        Execute single training iteration.

        Args:
            trajectories: Batch of trajectories
            log_probs: Log-probabilities under current policy

        Returns:
            Training step metrics
        """
        # Compute loss and gradients
        loss, gradient_info = self.compute_policy_gradient(trajectories, log_probs)

        # Compute reward statistics
        rewards = self.reward_engine.compute_rewards(trajectories)
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        variance = (
            sum((r - avg_reward) ** 2 for r in rewards) / len(rewards)
            if rewards
            else 0.0
        )
        std_reward = variance**0.5

        # Record metrics
        self.metrics.record(
            loss=loss,
            avg_reward=avg_reward,
            std_reward=std_reward,
            positive_count=gradient_info["positive_count"],
        )

        step_metrics = {
            "loss": loss,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "positive_advantages": gradient_info["positive_count"],
            "total_trajectories": len(trajectories),
            "iteration": self.metrics.iterations,
        }

        logger.info(
            "training_step_complete",
            **step_metrics,
        )

        return step_metrics

    def should_continue_training(
        self,
        max_iterations: int,
        convergence_check: bool = True,
    ) -> bool:
        """
        Check if training should continue.

        Args:
            max_iterations: Maximum allowed iterations
            convergence_check: Whether to check convergence

        Returns:
            True if training should continue
        """
        # Check iteration limit
        if self.metrics.iterations >= max_iterations:
            logger.info(
                "training_iteration_limit_reached",
                iterations=self.metrics.iterations,
                max_iterations=max_iterations,
            )
            return False

        # Check convergence
        if convergence_check and self.metrics.is_converged():
            logger.info(
                "training_converged",
                iterations=self.metrics.iterations,
                final_loss=self.metrics.losses[-1] if self.metrics.losses else 0.0,
            )
            return False

        return True

    def get_metrics(self) -> dict[str, Any]:
        """
        Get current training metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "iterations": self.metrics.iterations,
            "latest": self.metrics.get_latest(),
            "convergence": {
                "converged": self.metrics.is_converged(),
                "losses": self.metrics.losses[-10:] if self.metrics.losses else [],
            },
        }

    def _clip_gradients(self, gradients: list[float]) -> list[float]:
        """
        Apply gradient clipping.

        Args:
            gradients: Raw gradients

        Returns:
            Clipped gradients
        """
        clip_value = self.config.gradient_clip_value

        clipped = []
        clipped_count = 0

        for grad in gradients:
            if grad > clip_value:
                clipped.append(clip_value)
                clipped_count += 1
            elif grad < -clip_value:
                clipped.append(-clip_value)
                clipped_count += 1
            else:
                clipped.append(grad)

        if clipped_count > 0:
            logger.debug(
                "gradients_clipped",
                total=len(gradients),
                clipped_count=clipped_count,
                clip_value=clip_value,
            )

        return clipped

    def reset_metrics(self) -> None:
        """Reset training metrics."""
        self.metrics = TrainingMetrics()
        logger.info("training_metrics_reset")
