"""
Multi-step credit assignment with temporal difference rewards.

Implements per-step reward discounting to assign credit based on temporal proximity
to final outcomes. Earlier actions receive less credit than later actions.
"""

from __future__ import annotations

import structlog

from agentcore.training.models import Trajectory

logger = structlog.get_logger()


class CreditAssignmentConfig:
    """Configuration for credit assignment."""

    def __init__(
        self,
        gamma: float = 0.99,
        enable_td_rewards: bool = True,
    ) -> None:
        """
        Initialize credit assignment configuration.

        Args:
            gamma: Discount factor for temporal difference rewards (0 < gamma <= 1)
            enable_td_rewards: Whether to use TD rewards (False = uniform rewards)
        """
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        self.gamma = gamma
        self.enable_td_rewards = enable_td_rewards


class CreditAssignment:
    """
    Multi-step credit assignment with temporal difference rewards.

    Implements TD learning where each step receives a discounted reward based on
    temporal distance from the final outcome. Earlier steps receive less credit
    than later steps, improving convergence by properly assigning credit to actions.
    """

    def __init__(self, config: CreditAssignmentConfig | None = None) -> None:
        """
        Initialize credit assignment.

        Args:
            config: Credit assignment configuration (uses defaults if not provided)
        """
        self.config = config or CreditAssignmentConfig()

        logger.info(
            "credit_assignment_initialized",
            gamma=self.config.gamma,
            td_enabled=self.config.enable_td_rewards,
        )

    def compute_step_rewards(
        self,
        trajectory: Trajectory,
    ) -> list[float]:
        """
        Compute per-step rewards using temporal difference discounting.

        For a trajectory with n steps and final reward R:
        - Step i receives: R * gamma^(n-i-1)
        - Later steps (higher i) receive more credit
        - Earlier steps (lower i) receive less credit

        Args:
            trajectory: Execution trajectory with reward

        Returns:
            List of per-step rewards (length = num_steps)

        Example:
            trajectory with 3 steps, reward=1.0, gamma=0.99:
            - step 0: 1.0 * 0.99^2 = 0.9801
            - step 1: 1.0 * 0.99^1 = 0.99
            - step 2: 1.0 * 0.99^0 = 1.0
        """
        if not self.config.enable_td_rewards:
            # Uniform rewards: all steps get final reward
            num_steps = len(trajectory.steps)
            return [trajectory.reward] * num_steps

        num_steps = len(trajectory.steps)
        final_reward = trajectory.reward
        gamma = self.config.gamma

        # Compute discounted rewards: step i gets final_reward * gamma^(n-i-1)
        step_rewards = []
        for i in range(num_steps):
            discount = gamma ** (num_steps - i - 1)
            step_reward = final_reward * discount
            step_rewards.append(step_reward)

        logger.debug(
            "step_rewards_computed",
            num_steps=num_steps,
            final_reward=final_reward,
            gamma=gamma,
            step_rewards=step_rewards,
        )

        return step_rewards

    def compute_step_advantages(
        self,
        trajectories: list[Trajectory],
        normalize: bool = True,
    ) -> list[list[float]]:
        """
        Compute per-step advantages using TD rewards.

        Advantages are computed by:
        1. Computing step-wise rewards for each trajectory
        2. Normalizing across all steps in all trajectories (if normalize=True)
        3. Computing advantage = normalized_step_reward - mean_step_reward

        Args:
            trajectories: List of execution trajectories
            normalize: Whether to normalize advantages across all steps

        Returns:
            List of per-step advantage lists (one list per trajectory)

        Example:
            trajectories = [traj1(3 steps), traj2(2 steps)]
            Returns: [[adv1_0, adv1_1, adv1_2], [adv2_0, adv2_1]]
        """
        if not trajectories:
            return []

        # Compute step rewards for all trajectories
        all_step_rewards = []
        for traj in trajectories:
            step_rewards = self.compute_step_rewards(traj)
            all_step_rewards.append(step_rewards)

        if not normalize:
            # Return raw step rewards as advantages
            return all_step_rewards

        # Flatten all step rewards for normalization
        flat_rewards = []
        for step_rewards in all_step_rewards:
            flat_rewards.extend(step_rewards)

        if not flat_rewards:
            return []

        # Compute normalization statistics
        mean_reward = sum(flat_rewards) / len(flat_rewards)
        variance = sum((r - mean_reward) ** 2 for r in flat_rewards) / len(flat_rewards)
        std_reward = variance**0.5 if variance > 0 else 1.0

        # Compute normalized advantages
        all_advantages = []
        for step_rewards in all_step_rewards:
            step_advantages = []
            for reward in step_rewards:
                # Normalize: (reward - mean) / std
                normalized_reward = (reward - mean_reward) / std_reward
                step_advantages.append(normalized_reward)
            all_advantages.append(step_advantages)

        logger.debug(
            "step_advantages_computed",
            num_trajectories=len(trajectories),
            total_steps=len(flat_rewards),
            mean_reward=mean_reward,
            std_reward=std_reward,
        )

        return all_advantages

    def compute_trajectory_advantage(
        self,
        trajectory: Trajectory,
        baseline: float = 0.0,
    ) -> float:
        """
        Compute single trajectory-level advantage.

        This is a simplified version that computes a single advantage value
        for the entire trajectory by summing discounted step rewards.

        Args:
            trajectory: Execution trajectory
            baseline: Baseline value (typically mean trajectory reward)

        Returns:
            Trajectory-level advantage
        """
        step_rewards = self.compute_step_rewards(trajectory)
        total_reward = sum(step_rewards)
        advantage = total_reward - baseline

        logger.debug(
            "trajectory_advantage_computed",
            num_steps=len(trajectory.steps),
            total_reward=total_reward,
            baseline=baseline,
            advantage=advantage,
        )

        return advantage

    def get_config(self) -> dict[str, any]:
        """
        Get current configuration.

        Returns:
            Dictionary with configuration values
        """
        return {
            "gamma": self.config.gamma,
            "td_enabled": self.config.enable_td_rewards,
        }
