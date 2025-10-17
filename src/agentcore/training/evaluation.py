"""
Evaluation framework for GRPO training.

Implements held-out evaluation with metrics computation and statistical
significance testing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from scipy import stats  # type: ignore

from agentcore.training.models import Trajectory, TrainingQuery


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for training performance assessment."""

    success_rate: float
    avg_reward: float
    avg_steps: float
    tool_accuracy: float | None = None
    sample_size: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary format."""
        result: dict[str, float | int] = {
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "avg_steps": self.avg_steps,
            "sample_size": self.sample_size,
        }
        if self.tool_accuracy is not None:
            result["tool_accuracy"] = self.tool_accuracy
        return result


@dataclass
class StatisticalTest:
    """Statistical significance test result."""

    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05
    baseline_mean: float
    trained_mean: float
    improvement: float  # Percentage improvement


class EvaluationFramework:
    """
    Evaluation framework for GRPO training.

    Implements held-out evaluation with metrics computation and statistical
    significance testing for trained vs baseline agent comparison.
    """

    def __init__(self, evaluation_interval: int = 10):
        """
        Initialize evaluation framework.

        Args:
            evaluation_interval: Run evaluation every N iterations (default: 10)
        """
        self.evaluation_interval = evaluation_interval

    def split_training_data(
        self,
        training_queries: list[TrainingQuery],
        held_out_ratio: float = 0.2,
    ) -> tuple[list[TrainingQuery], list[TrainingQuery]]:
        """
        Split training data into train and held-out evaluation sets.

        Args:
            training_queries: Complete training dataset
            held_out_ratio: Fraction to reserve for evaluation (default: 0.2)

        Returns:
            Tuple of (train_queries, eval_queries)
        """
        if not 0 < held_out_ratio < 1:
            raise ValueError(f"held_out_ratio must be between 0 and 1, got {held_out_ratio}")

        split_idx = int(len(training_queries) * (1 - held_out_ratio))

        if split_idx < 1:
            raise ValueError(
                f"Training set too small to split with ratio {held_out_ratio}. "
                f"Need at least {int(1 / held_out_ratio)} queries."
            )

        train_queries = training_queries[:split_idx]
        eval_queries = training_queries[split_idx:]

        return train_queries, eval_queries

    def compute_metrics(
        self,
        trajectories: list[Trajectory],
    ) -> EvaluationMetrics:
        """
        Compute evaluation metrics from trajectories.

        Computes:
        - success_rate: Fraction of successful trajectories
        - avg_reward: Mean reward across trajectories
        - avg_steps: Mean number of steps per trajectory
        - tool_accuracy: Fraction of trajectories using tools correctly (if applicable)

        Args:
            trajectories: List of evaluation trajectories

        Returns:
            EvaluationMetrics with computed statistics
        """
        if not trajectories:
            raise ValueError("Cannot compute metrics from empty trajectory list")

        # Success rate
        successful = [t for t in trajectories if t.success is True]
        success_rate = len(successful) / len(trajectories)

        # Average reward
        rewards = [t.reward for t in trajectories]
        avg_reward = statistics.mean(rewards)

        # Average steps
        step_counts = [len(t.steps) for t in trajectories]
        avg_steps = statistics.mean(step_counts)

        # Tool accuracy (optional - depends on trajectory metadata)
        tool_accuracy: float | None = None
        tool_using_trajectories = [
            t for t in trajectories
            if any(
                step.action.get("type") == "tool_call"
                for step in t.steps
            )
        ]
        if tool_using_trajectories:
            correct_tool_usage = [
                t for t in tool_using_trajectories
                if self._check_tool_correctness(t)
            ]
            tool_accuracy = len(correct_tool_usage) / len(tool_using_trajectories)

        return EvaluationMetrics(
            success_rate=success_rate,
            avg_reward=avg_reward,
            avg_steps=avg_steps,
            tool_accuracy=tool_accuracy,
            sample_size=len(trajectories),
        )

    def _check_tool_correctness(self, trajectory: Trajectory) -> bool:
        """
        Check if tools were used correctly in a trajectory.

        Heuristic: trajectory succeeded and used tools without errors.

        Args:
            trajectory: Trajectory to check

        Returns:
            True if tools used correctly, False otherwise
        """
        if not trajectory.success:
            return False

        # Check for tool call errors in steps
        for step in trajectory.steps:
            if step.action.get("type") == "tool_call":
                result = step.result
                if isinstance(result, dict) and result.get("error"):
                    return False

        return True

    def compare_with_baseline(
        self,
        baseline_trajectories: list[Trajectory],
        trained_trajectories: list[Trajectory],
        metric_key: str = "reward",
    ) -> StatisticalTest:
        """
        Compare trained agent performance with baseline using t-test.

        Args:
            baseline_trajectories: Trajectories from untrained agent
            trained_trajectories: Trajectories from trained agent
            metric_key: Metric to compare ("reward", "steps", "success")

        Returns:
            StatisticalTest with t-statistic, p-value, and significance
        """
        if not baseline_trajectories or not trained_trajectories:
            raise ValueError("Both baseline and trained trajectories required for comparison")

        # Extract metric values
        baseline_values = self._extract_metric(baseline_trajectories, metric_key)
        trained_values = self._extract_metric(trained_trajectories, metric_key)

        # Compute t-test
        t_statistic, p_value = stats.ttest_ind(trained_values, baseline_values)

        # Check significance (p < 0.05)
        is_significant = p_value < 0.05

        # Compute means and improvement
        baseline_mean = statistics.mean(baseline_values)
        trained_mean = statistics.mean(trained_values)

        if baseline_mean == 0:
            improvement = 0.0 if trained_mean == 0 else 100.0
        else:
            improvement = ((trained_mean - baseline_mean) / abs(baseline_mean)) * 100

        return StatisticalTest(
            t_statistic=float(t_statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            baseline_mean=baseline_mean,
            trained_mean=trained_mean,
            improvement=improvement,
        )

    def _extract_metric(
        self,
        trajectories: list[Trajectory],
        metric_key: str,
    ) -> list[float]:
        """
        Extract metric values from trajectories.

        Args:
            trajectories: List of trajectories
            metric_key: Metric to extract ("reward", "steps", "success")

        Returns:
            List of metric values
        """
        if metric_key == "reward":
            return [t.reward for t in trajectories]
        elif metric_key == "steps":
            return [float(len(t.steps)) for t in trajectories]
        elif metric_key == "success":
            return [1.0 if t.success else 0.0 for t in trajectories]
        else:
            raise ValueError(f"Unknown metric_key: {metric_key}")

    def should_evaluate(self, current_iteration: int) -> bool:
        """
        Check if evaluation should run at current iteration.

        Args:
            current_iteration: Current training iteration

        Returns:
            True if evaluation should run, False otherwise
        """
        return current_iteration > 0 and current_iteration % self.evaluation_interval == 0

    def run_evaluation(
        self,
        eval_queries: list[TrainingQuery],
        baseline_trajectories: list[Trajectory],
        trained_trajectories: list[Trajectory],
    ) -> dict[str, Any]:
        """
        Run complete evaluation workflow.

        Computes metrics for both baseline and trained agents, then performs
        statistical significance testing.

        Args:
            eval_queries: Held-out evaluation queries
            baseline_trajectories: Trajectories from untrained agent
            trained_trajectories: Trajectories from trained agent

        Returns:
            Dictionary with evaluation results including metrics and statistical tests
        """
        # Compute metrics
        baseline_metrics = self.compute_metrics(baseline_trajectories)
        trained_metrics = self.compute_metrics(trained_trajectories)

        # Statistical significance tests
        reward_test = self.compare_with_baseline(
            baseline_trajectories,
            trained_trajectories,
            metric_key="reward",
        )

        success_test = self.compare_with_baseline(
            baseline_trajectories,
            trained_trajectories,
            metric_key="success",
        )

        steps_test = self.compare_with_baseline(
            baseline_trajectories,
            trained_trajectories,
            metric_key="steps",
        )

        return {
            "baseline_metrics": baseline_metrics.to_dict(),
            "trained_metrics": trained_metrics.to_dict(),
            "statistical_tests": {
                "reward": {
                    "t_statistic": reward_test.t_statistic,
                    "p_value": reward_test.p_value,
                    "is_significant": reward_test.is_significant,
                    "improvement_percent": reward_test.improvement,
                },
                "success": {
                    "t_statistic": success_test.t_statistic,
                    "p_value": success_test.p_value,
                    "is_significant": success_test.is_significant,
                    "improvement_percent": success_test.improvement,
                },
                "steps": {
                    "t_statistic": steps_test.t_statistic,
                    "p_value": steps_test.p_value,
                    "is_significant": steps_test.is_significant,
                    "improvement_percent": steps_test.improvement,
                },
            },
            "eval_query_count": len(eval_queries),
        }
