"""
Base optimizer class for all optimization algorithms
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import dspy

from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    PerformanceMetrics,
)


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms

    All optimizers must implement the optimize method to perform
    systematic optimization of agent prompts and behaviors.
    """

    def __init__(self, llm: dspy.LM | None = None) -> None:
        """
        Initialize optimizer

        Args:
            llm: DSPy language model for optimization
        """
        self.llm = llm or dspy.LM("openai/gpt-5-mini")

    @abstractmethod
    async def optimize(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Perform optimization

        Args:
            request: Optimization request with target and objectives
            baseline_metrics: Current performance metrics
            training_data: Training examples for optimization

        Returns:
            OptimizationResult with improvements and details
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the optimization algorithm"""
        pass

    def calculate_improvement(
        self, baseline: PerformanceMetrics, optimized: PerformanceMetrics
    ) -> float:
        """
        Calculate overall improvement percentage

        Args:
            baseline: Baseline performance metrics
            optimized: Optimized performance metrics

        Returns:
            Improvement percentage (0-100)
        """
        improvements = []

        # Success rate improvement
        if baseline.success_rate > 0:
            success_improvement = (
                (optimized.success_rate - baseline.success_rate)
                / baseline.success_rate
                * 100
            )
            improvements.append(success_improvement)

        # Cost efficiency improvement (lower is better)
        if baseline.avg_cost_per_task > 0:
            cost_improvement = (
                (baseline.avg_cost_per_task - optimized.avg_cost_per_task)
                / baseline.avg_cost_per_task
                * 100
            )
            improvements.append(cost_improvement)

        # Latency improvement (lower is better)
        if baseline.avg_latency_ms > 0:
            latency_improvement = (
                (baseline.avg_latency_ms - optimized.avg_latency_ms)
                / baseline.avg_latency_ms
                * 100
            )
            improvements.append(latency_improvement)

        # Quality score improvement
        if baseline.quality_score > 0:
            quality_improvement = (
                (optimized.quality_score - baseline.quality_score)
                / baseline.quality_score
                * 100
            )
            improvements.append(quality_improvement)

        # Return average improvement
        return sum(improvements) / len(improvements) if improvements else 0.0
