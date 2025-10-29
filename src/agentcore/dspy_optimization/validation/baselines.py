"""
Baseline optimizers for comparison

Provides simple baseline algorithms (random search, grid search) for
comparing against research-backed optimization algorithms.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

import dspy
from pydantic import BaseModel

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)


class BaselineOptimizer(BaseOptimizer, ABC):
    """Base class for baseline optimizers"""

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name"""
        pass


class RandomSearchOptimizer(BaselineOptimizer):
    """
    Random search baseline

    Randomly samples configurations without any intelligent search strategy.
    Serves as lower bound for optimization performance.
    """

    def __init__(
        self,
        llm: dspy.LM | None = None,
        num_trials: int = 10,
        seed: int | None = None,
    ) -> None:
        """
        Initialize random search optimizer

        Args:
            llm: DSPy language model
            num_trials: Number of random trials
            seed: Random seed for reproducibility
        """
        super().__init__(llm)
        self.num_trials = num_trials
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    async def optimize(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Perform random search optimization

        Args:
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data

        Returns:
            Optimization result
        """
        result = OptimizationResult(
            status=OptimizationStatus.IN_PROGRESS,
            baseline_performance=baseline_metrics,
        )

        try:
            best_metrics = baseline_metrics
            best_config: dict[str, Any] = {}

            # Random search over configurations
            for trial in range(self.num_trials):
                # Generate random configuration
                config = self._generate_random_config()

                # Evaluate configuration
                metrics = await self._evaluate_config(
                    config,
                    training_data,
                    baseline_metrics,
                )

                # Update best if improved
                if metrics.success_rate > best_metrics.success_rate:
                    best_metrics = metrics
                    best_config = config

            # Calculate improvement
            improvement = self.calculate_improvement(baseline_metrics, best_metrics)

            # Update result
            result.status = OptimizationStatus.COMPLETED
            result.optimized_performance = best_metrics
            result.improvement_percentage = improvement
            result.statistical_significance = 0.1  # Lower confidence for random search
            result.optimization_details = OptimizationDetails(
                algorithm_used=self.get_algorithm_name(),
                iterations=self.num_trials,
                key_improvements=[
                    f"Evaluated {self.num_trials} random configurations",
                    f"Best config: {best_config}",
                ],
                parameters={
                    "num_trials": self.num_trials,
                    "seed": self.seed,
                    "best_config": best_config,
                },
            )

        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.error_message = str(e)

        return result

    def get_algorithm_name(self) -> str:
        """Get algorithm name"""
        return "random_search"

    def _generate_random_config(self) -> dict[str, Any]:
        """
        Generate random configuration

        Returns:
            Random configuration dict
        """
        return {
            "temperature": random.uniform(0.1, 1.0),
            "max_tokens": random.randint(100, 2000),
            "reasoning_depth": random.randint(1, 5),
            "tool_usage_threshold": random.uniform(0.3, 0.9),
        }

    async def _evaluate_config(
        self,
        config: dict[str, Any],
        training_data: list[dict[str, Any]],
        baseline_metrics: PerformanceMetrics,
    ) -> PerformanceMetrics:
        """
        Evaluate configuration

        Args:
            config: Configuration to evaluate
            training_data: Training data
            baseline_metrics: Baseline metrics

        Returns:
            Performance metrics
        """
        # Simulate evaluation with random variation
        improvement_factor = random.uniform(0.8, 1.1)

        return PerformanceMetrics(
            success_rate=min(baseline_metrics.success_rate * improvement_factor, 1.0),
            avg_cost_per_task=baseline_metrics.avg_cost_per_task * random.uniform(0.9, 1.1),
            avg_latency_ms=int(baseline_metrics.avg_latency_ms * random.uniform(0.9, 1.1)),
            quality_score=min(baseline_metrics.quality_score * improvement_factor, 1.0),
        )


class GridSearchOptimizer(BaselineOptimizer):
    """
    Grid search baseline

    Exhaustively searches over a predefined grid of configurations.
    More systematic than random search but lacks intelligence.
    """

    def __init__(
        self,
        llm: dspy.LM | None = None,
        grid_size: int = 3,
    ) -> None:
        """
        Initialize grid search optimizer

        Args:
            llm: DSPy language model
            grid_size: Number of values per parameter
        """
        super().__init__(llm)
        self.grid_size = grid_size

    async def optimize(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Perform grid search optimization

        Args:
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data

        Returns:
            Optimization result
        """
        result = OptimizationResult(
            status=OptimizationStatus.IN_PROGRESS,
            baseline_performance=baseline_metrics,
        )

        try:
            # Generate grid
            grid = self._generate_grid()

            best_metrics = baseline_metrics
            best_config: dict[str, Any] = {}
            evaluated_count = 0

            # Evaluate all grid points
            for config in grid:
                metrics = await self._evaluate_config(
                    config,
                    training_data,
                    baseline_metrics,
                )

                evaluated_count += 1

                # Update best if improved
                if metrics.success_rate > best_metrics.success_rate:
                    best_metrics = metrics
                    best_config = config

            # Calculate improvement
            improvement = self.calculate_improvement(baseline_metrics, best_metrics)

            # Update result
            result.status = OptimizationStatus.COMPLETED
            result.optimized_performance = best_metrics
            result.improvement_percentage = improvement
            result.statistical_significance = 0.05  # Medium confidence for grid search
            result.optimization_details = OptimizationDetails(
                algorithm_used=self.get_algorithm_name(),
                iterations=evaluated_count,
                key_improvements=[
                    f"Exhaustively evaluated {evaluated_count} grid configurations",
                    f"Best config: {best_config}",
                ],
                parameters={
                    "grid_size": self.grid_size,
                    "total_configs": evaluated_count,
                    "best_config": best_config,
                },
            )

        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.error_message = str(e)

        return result

    def get_algorithm_name(self) -> str:
        """Get algorithm name"""
        return "grid_search"

    def _generate_grid(self) -> list[dict[str, Any]]:
        """
        Generate parameter grid

        Returns:
            List of configurations
        """
        import itertools

        # Define parameter grids
        temperatures = [0.1 + i * (0.9 / (self.grid_size - 1)) for i in range(self.grid_size)]
        max_tokens_values = [100 + i * (1900 // (self.grid_size - 1)) for i in range(self.grid_size)]
        reasoning_depths = list(range(1, self.grid_size + 1))

        # Generate all combinations
        grid = []
        for temp, tokens, depth in itertools.product(
            temperatures, max_tokens_values, reasoning_depths
        ):
            grid.append({
                "temperature": temp,
                "max_tokens": int(tokens),
                "reasoning_depth": depth,
            })

        return grid

    async def _evaluate_config(
        self,
        config: dict[str, Any],
        training_data: list[dict[str, Any]],
        baseline_metrics: PerformanceMetrics,
    ) -> PerformanceMetrics:
        """
        Evaluate configuration

        Args:
            config: Configuration to evaluate
            training_data: Training data
            baseline_metrics: Baseline metrics

        Returns:
            Performance metrics
        """
        # Simulate evaluation with deterministic variation based on config
        # Better configs (mid-range temperature, higher reasoning depth) perform better
        temp_score = 1.0 - abs(config["temperature"] - 0.5)  # Best at 0.5
        depth_score = min(config["reasoning_depth"] / 5.0, 1.0)  # Better with more depth

        improvement_factor = 0.9 + (temp_score + depth_score) * 0.15

        return PerformanceMetrics(
            success_rate=min(baseline_metrics.success_rate * improvement_factor, 1.0),
            avg_cost_per_task=baseline_metrics.avg_cost_per_task * (1.0 / improvement_factor),
            avg_latency_ms=int(baseline_metrics.avg_latency_ms * (1.0 / improvement_factor)),
            quality_score=min(baseline_metrics.quality_score * improvement_factor, 1.0),
        )


class BaselineComparison(BaseModel):
    """Comparison of algorithm against baselines"""

    algorithm_name: str
    algorithm_improvement: float
    random_search_improvement: float
    grid_search_improvement: float
    beats_random_search: bool
    beats_grid_search: bool
    improvement_over_random: float
    improvement_over_grid: float

    @property
    def beats_both_baselines(self) -> bool:
        """Check if beats both baseline methods"""
        return self.beats_random_search and self.beats_grid_search
