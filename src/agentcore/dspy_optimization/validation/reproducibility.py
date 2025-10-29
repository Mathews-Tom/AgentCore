"""
Reproducibility validation

Validates that optimization algorithms produce consistent results across
multiple runs with same seed, ensuring deterministic behavior.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    PerformanceMetrics,
)


class ReproducibilityResult(BaseModel):
    """Result of reproducibility validation"""

    algorithm_name: str
    num_runs: int
    seed: int
    is_reproducible: bool
    variance: float
    mean_improvement: float
    std_deviation: float
    coefficient_of_variation: float
    all_results: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReproducibilityAnalysis(BaseModel):
    """Analysis of reproducibility across algorithms"""

    total_algorithms: int
    reproducible_algorithms: int
    reproducibility_rate: float
    results_by_algorithm: dict[str, ReproducibilityResult] = Field(default_factory=dict)
    summary: str


class ReproducibilityValidator:
    """
    Validates reproducibility of optimization algorithms

    Ensures that algorithms produce consistent results across multiple
    runs with the same seed, which is critical for:
    - Scientific validity
    - Debugging
    - Reliable comparisons
    """

    def __init__(
        self,
        num_runs: int = 5,
        variance_threshold: float = 0.01,
    ) -> None:
        """
        Initialize reproducibility validator

        Args:
            num_runs: Number of runs to perform for validation
            variance_threshold: Maximum acceptable variance for reproducibility
        """
        self.num_runs = num_runs
        self.variance_threshold = variance_threshold

    async def validate_algorithm(
        self,
        optimizer: BaseOptimizer,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
        seed: int = 42,
    ) -> ReproducibilityResult:
        """
        Validate reproducibility of algorithm

        Args:
            optimizer: Optimizer to validate
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data
            seed: Random seed for reproducibility

        Returns:
            Reproducibility result
        """
        improvements: list[float] = []
        all_results: list[dict[str, Any]] = []

        # Run optimization multiple times with same seed
        for run_idx in range(self.num_runs):
            # Set seed (implementation depends on optimizer)
            self._set_optimizer_seed(optimizer, seed)

            # Run optimization
            result = await optimizer.optimize(
                request=request,
                baseline_metrics=baseline_metrics,
                training_data=training_data,
            )

            # Record improvement
            improvement = result.improvement_percentage
            improvements.append(improvement)

            all_results.append({
                "run": run_idx + 1,
                "improvement": improvement,
                "success_rate": (
                    result.optimized_performance.success_rate
                    if result.optimized_performance
                    else 0.0
                ),
                "status": result.status.value,
            })

        # Calculate statistics
        mean_improvement = sum(improvements) / len(improvements)
        variance = sum((x - mean_improvement) ** 2 for x in improvements) / len(
            improvements
        )
        std_deviation = variance**0.5

        # Calculate coefficient of variation (CV)
        coefficient_of_variation = (
            std_deviation / mean_improvement if mean_improvement != 0 else float("inf")
        )

        # Check reproducibility
        is_reproducible = variance <= self.variance_threshold

        return ReproducibilityResult(
            algorithm_name=optimizer.get_algorithm_name(),
            num_runs=self.num_runs,
            seed=seed,
            is_reproducible=is_reproducible,
            variance=variance,
            mean_improvement=mean_improvement,
            std_deviation=std_deviation,
            coefficient_of_variation=coefficient_of_variation,
            all_results=all_results,
            metadata={
                "variance_threshold": self.variance_threshold,
                "min_improvement": min(improvements),
                "max_improvement": max(improvements),
            },
        )

    async def validate_multiple_algorithms(
        self,
        optimizers: list[BaseOptimizer],
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
        seed: int = 42,
    ) -> ReproducibilityAnalysis:
        """
        Validate reproducibility of multiple algorithms

        Args:
            optimizers: List of optimizers to validate
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data
            seed: Random seed

        Returns:
            Reproducibility analysis
        """
        results_by_algorithm: dict[str, ReproducibilityResult] = {}

        for optimizer in optimizers:
            result = await self.validate_algorithm(
                optimizer=optimizer,
                request=request,
                baseline_metrics=baseline_metrics,
                training_data=training_data,
                seed=seed,
            )
            results_by_algorithm[optimizer.get_algorithm_name()] = result

        # Calculate summary statistics
        total = len(results_by_algorithm)
        reproducible = sum(
            1 for r in results_by_algorithm.values() if r.is_reproducible
        )
        reproducibility_rate = reproducible / total if total > 0 else 0.0

        # Generate summary
        summary = self._generate_summary(
            results_by_algorithm,
            reproducible,
            total,
            reproducibility_rate,
        )

        return ReproducibilityAnalysis(
            total_algorithms=total,
            reproducible_algorithms=reproducible,
            reproducibility_rate=reproducibility_rate,
            results_by_algorithm=results_by_algorithm,
            summary=summary,
        )

    async def validate_cross_run_consistency(
        self,
        optimizer: BaseOptimizer,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
        num_different_seeds: int = 3,
    ) -> dict[str, Any]:
        """
        Validate consistency across different seeds

        Args:
            optimizer: Optimizer to validate
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data
            num_different_seeds: Number of different seeds to test

        Returns:
            Cross-run consistency analysis
        """
        seed_results: list[ReproducibilityResult] = []

        for seed_idx in range(num_different_seeds):
            seed = 42 + seed_idx * 100
            result = await self.validate_algorithm(
                optimizer=optimizer,
                request=request,
                baseline_metrics=baseline_metrics,
                training_data=training_data,
                seed=seed,
            )
            seed_results.append(result)

        # Analyze variance across different seeds
        mean_improvements = [r.mean_improvement for r in seed_results]
        overall_mean = sum(mean_improvements) / len(mean_improvements)
        overall_variance = sum((x - overall_mean) ** 2 for x in mean_improvements) / len(
            mean_improvements
        )

        return {
            "algorithm_name": optimizer.get_algorithm_name(),
            "num_seeds_tested": num_different_seeds,
            "seed_results": [
                {
                    "seed": r.seed,
                    "mean_improvement": r.mean_improvement,
                    "is_reproducible": r.is_reproducible,
                    "variance": r.variance,
                }
                for r in seed_results
            ],
            "cross_seed_mean": overall_mean,
            "cross_seed_variance": overall_variance,
            "consistent_reproducibility": all(r.is_reproducible for r in seed_results),
        }

    def _set_optimizer_seed(
        self,
        optimizer: BaseOptimizer,
        seed: int,
    ) -> None:
        """
        Set random seed for optimizer

        Args:
            optimizer: Optimizer to seed
            seed: Random seed
        """
        import random

        # Set Python random seed
        random.seed(seed)

        # Set numpy seed if available
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

        # Set optimizer-specific seed if supported
        if hasattr(optimizer, "seed"):
            optimizer.seed = seed

    def _generate_summary(
        self,
        results_by_algorithm: dict[str, ReproducibilityResult],
        reproducible: int,
        total: int,
        reproducibility_rate: float,
    ) -> str:
        """
        Generate reproducibility summary

        Args:
            results_by_algorithm: Results by algorithm
            reproducible: Number of reproducible algorithms
            total: Total algorithms
            reproducibility_rate: Reproducibility rate

        Returns:
            Summary string
        """
        summary_lines = [
            f"Reproducibility Validation: {reproducible}/{total} algorithms reproducible ({reproducibility_rate:.1%})",
            "",
            "Algorithm Details:",
        ]

        for algo_name, result in results_by_algorithm.items():
            status = "✓ Reproducible" if result.is_reproducible else "✗ Non-reproducible"
            summary_lines.append(
                f"  {algo_name}: {status} "
                f"(variance: {result.variance:.4f}, CV: {result.coefficient_of_variation:.4f})"
            )

        return "\n".join(summary_lines)
