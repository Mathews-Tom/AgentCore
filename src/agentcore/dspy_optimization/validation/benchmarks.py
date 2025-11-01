"""
Benchmark suite for algorithm validation

Provides standardized benchmarks for validating optimization algorithms
against research paper claims (MIPROv2, GEPA, genetic algorithms).
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    PerformanceMetrics,
)


class BenchmarkType(str, Enum):
    """Types of benchmarks"""

    MIPROV2_STANDARD = "miprov2_standard"
    GEPA_EFFICIENCY = "gepa_efficiency"
    GENETIC_CONVERGENCE = "genetic_convergence"
    MULTI_OBJECTIVE = "multi_objective"
    SCALABILITY = "scalability"


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark"""

    name: str
    benchmark_type: BenchmarkType
    training_samples: int = Field(default=100)
    test_samples: int = Field(default=50)
    max_rollouts: int = Field(default=100)
    timeout_seconds: int = Field(default=300)
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    """Result of benchmark execution"""

    benchmark_name: str
    benchmark_type: BenchmarkType
    algorithm_name: str
    success: bool
    baseline_performance: PerformanceMetrics
    final_performance: PerformanceMetrics
    improvement_percentage: float
    rollouts_used: int
    execution_time_seconds: float
    meets_research_claims: bool
    research_claim_details: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Benchmark:
    """
    Single benchmark for algorithm validation

    Executes a standardized test to validate algorithm performance
    against research paper claims.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """
        Initialize benchmark

        Args:
            config: Benchmark configuration
        """
        self.config = config

    async def run(
        self,
        optimizer: BaseOptimizer,
        request: OptimizationRequest,
    ) -> BenchmarkResult:
        """
        Run benchmark

        Args:
            optimizer: Optimizer to test
            request: Optimization request

        Returns:
            Benchmark result
        """
        start_time = time.time()

        # Generate synthetic training data
        training_data = self._generate_training_data(
            self.config.training_samples,
            self.config.seed,
        )

        # Create baseline metrics
        baseline_metrics = PerformanceMetrics(
            success_rate=0.70,
            avg_cost_per_task=0.05,
            avg_latency_ms=500,
            quality_score=0.75,
        )

        try:
            # Run optimization
            result = await optimizer.optimize(
                request=request,
                baseline_metrics=baseline_metrics,
                training_data=training_data,
            )

            execution_time = time.time() - start_time

            # Extract final performance
            final_performance = (
                result.optimized_performance or baseline_metrics
            )

            # Calculate improvement
            improvement = (
                (final_performance.success_rate - baseline_metrics.success_rate)
                / baseline_metrics.success_rate
                * 100
            )

            # Check against research claims
            meets_claims, claim_details = self._validate_research_claims(
                optimizer_name=optimizer.get_algorithm_name(),
                improvement=improvement,
                rollouts=result.optimization_details.iterations if result.optimization_details else 0,
            )

            return BenchmarkResult(
                benchmark_name=self.config.name,
                benchmark_type=self.config.benchmark_type,
                algorithm_name=optimizer.get_algorithm_name(),
                success=True,
                baseline_performance=baseline_metrics,
                final_performance=final_performance,
                improvement_percentage=improvement,
                rollouts_used=result.optimization_details.iterations if result.optimization_details else 0,
                execution_time_seconds=execution_time,
                meets_research_claims=meets_claims,
                research_claim_details=claim_details,
                metadata={
                    "training_samples": self.config.training_samples,
                    "seed": self.config.seed,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return BenchmarkResult(
                benchmark_name=self.config.name,
                benchmark_type=self.config.benchmark_type,
                algorithm_name=optimizer.get_algorithm_name(),
                success=False,
                baseline_performance=baseline_metrics,
                final_performance=baseline_metrics,
                improvement_percentage=0.0,
                rollouts_used=0,
                execution_time_seconds=execution_time,
                meets_research_claims=False,
                research_claim_details={"error": str(e)},
            )

    def _generate_training_data(
        self,
        num_samples: int,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate synthetic training data

        Args:
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Training data samples
        """
        import random

        if seed is not None:
            random.seed(seed)

        data = []
        for i in range(num_samples):
            data.append({
                "question": f"Sample question {i}",
                "answer": f"Sample answer {i}",
                "success_rate": random.uniform(0.6, 0.9),
                "cost": random.uniform(0.02, 0.08),
                "latency": random.randint(300, 700),
            })

        return data

    def _validate_research_claims(
        self,
        optimizer_name: str,
        improvement: float,
        rollouts: int,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate against research paper claims

        Args:
            optimizer_name: Name of optimizer
            improvement: Improvement percentage
            rollouts: Number of rollouts used

        Returns:
            Tuple of (meets_claims, claim_details)
        """
        claim_details: dict[str, Any] = {}

        if optimizer_name == "miprov2":
            # MIPROv2 claims: systematic instruction improvement
            expected_min_improvement = 10.0  # 10% improvement
            meets_claims = improvement >= expected_min_improvement

            claim_details = {
                "expected_min_improvement": expected_min_improvement,
                "actual_improvement": improvement,
                "claim_met": meets_claims,
                "research_paper": "MIPROv2: Multiprompt Instruction Proposal Optimizer v2",
            }

        elif optimizer_name == "gepa":
            # GEPA claims: 10%+ gains with 35x fewer rollouts than MIPROv2
            expected_min_improvement = 10.0
            expected_max_rollouts = 100 // 35  # ~3 rollouts
            meets_claims = (
                improvement >= expected_min_improvement
                and rollouts <= expected_max_rollouts
            )

            claim_details = {
                "expected_min_improvement": expected_min_improvement,
                "actual_improvement": improvement,
                "expected_max_rollouts": expected_max_rollouts,
                "actual_rollouts": rollouts,
                "efficiency_ratio": 100 / max(rollouts, 1),  # Compared to 100 rollouts baseline
                "claim_met": meets_claims,
                "research_paper": "GEPA: Generalized Enhancement through Prompt Adaptation",
            }

        elif optimizer_name == "genetic":
            # Genetic algorithm claims: convergence within reasonable iterations
            expected_max_generations = 50
            expected_min_improvement = 5.0  # 5% improvement
            meets_claims = (
                improvement >= expected_min_improvement
                and rollouts <= expected_max_generations
            )

            claim_details = {
                "expected_min_improvement": expected_min_improvement,
                "actual_improvement": improvement,
                "expected_max_generations": expected_max_generations,
                "actual_generations": rollouts,
                "claim_met": meets_claims,
                "research_basis": "Evolutionary computation for prompt optimization",
            }

        else:
            claim_details = {
                "error": f"No research claims defined for {optimizer_name}",
                "claim_met": False,
            }
            meets_claims = False

        return meets_claims, claim_details


class BenchmarkSuite:
    """
    Suite of benchmarks for comprehensive validation

    Runs multiple standardized benchmarks to validate algorithm
    performance against research claims.
    """

    def __init__(self) -> None:
        """Initialize benchmark suite"""
        self.benchmarks: list[Benchmark] = []

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """
        Add benchmark to suite

        Args:
            benchmark: Benchmark to add
        """
        self.benchmarks.append(benchmark)

    def add_standard_benchmarks(self) -> None:
        """Add standard benchmarks for all algorithms"""
        # MIPROv2 standard benchmark
        self.add_benchmark(
            Benchmark(
                BenchmarkConfig(
                    name="MIPROv2 Standard Test",
                    benchmark_type=BenchmarkType.MIPROV2_STANDARD,
                    training_samples=100,
                    test_samples=50,
                    max_rollouts=100,
                    seed=42,
                )
            )
        )

        # GEPA efficiency benchmark
        self.add_benchmark(
            Benchmark(
                BenchmarkConfig(
                    name="GEPA Efficiency Test",
                    benchmark_type=BenchmarkType.GEPA_EFFICIENCY,
                    training_samples=100,
                    test_samples=50,
                    max_rollouts=10,  # Should achieve results with fewer rollouts
                    seed=42,
                )
            )
        )

        # Genetic convergence benchmark
        self.add_benchmark(
            Benchmark(
                BenchmarkConfig(
                    name="Genetic Convergence Test",
                    benchmark_type=BenchmarkType.GENETIC_CONVERGENCE,
                    training_samples=100,
                    test_samples=50,
                    max_rollouts=50,
                    seed=42,
                )
            )
        )

    async def run_all(
        self,
        optimizers: list[BaseOptimizer],
        request: OptimizationRequest,
    ) -> list[BenchmarkResult]:
        """
        Run all benchmarks for all optimizers

        Args:
            optimizers: List of optimizers to test
            request: Optimization request

        Returns:
            List of benchmark results
        """
        results: list[BenchmarkResult] = []

        for optimizer in optimizers:
            for benchmark in self.benchmarks:
                result = await benchmark.run(optimizer, request)
                results.append(result)

        return results

    def get_summary(self, results: list[BenchmarkResult]) -> dict[str, Any]:
        """
        Get summary of benchmark results

        Args:
            results: List of benchmark results

        Returns:
            Summary dictionary
        """
        total = len(results)
        successful = sum(1 for r in results if r.success)
        meets_claims = sum(1 for r in results if r.meets_research_claims)

        by_algorithm: dict[str, dict[str, Any]] = {}
        for result in results:
            if result.algorithm_name not in by_algorithm:
                by_algorithm[result.algorithm_name] = {
                    "total_benchmarks": 0,
                    "successful": 0,
                    "meets_claims": 0,
                    "avg_improvement": 0.0,
                    "avg_rollouts": 0.0,
                    "avg_execution_time": 0.0,
                }

            stats = by_algorithm[result.algorithm_name]
            stats["total_benchmarks"] += 1
            if result.success:
                stats["successful"] += 1
            if result.meets_research_claims:
                stats["meets_claims"] += 1
            stats["avg_improvement"] += result.improvement_percentage
            stats["avg_rollouts"] += result.rollouts_used
            stats["avg_execution_time"] += result.execution_time_seconds

        # Calculate averages
        for stats in by_algorithm.values():
            total_benchmarks = stats["total_benchmarks"]
            if total_benchmarks > 0:
                stats["avg_improvement"] /= total_benchmarks
                stats["avg_rollouts"] /= total_benchmarks
                stats["avg_execution_time"] /= total_benchmarks

        return {
            "total_benchmarks": total,
            "successful_benchmarks": successful,
            "meets_research_claims": meets_claims,
            "success_rate": successful / total if total > 0 else 0.0,
            "claims_met_rate": meets_claims / total if total > 0 else 0.0,
            "by_algorithm": by_algorithm,
        }
