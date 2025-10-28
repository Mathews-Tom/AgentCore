"""
Optimization pipeline orchestrator

Manages end-to-end optimization workflows including algorithm selection,
execution, validation, and result tracking.
"""

from __future__ import annotations

import asyncio
from typing import Any

import dspy

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.algorithms.gepa import GEPAOptimizer
from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)


class OptimizationPipeline:
    """
    Orchestrates optimization workflows

    Handles algorithm selection, execution, result comparison,
    and best optimization selection based on objectives.
    """

    def __init__(self, llm: dspy.LM | None = None) -> None:
        """
        Initialize optimization pipeline

        Args:
            llm: DSPy language model for optimization
        """
        self.llm = llm or dspy.LM("openai/gpt-4.1-mini")
        self.optimizers: dict[str, BaseOptimizer] = {
            "miprov2": MIPROv2Optimizer(llm=self.llm),
            "gepa": GEPAOptimizer(llm=self.llm),
        }

    async def run_optimization(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Run optimization pipeline

        Args:
            request: Optimization request
            baseline_metrics: Baseline performance metrics
            training_data: Training data for optimization

        Returns:
            Best optimization result
        """
        # Validate request
        self._validate_request(request)

        # Run optimizations with requested algorithms
        results = await self._run_algorithms(
            request, baseline_metrics, training_data
        )

        # Select best result
        best_result = self._select_best_result(results, request)

        return best_result

    async def _run_algorithms(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> list[OptimizationResult]:
        """
        Run multiple optimization algorithms

        Args:
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data

        Returns:
            List of optimization results
        """
        tasks = []

        for algorithm_name in request.algorithms:
            if algorithm_name in self.optimizers:
                optimizer = self.optimizers[algorithm_name]
                task = optimizer.optimize(request, baseline_metrics, training_data)
                tasks.append(task)

        # Run algorithms concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed results
        valid_results = [
            r for r in results
            if isinstance(r, OptimizationResult)
            and r.status == OptimizationStatus.COMPLETED
        ]

        return valid_results

    def _select_best_result(
        self,
        results: list[OptimizationResult],
        request: OptimizationRequest,
    ) -> OptimizationResult:
        """
        Select best optimization result based on objectives

        Args:
            results: List of optimization results
            request: Original optimization request

        Returns:
            Best optimization result
        """
        if not results:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                error_message="No successful optimizations",
            )

        # Score each result based on objectives
        scored_results = [
            (self._score_result(r, request), r) for r in results
        ]

        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)

        return scored_results[0][1]

    def _score_result(
        self, result: OptimizationResult, request: OptimizationRequest
    ) -> float:
        """
        Score optimization result based on objectives

        Args:
            result: Optimization result to score
            request: Optimization request with objectives

        Returns:
            Weighted score
        """
        if not result.optimized_performance:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for objective in request.objectives:
            metric_value = 0.0

            if objective.metric.value == "success_rate":
                metric_value = result.optimized_performance.success_rate
            elif objective.metric.value == "cost_efficiency":
                # Inverse - lower cost is better
                metric_value = 1.0 - min(
                    result.optimized_performance.avg_cost_per_task, 1.0
                )
            elif objective.metric.value == "latency":
                # Inverse - lower latency is better
                metric_value = 1.0 - min(
                    result.optimized_performance.avg_latency_ms / 10000, 1.0
                )
            elif objective.metric.value == "quality_score":
                metric_value = result.optimized_performance.quality_score

            # Calculate weighted contribution
            score_contribution = (metric_value / objective.target_value) * objective.weight
            total_score += score_contribution
            total_weight += objective.weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _validate_request(self, request: OptimizationRequest) -> None:
        """
        Validate optimization request

        Args:
            request: Request to validate

        Raises:
            ValueError: If request is invalid
        """
        if not request.objectives:
            raise ValueError("At least one optimization objective is required")

        if not request.algorithms:
            raise ValueError("At least one optimization algorithm must be specified")

        # Validate algorithm names
        invalid_algorithms = [
            alg for alg in request.algorithms if alg not in self.optimizers
        ]
        if invalid_algorithms:
            raise ValueError(
                f"Invalid algorithms: {invalid_algorithms}. "
                f"Available: {list(self.optimizers.keys())}"
            )

    def register_optimizer(self, name: str, optimizer: BaseOptimizer) -> None:
        """
        Register custom optimizer

        Args:
            name: Algorithm name
            optimizer: Optimizer instance
        """
        self.optimizers[name] = optimizer

    def get_available_algorithms(self) -> list[str]:
        """
        Get list of available optimization algorithms

        Returns:
            List of algorithm names
        """
        return list(self.optimizers.keys())
