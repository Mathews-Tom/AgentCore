"""
Performance comparison tools for optimizer plugins
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentcore.dspy_optimization.models import (
    OptimizationResult,
    PerformanceMetrics,
)


@dataclass
class MetricComparison:
    """Comparison of a single metric between algorithms"""

    metric_name: str
    baseline_value: float
    algorithm_values: dict[str, float]
    best_algorithm: str
    best_value: float
    improvement_from_baseline: float


@dataclass
class AlgorithmComparison:
    """Comparison between multiple algorithms"""

    baseline_metrics: PerformanceMetrics
    algorithm_results: dict[str, OptimizationResult]
    metric_comparisons: list[MetricComparison]
    overall_winner: str
    summary: str


class PerformanceComparator:
    """
    Compare performance across multiple optimizer algorithms

    Provides detailed analysis of relative performance, statistical
    significance, and recommendations.
    """

    def __init__(self) -> None:
        """Initialize performance comparator"""
        pass

    def compare_results(
        self,
        baseline_metrics: PerformanceMetrics,
        results: dict[str, OptimizationResult],
    ) -> AlgorithmComparison:
        """
        Compare optimization results across algorithms

        Args:
            baseline_metrics: Baseline performance metrics
            results: Dict of algorithm name to optimization result

        Returns:
            AlgorithmComparison with detailed comparison
        """
        # Extract metrics for comparison
        metric_comparisons = self._compare_metrics(baseline_metrics, results)

        # Determine overall winner
        overall_winner = self._determine_winner(metric_comparisons)

        # Generate summary
        summary = self._generate_summary(
            baseline_metrics, results, metric_comparisons, overall_winner
        )

        return AlgorithmComparison(
            baseline_metrics=baseline_metrics,
            algorithm_results=results,
            metric_comparisons=metric_comparisons,
            overall_winner=overall_winner,
            summary=summary,
        )

    def _compare_metrics(
        self,
        baseline_metrics: PerformanceMetrics,
        results: dict[str, OptimizationResult],
    ) -> list[MetricComparison]:
        """Compare individual metrics across algorithms"""
        comparisons: list[MetricComparison] = []

        # Success rate comparison
        success_values = {}
        for alg_name, result in results.items():
            if result.optimized_performance:
                success_values[alg_name] = result.optimized_performance.success_rate

        if success_values:
            best_alg = max(success_values.items(), key=lambda x: x[1])
            comparisons.append(
                MetricComparison(
                    metric_name="success_rate",
                    baseline_value=baseline_metrics.success_rate,
                    algorithm_values=success_values,
                    best_algorithm=best_alg[0],
                    best_value=best_alg[1],
                    improvement_from_baseline=(
                        (best_alg[1] - baseline_metrics.success_rate)
                        / baseline_metrics.success_rate
                        * 100
                        if baseline_metrics.success_rate > 0
                        else 0.0
                    ),
                )
            )

        # Cost efficiency comparison (lower is better)
        cost_values = {}
        for alg_name, result in results.items():
            if result.optimized_performance:
                cost_values[alg_name] = result.optimized_performance.avg_cost_per_task

        if cost_values:
            best_alg = min(cost_values.items(), key=lambda x: x[1])
            comparisons.append(
                MetricComparison(
                    metric_name="avg_cost_per_task",
                    baseline_value=baseline_metrics.avg_cost_per_task,
                    algorithm_values=cost_values,
                    best_algorithm=best_alg[0],
                    best_value=best_alg[1],
                    improvement_from_baseline=(
                        (baseline_metrics.avg_cost_per_task - best_alg[1])
                        / baseline_metrics.avg_cost_per_task
                        * 100
                        if baseline_metrics.avg_cost_per_task > 0
                        else 0.0
                    ),
                )
            )

        # Latency comparison (lower is better)
        latency_values = {}
        for alg_name, result in results.items():
            if result.optimized_performance:
                latency_values[alg_name] = float(
                    result.optimized_performance.avg_latency_ms
                )

        if latency_values:
            best_alg = min(latency_values.items(), key=lambda x: x[1])
            comparisons.append(
                MetricComparison(
                    metric_name="avg_latency_ms",
                    baseline_value=float(baseline_metrics.avg_latency_ms),
                    algorithm_values=latency_values,
                    best_algorithm=best_alg[0],
                    best_value=best_alg[1],
                    improvement_from_baseline=(
                        (float(baseline_metrics.avg_latency_ms) - best_alg[1])
                        / float(baseline_metrics.avg_latency_ms)
                        * 100
                        if baseline_metrics.avg_latency_ms > 0
                        else 0.0
                    ),
                )
            )

        # Quality score comparison
        quality_values = {}
        for alg_name, result in results.items():
            if result.optimized_performance:
                quality_values[alg_name] = result.optimized_performance.quality_score

        if quality_values:
            best_alg = max(quality_values.items(), key=lambda x: x[1])
            comparisons.append(
                MetricComparison(
                    metric_name="quality_score",
                    baseline_value=baseline_metrics.quality_score,
                    algorithm_values=quality_values,
                    best_algorithm=best_alg[0],
                    best_value=best_alg[1],
                    improvement_from_baseline=(
                        (best_alg[1] - baseline_metrics.quality_score)
                        / baseline_metrics.quality_score
                        * 100
                        if baseline_metrics.quality_score > 0
                        else 0.0
                    ),
                )
            )

        return comparisons

    def _determine_winner(
        self, metric_comparisons: list[MetricComparison]
    ) -> str:
        """Determine overall winner based on metric comparisons"""
        if not metric_comparisons:
            return "none"

        # Count wins per algorithm
        wins: dict[str, int] = {}
        for comparison in metric_comparisons:
            alg = comparison.best_algorithm
            wins[alg] = wins.get(alg, 0) + 1

        # Return algorithm with most wins
        return max(wins.items(), key=lambda x: x[1])[0]

    def _generate_summary(
        self,
        baseline_metrics: PerformanceMetrics,
        results: dict[str, OptimizationResult],
        metric_comparisons: list[MetricComparison],
        overall_winner: str,
    ) -> str:
        """Generate human-readable summary"""
        lines = [
            f"Performance Comparison across {len(results)} algorithms:",
            "",
            f"Overall Winner: {overall_winner}",
            "",
            "Metric Comparisons:",
        ]

        for comp in metric_comparisons:
            lines.append(
                f"  - {comp.metric_name}: {comp.best_algorithm} "
                f"({comp.best_value:.4f}, {comp.improvement_from_baseline:+.2f}% vs baseline)"
            )

        return "\n".join(lines)

    def rank_algorithms(
        self,
        results: dict[str, OptimizationResult],
        weights: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rank algorithms by weighted performance score

        Args:
            results: Dict of algorithm name to optimization result
            weights: Optional metric weights (default: equal weights)

        Returns:
            List of (algorithm_name, score) tuples, sorted by score descending
        """
        if not weights:
            weights = {
                "success_rate": 0.25,
                "cost_efficiency": 0.25,
                "latency": 0.25,
                "quality_score": 0.25,
            }

        scores: dict[str, float] = {}

        for alg_name, result in results.items():
            if not result.optimized_performance:
                scores[alg_name] = 0.0
                continue

            metrics = result.optimized_performance
            score = 0.0

            # Success rate (higher is better)
            score += weights.get("success_rate", 0.0) * metrics.success_rate

            # Cost efficiency (lower is better, normalize to 0-1)
            if metrics.avg_cost_per_task > 0:
                cost_score = 1.0 / (1.0 + metrics.avg_cost_per_task)
                score += weights.get("cost_efficiency", 0.0) * cost_score

            # Latency (lower is better, normalize to 0-1)
            if metrics.avg_latency_ms > 0:
                latency_score = 1.0 / (1.0 + metrics.avg_latency_ms / 1000.0)
                score += weights.get("latency", 0.0) * latency_score

            # Quality score (higher is better)
            score += weights.get("quality_score", 0.0) * metrics.quality_score

            scores[alg_name] = score

        # Sort by score descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def generate_comparison_report(
        self,
        comparison: AlgorithmComparison,
        format: str = "text",
    ) -> str:
        """
        Generate formatted comparison report

        Args:
            comparison: Algorithm comparison results
            format: Output format ("text" or "markdown")

        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._generate_markdown_report(comparison)
        else:
            return self._generate_text_report(comparison)

    def _generate_text_report(
        self, comparison: AlgorithmComparison
    ) -> str:
        """Generate plain text report"""
        lines = [
            "=" * 80,
            "OPTIMIZER PERFORMANCE COMPARISON",
            "=" * 80,
            "",
            "BASELINE METRICS",
            "-" * 80,
            f"Success Rate:      {comparison.baseline_metrics.success_rate:.2%}",
            f"Avg Cost/Task:     ${comparison.baseline_metrics.avg_cost_per_task:.4f}",
            f"Avg Latency:       {comparison.baseline_metrics.avg_latency_ms}ms",
            f"Quality Score:     {comparison.baseline_metrics.quality_score:.2f}",
            "",
            "ALGORITHM RESULTS",
            "-" * 80,
        ]

        for alg_name, result in comparison.algorithm_results.items():
            if result.optimized_performance:
                metrics = result.optimized_performance
                lines.extend(
                    [
                        f"\n{alg_name}:",
                        f"  Success Rate:    {metrics.success_rate:.2%}",
                        f"  Avg Cost/Task:   ${metrics.avg_cost_per_task:.4f}",
                        f"  Avg Latency:     {metrics.avg_latency_ms}ms",
                        f"  Quality Score:   {metrics.quality_score:.2f}",
                        f"  Improvement:     {result.improvement_percentage:+.2f}%",
                    ]
                )

        lines.extend(
            [
                "",
                "OVERALL WINNER",
                "-" * 80,
                f"{comparison.overall_winner}",
                "",
                "=" * 80,
            ]
        )

        return "\n".join(lines)

    def _generate_markdown_report(
        self, comparison: AlgorithmComparison
    ) -> str:
        """Generate markdown report"""
        lines = [
            "# Optimizer Performance Comparison",
            "",
            "## Baseline Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Success Rate | {comparison.baseline_metrics.success_rate:.2%} |",
            f"| Avg Cost/Task | ${comparison.baseline_metrics.avg_cost_per_task:.4f} |",
            f"| Avg Latency | {comparison.baseline_metrics.avg_latency_ms}ms |",
            f"| Quality Score | {comparison.baseline_metrics.quality_score:.2f} |",
            "",
            "## Algorithm Results",
            "",
            "| Algorithm | Success Rate | Avg Cost | Avg Latency | Quality | Improvement |",
            "|-----------|--------------|----------|-------------|---------|-------------|",
        ]

        for alg_name, result in comparison.algorithm_results.items():
            if result.optimized_performance:
                metrics = result.optimized_performance
                lines.append(
                    f"| {alg_name} | {metrics.success_rate:.2%} | "
                    f"${metrics.avg_cost_per_task:.4f} | "
                    f"{metrics.avg_latency_ms}ms | "
                    f"{metrics.quality_score:.2f} | "
                    f"{result.improvement_percentage:+.2f}% |"
                )

        lines.extend(
            [
                "",
                f"## Overall Winner: **{comparison.overall_winner}**",
                "",
            ]
        )

        return "\n".join(lines)
