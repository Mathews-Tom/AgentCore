"""
Tests for performance comparison tools
"""

from __future__ import annotations

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.plugins.comparison import (
    AlgorithmComparison,
    MetricComparison,
    PerformanceComparator,
)


@pytest.fixture
def baseline_metrics() -> PerformanceMetrics:
    """Create baseline metrics"""
    return PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.85,
    )


@pytest.fixture
def mipro_result() -> OptimizationResult:
    """Create MIPROv2 optimization result"""
    return OptimizationResult(
        status=OptimizationStatus.COMPLETED,
        baseline_performance=PerformanceMetrics(
            success_rate=0.80,
            avg_cost_per_task=0.10,
            avg_latency_ms=2000,
            quality_score=0.85,
        ),
        optimized_performance=PerformanceMetrics(
            success_rate=0.90,
            avg_cost_per_task=0.08,
            avg_latency_ms=1800,
            quality_score=0.92,
        ),
        improvement_percentage=12.5,
        optimization_details=OptimizationDetails(
            algorithm_used="miprov2",
            iterations=10,
        ),
    )


@pytest.fixture
def gepa_result() -> OptimizationResult:
    """Create GEPA optimization result"""
    return OptimizationResult(
        status=OptimizationStatus.COMPLETED,
        baseline_performance=PerformanceMetrics(
            success_rate=0.80,
            avg_cost_per_task=0.10,
            avg_latency_ms=2000,
            quality_score=0.85,
        ),
        optimized_performance=PerformanceMetrics(
            success_rate=0.88,
            avg_cost_per_task=0.09,
            avg_latency_ms=1900,
            quality_score=0.90,
        ),
        improvement_percentage=10.0,
        optimization_details=OptimizationDetails(
            algorithm_used="gepa",
            iterations=8,
        ),
    )


@pytest.fixture
def comparator() -> PerformanceComparator:
    """Create performance comparator"""
    return PerformanceComparator()


class TestPerformanceComparator:
    """Tests for PerformanceComparator"""

    def test_compare_results(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test comparing optimization results"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}

        comparison = comparator.compare_results(baseline_metrics, results)

        assert isinstance(comparison, AlgorithmComparison)
        assert comparison.baseline_metrics == baseline_metrics
        assert len(comparison.algorithm_results) == 2
        assert len(comparison.metric_comparisons) == 4  # 4 metrics
        assert comparison.overall_winner in ["miprov2", "gepa"]
        assert len(comparison.summary) > 0

    def test_compare_metrics(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test comparing individual metrics"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}

        metric_comparisons = comparator._compare_metrics(
            baseline_metrics, results
        )

        assert len(metric_comparisons) == 4

        # Check success rate comparison
        success_comp = next(
            c for c in metric_comparisons if c.metric_name == "success_rate"
        )
        assert success_comp.baseline_value == 0.80
        assert success_comp.best_algorithm == "miprov2"
        assert success_comp.best_value == 0.90
        assert success_comp.improvement_from_baseline > 0

        # Check cost comparison (lower is better)
        cost_comp = next(
            c for c in metric_comparisons if c.metric_name == "avg_cost_per_task"
        )
        assert cost_comp.baseline_value == 0.10
        assert cost_comp.best_algorithm == "miprov2"
        assert cost_comp.best_value == 0.08

    def test_determine_winner(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test determining overall winner"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}
        metric_comparisons = comparator._compare_metrics(
            baseline_metrics, results
        )

        winner = comparator._determine_winner(metric_comparisons)

        assert winner == "miprov2"  # MIPROv2 should win more metrics

    def test_determine_winner_empty(
        self, comparator: PerformanceComparator
    ) -> None:
        """Test determining winner with no comparisons"""
        winner = comparator._determine_winner([])
        assert winner == "none"

    def test_rank_algorithms(
        self,
        comparator: PerformanceComparator,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test ranking algorithms by weighted score"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}

        rankings = comparator.rank_algorithms(results)

        assert len(rankings) == 2
        assert rankings[0][0] == "miprov2"  # Best algorithm first
        assert rankings[1][0] == "gepa"
        assert rankings[0][1] > rankings[1][1]  # Higher score first

    def test_rank_algorithms_with_weights(
        self,
        comparator: PerformanceComparator,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test ranking with custom weights"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}

        # Weight success rate heavily
        weights = {
            "success_rate": 0.70,
            "cost_efficiency": 0.10,
            "latency": 0.10,
            "quality_score": 0.10,
        }

        rankings = comparator.rank_algorithms(results, weights=weights)

        assert len(rankings) == 2
        assert rankings[0][0] == "miprov2"  # Should still win

    def test_generate_text_report(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test generating text report"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}
        comparison = comparator.compare_results(baseline_metrics, results)

        report = comparator.generate_comparison_report(comparison, format="text")

        assert "OPTIMIZER PERFORMANCE COMPARISON" in report
        assert "BASELINE METRICS" in report
        assert "ALGORITHM RESULTS" in report
        assert "miprov2" in report
        assert "gepa" in report
        assert "OVERALL WINNER" in report

    def test_generate_markdown_report(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test generating markdown report"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}
        comparison = comparator.compare_results(baseline_metrics, results)

        report = comparator.generate_comparison_report(
            comparison, format="markdown"
        )

        assert "# Optimizer Performance Comparison" in report
        assert "## Baseline Metrics" in report
        assert "## Algorithm Results" in report
        assert "| Algorithm |" in report
        assert "miprov2" in report
        assert "gepa" in report

    def test_comparison_with_incomplete_results(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test comparison with incomplete results"""
        incomplete_result = OptimizationResult(
            status=OptimizationStatus.FAILED,
            baseline_performance=baseline_metrics,
            optimized_performance=None,
        )

        results = {"incomplete": incomplete_result}
        comparison = comparator.compare_results(baseline_metrics, results)

        assert len(comparison.metric_comparisons) == 0
        assert comparison.overall_winner == "none"

    def test_metric_comparison_structure(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
        mipro_result: OptimizationResult,
    ) -> None:
        """Test metric comparison structure"""
        results = {"miprov2": mipro_result}
        metric_comparisons = comparator._compare_metrics(
            baseline_metrics, results
        )

        for comp in metric_comparisons:
            assert isinstance(comp, MetricComparison)
            assert isinstance(comp.metric_name, str)
            assert isinstance(comp.baseline_value, float)
            assert isinstance(comp.algorithm_values, dict)
            assert isinstance(comp.best_algorithm, str)
            assert isinstance(comp.best_value, float)
            assert isinstance(comp.improvement_from_baseline, float)

    def test_generate_summary(
        self,
        comparator: PerformanceComparator,
        baseline_metrics: PerformanceMetrics,
        mipro_result: OptimizationResult,
        gepa_result: OptimizationResult,
    ) -> None:
        """Test summary generation"""
        results = {"miprov2": mipro_result, "gepa": gepa_result}
        metric_comparisons = comparator._compare_metrics(
            baseline_metrics, results
        )
        winner = comparator._determine_winner(metric_comparisons)

        summary = comparator._generate_summary(
            baseline_metrics, results, metric_comparisons, winner
        )

        assert "Performance Comparison across 2 algorithms" in summary
        assert "Overall Winner:" in summary
        assert "Metric Comparisons:" in summary
        assert "miprov2" in summary.lower() or "gepa" in summary.lower()
