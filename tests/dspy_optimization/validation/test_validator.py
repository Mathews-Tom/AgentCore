"""Tests for comprehensive algorithm validator"""

import pytest

from agentcore.dspy_optimization.algorithms.gepa import GEPAOptimizer
from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer
from agentcore.dspy_optimization.models import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationTarget,
    MetricType,
    PerformanceMetrics,
    OptimizationTargetType,
)
from agentcore.dspy_optimization.validation.validator import AlgorithmValidator


@pytest.fixture
def optimization_request() -> OptimizationRequest:
    """Create test optimization request"""
    return OptimizationRequest(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent",
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE,
                target_value=0.85,
                weight=1.0,
            )
        ],
        constraints=OptimizationConstraints(
            max_cost_per_task=0.10,
            max_latency_ms=1000,
            min_improvement_threshold=0.05,
        ),
    )


@pytest.fixture
def baseline_metrics() -> PerformanceMetrics:
    """Create baseline metrics"""
    return PerformanceMetrics(
        success_rate=0.70,
        avg_cost_per_task=0.05,
        avg_latency_ms=500,
        quality_score=0.75,
    )


@pytest.fixture
def training_data() -> list[dict]:
    """Create training data"""
    return [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]


@pytest.mark.asyncio
async def test_validate_single_algorithm(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test validating a single algorithm"""
    validator = AlgorithmValidator()
    optimizer = MIPROv2Optimizer(num_candidates=5)

    result = await validator.validate_algorithm(
        optimizer=optimizer,
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    assert result.algorithm_name == "miprov2"
    assert 0 <= result.overall_score <= 100
    assert len(result.benchmark_results) > 0
    assert result.baseline_comparison is not None
    assert result.statistical_significance is not None
    assert result.reproducibility is not None
    assert len(result.strengths) > 0 or len(result.weaknesses) > 0
    assert result.recommendation


@pytest.mark.asyncio
async def test_validate_multiple_algorithms(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test validating multiple algorithms"""
    validator = AlgorithmValidator()

    optimizers = [
        MIPROv2Optimizer(num_candidates=5),
        GEPAOptimizer(max_iterations=3),
    ]

    report = await validator.validate_multiple_algorithms(
        optimizers=optimizers,
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    assert len(report.validation_results) == 2
    assert len(report.algorithm_rankings) == 2
    assert report.summary
    assert report.timestamp
    assert "miprov2" in [r.algorithm_name for r in report.validation_results]
    assert "gepa" in [r.algorithm_name for r in report.validation_results]


@pytest.mark.asyncio
async def test_baseline_comparison(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test baseline comparison functionality"""
    validator = AlgorithmValidator()
    optimizer = GEPAOptimizer(max_iterations=3)

    comparison = await validator._compare_baselines(
        optimizer=optimizer,
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    assert comparison.algorithm_name == "gepa"
    assert comparison.algorithm_improvement >= 0
    assert comparison.random_search_improvement >= 0
    assert comparison.grid_search_improvement >= 0
    assert isinstance(comparison.beats_random_search, bool)
    assert isinstance(comparison.beats_grid_search, bool)


@pytest.mark.asyncio
async def test_statistical_significance(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test statistical significance testing"""
    validator = AlgorithmValidator()
    optimizer = MIPROv2Optimizer(num_candidates=5)

    significance = await validator._test_statistical_significance(
        optimizer=optimizer,
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    assert significance.p_value >= 0
    assert isinstance(significance.is_significant, bool)
    assert significance.confidence_level > 0
    assert len(significance.sample_sizes) > 0


def test_overall_score_calculation() -> None:
    """Test overall score calculation"""
    from agentcore.dspy_optimization.monitoring.statistics import (
        SignificanceResult,
        SignificanceTest,
    )
    from agentcore.dspy_optimization.validation.baselines import BaselineComparison
    from agentcore.dspy_optimization.validation.benchmarks import (
        BenchmarkResult,
        BenchmarkType,
    )
    from agentcore.dspy_optimization.validation.reproducibility import (
        ReproducibilityResult,
    )

    validator = AlgorithmValidator()

    benchmark_results = [
        BenchmarkResult(
            benchmark_name="Test",
            benchmark_type=BenchmarkType.MIPROV2_STANDARD,
            algorithm_name="test",
            success=True,
            baseline_performance=PerformanceMetrics(
                success_rate=0.7,
                avg_cost_per_task=0.05,
                avg_latency_ms=500,
                quality_score=0.75,
            ),
            final_performance=PerformanceMetrics(
                success_rate=0.85,
                avg_cost_per_task=0.045,
                avg_latency_ms=450,
                quality_score=0.85,
            ),
            improvement_percentage=15.0,
            rollouts_used=10,
            execution_time_seconds=5.0,
            meets_research_claims=True,
        )
    ]

    baseline_comparison = BaselineComparison(
        algorithm_name="test",
        algorithm_improvement=15.0,
        random_search_improvement=5.0,
        grid_search_improvement=10.0,
        beats_random_search=True,
        beats_grid_search=True,
        improvement_over_random=10.0,
        improvement_over_grid=5.0,
    )

    statistical_significance = SignificanceResult(
        test_type=SignificanceTest.WELCH_T_TEST,
        p_value=0.01,
        is_significant=True,
        confidence_level=0.95,
    )

    reproducibility = ReproducibilityResult(
        algorithm_name="test",
        num_runs=5,
        seed=42,
        is_reproducible=True,
        variance=0.005,
        mean_improvement=15.0,
        std_deviation=0.07,
        coefficient_of_variation=0.0047,
    )

    score = validator._calculate_overall_score(
        benchmark_results=benchmark_results,
        baseline_comparison=baseline_comparison,
        statistical_significance=statistical_significance,
        reproducibility=reproducibility,
    )

    # Should get high score: 25 (benchmark) + 25 (baseline) + 20 (stats) + 15 (repro) = 85
    assert score >= 80
    assert score <= 100


def test_validation_passing_criteria() -> None:
    """Test validation passing criteria"""
    from agentcore.dspy_optimization.monitoring.statistics import (
        SignificanceResult,
        SignificanceTest,
    )
    from agentcore.dspy_optimization.validation.baselines import BaselineComparison
    from agentcore.dspy_optimization.validation.benchmarks import (
        BenchmarkResult,
        BenchmarkType,
    )
    from agentcore.dspy_optimization.validation.reproducibility import (
        ReproducibilityResult,
    )

    validator = AlgorithmValidator()

    # Passing case
    passing_result = validator._check_validation_passing(
        overall_score=85.0,
        benchmark_results=[
            BenchmarkResult(
                benchmark_name="Test",
                benchmark_type=BenchmarkType.MIPROV2_STANDARD,
                algorithm_name="test",
                success=True,
                baseline_performance=PerformanceMetrics(
                    success_rate=0.7,
                    avg_cost_per_task=0.05,
                    avg_latency_ms=500,
                    quality_score=0.75,
                ),
                final_performance=PerformanceMetrics(
                    success_rate=0.85,
                    avg_cost_per_task=0.045,
                    avg_latency_ms=450,
                    quality_score=0.85,
                ),
                improvement_percentage=15.0,
                rollouts_used=10,
                execution_time_seconds=5.0,
                meets_research_claims=True,
            )
        ],
        baseline_comparison=BaselineComparison(
            algorithm_name="test",
            algorithm_improvement=15.0,
            random_search_improvement=5.0,
            grid_search_improvement=10.0,
            beats_random_search=True,
            beats_grid_search=True,
            improvement_over_random=10.0,
            improvement_over_grid=5.0,
        ),
        statistical_significance=SignificanceResult(
            test_type=SignificanceTest.WELCH_T_TEST,
            p_value=0.01,
            is_significant=True,
            confidence_level=0.95,
        ),
        reproducibility=ReproducibilityResult(
            algorithm_name="test",
            num_runs=5,
            seed=42,
            is_reproducible=True,
            variance=0.005,
            mean_improvement=15.0,
            std_deviation=0.07,
            coefficient_of_variation=0.0047,
        ),
    )

    assert passing_result

    # Failing case (low score)
    failing_result = validator._check_validation_passing(
        overall_score=40.0,
        benchmark_results=[],
        baseline_comparison=BaselineComparison(
            algorithm_name="test",
            algorithm_improvement=5.0,
            random_search_improvement=10.0,
            grid_search_improvement=15.0,
            beats_random_search=False,
            beats_grid_search=False,
            improvement_over_random=-5.0,
            improvement_over_grid=-10.0,
        ),
        statistical_significance=SignificanceResult(
            test_type=SignificanceTest.WELCH_T_TEST,
            p_value=0.5,
            is_significant=False,
            confidence_level=0.95,
        ),
        reproducibility=ReproducibilityResult(
            algorithm_name="test",
            num_runs=5,
            seed=42,
            is_reproducible=False,
            variance=0.05,
            mean_improvement=5.0,
            std_deviation=0.22,
            coefficient_of_variation=0.044,
        ),
    )

    assert not failing_result
