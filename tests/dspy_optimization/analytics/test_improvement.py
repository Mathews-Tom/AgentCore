"""
Tests for improvement validation and analysis
"""

import pytest

from agentcore.dspy_optimization.analytics.improvement import (
    ImprovementAnalyzer,
    ImprovementStatus,
    ImprovementValidationConfig,
)
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationResult,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)


@pytest.fixture
def analyzer() -> ImprovementAnalyzer:
    """Create improvement analyzer"""
    return ImprovementAnalyzer()


@pytest.fixture
def target() -> OptimizationTarget:
    """Create optimization target"""
    return OptimizationTarget(type=OptimizationTargetType.AGENT, id="test-agent")


@pytest.fixture
def baseline_metrics() -> PerformanceMetrics:
    """Create baseline metrics"""
    return PerformanceMetrics(
        success_rate=0.70,
        avg_cost_per_task=0.50,
        avg_latency_ms=500,
        quality_score=0.75,
    )


@pytest.fixture
def optimized_metrics() -> PerformanceMetrics:
    """Create optimized metrics with 25% improvement"""
    return PerformanceMetrics(
        success_rate=0.875,  # 25% improvement
        avg_cost_per_task=0.375,  # 25% reduction
        avg_latency_ms=375,  # 25% reduction
        quality_score=0.9375,  # 25% improvement
    )


@pytest.fixture
def baseline_samples() -> list[dict[str, float]]:
    """Create baseline samples"""
    return [
        {
            "success_rate": 0.70 + i * 0.01,
            "avg_cost_per_task": 0.50,
            "avg_latency_ms": 500,
            "quality_score": 0.75,
        }
        for i in range(30)
    ]


@pytest.fixture
def optimized_samples() -> list[dict[str, float]]:
    """Create optimized samples"""
    return [
        {
            "success_rate": 0.875 + i * 0.01,
            "avg_cost_per_task": 0.375,
            "avg_latency_ms": 375,
            "quality_score": 0.9375,
        }
        for i in range(30)
    ]


@pytest.fixture
def optimization_result(
    target: OptimizationTarget,
    baseline_metrics: PerformanceMetrics,
    optimized_metrics: PerformanceMetrics,
) -> OptimizationResult:
    """Create optimization result"""
    return OptimizationResult(
        optimization_id="test-opt-001",
        status=OptimizationStatus.COMPLETED,
        baseline_performance=baseline_metrics,
        optimized_performance=optimized_metrics,
        improvement_percentage=0.25,
        statistical_significance=0.95,
        optimization_details=OptimizationDetails(
            algorithm_used="miprov2",
            iterations=10,
            key_improvements=["prompt optimization", "better examples"],
            parameters={"target": target},
        ),
    )


@pytest.mark.asyncio
async def test_validate_improvement_target_met(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test improvement validation when target is met"""
    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        optimized_samples,
    )

    assert validation.status == ImprovementStatus.TARGET_MET
    assert validation.meets_target is True
    assert validation.improvement_metrics.overall_improvement >= 0.20
    assert validation.improvement_metrics.overall_improvement <= 0.30
    assert validation.is_statistically_significant is True


@pytest.mark.asyncio
async def test_validate_improvement_excellent(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test improvement validation with excellent results"""
    # Modify for 35% improvement across all metrics
    optimization_result.optimized_performance.success_rate = 0.945  # 35% improvement
    optimization_result.optimized_performance.avg_cost_per_task = 0.325  # 35% reduction
    optimization_result.optimized_performance.avg_latency_ms = 325  # 35% reduction
    optimization_result.optimized_performance.quality_score = 1.0  # 33% improvement (capped at 1.0)
    optimization_result.improvement_percentage = 0.35

    # Update samples to match
    excellent_samples = [
        {
            "success_rate": 0.945 + i * 0.001,
            "avg_cost_per_task": 0.325,
            "avg_latency_ms": 325,
            "quality_score": 1.0,
        }
        for i in range(30)
    ]

    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        excellent_samples,
    )

    assert validation.status == ImprovementStatus.EXCELLENT
    assert validation.exceeds_target is True


@pytest.mark.asyncio
async def test_validate_improvement_insufficient(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test improvement validation with insufficient results"""
    # Modify for 3% improvement across all metrics
    optimization_result.optimized_performance.success_rate = 0.721  # 3% improvement
    optimization_result.optimized_performance.avg_cost_per_task = 0.485  # 3% reduction
    optimization_result.optimized_performance.avg_latency_ms = 485  # 3% reduction
    optimization_result.optimized_performance.quality_score = 0.7725  # 3% improvement
    optimization_result.improvement_percentage = 0.03

    # Update samples to match
    insufficient_samples = [
        {
            "success_rate": 0.721 + i * 0.001,
            "avg_cost_per_task": 0.485,
            "avg_latency_ms": 485,
            "quality_score": 0.7725,
        }
        for i in range(30)
    ]

    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        insufficient_samples,
    )

    assert validation.status == ImprovementStatus.INSUFFICIENT
    assert validation.meets_target is False


@pytest.mark.asyncio
async def test_validate_improvement_degradation(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test improvement validation with performance degradation"""
    # Modify for negative improvement across all metrics
    optimization_result.optimized_performance.success_rate = 0.65  # -7% degradation
    optimization_result.optimized_performance.avg_cost_per_task = 0.54  # 8% increase (worse)
    optimization_result.optimized_performance.avg_latency_ms = 540  # 8% increase (worse)
    optimization_result.optimized_performance.quality_score = 0.70  # -7% degradation
    optimization_result.improvement_percentage = -0.07

    # Update samples to match
    degraded_samples = [
        {
            "success_rate": 0.65 + i * 0.001,
            "avg_cost_per_task": 0.54,
            "avg_latency_ms": 540,
            "quality_score": 0.70,
        }
        for i in range(30)
    ]

    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        degraded_samples,
    )

    assert validation.status == ImprovementStatus.DEGRADATION
    # Check that degradation is mentioned in recommendations
    recommendations_text = " ".join(validation.recommendations).lower()
    assert "degraded" in recommendations_text or "degradation" in recommendations_text


@pytest.mark.asyncio
async def test_validate_improvement_insufficient_samples(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
) -> None:
    """Test validation fails with insufficient samples"""
    small_samples = [{"success_rate": 0.7}] * 10  # Only 10 samples

    with pytest.raises(ValueError, match="Insufficient baseline samples"):
        await analyzer.validate_improvement(
            optimization_result,
            small_samples,
            small_samples,
        )


@pytest.mark.asyncio
async def test_validate_improvement_with_weights(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test improvement validation with custom weights"""
    weights = {
        "success_rate": 0.6,  # Higher weight on success rate
        "cost": 0.2,
        "latency": 0.1,
        "quality": 0.1,
    }

    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        optimized_samples,
        weights=weights,
    )

    assert validation.improvement_metrics.weighted_improvement >= 0.0
    assert validation.status in [
        ImprovementStatus.EXCELLENT,
        ImprovementStatus.TARGET_MET,
        ImprovementStatus.ACCEPTABLE,
    ]


@pytest.mark.asyncio
async def test_validate_multiple_results(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test validation of multiple results"""
    results = [
        (optimization_result, baseline_samples, optimized_samples),
        (optimization_result, baseline_samples, optimized_samples),
    ]

    validations = await analyzer.validate_multiple_results(results)

    assert len(validations) == 2
    assert all(v.is_statistically_significant for v in validations)


@pytest.mark.asyncio
async def test_get_improvement_summary(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test improvement summary generation"""
    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        optimized_samples,
    )

    summary = await analyzer.get_improvement_summary([validation])

    assert summary["total_validations"] == 1
    assert summary["avg_improvement"] >= 0.20
    assert summary["target_met_count"] >= 0
    assert "status_distribution" in summary


@pytest.mark.asyncio
async def test_recommendations_insufficient_improvement(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test recommendations for insufficient improvement"""
    # Modify for 5% improvement across all metrics (marginal)
    optimization_result.optimized_performance.success_rate = 0.735  # 5% improvement
    optimization_result.optimized_performance.avg_cost_per_task = 0.475  # 5% reduction
    optimization_result.optimized_performance.avg_latency_ms = 475  # 5% reduction
    optimization_result.optimized_performance.quality_score = 0.7875  # 5% improvement
    optimization_result.improvement_percentage = 0.05

    # Update samples to match
    marginal_samples = [
        {
            "success_rate": 0.735 + i * 0.001,
            "avg_cost_per_task": 0.475,
            "avg_latency_ms": 475,
            "quality_score": 0.7875,
        }
        for i in range(30)
    ]

    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        marginal_samples,
    )

    assert len(validation.recommendations) > 0
    # Should recommend trying different algorithm or parameters
    assert any(
        "algorithm" in rec.lower() or "parameter" in rec.lower() or "optimization" in rec.lower()
        for rec in validation.recommendations
    )


@pytest.mark.asyncio
async def test_recommendations_target_met(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test recommendations when target is met"""
    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        optimized_samples,
    )

    assert validation.meets_target is True
    assert any("deploy" in rec.lower() for rec in validation.recommendations)


def test_custom_config() -> None:
    """Test custom configuration"""
    config = ImprovementValidationConfig(
        target_min=0.15,
        target_max=0.25,
        acceptable_min=0.08,
        require_statistical_significance=False,
    )

    analyzer = ImprovementAnalyzer(config=config)

    assert analyzer.config.target_min == 0.15
    assert analyzer.config.target_max == 0.25
    assert analyzer.config.require_statistical_significance is False


@pytest.mark.asyncio
async def test_improvement_metrics_calculation(
    analyzer: ImprovementAnalyzer,
    baseline_metrics: PerformanceMetrics,
    optimized_metrics: PerformanceMetrics,
) -> None:
    """Test detailed improvement metrics calculation"""
    metrics = analyzer._calculate_improvement_metrics(
        baseline_metrics,
        optimized_metrics,
    )

    assert metrics.success_rate_improvement == pytest.approx(0.25, rel=0.01)
    assert metrics.cost_reduction_percentage == pytest.approx(25.0, rel=0.01)
    assert metrics.latency_reduction_percentage == pytest.approx(25.0, rel=0.01)
    assert metrics.overall_improvement >= 0.20


@pytest.mark.asyncio
async def test_sample_size_tracking(
    analyzer: ImprovementAnalyzer,
    optimization_result: OptimizationResult,
    baseline_samples: list[dict[str, float]],
    optimized_samples: list[dict[str, float]],
) -> None:
    """Test sample size tracking in validation"""
    validation = await analyzer.validate_improvement(
        optimization_result,
        baseline_samples,
        optimized_samples,
    )

    assert validation.sample_sizes["baseline"] == 30
    assert validation.sample_sizes["optimized"] == 30
