"""
Tests for ROI calculation
"""

import pytest

from agentcore.dspy_optimization.analytics.roi import (
    CostModel,
    ROICalculator,
)
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)


@pytest.fixture
def calculator() -> ROICalculator:
    """Create ROI calculator"""
    return ROICalculator()


@pytest.fixture
def optimization_result() -> OptimizationResult:
    """Create optimization result with 25% improvement"""
    return OptimizationResult(
        optimization_id="test-opt-001",
        status=OptimizationStatus.COMPLETED,
        baseline_performance=PerformanceMetrics(
            success_rate=0.70,
            avg_cost_per_task=0.50,
            avg_latency_ms=500,
            quality_score=0.75,
        ),
        optimized_performance=PerformanceMetrics(
            success_rate=0.875,  # 25% improvement
            avg_cost_per_task=0.375,  # 25% reduction
            avg_latency_ms=375,  # 25% reduction
            quality_score=0.9375,  # 25% improvement
        ),
        improvement_percentage=0.25,
        statistical_significance=0.95,
        optimization_details=OptimizationDetails(
            algorithm_used="miprov2",
            iterations=10,
            key_improvements=["prompt optimization"],
            parameters={
                "target_type": "agent",
                "target_id": "test-agent",
            },
        ),
    )


@pytest.fixture
def optimization_costs() -> dict[str, float]:
    """Create optimization costs"""
    return {
        "compute_hours": 5.0,
        "tokens": 1_000_000,
        "infrastructure": 10.0,
        "engineer_hours": 2.0,
    }


@pytest.mark.asyncio
async def test_calculate_roi_positive(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test ROI calculation with positive return"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,  # 1000 requests per day
        forecast_period_days=365,
    )

    assert report.is_profitable is True
    assert report.metrics.roi_percentage > 0
    assert report.metrics.payback_period_days < 365
    assert report.break_even_date is not None


@pytest.mark.asyncio
async def test_calculate_roi_cost_savings(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test cost savings calculation"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
    )

    # 25% cost reduction should generate savings
    assert report.metrics.cost_savings_per_period > 0

    # Optimized cost should be lower than baseline
    assert report.optimized_costs.total_cost < report.baseline_costs.total_cost


@pytest.mark.asyncio
async def test_calculate_roi_performance_value(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test performance gain valuation"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
        business_value_factors={
            "success_rate": 0.20,  # Higher value per success rate improvement
            "quality": 0.10,
        },
    )

    # Performance improvements should generate value
    assert report.metrics.performance_gain_value >= 0


@pytest.mark.asyncio
async def test_payback_period_calculation(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test payback period calculation"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
    )

    # Payback period should be reasonable for 25% improvement
    assert report.metrics.payback_period_days > 0
    assert report.metrics.payback_period_days < 365


@pytest.mark.asyncio
async def test_npv_calculation(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test NPV calculation"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
        forecast_period_days=365,
    )

    # NPV should be positive for profitable investment
    if report.is_profitable:
        assert report.metrics.net_present_value > 0


@pytest.mark.asyncio
async def test_cost_breakdown(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test cost breakdown calculation"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
    )

    # Optimization costs breakdown
    assert report.optimization_costs.compute_cost > 0
    assert report.optimization_costs.token_cost > 0
    assert report.optimization_costs.human_time_cost > 0
    assert report.optimization_costs.total_cost > 0

    # Total should equal sum of components
    total = (
        report.optimization_costs.compute_cost
        + report.optimization_costs.token_cost
        + report.optimization_costs.infrastructure_cost
        + report.optimization_costs.human_time_cost
    )
    assert report.optimization_costs.total_cost == pytest.approx(total, rel=0.01)


@pytest.mark.asyncio
async def test_recommendations_profitable(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test recommendations for profitable optimization"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
    )

    if report.is_profitable:
        assert len(report.recommendations) > 0
        assert any("deploy" in rec.lower() or "roi" in rec.lower() for rec in report.recommendations)


@pytest.mark.asyncio
async def test_recommendations_unprofitable(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
) -> None:
    """Test recommendations for unprofitable optimization"""
    # Very high optimization costs
    expensive_costs = {
        "compute_hours": 500.0,
        "tokens": 100_000_000,
        "infrastructure": 1000.0,
        "engineer_hours": 100.0,
    }

    report = await calculator.calculate_roi(
        optimization_result,
        expensive_costs,
        baseline_volume=10,  # Low volume
    )

    if not report.is_profitable:
        assert any("negative" in rec.lower() or "reconsider" in rec.lower() for rec in report.recommendations)


@pytest.mark.asyncio
async def test_compare_roi(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test ROI comparison across optimizations"""
    report1 = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
    )

    report2 = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=2000,  # Higher volume = better ROI
    )

    comparison = await calculator.compare_roi([report1, report2])

    assert comparison["total_reports"] == 2
    assert "avg_roi_percentage" in comparison
    assert "best_roi" in comparison
    assert "worst_roi" in comparison


@pytest.mark.asyncio
async def test_different_cost_models(
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test different cost models"""
    for model in [CostModel.TOKEN_BASED, CostModel.TIME_BASED, CostModel.HYBRID]:
        calculator = ROICalculator(cost_model=model)

        report = await calculator.calculate_roi(
            optimization_result,
            optimization_costs,
            baseline_volume=1000,
        )

        assert report.assumptions["cost_model"] == model.value


@pytest.mark.asyncio
async def test_performance_improvements_tracking(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test performance improvements are tracked"""
    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
    )

    assert "success_rate" in report.performance_improvements
    assert "cost_reduction" in report.performance_improvements
    assert "latency_reduction" in report.performance_improvements
    assert "quality_improvement" in report.performance_improvements

    # Check improvements are calculated correctly
    assert report.performance_improvements["success_rate"] > 0
    assert report.performance_improvements["cost_reduction"] > 0


@pytest.mark.asyncio
async def test_custom_discount_rate(
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test custom discount rate"""
    calculator = ROICalculator(discount_rate=0.15)

    report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=1000,
    )

    assert report.assumptions["discount_rate"] == 0.15


@pytest.mark.asyncio
async def test_volume_impact_on_roi(
    calculator: ROICalculator,
    optimization_result: OptimizationResult,
    optimization_costs: dict[str, float],
) -> None:
    """Test impact of volume on ROI"""
    low_volume_report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=100,
    )

    high_volume_report = await calculator.calculate_roi(
        optimization_result,
        optimization_costs,
        baseline_volume=10000,
    )

    # Higher volume should result in better ROI
    assert high_volume_report.metrics.roi_percentage > low_volume_report.metrics.roi_percentage
    assert high_volume_report.metrics.payback_period_days < low_volume_report.metrics.payback_period_days
