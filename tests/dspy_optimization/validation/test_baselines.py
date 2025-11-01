"""Tests for baseline optimizers"""

import pytest

from agentcore.dspy_optimization.models import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationTarget,
    MetricType,
    PerformanceMetrics,
    OptimizationTargetType,
)
from agentcore.dspy_optimization.validation.baselines import (
    BaselineComparison,
    GridSearchOptimizer,
    RandomSearchOptimizer,
)


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
    return [
        {"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)
    ]


@pytest.mark.asyncio
async def test_random_search_optimizer(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test random search optimizer"""
    optimizer = RandomSearchOptimizer(num_trials=10, seed=42)

    result = await optimizer.optimize(
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    assert result.status.value == "completed"
    assert result.optimization_details is not None
    assert result.optimization_details.algorithm_used == "random_search"
    assert result.optimization_details.iterations == 10


@pytest.mark.asyncio
async def test_random_search_reproducibility(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test random search produces similar results across runs"""
    results = []

    for _ in range(3):
        optimizer = RandomSearchOptimizer(num_trials=10, seed=42)
        result = await optimizer.optimize(
            request=optimization_request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )
        results.append(result.improvement_percentage)

    # Results should be within reasonable range (not deterministic due to evaluation randomness)
    mean = sum(results) / len(results)
    variance = sum((x - mean) ** 2 for x in results) / len(results)

    # Variance should be low for reproducibility
    assert variance < 10.0  # Allow some variation


@pytest.mark.asyncio
async def test_grid_search_optimizer(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test grid search optimizer"""
    optimizer = GridSearchOptimizer(grid_size=3)

    result = await optimizer.optimize(
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    assert result.status.value == "completed"
    assert result.optimization_details is not None
    assert result.optimization_details.algorithm_used == "grid_search"
    # Grid size 3 with 3 parameters should evaluate 3^3 = 27 configs
    assert result.optimization_details.iterations == 27


@pytest.mark.asyncio
async def test_grid_search_deterministic(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test grid search is deterministic"""
    optimizer1 = GridSearchOptimizer(grid_size=3)
    optimizer2 = GridSearchOptimizer(grid_size=3)

    result1 = await optimizer1.optimize(
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    result2 = await optimizer2.optimize(
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
    )

    # Grid search should be deterministic
    assert result1.improvement_percentage == result2.improvement_percentage


def test_baseline_comparison_beats_both() -> None:
    """Test baseline comparison property"""
    comparison = BaselineComparison(
        algorithm_name="test",
        algorithm_improvement=20.0,
        random_search_improvement=5.0,
        grid_search_improvement=10.0,
        beats_random_search=True,
        beats_grid_search=True,
        improvement_over_random=15.0,
        improvement_over_grid=10.0,
    )

    assert comparison.beats_both_baselines


def test_baseline_comparison_not_beats_both() -> None:
    """Test baseline comparison when not beating both"""
    comparison = BaselineComparison(
        algorithm_name="test",
        algorithm_improvement=8.0,
        random_search_improvement=5.0,
        grid_search_improvement=10.0,
        beats_random_search=True,
        beats_grid_search=False,
        improvement_over_random=3.0,
        improvement_over_grid=-2.0,
    )

    assert not comparison.beats_both_baselines
