"""Tests for reproducibility validation"""

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
from agentcore.dspy_optimization.validation.reproducibility import (
    ReproducibilityValidator,
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
    return [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]


@pytest.mark.asyncio
async def test_validate_algorithm_reproducibility(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test algorithm reproducibility validation"""
    validator = ReproducibilityValidator(num_runs=3, variance_threshold=0.01)
    optimizer = MIPROv2Optimizer(num_candidates=5)

    result = await validator.validate_algorithm(
        optimizer=optimizer,
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
        seed=42,
    )

    assert result.algorithm_name == "miprov2"
    assert result.num_runs == 3
    assert result.seed == 42
    assert len(result.all_results) == 3
    assert result.variance >= 0
    assert result.mean_improvement > 0
    assert result.std_deviation >= 0


@pytest.mark.asyncio
async def test_validate_multiple_algorithms(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test validating multiple algorithms"""
    validator = ReproducibilityValidator(num_runs=3, variance_threshold=0.01)

    optimizers = [
        MIPROv2Optimizer(num_candidates=5),
        GEPAOptimizer(max_iterations=3),
    ]

    analysis = await validator.validate_multiple_algorithms(
        optimizers=optimizers,
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
        seed=42,
    )

    assert analysis.total_algorithms == 2
    assert len(analysis.results_by_algorithm) == 2
    assert "miprov2" in analysis.results_by_algorithm
    assert "gepa" in analysis.results_by_algorithm
    assert 0.0 <= analysis.reproducibility_rate <= 1.0
    assert analysis.summary


@pytest.mark.asyncio
async def test_validate_cross_run_consistency(
    optimization_request: OptimizationRequest,
    baseline_metrics: PerformanceMetrics,
    training_data: list[dict],
) -> None:
    """Test cross-run consistency validation"""
    validator = ReproducibilityValidator(num_runs=3)
    optimizer = GEPAOptimizer(max_iterations=3)

    result = await validator.validate_cross_run_consistency(
        optimizer=optimizer,
        request=optimization_request,
        baseline_metrics=baseline_metrics,
        training_data=training_data,
        num_different_seeds=2,
    )

    assert result["algorithm_name"] == "gepa"
    assert result["num_seeds_tested"] == 2
    assert len(result["seed_results"]) == 2
    assert "cross_seed_mean" in result
    assert "cross_seed_variance" in result
    assert "consistent_reproducibility" in result


def test_reproducibility_result_interpretation() -> None:
    """Test reproducibility result interpretation"""
    from agentcore.dspy_optimization.validation.reproducibility import (
        ReproducibilityResult,
    )

    # Low variance = reproducible
    result_reproducible = ReproducibilityResult(
        algorithm_name="test",
        num_runs=5,
        seed=42,
        is_reproducible=True,
        variance=0.005,
        mean_improvement=15.0,
        std_deviation=0.07,
        coefficient_of_variation=0.0047,
    )

    assert result_reproducible.is_reproducible
    assert result_reproducible.variance < 0.01

    # High variance = not reproducible
    result_not_reproducible = ReproducibilityResult(
        algorithm_name="test",
        num_runs=5,
        seed=42,
        is_reproducible=False,
        variance=0.05,
        mean_improvement=15.0,
        std_deviation=0.22,
        coefficient_of_variation=0.0147,
    )

    assert not result_not_reproducible.is_reproducible
    assert result_not_reproducible.variance > 0.01
