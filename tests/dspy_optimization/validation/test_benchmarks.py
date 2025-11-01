"""Tests for benchmark suite"""

import pytest

from agentcore.dspy_optimization.algorithms.gepa import GEPAOptimizer
from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer
from agentcore.dspy_optimization.evolutionary.genetic import GeneticOptimizer
from agentcore.dspy_optimization.models import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationTarget,
    MetricType,
    PerformanceMetrics,
    OptimizationTargetType,
)
from agentcore.dspy_optimization.validation.benchmarks import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkSuite,
    BenchmarkType,
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


@pytest.mark.asyncio
async def test_benchmark_miprov2(optimization_request: OptimizationRequest) -> None:
    """Test MIPROv2 benchmark"""
    benchmark = Benchmark(
        BenchmarkConfig(
            name="MIPROv2 Test",
            benchmark_type=BenchmarkType.MIPROV2_STANDARD,
            training_samples=50,
            seed=42,
        )
    )

    optimizer = MIPROv2Optimizer(num_candidates=5)
    result = await benchmark.run(optimizer, optimization_request)

    assert result.success
    assert result.algorithm_name == "miprov2"
    assert result.improvement_percentage > 0
    assert result.rollouts_used > 0
    assert result.execution_time_seconds > 0


@pytest.mark.asyncio
async def test_benchmark_gepa(optimization_request: OptimizationRequest) -> None:
    """Test GEPA benchmark"""
    benchmark = Benchmark(
        BenchmarkConfig(
            name="GEPA Test",
            benchmark_type=BenchmarkType.GEPA_EFFICIENCY,
            training_samples=50,
            seed=42,
        )
    )

    optimizer = GEPAOptimizer(max_iterations=3)
    result = await benchmark.run(optimizer, optimization_request)

    assert result.success
    assert result.algorithm_name == "gepa"
    assert result.improvement_percentage > 0
    # GEPA should use fewer rollouts than MIPROv2
    assert result.rollouts_used <= 10


@pytest.mark.asyncio
async def test_benchmark_genetic(optimization_request: OptimizationRequest) -> None:
    """Test genetic algorithm benchmark"""
    benchmark = Benchmark(
        BenchmarkConfig(
            name="Genetic Test",
            benchmark_type=BenchmarkType.GENETIC_CONVERGENCE,
            training_samples=50,
            seed=42,
        )
    )

    optimizer = GeneticOptimizer()
    result = await benchmark.run(optimizer, optimization_request)

    assert result.success
    assert result.algorithm_name == "genetic"
    assert result.improvement_percentage > 0


def test_benchmark_research_claims() -> None:
    """Test validation against research claims"""
    benchmark = Benchmark(
        BenchmarkConfig(
            name="Claims Test",
            benchmark_type=BenchmarkType.GEPA_EFFICIENCY,
            training_samples=50,
            seed=42,
        )
    )

    # Test GEPA claims validation - with sufficient rollouts
    meets_claims, details = benchmark._validate_research_claims(
        optimizer_name="gepa",
        improvement=15.0,  # 15% improvement
        rollouts=2,  # Very few rollouts (100 / 2 = 50x efficiency)
    )

    assert meets_claims, f"GEPA claims not met: {details}"
    assert details["claim_met"]
    assert details["efficiency_ratio"] >= 35  # 35x efficiency claim (100 / 2 = 50x)


def test_benchmark_suite_standard_benchmarks() -> None:
    """Test adding standard benchmarks"""
    suite = BenchmarkSuite()
    suite.add_standard_benchmarks()

    assert len(suite.benchmarks) == 3
    assert any(
        b.config.benchmark_type == BenchmarkType.MIPROV2_STANDARD
        for b in suite.benchmarks
    )
    assert any(
        b.config.benchmark_type == BenchmarkType.GEPA_EFFICIENCY for b in suite.benchmarks
    )
    assert any(
        b.config.benchmark_type == BenchmarkType.GENETIC_CONVERGENCE
        for b in suite.benchmarks
    )


@pytest.mark.asyncio
async def test_benchmark_suite_run_all(optimization_request: OptimizationRequest) -> None:
    """Test running all benchmarks"""
    suite = BenchmarkSuite()
    suite.add_standard_benchmarks()

    optimizers = [
        MIPROv2Optimizer(num_candidates=5),
        GEPAOptimizer(max_iterations=3),
    ]

    results = await suite.run_all(optimizers, optimization_request)

    # Should have results for each optimizer x benchmark combination
    assert len(results) == len(optimizers) * len(suite.benchmarks)
    assert all(r.success for r in results)


def test_benchmark_suite_summary() -> None:
    """Test benchmark suite summary generation"""
    from agentcore.dspy_optimization.validation.benchmarks import BenchmarkResult

    results = [
        BenchmarkResult(
            benchmark_name="Test 1",
            benchmark_type=BenchmarkType.MIPROV2_STANDARD,
            algorithm_name="miprov2",
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
        ),
        BenchmarkResult(
            benchmark_name="Test 2",
            benchmark_type=BenchmarkType.GEPA_EFFICIENCY,
            algorithm_name="gepa",
            success=True,
            baseline_performance=PerformanceMetrics(
                success_rate=0.7,
                avg_cost_per_task=0.05,
                avg_latency_ms=500,
                quality_score=0.75,
            ),
            final_performance=PerformanceMetrics(
                success_rate=0.88,
                avg_cost_per_task=0.042,
                avg_latency_ms=420,
                quality_score=0.88,
            ),
            improvement_percentage=20.0,
            rollouts_used=3,
            execution_time_seconds=3.0,
            meets_research_claims=True,
        ),
    ]

    suite = BenchmarkSuite()
    summary = suite.get_summary(results)

    assert summary["total_benchmarks"] == 2
    assert summary["successful_benchmarks"] == 2
    assert summary["meets_research_claims"] == 2
    assert summary["success_rate"] == 1.0
    assert summary["claims_met_rate"] == 1.0
    assert "miprov2" in summary["by_algorithm"]
    assert "gepa" in summary["by_algorithm"]
    assert summary["by_algorithm"]["gepa"]["avg_improvement"] == 20.0
