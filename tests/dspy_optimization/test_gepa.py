"""
Tests for GEPA optimizer
"""

from __future__ import annotations

import pytest
import dspy

from agentcore.dspy_optimization.algorithms.gepa import GEPAOptimizer
from agentcore.dspy_optimization.models import (
    MetricType,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationScope,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)


@pytest.fixture
def optimizer() -> GEPAOptimizer:
    """Create GEPA optimizer instance"""
    mock_llm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key", cache_seed=42)
    return GEPAOptimizer(llm=mock_llm, max_iterations=3)


@pytest.fixture
def baseline_metrics() -> PerformanceMetrics:
    """Create baseline performance metrics"""
    return PerformanceMetrics(
        success_rate=0.70,
        avg_cost_per_task=0.18,
        avg_latency_ms=3500,
        quality_score=0.75,
    )


@pytest.fixture
def optimization_request() -> OptimizationRequest:
    """Create optimization request"""
    return OptimizationRequest(
        target=OptimizationTarget(
            type=OptimizationTargetType.WORKFLOW,
            id="test-workflow",
            scope=OptimizationScope.INDIVIDUAL,
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE,
                target_value=0.90,
                weight=0.5,
            ),
            OptimizationObjective(
                metric=MetricType.QUALITY_SCORE,
                target_value=0.85,
                weight=0.5,
            ),
        ],
        algorithms=["gepa"],
    )


@pytest.fixture
def training_data() -> list[dict[str, str]]:
    """Create training data"""
    return [
        {"question": "Analyze user behavior", "answer": "User shows high engagement"},
        {"question": "Optimize workflow", "answer": "Reduce latency by 20%"},
        {"question": "Improve quality", "answer": "Increase success rate to 95%"},
    ]


class TestGEPAOptimizer:
    """Tests for GEPAOptimizer class"""

    def test_initialization(self, optimizer: GEPAOptimizer) -> None:
        """Test optimizer initialization"""
        assert optimizer.max_iterations == 3
        assert optimizer.reflection_depth == 3
        assert optimizer.llm is not None
        assert optimizer.reflection_module is not None

    def test_algorithm_name(self, optimizer: GEPAOptimizer) -> None:
        """Test algorithm name"""
        assert optimizer.get_algorithm_name() == "gepa"

    async def test_optimize_success(
        self,
        optimizer: GEPAOptimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test successful optimization"""
        result = await optimizer.optimize(
            optimization_request, baseline_metrics, training_data
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.baseline_performance == baseline_metrics
        assert result.optimized_performance is not None
        assert result.improvement_percentage > 0
        assert result.optimization_details is not None
        assert result.optimization_details.algorithm_used == "gepa"

    async def test_optimize_better_than_baseline(
        self,
        optimizer: GEPAOptimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test that GEPA optimization improves over baseline"""
        result = await optimizer.optimize(
            optimization_request, baseline_metrics, training_data
        )

        if result.optimized_performance:
            # GEPA should achieve significant improvements
            assert result.optimized_performance.success_rate > baseline_metrics.success_rate
            assert result.optimized_performance.avg_cost_per_task < baseline_metrics.avg_cost_per_task
            assert result.optimized_performance.avg_latency_ms < baseline_metrics.avg_latency_ms
            assert result.optimized_performance.quality_score > baseline_metrics.quality_score

    def test_create_initial_prompt(
        self, optimizer: GEPAOptimizer, optimization_request: OptimizationRequest
    ) -> None:
        """Test initial prompt creation"""
        prompt = optimizer._create_initial_prompt(optimization_request)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "workflow" in prompt.lower()

    def test_summarize_performance(
        self,
        optimizer: GEPAOptimizer,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test performance summarization"""
        summary = optimizer._summarize_performance(baseline_metrics, training_data)

        assert isinstance(summary, str)
        assert "0.70" in summary or "70" in summary  # Success rate
        assert "3500" in summary  # Latency
        assert str(len(training_data)) in summary

    def test_apply_improvement(self, optimizer: GEPAOptimizer) -> None:
        """Test improvement application"""
        current_prompt = "Original prompt"
        suggestion = "Add more context"

        improved = optimizer._apply_improvement(current_prompt, suggestion)

        assert len(improved) > len(current_prompt)
        assert current_prompt in improved
        assert suggestion in improved

    async def test_evaluate_prompt(
        self,
        optimizer: GEPAOptimizer,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test prompt evaluation"""
        metrics = await optimizer._evaluate_prompt(
            "Test prompt", training_data, baseline_metrics
        )

        assert metrics.success_rate >= baseline_metrics.success_rate
        assert metrics.avg_cost_per_task <= baseline_metrics.avg_cost_per_task
        assert metrics.avg_latency_ms <= baseline_metrics.avg_latency_ms

    def test_check_objectives_met(
        self, optimizer: GEPAOptimizer, optimization_request: OptimizationRequest
    ) -> None:
        """Test objectives checking"""
        # Metrics that meet objectives
        good_metrics = PerformanceMetrics(
            success_rate=0.95,
            avg_cost_per_task=0.10,
            avg_latency_ms=2000,
            quality_score=0.90,
        )

        assert optimizer._check_objectives_met(optimization_request, good_metrics)

        # Metrics that don't meet objectives
        bad_metrics = PerformanceMetrics(
            success_rate=0.60,
            avg_cost_per_task=0.20,
            avg_latency_ms=4000,
            quality_score=0.50,
        )

        assert not optimizer._check_objectives_met(optimization_request, bad_metrics)

    def test_custom_parameters(self) -> None:
        """Test optimizer with custom parameters"""
        optimizer = GEPAOptimizer(
            max_iterations=10,
            reflection_depth=5,
        )

        assert optimizer.max_iterations == 10
        assert optimizer.reflection_depth == 5

    async def test_optimization_with_no_improvements(
        self, optimizer: GEPAOptimizer, baseline_metrics: PerformanceMetrics
    ) -> None:
        """Test optimization when no improvements are needed"""
        # Request with already-met objectives
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.50,  # Already exceeded
                    weight=1.0,
                ),
            ],
            algorithms=["gepa"],
        )

        result = await optimizer.optimize(request, baseline_metrics, [])

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimized_performance is not None

    def test_reflection_signature(self) -> None:
        """Test that reflection signature is properly defined"""
        from agentcore.dspy_optimization.algorithms.gepa import ReflectionSignature

        sig = ReflectionSignature
        assert hasattr(sig, "__annotations__")
        assert "current_prompt" in sig.__annotations__
        assert "performance_data" in sig.__annotations__
        assert "improvement_suggestion" in sig.__annotations__
        assert "reasoning" in sig.__annotations__
