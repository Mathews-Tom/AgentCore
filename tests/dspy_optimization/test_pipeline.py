"""
Tests for optimization pipeline
"""

from __future__ import annotations

import pytest
import dspy

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.models import (
    MetricType,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    OptimizationScope,
    OptimizationStatus,
    OptimizationTarget,
    OptimizationTargetType,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.pipeline import OptimizationPipeline

# Skip tests requiring MLflow server
pytestmark = pytest.mark.skip(
    reason="MLflow server not available - run manually with MLflow server running"
)


@pytest.fixture
def pipeline() -> OptimizationPipeline:
    """Create optimization pipeline instance"""
    mock_llm = dspy.LM("openai/gpt-5-mini", api_key="test-key", cache_seed=42)
    return OptimizationPipeline(llm=mock_llm)


@pytest.fixture
def baseline_metrics() -> PerformanceMetrics:
    """Create baseline performance metrics"""
    return PerformanceMetrics(
        success_rate=0.75,
        avg_cost_per_task=0.15,
        avg_latency_ms=3000,
        quality_score=0.8,
    )


@pytest.fixture
def optimization_request() -> OptimizationRequest:
    """Create optimization request"""
    return OptimizationRequest(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="test-agent",
            scope=OptimizationScope.INDIVIDUAL,
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE,
                target_value=0.85,
                weight=0.6,
            ),
            OptimizationObjective(
                metric=MetricType.COST_EFFICIENCY,
                target_value=0.9,
                weight=0.4,
            ),
        ],
        algorithms=["miprov2", "gepa"],
    )


@pytest.fixture
def training_data() -> list[dict[str, str]]:
    """Create training data"""
    return [
        {"question": "What is AI?", "answer": "Artificial Intelligence"},
        {"question": "What is ML?", "answer": "Machine Learning"},
    ]


class TestOptimizationPipeline:
    """Tests for OptimizationPipeline class"""

    def test_initialization(self, pipeline: OptimizationPipeline) -> None:
        """Test pipeline initialization"""
        assert pipeline.llm is not None
        assert "miprov2" in pipeline.optimizers
        assert "gepa" in pipeline.optimizers

    def test_get_available_algorithms(self, pipeline: OptimizationPipeline) -> None:
        """Test getting available algorithms"""
        algorithms = pipeline.get_available_algorithms()

        assert "miprov2" in algorithms
        assert "gepa" in algorithms
        assert len(algorithms) == 2

    async def test_run_optimization_success(
        self,
        pipeline: OptimizationPipeline,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test successful optimization pipeline execution"""
        result = await pipeline.run_optimization(
            optimization_request, baseline_metrics, training_data
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.baseline_performance == baseline_metrics
        assert result.optimized_performance is not None
        assert result.improvement_percentage > 0

    async def test_run_optimization_single_algorithm(
        self,
        pipeline: OptimizationPipeline,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test optimization with single algorithm"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.85,
                ),
            ],
            algorithms=["miprov2"],
        )

        result = await pipeline.run_optimization(request, baseline_metrics, training_data)

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimization_details is not None
        assert result.optimization_details.algorithm_used == "miprov2"

    def test_validate_request_no_objectives(self, pipeline: OptimizationPipeline) -> None:
        """Test validation fails with no objectives"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test",
            ),
            objectives=[],
            algorithms=["miprov2"],
        )

        with pytest.raises(ValueError, match="At least one optimization objective"):
            pipeline._validate_request(request)

    def test_validate_request_no_algorithms(self, pipeline: OptimizationPipeline) -> None:
        """Test validation fails with no algorithms"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.85,
                ),
            ],
            algorithms=[],
        )

        with pytest.raises(ValueError, match="At least one optimization algorithm"):
            pipeline._validate_request(request)

    def test_validate_request_invalid_algorithm(
        self, pipeline: OptimizationPipeline
    ) -> None:
        """Test validation fails with invalid algorithm"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.85,
                ),
            ],
            algorithms=["invalid_algorithm"],
        )

        with pytest.raises(ValueError, match="Invalid algorithms"):
            pipeline._validate_request(request)

    def test_score_result(
        self, pipeline: OptimizationPipeline, optimization_request: OptimizationRequest
    ) -> None:
        """Test result scoring"""
        result = OptimizationResult(
            status=OptimizationStatus.COMPLETED,
            optimized_performance=PerformanceMetrics(
                success_rate=0.90,
                avg_cost_per_task=0.10,
                avg_latency_ms=2000,
                quality_score=0.95,
            ),
        )

        score = pipeline._score_result(result, optimization_request)

        assert score > 0
        assert isinstance(score, float)

    def test_score_result_no_performance(
        self, pipeline: OptimizationPipeline, optimization_request: OptimizationRequest
    ) -> None:
        """Test scoring with no performance data"""
        result = OptimizationResult(
            status=OptimizationStatus.FAILED,
            optimized_performance=None,
        )

        score = pipeline._score_result(result, optimization_request)

        assert score == 0.0

    def test_select_best_result(
        self, pipeline: OptimizationPipeline, optimization_request: OptimizationRequest
    ) -> None:
        """Test best result selection"""
        results = [
            OptimizationResult(
                status=OptimizationStatus.COMPLETED,
                optimized_performance=PerformanceMetrics(
                    success_rate=0.85,
                    avg_cost_per_task=0.12,
                    avg_latency_ms=2500,
                ),
                improvement_percentage=15.0,
            ),
            OptimizationResult(
                status=OptimizationStatus.COMPLETED,
                optimized_performance=PerformanceMetrics(
                    success_rate=0.92,
                    avg_cost_per_task=0.10,
                    avg_latency_ms=2200,
                ),
                improvement_percentage=25.0,
            ),
        ]

        best = pipeline._select_best_result(results, optimization_request)

        assert best == results[1]  # Second result is better
        assert best.improvement_percentage == 25.0

    def test_select_best_result_empty(
        self, pipeline: OptimizationPipeline, optimization_request: OptimizationRequest
    ) -> None:
        """Test selection with no results"""
        best = pipeline._select_best_result([], optimization_request)

        assert best.status == OptimizationStatus.FAILED
        assert "No successful optimizations" in best.error_message

    def test_register_custom_optimizer(self, pipeline: OptimizationPipeline) -> None:
        """Test registering custom optimizer"""

        class CustomOptimizer(BaseOptimizer):
            async def optimize(self, request, baseline_metrics, training_data):
                return OptimizationResult(status=OptimizationStatus.COMPLETED)

            def get_algorithm_name(self) -> str:
                return "custom"

        custom = CustomOptimizer()
        pipeline.register_optimizer("custom", custom)

        assert "custom" in pipeline.optimizers
        assert pipeline.optimizers["custom"] == custom

    async def test_run_optimization_with_custom_algorithm(
        self,
        pipeline: OptimizationPipeline,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test optimization with custom registered algorithm"""

        class CustomOptimizer(BaseOptimizer):
            async def optimize(self, request, baseline_metrics, training_data):
                return OptimizationResult(
                    status=OptimizationStatus.COMPLETED,
                    baseline_performance=baseline_metrics,
                    optimized_performance=PerformanceMetrics(
                        success_rate=0.95,
                        avg_cost_per_task=0.08,
                        avg_latency_ms=1800,
                    ),
                    improvement_percentage=30.0,
                )

            def get_algorithm_name(self) -> str:
                return "custom"

        pipeline.register_optimizer("custom", CustomOptimizer())

        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.85,
                ),
            ],
            algorithms=["custom"],
        )

        result = await pipeline.run_optimization(request, baseline_metrics, training_data)

        assert result.status == OptimizationStatus.COMPLETED
        assert result.improvement_percentage == 30.0

    async def test_run_algorithms_concurrent_execution(
        self,
        pipeline: OptimizationPipeline,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test that algorithms run concurrently"""
        results = await pipeline._run_algorithms(
            optimization_request, baseline_metrics, training_data
        )

        # Both algorithms should complete
        assert len(results) == 2
        assert all(r.status == OptimizationStatus.COMPLETED for r in results)

    def test_score_result_multiple_metrics(
        self, pipeline: OptimizationPipeline
    ) -> None:
        """Test scoring with multiple objectives"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.85,
                    weight=0.3,
                ),
                OptimizationObjective(
                    metric=MetricType.COST_EFFICIENCY,
                    target_value=0.90,
                    weight=0.3,
                ),
                OptimizationObjective(
                    metric=MetricType.LATENCY,
                    target_value=0.85,
                    weight=0.2,
                ),
                OptimizationObjective(
                    metric=MetricType.QUALITY_SCORE,
                    target_value=0.90,
                    weight=0.2,
                ),
            ],
        )

        result = OptimizationResult(
            status=OptimizationStatus.COMPLETED,
            optimized_performance=PerformanceMetrics(
                success_rate=0.90,
                avg_cost_per_task=0.10,
                avg_latency_ms=2000,
                quality_score=0.95,
            ),
        )

        score = pipeline._score_result(result, request)

        assert score > 0
        assert isinstance(score, float)
