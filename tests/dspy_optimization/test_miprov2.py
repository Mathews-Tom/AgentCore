"""
Tests for MIPROv2 optimizer
"""

from __future__ import annotations

import pytest
import dspy

from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer
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
def optimizer() -> MIPROv2Optimizer:
    """Create MIPROv2 optimizer instance"""
    # Use a mock LM for testing to avoid API calls
    mock_llm = dspy.LM("openai/gpt-4.1-mini", api_key="test-key", cache_seed=42)
    return MIPROv2Optimizer(llm=mock_llm, num_candidates=3)


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
        ],
        algorithms=["miprov2"],
    )


@pytest.fixture
def training_data() -> list[dict[str, str]]:
    """Create training data"""
    return [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Hamlet?", "answer": "Shakespeare"},
    ]


class TestMIPROv2Optimizer:
    """Tests for MIPROv2Optimizer class"""

    def test_initialization(self, optimizer: MIPROv2Optimizer) -> None:
        """Test optimizer initialization"""
        assert optimizer.num_candidates == 3
        assert optimizer.max_bootstrapped_demos == 5
        assert optimizer.max_labeled_demos == 10
        assert optimizer.llm is not None

    def test_algorithm_name(self, optimizer: MIPROv2Optimizer) -> None:
        """Test algorithm name"""
        assert optimizer.get_algorithm_name() == "miprov2"

    async def test_optimize_success(
        self,
        optimizer: MIPROv2Optimizer,
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
        assert result.optimization_details.algorithm_used == "miprov2"
        assert result.optimization_details.iterations == 3

    async def test_optimize_improvement(
        self,
        optimizer: MIPROv2Optimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test that optimization improves performance"""
        result = await optimizer.optimize(
            optimization_request, baseline_metrics, training_data
        )

        if result.optimized_performance:
            # Check improvements
            assert result.optimized_performance.success_rate >= baseline_metrics.success_rate
            assert result.optimized_performance.avg_cost_per_task <= baseline_metrics.avg_cost_per_task
            assert result.optimized_performance.avg_latency_ms <= baseline_metrics.avg_latency_ms

    def test_calculate_improvement(
        self, optimizer: MIPROv2Optimizer, baseline_metrics: PerformanceMetrics
    ) -> None:
        """Test improvement calculation"""
        optimized = PerformanceMetrics(
            success_rate=0.90,
            avg_cost_per_task=0.12,
            avg_latency_ms=2400,
            quality_score=0.95,
        )

        improvement = optimizer.calculate_improvement(baseline_metrics, optimized)

        assert improvement > 0
        assert isinstance(improvement, float)

    def test_prepare_training_data(
        self, optimizer: MIPROv2Optimizer, training_data: list[dict[str, str]]
    ) -> None:
        """Test training data preparation"""
        examples = optimizer._prepare_training_data(training_data)

        assert len(examples) == 3
        assert all(isinstance(ex, dspy.Example) for ex in examples)
        assert examples[0].question == "What is 2+2?"
        assert examples[0].answer == "4"

    def test_prepare_training_data_input_output_format(
        self, optimizer: MIPROv2Optimizer
    ) -> None:
        """Test training data with input/output format"""
        data = [
            {"input": "Test input 1", "output": "Test output 1"},
            {"input": "Test input 2", "output": "Test output 2"},
        ]

        examples = optimizer._prepare_training_data(data)

        assert len(examples) == 2
        assert examples[0].question == "Test input 1"
        assert examples[0].answer == "Test output 1"

    def test_prepare_training_data_empty(self, optimizer: MIPROv2Optimizer) -> None:
        """Test empty training data returns default example"""
        examples = optimizer._prepare_training_data([])

        assert len(examples) == 1
        assert examples[0].question == "test"
        assert examples[0].answer == "test"

    def test_create_optimization_program(
        self, optimizer: MIPROv2Optimizer, optimization_request: OptimizationRequest
    ) -> None:
        """Test program creation"""
        program = optimizer._create_optimization_program(optimization_request)

        assert isinstance(program, dspy.Module)
        assert hasattr(program, "forward")

    async def test_evaluate_program(
        self,
        optimizer: MIPROv2Optimizer,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test program evaluation"""
        program = optimizer._create_optimization_program(
            OptimizationRequest(
                target=OptimizationTarget(
                    type=OptimizationTargetType.AGENT,
                    id="test",
                ),
                objectives=[],
            )
        )

        metrics = await optimizer._evaluate_program(
            program, training_data, baseline_metrics
        )

        assert metrics.success_rate >= baseline_metrics.success_rate
        assert metrics.avg_cost_per_task <= baseline_metrics.avg_cost_per_task
        assert metrics.avg_latency_ms <= baseline_metrics.avg_latency_ms

    def test_custom_parameters(self) -> None:
        """Test optimizer with custom parameters"""
        optimizer = MIPROv2Optimizer(
            num_candidates=20,
            max_bootstrapped_demos=8,
            max_labeled_demos=15,
        )

        assert optimizer.num_candidates == 20
        assert optimizer.max_bootstrapped_demos == 8
        assert optimizer.max_labeled_demos == 15
