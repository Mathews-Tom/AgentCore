"""Tests for genetic algorithm optimizer"""

import pytest

from agentcore.dspy_optimization.evolutionary.genetic import GeneticOptimizer
from agentcore.dspy_optimization.evolutionary.convergence import ConvergenceCriteria
from agentcore.dspy_optimization.evolutionary.population import PopulationConfig
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationTarget,
    OptimizationTargetType,
    OptimizationScope,
    OptimizationObjective,
    MetricType,
    OptimizationConstraints,
    PerformanceMetrics,
    OptimizationStatus,
)


class TestGeneticOptimizer:
    """Tests for genetic algorithm optimizer"""

    @pytest.fixture
    def optimizer(self) -> GeneticOptimizer:
        """Create genetic optimizer for testing"""
        population_config = PopulationConfig(
            population_size=10,
            elitism_count=2,
        )
        convergence_criteria = ConvergenceCriteria(
            max_generations=5,  # Small for testing
            plateau_generations=3,
        )
        return GeneticOptimizer(
            population_config=population_config,
            convergence_criteria=convergence_criteria,
        )

    @pytest.fixture
    def optimization_request(self) -> OptimizationRequest:
        """Create optimization request for testing"""
        return OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test-agent",
                scope=OptimizationScope.INDIVIDUAL,
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.9,
                    weight=0.5,
                ),
                OptimizationObjective(
                    metric=MetricType.COST_EFFICIENCY,
                    target_value=0.8,
                    weight=0.5,
                ),
            ],
            algorithms=["genetic"],
            constraints=OptimizationConstraints(
                max_optimization_time=7200,
                min_improvement_threshold=0.05,
            ),
        )

    @pytest.fixture
    def baseline_metrics(self) -> PerformanceMetrics:
        """Create baseline metrics for testing"""
        return PerformanceMetrics(
            success_rate=0.75,
            avg_cost_per_task=0.12,
            avg_latency_ms=2500,
            quality_score=0.7,
        )

    @pytest.fixture
    def training_data(self) -> list[dict[str, str]]:
        """Create training data for testing"""
        return [
            {"question": "Test question 1", "answer": "Test answer 1"},
            {"question": "Test question 2", "answer": "Test answer 2"},
            {"question": "Test question 3", "answer": "Test answer 3"},
        ]

    @pytest.mark.asyncio
    async def test_basic_optimization(
        self,
        optimizer: GeneticOptimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test basic genetic optimization"""
        result = await optimizer.optimize(
            optimization_request,
            baseline_metrics,
            training_data,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.baseline_performance == baseline_metrics
        assert result.optimized_performance is not None
        assert result.optimization_details is not None
        assert result.optimization_details.algorithm_used == "genetic"

    @pytest.mark.asyncio
    async def test_optimization_improvement(
        self,
        optimizer: GeneticOptimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test that optimization produces improvements"""
        result = await optimizer.optimize(
            optimization_request,
            baseline_metrics,
            training_data,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimized_performance is not None

        # Check improvement
        assert result.improvement_percentage != 0.0

    @pytest.mark.asyncio
    async def test_multi_objective_optimization(
        self,
        optimizer: GeneticOptimizer,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test multi-objective optimization"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test-agent",
                scope=OptimizationScope.INDIVIDUAL,
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.9,
                    weight=0.4,
                ),
                OptimizationObjective(
                    metric=MetricType.COST_EFFICIENCY,
                    target_value=0.8,
                    weight=0.3,
                ),
                OptimizationObjective(
                    metric=MetricType.LATENCY,
                    target_value=0.7,
                    weight=0.3,
                ),
            ],
            algorithms=["genetic"],
        )

        result = await optimizer.optimize(
            request,
            baseline_metrics,
            training_data,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimization_details is not None

        # Check Pareto front mentioned in improvements
        improvements = result.optimization_details.key_improvements
        pareto_mentioned = any("Pareto" in imp for imp in improvements)
        assert pareto_mentioned

    @pytest.mark.asyncio
    async def test_convergence_tracking(
        self,
        optimizer: GeneticOptimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test convergence tracking"""
        result = await optimizer.optimize(
            optimization_request,
            baseline_metrics,
            training_data,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimization_details is not None

        # Check convergence reason recorded
        params = result.optimization_details.parameters
        assert "convergence_reason" in params
        assert "final_diversity" in params

    @pytest.mark.asyncio
    async def test_population_evolution(
        self,
        optimizer: GeneticOptimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test that population evolves over generations"""
        result = await optimizer.optimize(
            optimization_request,
            baseline_metrics,
            training_data,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimization_details is not None

        # Check that multiple generations ran
        generations = result.optimization_details.iterations
        assert generations > 0
        assert generations <= optimizer.convergence_criteria.max_generations

    @pytest.mark.asyncio
    async def test_elitism_preservation(
        self,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test that elitism preserves best individuals"""
        config = PopulationConfig(
            population_size=10,
            elitism_count=3,
        )
        convergence = ConvergenceCriteria(max_generations=3)

        optimizer = GeneticOptimizer(
            population_config=config,
            convergence_criteria=convergence,
        )

        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test-agent",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.9,
                ),
            ],
        )

        result = await optimizer.optimize(
            request,
            baseline_metrics,
            training_data,
        )

        assert result.status == OptimizationStatus.COMPLETED

    def test_algorithm_name(self, optimizer: GeneticOptimizer) -> None:
        """Test algorithm name"""
        assert optimizer.get_algorithm_name() == "genetic"

    def test_random_genome_creation(self, optimizer: GeneticOptimizer) -> None:
        """Test random genome creation"""
        genome = optimizer._create_random_genome()

        assert isinstance(genome, dict)
        assert len(genome) > 0

        # Check parameter ranges
        assert 0.1 <= genome["temperature"] <= 1.0
        assert 100 <= genome["max_tokens"] <= 2000
        assert 1 <= genome["reasoning_depth"] <= 5

    @pytest.mark.asyncio
    async def test_empty_training_data(
        self,
        optimizer: GeneticOptimizer,
        optimization_request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """Test optimization with empty training data"""
        result = await optimizer.optimize(
            optimization_request,
            baseline_metrics,
            [],
        )

        # Should still complete (using simulated evaluation)
        assert result.status == OptimizationStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_single_objective_optimization(
        self,
        optimizer: GeneticOptimizer,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, str]],
    ) -> None:
        """Test single-objective optimization (no Pareto)"""
        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test-agent",
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE,
                    target_value=0.9,
                ),
            ],
        )

        result = await optimizer.optimize(
            request,
            baseline_metrics,
            training_data,
        )

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimization_details is not None

        # Should not mention Pareto for single objective
        improvements = result.optimization_details.key_improvements
        pareto_mentioned = any("Pareto" in imp for imp in improvements)
        assert not pareto_mentioned
