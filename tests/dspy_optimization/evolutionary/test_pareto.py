"""Tests for Pareto frontier and multi-objective optimization"""

import pytest

from agentcore.dspy_optimization.evolutionary.population import Individual
from agentcore.dspy_optimization.evolutionary.pareto import (
    ParetoFrontier,
    MultiObjectiveOptimizer,
)
from agentcore.dspy_optimization.models import (
    OptimizationObjective,
    MetricType,
)


class TestParetoFrontier:
    """Tests for Pareto frontier calculation"""

    @pytest.fixture
    def objectives(self) -> list[OptimizationObjective]:
        """Create test objectives"""
        return [
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
        ]

    def test_pareto_frontier_single_front(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test Pareto frontier with clear dominated solutions"""
        population = [
            Individual(
                genome={},
                objectives={"success_rate": 0.9, "cost_efficiency": 0.8},
            ),
            Individual(
                genome={},
                objectives={"success_rate": 0.7, "cost_efficiency": 0.6},
            ),
            Individual(
                genome={},
                objectives={"success_rate": 0.5, "cost_efficiency": 0.4},
            ),
        ]

        frontier = ParetoFrontier(objectives)
        optimal = frontier.get_pareto_optimal(population)

        # Only first individual is Pareto optimal
        assert len(optimal) == 1
        assert optimal[0].objectives["success_rate"] == 0.9

    def test_pareto_frontier_multiple_optimal(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test Pareto frontier with multiple optimal solutions"""
        population = [
            Individual(
                genome={},
                objectives={"success_rate": 0.9, "cost_efficiency": 0.6},
            ),
            Individual(
                genome={},
                objectives={"success_rate": 0.7, "cost_efficiency": 0.8},
            ),
            Individual(
                genome={},
                objectives={"success_rate": 0.5, "cost_efficiency": 0.5},
            ),
        ]

        frontier = ParetoFrontier(objectives)
        optimal = frontier.get_pareto_optimal(population)

        # First two are Pareto optimal (trade-offs)
        assert len(optimal) == 2

    def test_crowding_distance(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test crowding distance calculation"""
        population = [
            Individual(
                genome={},
                objectives={"success_rate": 0.9, "cost_efficiency": 0.6},
            ),
            Individual(
                genome={},
                objectives={"success_rate": 0.8, "cost_efficiency": 0.7},
            ),
            Individual(
                genome={},
                objectives={"success_rate": 0.7, "cost_efficiency": 0.8},
            ),
        ]

        frontier = ParetoFrontier(objectives)
        distances = frontier.calculate_crowding_distance(population)

        # Boundary points should have infinite distance
        assert distances[population[0].id] == float('inf')
        assert distances[population[2].id] == float('inf')

        # Middle point should have finite distance
        assert 0.0 < distances[population[1].id] < float('inf')

    def test_nsga2_selection(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test NSGA-II selection"""
        population = [
            Individual(
                genome={},
                fitness=0.9,
                objectives={"success_rate": 0.9, "cost_efficiency": 0.8},
            ),
            Individual(
                genome={},
                fitness=0.7,
                objectives={"success_rate": 0.7, "cost_efficiency": 0.6},
            ),
            Individual(
                genome={},
                fitness=0.5,
                objectives={"success_rate": 0.5, "cost_efficiency": 0.4},
            ),
        ]

        frontier = ParetoFrontier(objectives)
        selected = frontier.select_by_nsga2(population, count=2)

        # Should select best individuals
        assert len(selected) == 2

    def test_empty_population(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test Pareto frontier with empty population"""
        frontier = ParetoFrontier(objectives)
        optimal = frontier.get_pareto_optimal([])

        assert len(optimal) == 0


class TestMultiObjectiveOptimizer:
    """Tests for multi-objective optimizer"""

    @pytest.fixture
    def objectives(self) -> list[OptimizationObjective]:
        """Create test objectives"""
        return [
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
        ]

    def test_optimize_population(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test population optimization"""
        population = [
            Individual(
                genome={},
                objectives={
                    "success_rate": 0.9,
                    "cost_efficiency": 0.8,
                    "latency": 0.7,
                },
            ),
            Individual(
                genome={},
                objectives={
                    "success_rate": 0.7,
                    "cost_efficiency": 0.6,
                    "latency": 0.5,
                },
            ),
        ]

        optimizer = MultiObjectiveOptimizer(objectives)
        pareto_front = optimizer.optimize_population(population)

        assert len(pareto_front) >= 1

    def test_select_next_generation(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test next generation selection"""
        population = [
            Individual(
                genome={},
                fitness=0.8,
                objectives={
                    "success_rate": 0.8,
                    "cost_efficiency": 0.7,
                    "latency": 0.6,
                },
            ),
        ]

        offspring = [
            Individual(
                genome={},
                fitness=0.9,
                objectives={
                    "success_rate": 0.9,
                    "cost_efficiency": 0.8,
                    "latency": 0.7,
                },
            ),
        ]

        optimizer = MultiObjectiveOptimizer(objectives)
        next_gen = optimizer.select_for_next_generation(
            population,
            offspring,
            population_size=1,
        )

        assert len(next_gen) == 1

    def test_get_balanced_solution(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test balanced solution selection"""
        pareto_front = [
            Individual(
                genome={},
                objectives={
                    "success_rate": 0.9,
                    "cost_efficiency": 0.6,
                    "latency": 0.5,
                },
            ),
            Individual(
                genome={},
                objectives={
                    "success_rate": 0.7,
                    "cost_efficiency": 0.8,
                    "latency": 0.7,
                },
            ),
            Individual(
                genome={},
                objectives={
                    "success_rate": 0.8,
                    "cost_efficiency": 0.7,
                    "latency": 0.8,
                },
            ),
        ]

        optimizer = MultiObjectiveOptimizer(objectives)
        balanced = optimizer.get_balanced_solution(pareto_front)

        assert balanced is not None
        assert balanced in pareto_front

    def test_analyze_tradeoffs(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test trade-off analysis"""
        pareto_front = [
            Individual(
                genome={},
                objectives={
                    "success_rate": 0.9,
                    "cost_efficiency": 0.6,
                    "latency": 0.5,
                },
            ),
            Individual(
                genome={},
                objectives={
                    "success_rate": 0.7,
                    "cost_efficiency": 0.8,
                    "latency": 0.7,
                },
            ),
        ]

        optimizer = MultiObjectiveOptimizer(objectives)
        analyses = optimizer.analyze_tradeoffs(pareto_front)

        # Should have analysis for each objective pair
        assert len(analyses) == 3  # 3 choose 2 = 3 pairs

        for analysis in analyses:
            assert -1.0 <= analysis.correlation <= 1.0
            assert 0.0 <= analysis.diversity_score <= 1.0
            assert analysis.pareto_front_size == 2

    def test_empty_pareto_front(
        self,
        objectives: list[OptimizationObjective],
    ) -> None:
        """Test with empty Pareto front"""
        optimizer = MultiObjectiveOptimizer(objectives)
        balanced = optimizer.get_balanced_solution([])

        assert balanced is None
