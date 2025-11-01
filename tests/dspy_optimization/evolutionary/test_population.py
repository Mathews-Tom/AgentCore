"""Tests for population management"""

import pytest

from agentcore.dspy_optimization.evolutionary.population import (
    Individual,
    Population,
    PopulationConfig,
)
from agentcore.dspy_optimization.models import (
    PerformanceMetrics,
    OptimizationObjective,
    MetricType,
)


class TestIndividual:
    """Tests for Individual class"""

    def test_individual_creation(self) -> None:
        """Test creating an individual"""
        genome = {"param1": 0.5, "param2": 1.0}
        individual = Individual(genome=genome, fitness=0.8)

        assert individual.genome == genome
        assert individual.fitness == 0.8
        assert individual.generation == 0
        assert len(individual.id) > 0

    def test_individual_dominates(self) -> None:
        """Test Pareto dominance checking"""
        objectives = [
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

        ind1 = Individual(
            genome={},
            objectives={"success_rate": 0.9, "cost_efficiency": 0.8},
        )
        ind2 = Individual(
            genome={},
            objectives={"success_rate": 0.7, "cost_efficiency": 0.6},
        )

        # ind1 dominates ind2 (better in all objectives)
        assert ind1.dominates(ind2, objectives)
        assert not ind2.dominates(ind1, objectives)

    def test_individual_non_dominance(self) -> None:
        """Test non-dominating individuals"""
        objectives = [
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

        ind1 = Individual(
            genome={},
            objectives={"success_rate": 0.9, "cost_efficiency": 0.6},
        )
        ind2 = Individual(
            genome={},
            objectives={"success_rate": 0.7, "cost_efficiency": 0.8},
        )

        # Neither dominates (trade-off)
        assert not ind1.dominates(ind2, objectives)
        assert not ind2.dominates(ind1, objectives)

    def test_individual_clone(self) -> None:
        """Test cloning an individual"""
        original = Individual(
            genome={"param": 0.5},
            fitness=0.8,
            generation=5,
        )
        clone = original.clone()

        assert clone.id != original.id
        assert clone.genome == original.genome
        assert clone.fitness == original.fitness
        assert clone.generation == original.generation
        assert original.id in clone.parent_ids


class TestPopulation:
    """Tests for Population class"""

    def test_population_initialization(self) -> None:
        """Test population initialization"""
        config = PopulationConfig(population_size=10)
        population = Population(config)

        individuals = [
            Individual(genome={"param": i / 10}, fitness=i / 10)
            for i in range(10)
        ]

        population.initialize(individuals)

        assert len(population.individuals) == 10
        assert population.generation == 0
        assert population.best_individual is not None

    def test_population_elite_selection(self) -> None:
        """Test elite selection"""
        config = PopulationConfig(population_size=10, elitism_count=3)
        population = Population(config)

        individuals = [
            Individual(genome={}, fitness=float(i))
            for i in range(10)
        ]
        population.initialize(individuals)

        elite = population.get_elite()

        assert len(elite) == 3
        assert elite[0].fitness == 9.0  # Highest fitness
        assert elite[1].fitness == 8.0
        assert elite[2].fitness == 7.0

    def test_population_diversity(self) -> None:
        """Test diversity calculation"""
        config = PopulationConfig(population_size=10)
        population = Population(config)

        # Create diverse individuals
        individuals = [
            Individual(genome={"param1": i / 10, "param2": (10 - i) / 10})
            for i in range(10)
        ]
        population.initialize(individuals)

        diversity = population.calculate_diversity()

        # Should have high diversity
        assert 0.0 <= diversity <= 1.0
        assert diversity > 0.0

    def test_population_low_diversity(self) -> None:
        """Test low diversity detection"""
        config = PopulationConfig(population_size=10)
        population = Population(config)

        # Create identical individuals
        individuals = [
            Individual(genome={"param": 0.5})
            for _ in range(10)
        ]
        population.initialize(individuals)

        diversity = population.calculate_diversity()

        # Should have zero diversity
        assert diversity == 0.0

    def test_population_advance_generation(self) -> None:
        """Test advancing to next generation"""
        config = PopulationConfig(population_size=10, elitism_count=2)
        population = Population(config)

        # Initial population
        individuals = [
            Individual(genome={}, fitness=float(i))
            for i in range(10)
        ]
        population.initialize(individuals)

        initial_generation = population.generation
        initial_best = population.best_individual

        # New individuals for next generation
        new_individuals = [
            Individual(genome={}, fitness=float(i + 5))
            for i in range(8)
        ]

        population.advance_generation(new_individuals)

        # Check generation advanced
        assert population.generation == initial_generation + 1

        # Check elite preserved
        assert len(population.individuals) == 10

        # Check best updated if better individual found
        if initial_best:
            assert population.best_individual is not None

    def test_population_fitness_stats(self) -> None:
        """Test fitness statistics"""
        config = PopulationConfig(population_size=10)
        population = Population(config)

        individuals = [
            Individual(genome={}, fitness=float(i))
            for i in range(10)
        ]
        population.initialize(individuals)

        stats = population.get_fitness_stats()

        assert stats["min"] == 0.0
        assert stats["max"] == 9.0
        assert stats["mean"] == 4.5
        assert stats["std"] > 0.0

    def test_population_average_fitness(self) -> None:
        """Test average fitness calculation"""
        config = PopulationConfig(population_size=10)
        population = Population(config)

        individuals = [
            Individual(genome={}, fitness=float(i))
            for i in range(1, 6)  # 1, 2, 3, 4, 5
        ]
        population.initialize(individuals)

        avg = population.get_average_fitness()

        assert avg == 3.0  # (1+2+3+4+5)/5 = 3

    def test_population_empty(self) -> None:
        """Test population with no individuals"""
        config = PopulationConfig(population_size=10)
        population = Population(config)

        # Don't initialize, just test empty population methods
        assert population.get_average_fitness() == 0.0
        assert population.calculate_diversity() == 0.0

        stats = population.get_fitness_stats()
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
