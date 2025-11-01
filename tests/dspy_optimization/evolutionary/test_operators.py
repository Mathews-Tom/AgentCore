"""Tests for genetic operators"""

import pytest

from agentcore.dspy_optimization.evolutionary.population import Individual
from agentcore.dspy_optimization.evolutionary.operators import (
    UniformCrossover,
    SinglePointCrossover,
    TwoPointCrossover,
    GaussianMutation,
    UniformMutation,
    BitFlipMutation,
)


class TestUniformCrossover:
    """Tests for uniform crossover"""

    def test_uniform_crossover(self) -> None:
        """Test basic uniform crossover"""
        parent1 = Individual(
            genome={"a": 1.0, "b": 2.0, "c": 3.0},
            generation=1,
        )
        parent2 = Individual(
            genome={"a": 4.0, "b": 5.0, "c": 6.0},
            generation=1,
        )

        crossover = UniformCrossover(swap_probability=0.5)
        child1, child2 = crossover.crossover(parent1, parent2)

        # Check children have correct structure
        assert len(child1.genome) == 3
        assert len(child2.genome) == 3

        # Check generation incremented
        assert child1.generation == 2
        assert child2.generation == 2

        # Check parent IDs recorded
        assert parent1.id in child1.parent_ids
        assert parent2.id in child1.parent_ids


class TestSinglePointCrossover:
    """Tests for single-point crossover"""

    def test_single_point_crossover(self) -> None:
        """Test basic single-point crossover"""
        parent1 = Individual(
            genome={"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0},
            generation=1,
        )
        parent2 = Individual(
            genome={"a": 5.0, "b": 6.0, "c": 7.0, "d": 8.0},
            generation=1,
        )

        crossover = SinglePointCrossover()
        child1, child2 = crossover.crossover(parent1, parent2)

        # Check children created
        assert len(child1.genome) == 4
        assert len(child2.genome) == 4
        assert child1.generation == 2

    def test_single_point_minimal_genes(self) -> None:
        """Test single-point crossover with minimal genes"""
        parent1 = Individual(genome={"a": 1.0}, generation=1)
        parent2 = Individual(genome={"a": 2.0}, generation=1)

        crossover = SinglePointCrossover()
        child1, child2 = crossover.crossover(parent1, parent2)

        # Should return clones for minimal genes
        assert len(child1.genome) >= 1
        assert len(child2.genome) >= 1


class TestTwoPointCrossover:
    """Tests for two-point crossover"""

    def test_two_point_crossover(self) -> None:
        """Test basic two-point crossover"""
        parent1 = Individual(
            genome={"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0},
            generation=1,
        )
        parent2 = Individual(
            genome={"a": 6.0, "b": 7.0, "c": 8.0, "d": 9.0, "e": 10.0},
            generation=1,
        )

        crossover = TwoPointCrossover()
        child1, child2 = crossover.crossover(parent1, parent2)

        assert len(child1.genome) == 5
        assert len(child2.genome) == 5
        assert child1.generation == 2


class TestGaussianMutation:
    """Tests for Gaussian mutation"""

    def test_gaussian_mutation(self) -> None:
        """Test basic Gaussian mutation"""
        individual = Individual(
            genome={"a": 0.5, "b": 0.7, "c": 0.3},
            generation=1,
        )

        mutation = GaussianMutation(mutation_rate=1.0, sigma=0.1)
        mutated = mutation.mutate(individual)

        # Check mutation occurred
        assert mutated.id != individual.id
        assert len(mutated.genome) == 3

        # Values should be different but in valid range [0, 1]
        for key in mutated.genome:
            assert 0.0 <= mutated.genome[key] <= 1.0

    def test_gaussian_mutation_zero_rate(self) -> None:
        """Test Gaussian mutation with zero rate"""
        individual = Individual(
            genome={"a": 0.5, "b": 0.7},
            generation=1,
        )

        mutation = GaussianMutation(mutation_rate=0.0)
        mutated = mutation.mutate(individual)

        # No mutations should occur
        assert mutated.genome == individual.genome


class TestUniformMutation:
    """Tests for uniform mutation"""

    def test_uniform_mutation(self) -> None:
        """Test basic uniform mutation"""
        individual = Individual(
            genome={"a": 0.5, "b": 0.7, "c": 0.3},
            generation=1,
        )

        mutation = UniformMutation(mutation_rate=1.0, min_value=0.0, max_value=1.0)
        mutated = mutation.mutate(individual)

        # Check structure preserved
        assert len(mutated.genome) == 3

        # Values should be in specified range
        for key in mutated.genome:
            assert 0.0 <= mutated.genome[key] <= 1.0

    def test_uniform_mutation_custom_range(self) -> None:
        """Test uniform mutation with custom range"""
        individual = Individual(
            genome={"a": 50.0, "b": 75.0},
            generation=1,
        )

        mutation = UniformMutation(mutation_rate=1.0, min_value=0.0, max_value=100.0)
        mutated = mutation.mutate(individual)

        for key in mutated.genome:
            assert 0.0 <= mutated.genome[key] <= 100.0


class TestBitFlipMutation:
    """Tests for bit flip mutation"""

    def test_bit_flip_mutation(self) -> None:
        """Test basic bit flip mutation"""
        individual = Individual(
            genome={"a": True, "b": False, "c": True},
            generation=1,
        )

        mutation = BitFlipMutation(mutation_rate=1.0)
        mutated = mutation.mutate(individual)

        # All bits should be flipped
        assert mutated.genome["a"] == False
        assert mutated.genome["b"] == True
        assert mutated.genome["c"] == False

    def test_bit_flip_binary_integers(self) -> None:
        """Test bit flip with binary integers"""
        individual = Individual(
            genome={"a": 1, "b": 0, "c": 1},
            generation=1,
        )

        mutation = BitFlipMutation(mutation_rate=1.0)
        mutated = mutation.mutate(individual)

        # Binary integers should be flipped
        assert mutated.genome["a"] == 0
        assert mutated.genome["b"] == 1
        assert mutated.genome["c"] == 0

    def test_bit_flip_zero_rate(self) -> None:
        """Test bit flip with zero mutation rate"""
        individual = Individual(
            genome={"a": True, "b": False},
            generation=1,
        )

        mutation = BitFlipMutation(mutation_rate=0.0)
        mutated = mutation.mutate(individual)

        # No flips should occur
        assert mutated.genome == individual.genome
