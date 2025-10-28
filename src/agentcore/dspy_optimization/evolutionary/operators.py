"""
Genetic operators for crossover and mutation

Implements various crossover and mutation strategies for genetic algorithms.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from agentcore.dspy_optimization.evolutionary.population import Individual


class CrossoverOperator(ABC):
    """
    Abstract base class for crossover operators

    Crossover combines genetic material from two parents
    to create offspring with mixed characteristics.
    """

    @abstractmethod
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> tuple[Individual, Individual]:
        """
        Perform crossover between two parents

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Tuple of two offspring
        """
        pass


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover operator

    Each gene is randomly selected from either parent with equal probability.
    Provides high exploration of genetic combinations.
    """

    def __init__(self, swap_probability: float = 0.5) -> None:
        """
        Initialize uniform crossover

        Args:
            swap_probability: Probability of selecting gene from parent2
        """
        self.swap_probability = swap_probability

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> tuple[Individual, Individual]:
        """
        Perform uniform crossover

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring with mixed genes
        """
        # Get all unique keys from both genomes
        all_keys = set(parent1.genome.keys()) | set(parent2.genome.keys())

        # Create offspring genomes
        offspring1_genome: dict[str, Any] = {}
        offspring2_genome: dict[str, Any] = {}

        for key in all_keys:
            if random.random() < self.swap_probability:
                # Swap genes
                offspring1_genome[key] = parent2.genome.get(key, parent1.genome.get(key))
                offspring2_genome[key] = parent1.genome.get(key, parent2.genome.get(key))
            else:
                # Keep original
                offspring1_genome[key] = parent1.genome.get(key, parent2.genome.get(key))
                offspring2_genome[key] = parent2.genome.get(key, parent1.genome.get(key))

        # Create offspring individuals
        offspring1 = Individual(
            genome=offspring1_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )

        offspring2 = Individual(
            genome=offspring2_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )

        return offspring1, offspring2


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover operator

    Selects a random split point and swaps genetic material
    after that point between parents.
    """

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> tuple[Individual, Individual]:
        """
        Perform single-point crossover

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring with genes split at crossover point
        """
        # Get sorted keys for consistent ordering
        all_keys = sorted(set(parent1.genome.keys()) | set(parent2.genome.keys()))

        if len(all_keys) <= 1:
            # Not enough genes for crossover, return clones
            return parent1.clone(), parent2.clone()

        # Select random crossover point
        crossover_point = random.randint(1, len(all_keys) - 1)

        # Split genes at crossover point
        offspring1_genome: dict[str, Any] = {}
        offspring2_genome: dict[str, Any] = {}

        for i, key in enumerate(all_keys):
            if i < crossover_point:
                offspring1_genome[key] = parent1.genome.get(key, parent2.genome.get(key))
                offspring2_genome[key] = parent2.genome.get(key, parent1.genome.get(key))
            else:
                offspring1_genome[key] = parent2.genome.get(key, parent1.genome.get(key))
                offspring2_genome[key] = parent1.genome.get(key, parent2.genome.get(key))

        # Create offspring
        offspring1 = Individual(
            genome=offspring1_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )

        offspring2 = Individual(
            genome=offspring2_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )

        return offspring1, offspring2


class TwoPointCrossover(CrossoverOperator):
    """
    Two-point crossover operator

    Selects two random split points and swaps genetic material
    between them.
    """

    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> tuple[Individual, Individual]:
        """
        Perform two-point crossover

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring with genes split at two points
        """
        all_keys = sorted(set(parent1.genome.keys()) | set(parent2.genome.keys()))

        if len(all_keys) <= 2:
            return parent1.clone(), parent2.clone()

        # Select two random crossover points
        point1 = random.randint(1, len(all_keys) - 2)
        point2 = random.randint(point1 + 1, len(all_keys) - 1)

        # Create offspring genomes
        offspring1_genome: dict[str, Any] = {}
        offspring2_genome: dict[str, Any] = {}

        for i, key in enumerate(all_keys):
            if point1 <= i < point2:
                # Swap middle section
                offspring1_genome[key] = parent2.genome.get(key, parent1.genome.get(key))
                offspring2_genome[key] = parent1.genome.get(key, parent2.genome.get(key))
            else:
                # Keep original
                offspring1_genome[key] = parent1.genome.get(key, parent2.genome.get(key))
                offspring2_genome[key] = parent2.genome.get(key, parent1.genome.get(key))

        offspring1 = Individual(
            genome=offspring1_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )

        offspring2 = Individual(
            genome=offspring2_genome,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )

        return offspring1, offspring2


class MutationOperator(ABC):
    """
    Abstract base class for mutation operators

    Mutation introduces random variations to maintain
    genetic diversity and explore new solutions.
    """

    @abstractmethod
    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        pass


class GaussianMutation(MutationOperator):
    """
    Gaussian (normal distribution) mutation

    Adds random noise from Gaussian distribution to numeric genes.
    Suitable for continuous optimization problems.
    """

    def __init__(self, mutation_rate: float = 0.1, sigma: float = 0.1) -> None:
        """
        Initialize Gaussian mutation

        Args:
            mutation_rate: Probability of mutating each gene
            sigma: Standard deviation of Gaussian noise
        """
        self.mutation_rate = mutation_rate
        self.sigma = sigma

    def mutate(self, individual: Individual) -> Individual:
        """
        Apply Gaussian mutation

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        mutated_genome = individual.genome.copy()

        for key, value in mutated_genome.items():
            if random.random() < self.mutation_rate:
                # Apply Gaussian noise to numeric values
                if isinstance(value, (int, float)):
                    noise = random.gauss(0, self.sigma)
                    mutated_genome[key] = value + noise

                    # Clamp to [0, 1] for probabilities
                    if 0 <= value <= 1:
                        mutated_genome[key] = max(0.0, min(1.0, mutated_genome[key]))

        mutated = Individual(
            genome=mutated_genome,
            generation=individual.generation,
            parent_ids=[individual.id],
        )

        return mutated


class UniformMutation(MutationOperator):
    """
    Uniform random mutation

    Replaces genes with random values from uniform distribution.
    Provides high exploration capability.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> None:
        """
        Initialize uniform mutation

        Args:
            mutation_rate: Probability of mutating each gene
            min_value: Minimum value for mutations
            max_value: Maximum value for mutations
        """
        self.mutation_rate = mutation_rate
        self.min_value = min_value
        self.max_value = max_value

    def mutate(self, individual: Individual) -> Individual:
        """
        Apply uniform mutation

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        mutated_genome = individual.genome.copy()

        for key, value in mutated_genome.items():
            if random.random() < self.mutation_rate:
                # Replace with random value for numeric types
                if isinstance(value, (int, float)):
                    mutated_genome[key] = random.uniform(
                        self.min_value,
                        self.max_value
                    )

        mutated = Individual(
            genome=mutated_genome,
            generation=individual.generation,
            parent_ids=[individual.id],
        )

        return mutated


class BitFlipMutation(MutationOperator):
    """
    Bit flip mutation for binary genes

    Flips boolean or binary values with specified probability.
    """

    def __init__(self, mutation_rate: float = 0.1) -> None:
        """
        Initialize bit flip mutation

        Args:
            mutation_rate: Probability of flipping each bit
        """
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual) -> Individual:
        """
        Apply bit flip mutation

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        mutated_genome = individual.genome.copy()

        for key, value in mutated_genome.items():
            if random.random() < self.mutation_rate:
                # Flip boolean values
                if isinstance(value, bool):
                    mutated_genome[key] = not value
                # Flip binary integers (0/1)
                elif isinstance(value, int) and value in (0, 1):
                    mutated_genome[key] = 1 - value

        mutated = Individual(
            genome=mutated_genome,
            generation=individual.generation,
            parent_ids=[individual.id],
        )

        return mutated
