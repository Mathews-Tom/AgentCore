"""
Selection strategies for genetic algorithms

Implements tournament, roulette wheel, and elitism selection strategies.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from agentcore.dspy_optimization.evolutionary.population import Individual


class SelectionStrategy(ABC):
    """
    Abstract base class for selection strategies

    Selection determines which individuals reproduce to create
    the next generation based on fitness scores.
    """

    @abstractmethod
    def select(
        self,
        population: list[Individual],
        count: int,
    ) -> list[Individual]:
        """
        Select individuals from population

        Args:
            population: Population to select from
            count: Number of individuals to select

        Returns:
            Selected individuals
        """
        pass


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection strategy

    Randomly selects k individuals and chooses the best among them.
    Provides good balance between exploration and exploitation.
    """

    def __init__(self, tournament_size: int = 3) -> None:
        """
        Initialize tournament selection

        Args:
            tournament_size: Number of individuals in each tournament
        """
        self.tournament_size = tournament_size

    def select(
        self,
        population: list[Individual],
        count: int,
    ) -> list[Individual]:
        """
        Select individuals via tournament

        Args:
            population: Population to select from
            count: Number of individuals to select

        Returns:
            Selected individuals
        """
        selected: list[Individual] = []

        for _ in range(count):
            # Randomly select tournament participants
            tournament = random.sample(
                population,
                min(self.tournament_size, len(population))
            )

            # Select best from tournament
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)

        return selected


class RouletteWheelSelection(SelectionStrategy):
    """
    Roulette wheel (fitness proportionate) selection

    Selects individuals with probability proportional to their fitness.
    Higher fitness individuals have higher selection probability.
    """

    def select(
        self,
        population: list[Individual],
        count: int,
    ) -> list[Individual]:
        """
        Select individuals via roulette wheel

        Args:
            population: Population to select from
            count: Number of individuals to select

        Returns:
            Selected individuals
        """
        if not population:
            return []

        # Calculate total fitness
        total_fitness = sum(ind.fitness for ind in population)

        if total_fitness == 0:
            # If all fitness is zero, select randomly
            return random.sample(population, min(count, len(population)))

        # Normalize fitness to probabilities
        probabilities = [ind.fitness / total_fitness for ind in population]

        # Select with replacement based on probabilities
        selected = random.choices(
            population,
            weights=probabilities,
            k=count
        )

        return selected


class ElitismSelection(SelectionStrategy):
    """
    Elitism selection strategy

    Always selects the top-performing individuals.
    Ensures best solutions are never lost.
    """

    def select(
        self,
        population: list[Individual],
        count: int,
    ) -> list[Individual]:
        """
        Select elite individuals

        Args:
            population: Population to select from
            count: Number of individuals to select

        Returns:
            Top performing individuals
        """
        sorted_population = sorted(
            population,
            key=lambda ind: ind.fitness,
            reverse=True
        )

        return sorted_population[:count]


class RankSelection(SelectionStrategy):
    """
    Rank-based selection strategy

    Selects based on fitness rank rather than absolute fitness values.
    Reduces selection pressure when fitness differences are large.
    """

    def __init__(self, selection_pressure: float = 1.5) -> None:
        """
        Initialize rank selection

        Args:
            selection_pressure: Controls selection bias (1.0-2.0)
        """
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))

    def select(
        self,
        population: list[Individual],
        count: int,
    ) -> list[Individual]:
        """
        Select individuals via rank

        Args:
            population: Population to select from
            count: Number of individuals to select

        Returns:
            Selected individuals
        """
        if not population:
            return []

        # Sort by fitness (worst to best)
        sorted_population = sorted(population, key=lambda ind: ind.fitness)

        # Calculate rank-based probabilities
        n = len(sorted_population)
        probabilities = [
            (2 - self.selection_pressure) / n +
            (2 * rank * (self.selection_pressure - 1)) / (n * (n - 1))
            for rank in range(1, n + 1)
        ]

        # Normalize probabilities
        prob_sum = sum(probabilities)
        probabilities = [p / prob_sum for p in probabilities]

        # Select based on rank probabilities
        selected = random.choices(
            sorted_population,
            weights=probabilities,
            k=count
        )

        return selected


class StochasticUniversalSampling(SelectionStrategy):
    """
    Stochastic Universal Sampling (SUS)

    Similar to roulette wheel but with evenly spaced selection points.
    Provides lower variance in expected number of selections.
    """

    def select(
        self,
        population: list[Individual],
        count: int,
    ) -> list[Individual]:
        """
        Select individuals via SUS

        Args:
            population: Population to select from
            count: Number of individuals to select

        Returns:
            Selected individuals
        """
        if not population:
            return []

        # Calculate total fitness
        total_fitness = sum(ind.fitness for ind in population)

        if total_fitness == 0:
            return random.sample(population, min(count, len(population)))

        # Calculate selection interval
        interval = total_fitness / count

        # Random start point
        start = random.uniform(0, interval)

        # Generate evenly spaced pointers
        pointers = [start + i * interval for i in range(count)]

        # Select individuals at pointer positions
        selected: list[Individual] = []
        cumulative_fitness = 0.0
        pointer_idx = 0

        for individual in population:
            cumulative_fitness += individual.fitness

            while pointer_idx < len(pointers) and pointers[pointer_idx] < cumulative_fitness:
                selected.append(individual)
                pointer_idx += 1

            if pointer_idx >= len(pointers):
                break

        return selected
