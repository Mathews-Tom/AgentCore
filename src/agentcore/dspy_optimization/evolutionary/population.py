"""
Population management for genetic algorithms

Manages individuals, fitness evaluation, and population lifecycle operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import PerformanceMetrics, OptimizationObjective


@dataclass
class Individual:
    """
    Individual in genetic algorithm population

    Represents a candidate solution with genome (parameters),
    fitness scores, and performance metrics.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    genome: dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    objectives: dict[str, float] = field(default_factory=dict)
    metrics: PerformanceMetrics | None = None
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def dominates(self, other: Individual, objectives: list[OptimizationObjective]) -> bool:
        """
        Check if this individual dominates another in Pareto sense

        Args:
            other: Other individual to compare
            objectives: Optimization objectives for comparison

        Returns:
            True if this individual dominates the other
        """
        better_in_any = False
        worse_in_any = False

        for obj in objectives:
            metric_name = obj.metric.value
            self_value = self.objectives.get(metric_name, 0.0)
            other_value = other.objectives.get(metric_name, 0.0)

            if self_value > other_value:
                better_in_any = True
            elif self_value < other_value:
                worse_in_any = True

        # Dominates if better in at least one objective and not worse in any
        return better_in_any and not worse_in_any

    def clone(self) -> Individual:
        """Create a deep copy of the individual"""
        return Individual(
            id=str(uuid4()),
            genome=self.genome.copy(),
            fitness=self.fitness,
            objectives=self.objectives.copy(),
            metrics=self.metrics,
            generation=self.generation,
            parent_ids=[self.id],
            metadata=self.metadata.copy(),
        )


class PopulationConfig(BaseModel):
    """Configuration for population management"""

    population_size: int = Field(default=50, ge=10, le=500)
    elitism_count: int = Field(default=5, ge=0, le=20, description="Top individuals preserved")
    diversity_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Min diversity to maintain")
    max_generations: int = Field(default=100, ge=1, le=1000)
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)


class Population:
    """
    Manages population of individuals for genetic algorithm

    Handles population initialization, fitness evaluation,
    diversity maintenance, and generational transitions.
    """

    def __init__(self, config: PopulationConfig | None = None) -> None:
        """
        Initialize population

        Args:
            config: Population configuration
        """
        self.config = config or PopulationConfig()
        self.individuals: list[Individual] = []
        self.generation: int = 0
        self.best_individual: Individual | None = None
        self.fitness_history: list[dict[str, float]] = []

    def initialize(self, initial_individuals: list[Individual]) -> None:
        """
        Initialize population with individuals

        Args:
            initial_individuals: Initial population members
        """
        self.individuals = initial_individuals[: self.config.population_size]
        self.generation = 0
        self._update_best()

    def add_individual(self, individual: Individual) -> None:
        """
        Add individual to population

        Args:
            individual: Individual to add
        """
        individual.generation = self.generation
        self.individuals.append(individual)

    def add_individuals(self, individuals: list[Individual]) -> None:
        """
        Add multiple individuals to population

        Args:
            individuals: Individuals to add
        """
        for individual in individuals:
            self.add_individual(individual)

    def get_elite(self, count: int | None = None) -> list[Individual]:
        """
        Get elite individuals (top performers)

        Args:
            count: Number of elite individuals (uses config default if None)

        Returns:
            Top performing individuals
        """
        elite_count = count or self.config.elitism_count
        sorted_individuals = sorted(
            self.individuals,
            key=lambda ind: ind.fitness,
            reverse=True
        )
        return sorted_individuals[:elite_count]

    def get_worst(self, count: int) -> list[Individual]:
        """
        Get worst performing individuals

        Args:
            count: Number of worst individuals

        Returns:
            Bottom performing individuals
        """
        sorted_individuals = sorted(
            self.individuals,
            key=lambda ind: ind.fitness,
        )
        return sorted_individuals[:count]

    def calculate_diversity(self) -> float:
        """
        Calculate population diversity

        Measures genetic diversity based on genome variation.
        Higher values indicate more diverse population.

        Returns:
            Diversity score (0.0-1.0)
        """
        if len(self.individuals) < 2:
            return 0.0

        # Calculate pairwise genome differences
        total_difference = 0.0
        comparisons = 0

        for i, ind1 in enumerate(self.individuals):
            for ind2 in self.individuals[i + 1:]:
                difference = self._genome_difference(ind1.genome, ind2.genome)
                total_difference += difference
                comparisons += 1

        if comparisons == 0:
            return 0.0

        return min(total_difference / comparisons, 1.0)

    def advance_generation(self, new_individuals: list[Individual]) -> None:
        """
        Advance to next generation with new individuals

        Args:
            new_individuals: New generation individuals
        """
        # Preserve elite
        elite = self.get_elite()

        # Combine elite with new individuals
        self.individuals = elite + new_individuals

        # Maintain population size
        self.individuals = self.individuals[: self.config.population_size]

        # Update generation counter
        self.generation += 1

        # Update tracking
        self._update_best()
        self._record_fitness_history()

    def get_average_fitness(self) -> float:
        """
        Calculate average fitness of population

        Returns:
            Average fitness score
        """
        if not self.individuals:
            return 0.0

        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    def get_fitness_stats(self) -> dict[str, float]:
        """
        Get fitness statistics for population

        Returns:
            Dictionary with min, max, mean, std fitness
        """
        if not self.individuals:
            return {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
            }

        fitness_values = [ind.fitness for ind in self.individuals]
        mean = sum(fitness_values) / len(fitness_values)
        variance = sum((f - mean) ** 2 for f in fitness_values) / len(fitness_values)
        std = variance ** 0.5

        return {
            "min": min(fitness_values),
            "max": max(fitness_values),
            "mean": mean,
            "std": std,
        }

    def _update_best(self) -> None:
        """Update best individual tracker"""
        if not self.individuals:
            return

        current_best = max(self.individuals, key=lambda ind: ind.fitness)

        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best

    def _record_fitness_history(self) -> None:
        """Record fitness statistics for current generation"""
        stats = self.get_fitness_stats()
        stats["generation"] = self.generation
        stats["diversity"] = self.calculate_diversity()
        self.fitness_history.append(stats)

    def _genome_difference(self, genome1: dict[str, Any], genome2: dict[str, Any]) -> float:
        """
        Calculate difference between two genomes

        Args:
            genome1: First genome
            genome2: Second genome

        Returns:
            Difference score (0.0-1.0)
        """
        all_keys = set(genome1.keys()) | set(genome2.keys())

        if not all_keys:
            return 0.0

        differences = 0

        for key in all_keys:
            val1 = genome1.get(key)
            val2 = genome2.get(key)

            if val1 != val2:
                differences += 1

        return differences / len(all_keys)
