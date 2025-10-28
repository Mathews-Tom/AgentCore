"""
Convergence criteria and detection for genetic algorithms

Implements fitness plateau detection, generation limits,
and target fitness threshold checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.evolutionary.population import Population


class ConvergenceReason(str, Enum):
    """Reason for convergence"""

    FITNESS_PLATEAU = "fitness_plateau"
    GENERATION_LIMIT = "generation_limit"
    TARGET_FITNESS = "target_fitness"
    TIME_LIMIT = "time_limit"
    DIVERSITY_LOSS = "diversity_loss"
    NO_IMPROVEMENT = "no_improvement"


@dataclass
class ConvergenceStatus:
    """Status of convergence detection"""

    converged: bool
    reason: ConvergenceReason | None
    generation: int
    best_fitness: float
    message: str


class ConvergenceCriteria(BaseModel):
    """Configuration for convergence detection"""

    max_generations: int = Field(default=100, ge=1)
    target_fitness: float | None = Field(default=None, ge=0.0)
    plateau_generations: int = Field(default=10, ge=1)
    plateau_threshold: float = Field(default=0.001, ge=0.0)
    min_diversity: float = Field(default=0.05, ge=0.0, le=1.0)
    max_time_seconds: int | None = Field(default=None, ge=1)


class FitnessPlateauDetector:
    """
    Detects fitness plateau in genetic algorithm

    Monitors fitness improvements over generations and
    detects when optimization has stagnated.
    """

    def __init__(self, criteria: ConvergenceCriteria) -> None:
        """
        Initialize plateau detector

        Args:
            criteria: Convergence criteria
        """
        self.criteria = criteria
        self.fitness_history: list[float] = []
        self.best_fitness: float = float('-inf')
        self.generations_without_improvement: int = 0

    def update(self, population: Population) -> None:
        """
        Update detector with new generation

        Args:
            population: Current population
        """
        current_best = max(ind.fitness for ind in population.individuals) if population.individuals else 0.0

        self.fitness_history.append(current_best)

        # Check for improvement
        if current_best > self.best_fitness + self.criteria.plateau_threshold:
            self.best_fitness = current_best
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

    def has_converged(self) -> bool:
        """
        Check if fitness has plateaued

        Returns:
            True if plateau detected
        """
        return self.generations_without_improvement >= self.criteria.plateau_generations

    def get_plateau_duration(self) -> int:
        """
        Get number of generations without improvement

        Returns:
            Plateau duration in generations
        """
        return self.generations_without_improvement

    def get_improvement_rate(self, window: int = 10) -> float:
        """
        Calculate fitness improvement rate over recent window

        Args:
            window: Number of generations to analyze

        Returns:
            Improvement rate (fitness change per generation)
        """
        if len(self.fitness_history) < 2:
            return 0.0

        # Get recent history
        recent = self.fitness_history[-window:]

        if len(recent) < 2:
            return 0.0

        # Calculate linear trend
        initial = recent[0]
        final = recent[-1]

        return (final - initial) / len(recent)


class ConvergenceDetector:
    """
    Comprehensive convergence detection

    Monitors multiple convergence criteria including fitness plateau,
    generation limits, target fitness, diversity loss, and time limits.
    """

    def __init__(
        self,
        criteria: ConvergenceCriteria,
        start_time: float | None = None,
    ) -> None:
        """
        Initialize convergence detector

        Args:
            criteria: Convergence criteria
            start_time: Start time for time limit tracking
        """
        self.criteria = criteria
        self.start_time = start_time
        self.plateau_detector = FitnessPlateauDetector(criteria)

    def check_convergence(
        self,
        population: Population,
        current_time: float | None = None,
    ) -> ConvergenceStatus:
        """
        Check all convergence criteria

        Args:
            population: Current population
            current_time: Current time for time limit checking

        Returns:
            Convergence status
        """
        # Update plateau detector
        self.plateau_detector.update(population)

        best_fitness = self.plateau_detector.best_fitness
        generation = population.generation

        # Check generation limit
        if generation >= self.criteria.max_generations:
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.GENERATION_LIMIT,
                generation=generation,
                best_fitness=best_fitness,
                message=f"Reached maximum generations: {self.criteria.max_generations}",
            )

        # Check target fitness
        if self.criteria.target_fitness is not None:
            if best_fitness >= self.criteria.target_fitness:
                return ConvergenceStatus(
                    converged=True,
                    reason=ConvergenceReason.TARGET_FITNESS,
                    generation=generation,
                    best_fitness=best_fitness,
                    message=f"Reached target fitness: {self.criteria.target_fitness:.4f}",
                )

        # Check fitness plateau
        if self.plateau_detector.has_converged():
            plateau_duration = self.plateau_detector.get_plateau_duration()
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.FITNESS_PLATEAU,
                generation=generation,
                best_fitness=best_fitness,
                message=f"Fitness plateau detected: {plateau_duration} generations without improvement",
            )

        # Check diversity loss
        diversity = population.calculate_diversity()
        if diversity < self.criteria.min_diversity:
            return ConvergenceStatus(
                converged=True,
                reason=ConvergenceReason.DIVERSITY_LOSS,
                generation=generation,
                best_fitness=best_fitness,
                message=f"Diversity too low: {diversity:.4f} < {self.criteria.min_diversity:.4f}",
            )

        # Check time limit
        if self.criteria.max_time_seconds is not None and current_time is not None and self.start_time is not None:
            elapsed = current_time - self.start_time
            if elapsed >= self.criteria.max_time_seconds:
                return ConvergenceStatus(
                    converged=True,
                    reason=ConvergenceReason.TIME_LIMIT,
                    generation=generation,
                    best_fitness=best_fitness,
                    message=f"Time limit exceeded: {elapsed:.1f}s >= {self.criteria.max_time_seconds}s",
                )

        # Not converged
        return ConvergenceStatus(
            converged=False,
            reason=None,
            generation=generation,
            best_fitness=best_fitness,
            message="Optimization in progress",
        )

    def should_continue(
        self,
        population: Population,
        current_time: float | None = None,
    ) -> bool:
        """
        Check if optimization should continue

        Args:
            population: Current population
            current_time: Current time

        Returns:
            True if optimization should continue
        """
        status = self.check_convergence(population, current_time)
        return not status.converged

    def get_progress_info(self, population: Population) -> dict[str, float | int]:
        """
        Get progress information

        Args:
            population: Current population

        Returns:
            Dictionary with progress metrics
        """
        improvement_rate = self.plateau_detector.get_improvement_rate()
        diversity = population.calculate_diversity()

        return {
            "generation": population.generation,
            "max_generations": self.criteria.max_generations,
            "progress_percent": (population.generation / self.criteria.max_generations) * 100,
            "best_fitness": self.plateau_detector.best_fitness,
            "improvement_rate": improvement_rate,
            "diversity": diversity,
            "plateau_duration": self.plateau_detector.get_plateau_duration(),
        }


class AdaptiveConvergence:
    """
    Adaptive convergence detection

    Adjusts convergence criteria dynamically based on
    optimization progress and characteristics.
    """

    def __init__(self, base_criteria: ConvergenceCriteria) -> None:
        """
        Initialize adaptive convergence

        Args:
            base_criteria: Base convergence criteria
        """
        self.base_criteria = base_criteria
        self.adjusted_criteria = base_criteria.model_copy()
        self.adjustment_history: list[dict[str, float | int]] = []

    def adjust_criteria(
        self,
        population: Population,
        detector: ConvergenceDetector,
    ) -> ConvergenceCriteria:
        """
        Adjust convergence criteria based on progress

        Args:
            population: Current population
            detector: Convergence detector

        Returns:
            Adjusted convergence criteria
        """
        progress = detector.get_progress_info(population)

        # If making good progress, be more lenient with plateau
        improvement_rate = progress.get("improvement_rate", 0.0)
        if improvement_rate > self.base_criteria.plateau_threshold * 2:
            self.adjusted_criteria.plateau_generations = (
                self.base_criteria.plateau_generations + 5
            )
        else:
            self.adjusted_criteria.plateau_generations = (
                self.base_criteria.plateau_generations
            )

        # If diversity is high, allow more generations
        diversity = progress.get("diversity", 0.0)
        if diversity > 0.5:
            self.adjusted_criteria.max_generations = (
                int(self.base_criteria.max_generations * 1.2)
            )
        else:
            self.adjusted_criteria.max_generations = (
                self.base_criteria.max_generations
            )

        # Record adjustment
        self.adjustment_history.append({
            "generation": population.generation,
            "plateau_generations": self.adjusted_criteria.plateau_generations,
            "max_generations": self.adjusted_criteria.max_generations,
            "improvement_rate": improvement_rate,
            "diversity": diversity,
        })

        return self.adjusted_criteria

    def get_adjustment_history(self) -> list[dict[str, float | int]]:
        """
        Get history of criteria adjustments

        Returns:
            List of adjustment records
        """
        return self.adjustment_history
