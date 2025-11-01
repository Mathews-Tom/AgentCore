"""
Pareto frontier and multi-objective optimization

Implements NSGA-II inspired multi-objective optimization with
Pareto frontier calculation for trade-off analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.evolutionary.population import Individual
from agentcore.dspy_optimization.models import OptimizationObjective


@dataclass
class ParetoRank:
    """Pareto rank and crowding distance for individual"""

    individual: Individual
    rank: int
    crowding_distance: float


class ParetoFrontier:
    """
    Pareto frontier management

    Identifies and manages non-dominated solutions in
    multi-objective optimization problems.
    """

    def __init__(self, objectives: list[OptimizationObjective]) -> None:
        """
        Initialize Pareto frontier

        Args:
            objectives: Optimization objectives
        """
        self.objectives = objectives
        self.fronts: list[list[Individual]] = []

    def calculate_fronts(self, population: list[Individual]) -> list[list[Individual]]:
        """
        Calculate Pareto fronts using fast non-dominated sorting

        Args:
            population: Population to rank

        Returns:
            List of fronts, where front[0] is Pareto optimal
        """
        if not population:
            return []

        # Initialize domination relationships
        domination_count: dict[str, int] = {ind.id: 0 for ind in population}
        dominated_individuals: dict[str, list[Individual]] = {ind.id: [] for ind in population}

        # Calculate domination for all pairs
        for i, ind1 in enumerate(population):
            for ind2 in population[i + 1:]:
                if ind1.dominates(ind2, self.objectives):
                    dominated_individuals[ind1.id].append(ind2)
                    domination_count[ind2.id] += 1
                elif ind2.dominates(ind1, self.objectives):
                    dominated_individuals[ind2.id].append(ind1)
                    domination_count[ind1.id] += 1

        # First front: non-dominated individuals
        current_front = [
            ind for ind in population
            if domination_count[ind.id] == 0
        ]

        fronts = [current_front]

        # Calculate subsequent fronts
        while current_front:
            next_front: list[Individual] = []

            for ind in current_front:
                for dominated in dominated_individuals[ind.id]:
                    domination_count[dominated.id] -= 1

                    if domination_count[dominated.id] == 0:
                        next_front.append(dominated)

            if next_front:
                fronts.append(next_front)

            current_front = next_front

        self.fronts = fronts
        return fronts

    def get_pareto_optimal(self, population: list[Individual]) -> list[Individual]:
        """
        Get Pareto optimal solutions (first front)

        Args:
            population: Population to evaluate

        Returns:
            Pareto optimal individuals
        """
        fronts = self.calculate_fronts(population)
        return fronts[0] if fronts else []

    def calculate_crowding_distance(self, front: list[Individual]) -> dict[str, float]:
        """
        Calculate crowding distance for individuals in a front

        Crowding distance measures solution density in objective space.
        Higher values indicate more isolated solutions.

        Args:
            front: Pareto front individuals

        Returns:
            Dictionary mapping individual IDs to crowding distances
        """
        if not front:
            return {}

        distances: dict[str, float] = {ind.id: 0.0 for ind in front}

        # Calculate for each objective
        for obj in self.objectives:
            metric_name = obj.metric.value

            # Sort by objective value
            sorted_front = sorted(
                front,
                key=lambda ind: ind.objectives.get(metric_name, 0.0)
            )

            # Assign infinite distance to boundary points
            if len(sorted_front) > 0:
                distances[sorted_front[0].id] = float('inf')
                distances[sorted_front[-1].id] = float('inf')

            # Get objective range
            min_val = sorted_front[0].objectives.get(metric_name, 0.0)
            max_val = sorted_front[-1].objectives.get(metric_name, 0.0)
            objective_range = max_val - min_val

            if objective_range == 0:
                continue

            # Calculate crowding distance for middle points
            for i in range(1, len(sorted_front) - 1):
                prev_val = sorted_front[i - 1].objectives.get(metric_name, 0.0)
                next_val = sorted_front[i + 1].objectives.get(metric_name, 0.0)

                distance_contribution = (next_val - prev_val) / objective_range
                distances[sorted_front[i].id] += distance_contribution

        return distances

    def rank_population(self, population: list[Individual]) -> list[ParetoRank]:
        """
        Rank entire population with Pareto ranks and crowding distances

        Args:
            population: Population to rank

        Returns:
            List of ParetoRank objects with rankings
        """
        fronts = self.calculate_fronts(population)
        rankings: list[ParetoRank] = []

        for rank, front in enumerate(fronts):
            distances = self.calculate_crowding_distance(front)

            for individual in front:
                rankings.append(
                    ParetoRank(
                        individual=individual,
                        rank=rank,
                        crowding_distance=distances.get(individual.id, 0.0),
                    )
                )

        return rankings

    def select_by_nsga2(
        self,
        population: list[Individual],
        count: int,
    ) -> list[Individual]:
        """
        Select individuals using NSGA-II criterion

        Selects based on Pareto rank first, then crowding distance
        to maintain diversity.

        Args:
            population: Population to select from
            count: Number of individuals to select

        Returns:
            Selected individuals
        """
        rankings = self.rank_population(population)

        # Sort by rank, then by crowding distance (descending)
        rankings.sort(
            key=lambda r: (r.rank, -r.crowding_distance)
        )

        # Select top count individuals
        selected = [r.individual for r in rankings[:count]]

        return selected


class TradeOffAnalysis(BaseModel):
    """Analysis of trade-offs between objectives"""

    objective1: str
    objective2: str
    correlation: float = Field(ge=-1.0, le=1.0)
    pareto_front_size: int
    diversity_score: float = Field(ge=0.0, le=1.0)
    recommendations: list[str] = Field(default_factory=list)


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization with trade-off analysis

    Manages optimization with multiple competing objectives
    and provides trade-off analysis and recommendations.
    """

    def __init__(self, objectives: list[OptimizationObjective]) -> None:
        """
        Initialize multi-objective optimizer

        Args:
            objectives: Optimization objectives
        """
        self.objectives = objectives
        self.pareto_frontier = ParetoFrontier(objectives)

    def optimize_population(
        self,
        population: list[Individual],
    ) -> list[Individual]:
        """
        Optimize population for multiple objectives

        Args:
            population: Population to optimize

        Returns:
            Pareto optimal solutions
        """
        return self.pareto_frontier.get_pareto_optimal(population)

    def select_for_next_generation(
        self,
        population: list[Individual],
        offspring: list[Individual],
        population_size: int,
    ) -> list[Individual]:
        """
        Select next generation using NSGA-II

        Args:
            population: Current population
            offspring: Generated offspring
            population_size: Target population size

        Returns:
            Next generation individuals
        """
        # Combine parent and offspring populations
        combined = population + offspring

        # Select using NSGA-II criterion
        next_generation = self.pareto_frontier.select_by_nsga2(
            combined,
            population_size
        )

        return next_generation

    def analyze_tradeoffs(
        self,
        pareto_front: list[Individual],
    ) -> list[TradeOffAnalysis]:
        """
        Analyze trade-offs between objectives

        Args:
            pareto_front: Pareto optimal solutions

        Returns:
            Trade-off analysis for objective pairs
        """
        analyses: list[TradeOffAnalysis] = []

        if len(self.objectives) < 2:
            return analyses

        # Analyze each pair of objectives
        for i, obj1 in enumerate(self.objectives):
            for obj2 in self.objectives[i + 1:]:
                analysis = self._analyze_objective_pair(
                    pareto_front,
                    obj1,
                    obj2,
                )
                analyses.append(analysis)

        return analyses

    def get_balanced_solution(
        self,
        pareto_front: list[Individual],
    ) -> Individual | None:
        """
        Get most balanced solution from Pareto front

        Selects solution with best balance across all objectives
        weighted by their importance.

        Args:
            pareto_front: Pareto optimal solutions

        Returns:
            Most balanced individual or None
        """
        if not pareto_front:
            return None

        # Calculate weighted score for each individual
        best_individual = None
        best_score = float('-inf')

        for individual in pareto_front:
            score = 0.0

            for obj in self.objectives:
                metric_value = individual.objectives.get(obj.metric.value, 0.0)
                score += metric_value * obj.weight

            if score > best_score:
                best_score = score
                best_individual = individual

        return best_individual

    def _analyze_objective_pair(
        self,
        pareto_front: list[Individual],
        obj1: OptimizationObjective,
        obj2: OptimizationObjective,
    ) -> TradeOffAnalysis:
        """
        Analyze trade-off between two objectives

        Args:
            pareto_front: Pareto optimal solutions
            obj1: First objective
            obj2: Second objective

        Returns:
            Trade-off analysis
        """
        # Extract objective values
        values1 = [
            ind.objectives.get(obj1.metric.value, 0.0)
            for ind in pareto_front
        ]
        values2 = [
            ind.objectives.get(obj2.metric.value, 0.0)
            for ind in pareto_front
        ]

        # Calculate correlation
        correlation = self._calculate_correlation(values1, values2)

        # Calculate diversity
        diversity = self._calculate_diversity(values1, values2)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            obj1, obj2, correlation, diversity
        )

        return TradeOffAnalysis(
            objective1=obj1.metric.value,
            objective2=obj2.metric.value,
            correlation=correlation,
            pareto_front_size=len(pareto_front),
            diversity_score=diversity,
            recommendations=recommendations,
        )

    def _calculate_correlation(
        self,
        values1: list[float],
        values2: list[float],
    ) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(values1) < 2:
            return 0.0

        n = len(values1)
        mean1 = sum(values1) / n
        mean2 = sum(values2) / n

        numerator = sum(
            (v1 - mean1) * (v2 - mean2)
            for v1, v2 in zip(values1, values2)
        )

        denom1 = sum((v1 - mean1) ** 2 for v1 in values1) ** 0.5
        denom2 = sum((v2 - mean2) ** 2 for v2 in values2) ** 0.5

        if denom1 == 0 or denom2 == 0:
            return 0.0

        correlation = numerator / (denom1 * denom2)
        # Clamp to [-1, 1] to avoid floating point precision issues
        return max(-1.0, min(1.0, correlation))

    def _calculate_diversity(
        self,
        values1: list[float],
        values2: list[float],
    ) -> float:
        """Calculate diversity score based on value spread"""
        if not values1 or not values2:
            return 0.0

        range1 = max(values1) - min(values1)
        range2 = max(values2) - min(values2)

        # Normalize to [0, 1]
        diversity = (range1 + range2) / 2.0
        return min(diversity, 1.0)

    def _generate_recommendations(
        self,
        obj1: OptimizationObjective,
        obj2: OptimizationObjective,
        correlation: float,
        diversity: float,
    ) -> list[str]:
        """Generate optimization recommendations"""
        recommendations: list[str] = []

        if correlation < -0.5:
            recommendations.append(
                f"Strong trade-off exists between {obj1.metric.value} "
                f"and {obj2.metric.value} - prioritization required"
            )
        elif correlation > 0.5:
            recommendations.append(
                f"{obj1.metric.value} and {obj2.metric.value} can be "
                "optimized together without significant trade-offs"
            )

        if diversity < 0.3:
            recommendations.append(
                "Low solution diversity - consider relaxing constraints "
                "or adjusting objective weights"
            )
        elif diversity > 0.7:
            recommendations.append(
                "High solution diversity available - multiple good trade-off "
                "options exist"
            )

        return recommendations
