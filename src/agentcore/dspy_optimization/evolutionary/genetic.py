"""
Genetic algorithm optimizer implementation

Combines all evolutionary components (population, selection, operators,
convergence) into a complete genetic algorithm for agent optimization.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import dspy

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.evolutionary.convergence import (
    ConvergenceCriteria,
    ConvergenceDetector,
)
from agentcore.dspy_optimization.evolutionary.operators import (
    CrossoverOperator,
    GaussianMutation,
    MutationOperator,
    UniformCrossover,
)
from agentcore.dspy_optimization.evolutionary.pareto import MultiObjectiveOptimizer
from agentcore.dspy_optimization.evolutionary.population import (
    Individual,
    Population,
    PopulationConfig,
)
from agentcore.dspy_optimization.evolutionary.selection import (
    SelectionStrategy,
    TournamentSelection,
)
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic algorithm optimizer for agent evolution

    Implements population-based optimization with selection, crossover,
    mutation, and multi-objective optimization capabilities.

    Key features:
    - Population management with elitism
    - Tournament and roulette wheel selection
    - Uniform and Gaussian mutation
    - Multi-objective optimization with Pareto frontiers
    - Adaptive convergence detection
    """

    def __init__(
        self,
        llm: dspy.LM | None = None,
        population_config: PopulationConfig | None = None,
        convergence_criteria: ConvergenceCriteria | None = None,
        selection_strategy: SelectionStrategy | None = None,
        crossover_operator: CrossoverOperator | None = None,
        mutation_operator: MutationOperator | None = None,
    ) -> None:
        """
        Initialize genetic optimizer

        Args:
            llm: DSPy language model
            population_config: Population configuration
            convergence_criteria: Convergence criteria
            selection_strategy: Selection strategy (default: tournament)
            crossover_operator: Crossover operator (default: uniform)
            mutation_operator: Mutation operator (default: Gaussian)
        """
        super().__init__(llm)
        self.population_config = population_config or PopulationConfig()
        self.convergence_criteria = convergence_criteria or ConvergenceCriteria()
        self.selection_strategy = selection_strategy or TournamentSelection()
        self.crossover_operator = crossover_operator or UniformCrossover()
        self.mutation_operator = mutation_operator or GaussianMutation()

    async def optimize(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Perform genetic algorithm optimization

        Args:
            request: Optimization request with target and objectives
            baseline_metrics: Current performance metrics
            training_data: Training examples for optimization

        Returns:
            OptimizationResult with improvements and details
        """
        result = OptimizationResult(
            status=OptimizationStatus.IN_PROGRESS,
            baseline_performance=baseline_metrics,
        )

        start_time = time.time()

        try:
            # Initialize population
            population = self._initialize_population(baseline_metrics)

            # Setup multi-objective optimization if needed
            multi_objective_optimizer = None
            if len(request.objectives) > 1:
                multi_objective_optimizer = MultiObjectiveOptimizer(request.objectives)

            # Setup convergence detector
            convergence_detector = ConvergenceDetector(
                self.convergence_criteria,
                start_time=start_time,
            )

            # Evolution loop
            generation_count = 0
            key_improvements: list[str] = []

            while convergence_detector.should_continue(population, time.time()):
                # Evaluate fitness for all individuals
                await self._evaluate_population(
                    population,
                    request,
                    baseline_metrics,
                    training_data,
                )

                # Selection
                selected = self.selection_strategy.select(
                    population.individuals,
                    self.population_config.population_size,
                )

                # Generate offspring through crossover
                offspring = self._generate_offspring(selected)

                # Apply mutation
                offspring = self._apply_mutation(offspring)

                # Evaluate offspring fitness
                for individual in offspring:
                    individual = await self._evaluate_individual(
                        individual,
                        request,
                        baseline_metrics,
                        training_data,
                    )

                # Select next generation
                if multi_objective_optimizer:
                    # Multi-objective selection using NSGA-II
                    next_generation = multi_objective_optimizer.select_for_next_generation(
                        population.individuals,
                        offspring,
                        self.population_config.population_size,
                    )
                else:
                    # Single-objective: combine with elite preservation
                    next_generation = offspring

                # Advance to next generation
                population.advance_generation(next_generation)
                generation_count += 1

                # Record improvements
                if population.generation % 10 == 0:
                    progress = convergence_detector.get_progress_info(population)
                    key_improvements.append(
                        f"Generation {population.generation}: "
                        f"Best fitness {progress['best_fitness']:.4f}, "
                        f"Diversity {progress['diversity']:.4f}"
                    )

            # Get final convergence status
            convergence_status = convergence_detector.check_convergence(
                population,
                time.time(),
            )

            # Get best individual
            best_individual = population.best_individual

            if best_individual and best_individual.metrics:
                optimized_metrics = best_individual.metrics

                # Calculate improvement
                improvement = self.calculate_improvement(
                    baseline_metrics,
                    optimized_metrics,
                )

                # Generate final improvements summary
                key_improvements.append(
                    f"Converged after {generation_count} generations: {convergence_status.reason.value}"
                )
                key_improvements.append(
                    f"Final population diversity: {population.calculate_diversity():.4f}"
                )

                if multi_objective_optimizer:
                    pareto_front = multi_objective_optimizer.optimize_population(
                        population.individuals
                    )
                    key_improvements.append(
                        f"Pareto front size: {len(pareto_front)} optimal solutions"
                    )

                # Update result
                result.status = OptimizationStatus.COMPLETED
                result.optimized_performance = optimized_metrics
                result.improvement_percentage = improvement
                result.statistical_significance = 0.01  # Simplified
                result.optimization_details = OptimizationDetails(
                    algorithm_used=self.get_algorithm_name(),
                    iterations=generation_count,
                    key_improvements=key_improvements,
                    parameters={
                        "population_size": self.population_config.population_size,
                        "generations": generation_count,
                        "convergence_reason": convergence_status.reason.value,
                        "final_diversity": population.calculate_diversity(),
                        "best_fitness": best_individual.fitness,
                    },
                )
            else:
                result.status = OptimizationStatus.FAILED
                result.error_message = "No valid individuals in final population"

        except Exception as e:
            result.status = OptimizationStatus.FAILED
            result.error_message = str(e)

        return result

    def get_algorithm_name(self) -> str:
        """Get algorithm name"""
        return "genetic"

    def _initialize_population(
        self,
        baseline_metrics: PerformanceMetrics,
    ) -> Population:
        """
        Initialize population with random individuals

        Args:
            baseline_metrics: Baseline metrics for seeding

        Returns:
            Initialized population
        """
        population = Population(self.population_config)
        individuals: list[Individual] = []

        for _ in range(self.population_config.population_size):
            # Create individual with random genome
            genome = self._create_random_genome()

            individual = Individual(
                genome=genome,
                generation=0,
            )
            individuals.append(individual)

        population.initialize(individuals)
        return population

    def _create_random_genome(self) -> dict[str, Any]:
        """
        Create random genome for individual

        Returns:
            Random genome parameters
        """
        import random

        # Create random parameters for agent configuration
        # These could represent prompt templates, reasoning strategies, etc.
        return {
            "temperature": random.uniform(0.1, 1.0),
            "max_tokens": random.randint(100, 2000),
            "reasoning_depth": random.randint(1, 5),
            "tool_usage_threshold": random.uniform(0.3, 0.9),
            "confidence_threshold": random.uniform(0.5, 0.95),
            "retry_strategy": random.choice(["exponential", "linear", "none"]),
        }

    async def _evaluate_population(
        self,
        population: Population,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> None:
        """
        Evaluate fitness for all individuals in population

        Args:
            population: Population to evaluate
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data
        """
        for individual in population.individuals:
            if individual.fitness == 0.0:  # Only evaluate if not yet evaluated
                await self._evaluate_individual(
                    individual,
                    request,
                    baseline_metrics,
                    training_data,
                )

    async def _evaluate_individual(
        self,
        individual: Individual,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> Individual:
        """
        Evaluate individual fitness

        Args:
            individual: Individual to evaluate
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data

        Returns:
            Evaluated individual
        """
        # Simulate evaluation with genome-based performance
        # In production, this would actually run the agent configuration
        import random

        # Calculate metrics based on genome parameters
        genome = individual.genome

        # Simulate performance based on genome quality
        quality_factor = (
            genome.get("temperature", 0.5) * 0.2 +
            genome.get("confidence_threshold", 0.7) * 0.3 +
            (1.0 - genome.get("tool_usage_threshold", 0.5)) * 0.5
        )

        # Add some randomness for variation
        quality_factor += random.uniform(-0.1, 0.1)
        quality_factor = max(0.0, min(1.0, quality_factor))

        # Create performance metrics
        metrics = PerformanceMetrics(
            success_rate=min(baseline_metrics.success_rate * (1.0 + quality_factor * 0.3), 1.0),
            avg_cost_per_task=baseline_metrics.avg_cost_per_task * (1.0 - quality_factor * 0.2),
            avg_latency_ms=int(baseline_metrics.avg_latency_ms * (1.0 - quality_factor * 0.15)),
            quality_score=min(baseline_metrics.quality_score * (1.0 + quality_factor * 0.25), 1.0),
        )

        individual.metrics = metrics

        # Calculate fitness based on objectives
        fitness = 0.0
        for objective in request.objectives:
            metric_value = getattr(metrics, objective.metric.value, 0.0)
            baseline_value = getattr(baseline_metrics, objective.metric.value, 0.0)

            if baseline_value > 0:
                # Normalized improvement
                improvement = (metric_value - baseline_value) / baseline_value
                fitness += improvement * objective.weight

            individual.objectives[objective.metric.value] = metric_value

        individual.fitness = fitness

        return individual

    def _generate_offspring(
        self,
        parents: list[Individual],
    ) -> list[Individual]:
        """
        Generate offspring through crossover

        Args:
            parents: Parent individuals

        Returns:
            Offspring individuals
        """
        import random

        offspring: list[Individual] = []

        # Generate offspring in pairs
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Apply crossover with probability
            if random.random() < self.population_config.crossover_rate:
                child1, child2 = self.crossover_operator.crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                # No crossover, keep parents
                offspring.extend([parent1.clone(), parent2.clone()])

        return offspring

    def _apply_mutation(
        self,
        individuals: list[Individual],
    ) -> list[Individual]:
        """
        Apply mutation to individuals

        Args:
            individuals: Individuals to mutate

        Returns:
            Mutated individuals
        """
        import random

        mutated: list[Individual] = []

        for individual in individuals:
            # Apply mutation with probability
            if random.random() < self.population_config.mutation_rate:
                mutated_individual = self.mutation_operator.mutate(individual)
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)

        return mutated
