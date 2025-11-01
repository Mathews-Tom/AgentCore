"""
Evolutionary optimization module for genetic algorithm-based agent optimization

Provides population-based optimization with genetic operators (selection,
crossover, mutation) and multi-objective optimization with Pareto frontiers.
"""

from agentcore.dspy_optimization.evolutionary.genetic import GeneticOptimizer
from agentcore.dspy_optimization.evolutionary.population import (
    Individual,
    Population,
    PopulationConfig,
)
from agentcore.dspy_optimization.evolutionary.selection import (
    SelectionStrategy,
    TournamentSelection,
    RouletteWheelSelection,
    ElitismSelection,
)
from agentcore.dspy_optimization.evolutionary.operators import (
    CrossoverOperator,
    MutationOperator,
    UniformCrossover,
    SinglePointCrossover,
    GaussianMutation,
    UniformMutation,
)
from agentcore.dspy_optimization.evolutionary.pareto import (
    ParetoFrontier,
    MultiObjectiveOptimizer,
)
from agentcore.dspy_optimization.evolutionary.convergence import (
    ConvergenceCriteria,
    FitnessPlateauDetector,
)

__all__ = [
    "GeneticOptimizer",
    "Individual",
    "Population",
    "PopulationConfig",
    "SelectionStrategy",
    "TournamentSelection",
    "RouletteWheelSelection",
    "ElitismSelection",
    "CrossoverOperator",
    "MutationOperator",
    "UniformCrossover",
    "SinglePointCrossover",
    "GaussianMutation",
    "UniformMutation",
    "ParetoFrontier",
    "MultiObjectiveOptimizer",
    "ConvergenceCriteria",
    "FitnessPlateauDetector",
]
