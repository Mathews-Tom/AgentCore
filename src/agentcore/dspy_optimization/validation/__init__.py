"""
Algorithm validation and benchmarking

Validates DSPy optimization algorithms against research benchmarks with
statistical significance testing and reproducibility validation.
"""

from agentcore.dspy_optimization.validation.benchmarks import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
)
from agentcore.dspy_optimization.validation.baselines import (
    BaselineComparison,
    BaselineOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
)
from agentcore.dspy_optimization.validation.reproducibility import (
    ReproducibilityAnalysis,
    ReproducibilityResult,
    ReproducibilityValidator,
)
from agentcore.dspy_optimization.validation.validator import (
    AlgorithmValidator,
    ValidationReport,
    ValidationResult,
)

__all__ = [
    # Benchmarks
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
    # Baselines
    "BaselineComparison",
    "BaselineOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    # Reproducibility
    "ReproducibilityAnalysis",
    "ReproducibilityResult",
    "ReproducibilityValidator",
    # Validation
    "AlgorithmValidator",
    "ValidationReport",
    "ValidationResult",
]
