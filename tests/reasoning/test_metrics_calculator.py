"""
Unit tests for MetricsCalculator.

Tests the static utility methods for calculating compute savings,
token usage, and efficiency metrics.
"""

from __future__ import annotations

from src.agentcore.reasoning.models.reasoning_models import (
    BoundedContextConfig,
    BoundedContextIterationResult,
    BoundedContextResult,
    IterationMetrics,
)
from src.agentcore.reasoning.services.metrics_calculator import MetricsCalculator


def test_calculate_traditional_tokens() -> None:
    """Test traditional token calculation with quadratic growth."""
    # With 3 iterations, avg 1000 tokens per iteration, query 100 tokens:
    # Iteration 0: 100 + 1000 = 1100
    # Iteration 1: (100 + 1000) + 1000 = 2100
    # Iteration 2: (100 + 1000 + 1000) + 1000 = 3100
    # Total: 1100 + 2100 + 3100 = 6300
    result = MetricsCalculator.calculate_traditional_tokens(
        query_tokens=100,
        iterations=3,
        avg_generation_tokens=1000,
    )
    assert result == 6300


def test_calculate_traditional_tokens_single_iteration() -> None:
    """Test traditional tokens with single iteration."""
    result = MetricsCalculator.calculate_traditional_tokens(
        query_tokens=100,
        iterations=1,
        avg_generation_tokens=500,
    )
    # Iteration 0: 100 + 500 = 600
    assert result == 600


def test_calculate_traditional_tokens_no_iterations() -> None:
    """Test traditional tokens with no iterations."""
    result = MetricsCalculator.calculate_traditional_tokens(
        query_tokens=100,
        iterations=0,
        avg_generation_tokens=500,
    )
    assert result == 0


def test_calculate_bounded_tokens() -> None:
    """Test bounded context token calculation with linear growth."""
    # With chunk_size=8192, carryover_size=4096, iterations=5:
    # Iteration 0: 8192
    # Iterations 1-4: 4 × 8192 = 32768
    # Total: 8192 + 32768 = 40960
    result = MetricsCalculator.calculate_bounded_tokens(
        chunk_size=8192,
        carryover_size=4096,
        iterations=5,
    )
    assert result == 40960


def test_calculate_bounded_tokens_single_iteration() -> None:
    """Test bounded tokens with single iteration."""
    result = MetricsCalculator.calculate_bounded_tokens(
        chunk_size=8192,
        carryover_size=4096,
        iterations=1,
    )
    assert result == 8192


def test_calculate_bounded_tokens_no_iterations() -> None:
    """Test bounded tokens with no iterations."""
    result = MetricsCalculator.calculate_bounded_tokens(
        chunk_size=8192,
        carryover_size=4096,
        iterations=0,
    )
    assert result == 0


def test_calculate_compute_savings() -> None:
    """Test compute savings calculation."""
    # Create mock result
    iterations = [
        BoundedContextIterationResult(
            content="Step 1",
            has_answer=False,
            answer=None,
            carryover=None,
            metrics=IterationMetrics(
                iteration=0,
                tokens=1000,
                has_answer=False,
                carryover_generated=True,
                execution_time_ms=100,
            ),
        ),
        BoundedContextIterationResult(
            content="Step 2",
            has_answer=False,
            answer=None,
            carryover=None,
            metrics=IterationMetrics(
                iteration=1,
                tokens=1000,
                has_answer=False,
                carryover_generated=True,
                execution_time_ms=100,
            ),
        ),
        BoundedContextIterationResult(
            content="Final",
            has_answer=True,
            answer="Answer",
            carryover=None,
            metrics=IterationMetrics(
                iteration=2,
                tokens=1000,
                has_answer=True,
                carryover_generated=False,
                execution_time_ms=100,
            ),
        ),
    ]

    result = BoundedContextResult(
        answer="Answer",
        iterations=iterations,
        total_tokens=3000,
        total_iterations=3,
        compute_savings_pct=0.0,  # Will be calculated
        carryover_compressions=2,
        execution_time_ms=300,
    )

    config = BoundedContextConfig(
        chunk_size=8192,
        carryover_size=4096,
        max_iterations=5,
    )

    savings = MetricsCalculator.calculate_compute_savings(
        bounded_result=result,
        config=config,
        query_tokens=100,
    )

    # Traditional: (100 + 1000) + (100 + 1000 + 1000) + (100 + 1000 + 1000 + 1000) = 1100 + 2100 + 3100 = 6300
    # Bounded: 3000
    # Savings: (6300 - 3000) / 6300 = 3300 / 6300 = 52.38%
    assert savings > 50.0
    assert savings < 55.0


def test_calculate_compute_savings_zero_traditional() -> None:
    """Test compute savings when traditional would be zero."""
    iterations = [
        BoundedContextIterationResult(
            content="Quick",
            has_answer=True,
            answer="Quick",
            carryover=None,
            metrics=IterationMetrics(
                iteration=0,
                tokens=100,
                has_answer=True,
                carryover_generated=False,
                execution_time_ms=50,
            ),
        ),
    ]

    # Edge case: zero iterations in result
    result = BoundedContextResult(
        answer="Quick",
        iterations=iterations,
        total_tokens=100,
        total_iterations=1,
        compute_savings_pct=0.0,
        carryover_compressions=0,
        execution_time_ms=50,
    )

    config = BoundedContextConfig(
        chunk_size=8192,
        carryover_size=4096,
        max_iterations=5,
    )

    savings = MetricsCalculator.calculate_compute_savings(
        bounded_result=result,
        config=config,
        query_tokens=100,
    )

    # Should not crash with zero traditional tokens
    assert savings >= 0.0


def test_calculate_efficiency_ratio() -> None:
    """Test efficiency ratio calculation."""
    ratio = MetricsCalculator.calculate_efficiency_ratio(
        bounded_tokens=3000,
        traditional_tokens=10000,
    )
    assert ratio == 0.3


def test_calculate_efficiency_ratio_equal() -> None:
    """Test efficiency ratio when tokens are equal."""
    ratio = MetricsCalculator.calculate_efficiency_ratio(
        bounded_tokens=5000,
        traditional_tokens=5000,
    )
    assert ratio == 1.0


def test_calculate_efficiency_ratio_zero_traditional() -> None:
    """Test efficiency ratio with zero traditional tokens."""
    ratio = MetricsCalculator.calculate_efficiency_ratio(
        bounded_tokens=1000,
        traditional_tokens=0,
    )
    assert ratio == 1.0  # Fallback


def test_estimate_max_reasoning_capacity() -> None:
    """Test max reasoning capacity estimation."""
    # chunk_size=8192, carryover_size=4096, max_iterations=5
    # First iteration: 8192
    # Each subsequent: 8192 - 4096 = 4096 new content
    # Total: 8192 + (4 × 4096) = 8192 + 16384 = 24576
    capacity = MetricsCalculator.estimate_max_reasoning_capacity(
        chunk_size=8192,
        carryover_size=4096,
        max_iterations=5,
    )
    assert capacity == 24576


def test_estimate_max_reasoning_capacity_single_iteration() -> None:
    """Test capacity with single iteration."""
    capacity = MetricsCalculator.estimate_max_reasoning_capacity(
        chunk_size=8192,
        carryover_size=4096,
        max_iterations=1,
    )
    assert capacity == 8192


def test_estimate_max_reasoning_capacity_zero_iterations() -> None:
    """Test capacity with zero iterations."""
    capacity = MetricsCalculator.estimate_max_reasoning_capacity(
        chunk_size=8192,
        carryover_size=4096,
        max_iterations=0,
    )
    assert capacity == 0


def test_estimate_max_reasoning_capacity_no_carryover() -> None:
    """Test capacity when carryover is very small."""
    # chunk_size=8192, carryover_size=512, max_iterations=10
    # First: 8192
    # Each subsequent: 8192 - 512 = 7680
    # Total: 8192 + (9 × 7680) = 8192 + 69120 = 77312
    capacity = MetricsCalculator.estimate_max_reasoning_capacity(
        chunk_size=8192,
        carryover_size=512,
        max_iterations=10,
    )
    assert capacity == 77312
