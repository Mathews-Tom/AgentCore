"""
Metrics Calculator for Bounded Context Reasoning.

Calculates compute savings and efficiency metrics by comparing bounded
context strategy against traditional reasoning approaches.

The calculator demonstrates the computational advantage of maintaining
fixed context windows versus growing context in traditional approaches.
"""

from __future__ import annotations

import structlog

from ..models.reasoning_models import BoundedContextConfig, BoundedContextResult

logger = structlog.get_logger()


class MetricsCalculator:
    """
    Calculates performance metrics for bounded context reasoning.

    Compares bounded context approach (linear complexity, O(N)) against
    traditional reasoning approach (quadratic complexity, O(N²)) to
    demonstrate compute savings.
    """

    @staticmethod
    def calculate_traditional_tokens(
        query_tokens: int,
        iterations: int,
        avg_generation_tokens: int,
    ) -> int:
        """
        Calculate tokens used in traditional (unbounded) reasoning.

        Traditional approach maintains full context at each iteration,
        leading to quadratic token growth:
        - Iteration 0: query + generation_0
        - Iteration 1: query + generation_0 + generation_1
        - Iteration 2: query + generation_0 + generation_1 + generation_2
        - ...

        Args:
            query_tokens: Initial query token count
            iterations: Number of iterations performed
            avg_generation_tokens: Average tokens generated per iteration

        Returns:
            Total tokens that would be used in traditional approach
        """
        total = 0

        for i in range(iterations):
            # Context size grows linearly with each iteration
            # Iteration i processes: query + all previous generations
            context_size = query_tokens + (i * avg_generation_tokens)
            generation_size = avg_generation_tokens

            # Total tokens = input context + output generation
            iteration_tokens = context_size + generation_size
            total += iteration_tokens

        logger.debug(
            "traditional_tokens_calculated",
            query_tokens=query_tokens,
            iterations=iterations,
            avg_generation_tokens=avg_generation_tokens,
            total_tokens=total,
        )

        return total

    @staticmethod
    def calculate_bounded_tokens(
        chunk_size: int,
        carryover_size: int,
        iterations: int,
    ) -> int:
        """
        Calculate tokens used in bounded context reasoning.

        Bounded approach maintains fixed context at each iteration:
        - Iteration 0: up to chunk_size tokens
        - Iteration 1+: up to chunk_size tokens (query + carryover)

        Args:
            chunk_size: Maximum tokens per iteration
            carryover_size: Tokens carried between iterations
            iterations: Number of iterations performed

        Returns:
            Total tokens used in bounded approach
        """
        if iterations == 0:
            return 0

        # First iteration: full chunk_size
        first_iteration = chunk_size

        # Subsequent iterations: chunk_size (includes carryover in context)
        subsequent_iterations = (iterations - 1) * chunk_size

        total = first_iteration + subsequent_iterations

        logger.debug(
            "bounded_tokens_calculated",
            chunk_size=chunk_size,
            carryover_size=carryover_size,
            iterations=iterations,
            total_tokens=total,
        )

        return total

    @staticmethod
    def calculate_compute_savings(
        bounded_result: BoundedContextResult,
        config: BoundedContextConfig,
        query_tokens: int,
    ) -> float:
        """
        Calculate compute savings percentage for bounded context reasoning.

        Compares actual bounded context usage against theoretical traditional
        reasoning cost.

        Args:
            bounded_result: Result from bounded context execution
            config: Bounded context configuration used
            query_tokens: Token count of original query

        Returns:
            Compute savings percentage (0-100)
        """
        # Actual bounded context tokens
        bounded_tokens = bounded_result.total_tokens
        iterations = bounded_result.total_iterations

        # Calculate average generation tokens from actual result
        if iterations == 0:
            avg_generation = 0
        else:
            total_generated = sum(it.metrics.tokens for it in bounded_result.iterations)
            avg_generation = total_generated // iterations

        # Calculate traditional approach tokens
        traditional_tokens = MetricsCalculator.calculate_traditional_tokens(
            query_tokens=query_tokens,
            iterations=iterations,
            avg_generation_tokens=avg_generation,
        )

        # Avoid division by zero
        if traditional_tokens == 0:
            return 0.0

        # Calculate savings percentage
        savings_pct = ((traditional_tokens - bounded_tokens) / traditional_tokens) * 100

        # Ensure non-negative (bounded should always use <= traditional)
        savings_pct = max(0.0, savings_pct)

        logger.info(
            "compute_savings_calculated",
            bounded_tokens=bounded_tokens,
            traditional_tokens=traditional_tokens,
            savings_pct=savings_pct,
            iterations=iterations,
        )

        return savings_pct

    @staticmethod
    def calculate_efficiency_ratio(
        bounded_tokens: int,
        traditional_tokens: int,
    ) -> float:
        """
        Calculate efficiency ratio (bounded/traditional).

        Args:
            bounded_tokens: Tokens used in bounded approach
            traditional_tokens: Tokens in traditional approach

        Returns:
            Efficiency ratio (lower is better, e.g., 0.35 = 35% of traditional)
        """
        if traditional_tokens == 0:
            return 1.0

        ratio = bounded_tokens / traditional_tokens

        logger.debug(
            "efficiency_ratio_calculated",
            bounded_tokens=bounded_tokens,
            traditional_tokens=traditional_tokens,
            ratio=ratio,
        )

        return ratio

    @staticmethod
    def estimate_max_reasoning_capacity(
        chunk_size: int,
        carryover_size: int,
        max_iterations: int,
    ) -> int:
        """
        Estimate maximum reasoning capacity (total tokens processable).

        Bounded context total capacity formula:
        chunk_size + (max_iterations - 1) × (chunk_size - carryover_size)

        This represents the total unique content that can be processed
        across all iterations.

        Args:
            chunk_size: Tokens per iteration
            carryover_size: Tokens carried forward
            max_iterations: Maximum iterations allowed

        Returns:
            Maximum reasoning capacity in tokens
        """
        if max_iterations == 0:
            return 0

        # First iteration: full chunk_size
        # Each subsequent iteration adds: chunk_size - carryover_size new content
        capacity = chunk_size + (max_iterations - 1) * (chunk_size - carryover_size)

        logger.debug(
            "reasoning_capacity_estimated",
            chunk_size=chunk_size,
            carryover_size=carryover_size,
            max_iterations=max_iterations,
            capacity=capacity,
        )

        return capacity
        return capacity
