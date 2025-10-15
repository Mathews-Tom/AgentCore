"""
Prometheus metrics for Bounded Context Reasoning.

Tracks performance, usage, and compute savings metrics for the reasoning engine.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Request counters
reasoning_requests_total = Counter(
    "reasoning_bounded_context_requests_total",
    "Total number of reasoning requests",
    ["status"],  # success, error
)

reasoning_errors_total = Counter(
    "reasoning_bounded_context_errors_total",
    "Total number of reasoning errors by type",
    ["error_type"],  # validation_error, llm_error, timeout_error, internal_error
)

reasoning_llm_failures_total = Counter(
    "reasoning_bounded_context_llm_failures_total",
    "Total number of LLM service failures",
)

# Performance histograms
reasoning_duration_seconds = Histogram(
    "reasoning_bounded_context_duration_seconds",
    "Duration of reasoning requests in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

reasoning_tokens_total = Histogram(
    "reasoning_bounded_context_tokens_total",
    "Total tokens used per reasoning request",
    buckets=[100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000],
)

reasoning_compute_savings_pct = Histogram(
    "reasoning_bounded_context_compute_savings_pct",
    "Compute savings percentage compared to traditional reasoning",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)

reasoning_iterations_total = Histogram(
    "reasoning_bounded_context_iterations_total",
    "Number of iterations per reasoning request",
    buckets=[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50],
)


def record_reasoning_request(
    status: str,
    duration_seconds: float,
    total_tokens: int,
    compute_savings_pct: float,
    total_iterations: int,
) -> None:
    """
    Record metrics for a completed reasoning request.

    Args:
        status: Request status ('success' or 'error')
        duration_seconds: Request duration in seconds
        total_tokens: Total tokens used
        compute_savings_pct: Compute savings percentage
        total_iterations: Number of reasoning iterations
    """
    reasoning_requests_total.labels(status=status).inc()
    reasoning_duration_seconds.observe(duration_seconds)
    reasoning_tokens_total.observe(total_tokens)
    reasoning_compute_savings_pct.observe(compute_savings_pct)
    reasoning_iterations_total.observe(total_iterations)


def record_reasoning_error(error_type: str) -> None:
    """
    Record a reasoning error.

    Args:
        error_type: Type of error (validation_error, llm_error, timeout_error, internal_error)
    """
    reasoning_errors_total.labels(error_type=error_type).inc()


def record_llm_failure() -> None:
    """Record an LLM service failure."""
    reasoning_llm_failures_total.inc()
