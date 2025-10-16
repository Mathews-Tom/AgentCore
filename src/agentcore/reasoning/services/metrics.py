"""
Prometheus metrics for Bounded Context Reasoning.

Tracks performance, usage, and compute savings metrics for the reasoning engine.
"""

from __future__ import annotations

from typing import cast

from prometheus_client import REGISTRY, Counter, Histogram

# Module-level cache to prevent duplicate registration
_metrics_cache: dict[str, Counter | Histogram] = {}


def _get_or_create_counter(name: str, description: str, labelnames: list[str] | None = None) -> Counter:
    """Get existing counter or create new one, handling duplicates."""
    if name in _metrics_cache:
        return cast(Counter, _metrics_cache[name])

    try:
        counter = Counter(name, description, labelnames or [])
        _metrics_cache[name] = counter
        return counter
    except ValueError:
        # Metric already exists in registry, find and cache it
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = cast(Counter, collector)
                return cast(Counter, collector)
        raise


def _get_or_create_histogram(name: str, description: str, buckets: list[float]) -> Histogram:
    """Get existing histogram or create new one, handling duplicates."""
    if name in _metrics_cache:
        return cast(Histogram, _metrics_cache[name])

    try:
        histogram = Histogram(name, description, buckets=buckets)
        _metrics_cache[name] = histogram
        return histogram
    except ValueError:
        # Metric already exists in registry, find and cache it
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = cast(Histogram, collector)
                return cast(Histogram, collector)
        raise


# Lazy initialization - metrics are created on first use
_reasoning_requests_total: Counter | None = None
_reasoning_errors_total: Counter | None = None
_reasoning_llm_failures_total: Counter | None = None
_reasoning_duration_seconds: Histogram | None = None
_reasoning_tokens_total: Histogram | None = None
_reasoning_compute_savings_pct: Histogram | None = None
_reasoning_iterations_total: Histogram | None = None


def _get_reasoning_requests_total() -> Counter:
    """Get or create reasoning requests counter."""
    global _reasoning_requests_total
    if _reasoning_requests_total is None:
        _reasoning_requests_total = _get_or_create_counter(
            "reasoning_bounded_context_requests_total",
            "Total number of reasoning requests",
            ["status"],
        )
    return _reasoning_requests_total


def _get_reasoning_errors_total() -> Counter:
    """Get or create reasoning errors counter."""
    global _reasoning_errors_total
    if _reasoning_errors_total is None:
        _reasoning_errors_total = _get_or_create_counter(
            "reasoning_bounded_context_errors_total",
            "Total number of reasoning errors by type",
            ["error_type"],
        )
    return _reasoning_errors_total


def _get_reasoning_llm_failures_total() -> Counter:
    """Get or create LLM failures counter."""
    global _reasoning_llm_failures_total
    if _reasoning_llm_failures_total is None:
        _reasoning_llm_failures_total = _get_or_create_counter(
            "reasoning_bounded_context_llm_failures_total",
            "Total number of LLM service failures",
        )
    return _reasoning_llm_failures_total


def _get_reasoning_duration_seconds() -> Histogram:
    """Get or create duration histogram."""
    global _reasoning_duration_seconds
    if _reasoning_duration_seconds is None:
        _reasoning_duration_seconds = _get_or_create_histogram(
            "reasoning_bounded_context_duration_seconds",
            "Duration of reasoning requests in seconds",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )
    return _reasoning_duration_seconds


def _get_reasoning_tokens_total() -> Histogram:
    """Get or create tokens histogram."""
    global _reasoning_tokens_total
    if _reasoning_tokens_total is None:
        _reasoning_tokens_total = _get_or_create_histogram(
            "reasoning_bounded_context_tokens_total",
            "Total tokens used per reasoning request",
            buckets=[100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000],
        )
    return _reasoning_tokens_total


def _get_reasoning_compute_savings_pct() -> Histogram:
    """Get or create compute savings histogram."""
    global _reasoning_compute_savings_pct
    if _reasoning_compute_savings_pct is None:
        _reasoning_compute_savings_pct = _get_or_create_histogram(
            "reasoning_bounded_context_compute_savings_pct",
            "Compute savings percentage compared to traditional reasoning",
            buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        )
    return _reasoning_compute_savings_pct


def _get_reasoning_iterations_total() -> Histogram:
    """Get or create iterations histogram."""
    global _reasoning_iterations_total
    if _reasoning_iterations_total is None:
        _reasoning_iterations_total = _get_or_create_histogram(
            "reasoning_bounded_context_iterations_total",
            "Number of iterations per reasoning request",
            buckets=[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50],
        )
    return _reasoning_iterations_total


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
    _get_reasoning_requests_total().labels(status=status).inc()
    _get_reasoning_duration_seconds().observe(duration_seconds)
    _get_reasoning_tokens_total().observe(total_tokens)
    _get_reasoning_compute_savings_pct().observe(compute_savings_pct)
    _get_reasoning_iterations_total().observe(total_iterations)


def record_reasoning_error(error_type: str) -> None:
    """
    Record a reasoning error.

    Args:
        error_type: Type of error (validation_error, llm_error, timeout_error, internal_error)
    """
    _get_reasoning_errors_total().labels(error_type=error_type).inc()


def record_llm_failure() -> None:
    """Record an LLM service failure."""
    _get_reasoning_llm_failures_total().inc()


# Public API - define module-level accessors for backward compatibility
# These call the getter functions which handle lazy initialization
def __getattr__(name: str):
    """Module-level attribute access for lazy-initialized metrics."""
    if name == "reasoning_requests_total":
        return _get_reasoning_requests_total()
    elif name == "reasoning_errors_total":
        return _get_reasoning_errors_total()
    elif name == "reasoning_llm_failures_total":
        return _get_reasoning_llm_failures_total()
    elif name == "reasoning_duration_seconds":
        return _get_reasoning_duration_seconds()
    elif name == "reasoning_tokens_total":
        return _get_reasoning_tokens_total()
    elif name == "reasoning_compute_savings_pct":
        return _get_reasoning_compute_savings_pct()
    elif name == "reasoning_iterations_total":
        return _get_reasoning_iterations_total()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
