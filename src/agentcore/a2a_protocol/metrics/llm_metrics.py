"""Prometheus metrics for LLM operations.

Tracks all LLM operations for observability, cost monitoring, and performance analysis.
This module provides real-time metrics for:
- Request counts and success rates by provider/model
- Request latency percentiles
- Token usage tracking (prompt and completion)
- Error tracking by type
- Active request monitoring
- Governance violation tracking

All metrics follow Prometheus naming conventions and include appropriate labels
for aggregation and filtering.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, cast

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# Module-level cache to prevent duplicate registration
_metrics_cache: dict[str, Counter | Histogram | Gauge] = {}


def _get_or_create_counter(
    name: str, description: str, labelnames: list[str] | None = None
) -> Counter:
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


def _get_or_create_histogram(
    name: str,
    description: str,
    labelnames: list[str] | None = None,
    buckets: list[float] | None = None,
) -> Histogram:
    """Get existing histogram or create new one, handling duplicates."""
    if name in _metrics_cache:
        return cast(Histogram, _metrics_cache[name])

    try:
        histogram = Histogram(
            name, description, labelnames=labelnames or [], buckets=buckets or []
        )
        _metrics_cache[name] = histogram
        return histogram
    except ValueError:
        # Metric already exists in registry, find and cache it
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = cast(Histogram, collector)
                return cast(Histogram, collector)
        raise


def _get_or_create_gauge(
    name: str, description: str, labelnames: list[str] | None = None
) -> Gauge:
    """Get existing gauge or create new one, handling duplicates."""
    if name in _metrics_cache:
        return cast(Gauge, _metrics_cache[name])

    try:
        gauge = Gauge(name, description, labelnames or [])
        _metrics_cache[name] = gauge
        return gauge
    except ValueError:
        # Metric already exists in registry, find and cache it
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = cast(Gauge, collector)
                return cast(Gauge, collector)
        raise


# Lazy initialization - metrics are created on first use
_llm_requests_total: Counter | None = None
_llm_requests_duration_seconds: Histogram | None = None
_llm_tokens_total: Counter | None = None
_llm_errors_total: Counter | None = None
_llm_active_requests: Gauge | None = None
_llm_governance_violations_total: Counter | None = None


def _get_llm_requests_total() -> Counter:
    """Get or create LLM requests counter.

    Labels:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier (e.g., gpt-4.1-mini)
        status: Request status (success, error)
    """
    global _llm_requests_total
    if _llm_requests_total is None:
        _llm_requests_total = _get_or_create_counter(
            "llm_requests_total",
            "Total number of LLM requests by provider, model, and status",
            ["provider", "model", "status"],
        )
    return _llm_requests_total


def _get_llm_requests_duration_seconds() -> Histogram:
    """Get or create LLM request duration histogram.

    Labels:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier (e.g., gpt-4.1-mini)

    Buckets optimized for LLM latency patterns (100ms to 5min):
    - 0.1s: Very fast cached/simple responses
    - 0.5s: Fast responses
    - 1.0s: Normal responses
    - 2.5s: Longer responses
    - 5.0s: Complex responses
    - 10.0s: Very complex responses
    - 30.0s: Extended reasoning
    - 60.0s: Max typical timeout
    - 120.0s: Extended timeout
    - 300.0s: Maximum timeout
    """
    global _llm_requests_duration_seconds
    if _llm_requests_duration_seconds is None:
        _llm_requests_duration_seconds = _get_or_create_histogram(
            "llm_requests_duration_seconds",
            "Duration of LLM requests in seconds by provider and model",
            labelnames=["provider", "model"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )
    return _llm_requests_duration_seconds


def _get_llm_tokens_total() -> Counter:
    """Get or create LLM tokens counter.

    Labels:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier (e.g., gpt-4.1-mini)
        token_type: Token type (prompt, completion)
    """
    global _llm_tokens_total
    if _llm_tokens_total is None:
        _llm_tokens_total = _get_or_create_counter(
            "llm_tokens_total",
            "Total number of tokens used by provider, model, and type",
            ["provider", "model", "token_type"],
        )
    return _llm_tokens_total


def _get_llm_errors_total() -> Counter:
    """Get or create LLM errors counter.

    Labels:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier (e.g., gpt-4.1-mini)
        error_type: Error type (ProviderError, ProviderTimeoutError, ModelNotAllowedError, etc.)
    """
    global _llm_errors_total
    if _llm_errors_total is None:
        _llm_errors_total = _get_or_create_counter(
            "llm_errors_total",
            "Total number of LLM errors by provider, model, and error type",
            ["provider", "model", "error_type"],
        )
    return _llm_errors_total


def _get_llm_active_requests() -> Gauge:
    """Get or create LLM active requests gauge.

    Labels:
        provider: LLM provider (openai, anthropic, gemini)
    """
    global _llm_active_requests
    if _llm_active_requests is None:
        _llm_active_requests = _get_or_create_gauge(
            "llm_active_requests",
            "Number of currently active LLM requests by provider",
            ["provider"],
        )
    return _llm_active_requests


def _get_llm_governance_violations_total() -> Counter:
    """Get or create LLM governance violations counter.

    Labels:
        model: Model that was attempted (not in ALLOWED_MODELS)
        source_agent: Agent that attempted the request (for accountability)
    """
    global _llm_governance_violations_total
    if _llm_governance_violations_total is None:
        _llm_governance_violations_total = _get_or_create_counter(
            "llm_governance_violations_total",
            "Total number of governance violations (disallowed model attempts)",
            ["model", "source_agent"],
        )
    return _llm_governance_violations_total


# Public API - Recording functions for metrics


def record_llm_request(provider: str, model: str, status: str) -> None:
    """Record an LLM request completion.

    Args:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier
        status: Request status (success, error)
    """
    _get_llm_requests_total().labels(provider=provider, model=model, status=status).inc()


def record_llm_duration(provider: str, model: str, duration_seconds: float) -> None:
    """Record LLM request duration.

    Args:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier
        duration_seconds: Request duration in seconds
    """
    _get_llm_requests_duration_seconds().labels(provider=provider, model=model).observe(
        duration_seconds
    )


def record_llm_tokens(provider: str, model: str, token_type: str, count: int) -> None:
    """Record LLM token usage.

    Args:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier
        token_type: Token type (prompt, completion)
        count: Number of tokens
    """
    _get_llm_tokens_total().labels(
        provider=provider, model=model, token_type=token_type
    ).inc(count)


def record_llm_error(provider: str, model: str, error_type: str) -> None:
    """Record an LLM error.

    Args:
        provider: LLM provider (openai, anthropic, gemini)
        model: Model identifier
        error_type: Error type name (e.g., ProviderError, ProviderTimeoutError)
    """
    _get_llm_errors_total().labels(
        provider=provider, model=model, error_type=error_type
    ).inc()


def record_governance_violation(model: str, source_agent: str | None = None) -> None:
    """Record a governance violation (disallowed model attempt).

    Args:
        model: Model that was attempted
        source_agent: Agent that attempted the request (None if unknown)
    """
    agent = source_agent or "unknown"
    _get_llm_governance_violations_total().labels(model=model, source_agent=agent).inc()


@contextmanager
def track_active_requests(provider: str) -> Iterator[None]:
    """Context manager to track active requests.

    Increments gauge on entry, decrements on exit (even if exception occurs).

    Args:
        provider: LLM provider (openai, anthropic, gemini)

    Usage:
        with track_active_requests("openai"):
            # Make LLM request
            response = await client.complete(request)
    """
    gauge = _get_llm_active_requests()
    gauge.labels(provider=provider).inc()
    try:
        yield
    finally:
        gauge.labels(provider=provider).dec()


# Public API - Module-level accessors for backward compatibility
def __getattr__(name: str) -> Counter | Histogram | Gauge:
    """Module-level attribute access for lazy-initialized metrics."""
    if name == "llm_requests_total":
        return _get_llm_requests_total()
    elif name == "llm_requests_duration_seconds":
        return _get_llm_requests_duration_seconds()
    elif name == "llm_tokens_total":
        return _get_llm_tokens_total()
    elif name == "llm_errors_total":
        return _get_llm_errors_total()
    elif name == "llm_active_requests":
        return _get_llm_active_requests()
    elif name == "llm_governance_violations_total":
        return _get_llm_governance_violations_total()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
