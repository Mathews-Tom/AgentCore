"""Prometheus metrics for Coordination Service operations.

Tracks coordination and agent selection metrics for observability and performance analysis.
This module provides real-time metrics for:
- Signal registration counts by type and agent
- Active agent tracking
- Routing selections by strategy
- Signal registration latency
- Agent selection latency
- Overload prediction tracking

All metrics follow Prometheus naming conventions and include appropriate labels
for aggregation and filtering.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
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
            name, description, labelnames=labelnames or [], buckets=buckets
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
_coordination_signals_total: Counter | None = None
_coordination_agents_total: Gauge | None = None
_coordination_routing_selections_total: Counter | None = None
_coordination_signal_registration_duration_seconds: Histogram | None = None
_coordination_agent_selection_duration_seconds: Histogram | None = None
_coordination_overload_predictions_total: Counter | None = None


def get_coordination_signals_total() -> Counter:
    """Get or create coordination_signals_total counter.

    Tracks total signals registered by agent and signal type.
    Labels: agent_id, signal_type
    """
    global _coordination_signals_total
    if _coordination_signals_total is None:
        _coordination_signals_total = _get_or_create_counter(
            "coordination_signals_total",
            "Total coordination signals registered",
            labelnames=["agent_id", "signal_type"],
        )
    return _coordination_signals_total


def get_coordination_agents_total() -> Gauge:
    """Get or create coordination_agents_total gauge.

    Tracks number of active agents with coordination state.
    """
    global _coordination_agents_total
    if _coordination_agents_total is None:
        _coordination_agents_total = _get_or_create_gauge(
            "coordination_agents_total",
            "Number of active agents tracked by coordination service",
        )
    return _coordination_agents_total


def get_coordination_routing_selections_total() -> Counter:
    """Get or create coordination_routing_selections_total counter.

    Tracks total routing selections by strategy.
    Labels: strategy
    """
    global _coordination_routing_selections_total
    if _coordination_routing_selections_total is None:
        _coordination_routing_selections_total = _get_or_create_counter(
            "coordination_routing_selections_total",
            "Total agent routing selections by coordination service",
            labelnames=["strategy"],
        )
    return _coordination_routing_selections_total


def get_coordination_signal_registration_duration_seconds() -> Histogram:
    """Get or create coordination_signal_registration_duration_seconds histogram.

    Tracks signal registration latency in seconds.
    Labels: signal_type
    """
    global _coordination_signal_registration_duration_seconds
    if _coordination_signal_registration_duration_seconds is None:
        _coordination_signal_registration_duration_seconds = _get_or_create_histogram(
            "coordination_signal_registration_duration_seconds",
            "Duration of signal registration operations in seconds",
            labelnames=["signal_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
    return _coordination_signal_registration_duration_seconds


def get_coordination_agent_selection_duration_seconds() -> Histogram:
    """Get or create coordination_agent_selection_duration_seconds histogram.

    Tracks agent selection latency in seconds.
    Labels: strategy
    """
    global _coordination_agent_selection_duration_seconds
    if _coordination_agent_selection_duration_seconds is None:
        _coordination_agent_selection_duration_seconds = _get_or_create_histogram(
            "coordination_agent_selection_duration_seconds",
            "Duration of agent selection operations in seconds",
            labelnames=["strategy"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
    return _coordination_agent_selection_duration_seconds


def get_coordination_overload_predictions_total() -> Counter:
    """Get or create coordination_overload_predictions_total counter.

    Tracks overload predictions by agent and prediction outcome.
    Labels: agent_id, predicted
    """
    global _coordination_overload_predictions_total
    if _coordination_overload_predictions_total is None:
        _coordination_overload_predictions_total = _get_or_create_counter(
            "coordination_overload_predictions_total",
            "Total overload predictions made",
            labelnames=["agent_id", "predicted"],
        )
    return _coordination_overload_predictions_total


# Context managers for timing operations


@contextmanager
def track_signal_registration(signal_type: str) -> Iterator[None]:
    """Track signal registration duration.

    Args:
        signal_type: Type of signal being registered

    Yields:
        None

    Example:
        >>> with track_signal_registration("LOAD"):
        ...     # Signal registration code
        ...     pass
    """
    start_time = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start_time
        get_coordination_signal_registration_duration_seconds().labels(
            signal_type=signal_type
        ).observe(duration)


@contextmanager
def track_agent_selection(strategy: str) -> Iterator[None]:
    """Track agent selection duration.

    Args:
        strategy: Routing strategy being used

    Yields:
        None

    Example:
        >>> with track_agent_selection("ripple_coordination"):
        ...     # Agent selection code
        ...     pass
    """
    start_time = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start_time
        get_coordination_agent_selection_duration_seconds().labels(
            strategy=strategy
        ).observe(duration)


# Helper functions for incrementing metrics


def increment_signal_count(agent_id: str, signal_type: str) -> None:
    """Increment signal registration counter.

    Args:
        agent_id: Agent identifier
        signal_type: Type of signal
    """
    get_coordination_signals_total().labels(
        agent_id=agent_id, signal_type=signal_type
    ).inc()


def set_active_agents(count: int) -> None:
    """Set the number of active agents.

    Args:
        count: Number of active agents
    """
    get_coordination_agents_total().set(count)


def increment_routing_selection(strategy: str) -> None:
    """Increment routing selection counter.

    Args:
        strategy: Routing strategy used
    """
    get_coordination_routing_selections_total().labels(strategy=strategy).inc()


def increment_overload_prediction(agent_id: str, predicted: bool) -> None:
    """Increment overload prediction counter.

    Args:
        agent_id: Agent identifier
        predicted: Whether overload was predicted (True/False)
    """
    get_coordination_overload_predictions_total().labels(
        agent_id=agent_id, predicted=str(predicted).lower()
    ).inc()
