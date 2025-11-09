"""
Prometheus Metrics Exporter for ACE (Agent Context Engineering)

Exposes ACE performance monitoring, error tracking, and intervention metrics
to Prometheus for COMPASS validation and production monitoring.

Metrics cover:
- Stage-specific performance (planning, execution, reflection, verification)
- Error accumulation and detection
- Intervention effectiveness
- ACE-MEM coordination latency
- System overhead
"""

from __future__ import annotations

from typing import cast

from prometheus_client import REGISTRY, Counter, Gauge, Histogram

# Module-level cache to prevent duplicate registration
_metrics_cache: dict[str, Counter | Gauge | Histogram] = {}


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


def _get_or_create_histogram(
    name: str, description: str, buckets: list[float], labelnames: list[str] | None = None
) -> Histogram:
    """Get existing histogram or create new one, handling duplicates."""
    if name in _metrics_cache:
        return cast(Histogram, _metrics_cache[name])

    try:
        histogram = Histogram(name, description, labelnames=labelnames or [], buckets=buckets)
        _metrics_cache[name] = histogram
        return histogram
    except ValueError:
        # Metric already exists in registry, find and cache it
        for collector in list(REGISTRY._collector_to_names.keys()):
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = cast(Histogram, collector)
                return cast(Histogram, collector)
        raise


# ============================================================================
# COUNTER METRICS
# ============================================================================

# Lazy initialization - metrics are created on first use
_ace_performance_updates_total: Counter | None = None
_ace_errors_total: Counter | None = None
_ace_interventions_total: Counter | None = None


def _get_ace_performance_updates_total() -> Counter:
    """Get or create ACE performance updates counter."""
    global _ace_performance_updates_total
    if _ace_performance_updates_total is None:
        # Note: prometheus_client automatically appends _total to Counter names
        _ace_performance_updates_total = _get_or_create_counter(
            "ace_performance_updates",
            "Total number of ACE performance metric updates",
            ["agent_id", "stage"],
        )
    return _ace_performance_updates_total


def _get_ace_errors_total() -> Counter:
    """Get or create ACE errors counter."""
    global _ace_errors_total
    if _ace_errors_total is None:
        # Note: prometheus_client automatically appends _total to Counter names
        _ace_errors_total = _get_or_create_counter(
            "ace_errors",
            "Total number of errors detected by ACE",
            ["agent_id", "stage", "severity"],
        )
    return _ace_errors_total


def _get_ace_interventions_total() -> Counter:
    """Get or create ACE interventions counter."""
    global _ace_interventions_total
    if _ace_interventions_total is None:
        # Note: prometheus_client automatically appends _total to Counter names
        _ace_interventions_total = _get_or_create_counter(
            "ace_interventions",
            "Total number of ACE interventions triggered",
            ["agent_id", "type"],
        )
    return _ace_interventions_total


# ============================================================================
# GAUGE METRICS
# ============================================================================

_ace_baseline_deviation: Gauge | None = None
_ace_error_rate: Gauge | None = None
_ace_intervention_effectiveness: Gauge | None = None
_ace_context_staleness: Gauge | None = None


def _get_ace_baseline_deviation() -> Gauge:
    """Get or create ACE baseline deviation gauge."""
    global _ace_baseline_deviation
    if _ace_baseline_deviation is None:
        _ace_baseline_deviation = _get_or_create_gauge(
            "ace_baseline_deviation",
            "Deviation from performance baseline (percentage)",
            ["agent_id", "stage", "metric"],
        )
    return _ace_baseline_deviation


def _get_ace_error_rate() -> Gauge:
    """Get or create ACE error rate gauge."""
    global _ace_error_rate
    if _ace_error_rate is None:
        _ace_error_rate = _get_or_create_gauge(
            "ace_error_rate",
            "Current error rate by stage (0-1)",
            ["agent_id", "stage"],
        )
    return _ace_error_rate


def _get_ace_intervention_effectiveness() -> Gauge:
    """Get or create ACE intervention effectiveness gauge."""
    global _ace_intervention_effectiveness
    if _ace_intervention_effectiveness is None:
        _ace_intervention_effectiveness = _get_or_create_gauge(
            "ace_intervention_effectiveness",
            "Effectiveness of last intervention (0-1)",
            ["agent_id"],
        )
    return _ace_intervention_effectiveness


def _get_ace_context_staleness() -> Gauge:
    """Get or create ACE context staleness gauge."""
    global _ace_context_staleness
    if _ace_context_staleness is None:
        _ace_context_staleness = _get_or_create_gauge(
            "ace_context_staleness",
            "Context staleness score (0-1, higher = staler)",
            ["agent_id"],
        )
    return _ace_context_staleness


# ============================================================================
# HISTOGRAM METRICS
# ============================================================================

_ace_metric_computation_duration_seconds: Histogram | None = None
_ace_intervention_latency_seconds: Histogram | None = None
_ace_mem_query_duration_seconds: Histogram | None = None
_ace_stage_duration_seconds: Histogram | None = None


def _get_ace_metric_computation_duration_seconds() -> Histogram:
    """Get or create ACE metric computation duration histogram."""
    global _ace_metric_computation_duration_seconds
    if _ace_metric_computation_duration_seconds is None:
        _ace_metric_computation_duration_seconds = _get_or_create_histogram(
            "ace_metric_computation_duration_seconds",
            "Duration of ACE metric computations in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            labelnames=["operation"],
        )
    return _ace_metric_computation_duration_seconds


def _get_ace_intervention_latency_seconds() -> Histogram:
    """Get or create ACE intervention latency histogram."""
    global _ace_intervention_latency_seconds
    if _ace_intervention_latency_seconds is None:
        _ace_intervention_latency_seconds = _get_or_create_histogram(
            "ace_intervention_latency_seconds",
            "Latency of ACE interventions in seconds",
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            labelnames=["type"],
        )
    return _ace_intervention_latency_seconds


def _get_ace_mem_query_duration_seconds() -> Histogram:
    """Get or create ACE-MEM query duration histogram."""
    global _ace_mem_query_duration_seconds
    if _ace_mem_query_duration_seconds is None:
        _ace_mem_query_duration_seconds = _get_or_create_histogram(
            "ace_mem_query_duration_seconds",
            "Duration of ACE-MEM coordination queries in seconds",
            buckets=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0],
            labelnames=["query_type"],
        )
    return _ace_mem_query_duration_seconds


def _get_ace_stage_duration_seconds() -> Histogram:
    """Get or create ACE stage duration histogram."""
    global _ace_stage_duration_seconds
    if _ace_stage_duration_seconds is None:
        _ace_stage_duration_seconds = _get_or_create_histogram(
            "ace_stage_duration_seconds",
            "Duration of reasoning stages in seconds",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            labelnames=["agent_id", "stage"],
        )
    return _ace_stage_duration_seconds


# ============================================================================
# PUBLIC API - Recording Functions
# ============================================================================


def record_ace_performance_update(
    agent_id: str,
    stage: str,
    stage_success_rate: float,
    stage_error_rate: float,
    stage_duration_ms: int,
    baseline_delta: dict[str, float] | None = None,
    context_staleness_score: float | None = None,
    intervention_effectiveness: float | None = None,
) -> None:
    """
    Record ACE performance metrics update.

    Args:
        agent_id: Agent identifier
        stage: Reasoning stage (planning, execution, reflection, verification)
        stage_success_rate: Stage success rate (0-1)
        stage_error_rate: Stage error rate (0-1)
        stage_duration_ms: Stage duration in milliseconds
        baseline_delta: Optional baseline deviation dict (metric_name -> delta)
        context_staleness_score: Optional context staleness score (0-1)
        intervention_effectiveness: Optional intervention effectiveness (0-1)
    """
    # Counter: performance updates
    _get_ace_performance_updates_total().labels(agent_id=agent_id, stage=stage).inc()

    # Gauge: error rate
    _get_ace_error_rate().labels(agent_id=agent_id, stage=stage).set(stage_error_rate)

    # Histogram: stage duration
    _get_ace_stage_duration_seconds().labels(
        agent_id=agent_id, stage=stage
    ).observe(stage_duration_ms / 1000.0)

    # Gauge: baseline deviation (if provided)
    if baseline_delta:
        for metric_name, delta in baseline_delta.items():
            _get_ace_baseline_deviation().labels(
                agent_id=agent_id, stage=stage, metric=metric_name
            ).set(delta * 100)  # Convert to percentage

    # Gauge: context staleness (if provided)
    if context_staleness_score is not None:
        _get_ace_context_staleness().labels(agent_id=agent_id).set(context_staleness_score)

    # Gauge: intervention effectiveness (if provided)
    if intervention_effectiveness is not None:
        _get_ace_intervention_effectiveness().labels(agent_id=agent_id).set(
            intervention_effectiveness
        )


def record_ace_error(
    agent_id: str,
    stage: str,
    severity: str,
) -> None:
    """
    Record ACE error detection.

    Args:
        agent_id: Agent identifier
        stage: Reasoning stage where error occurred
        severity: Error severity (low, medium, high, critical)
    """
    _get_ace_errors_total().labels(
        agent_id=agent_id, stage=stage, severity=severity
    ).inc()


def record_ace_intervention(
    agent_id: str,
    intervention_type: str,
    latency_seconds: float,
) -> None:
    """
    Record ACE intervention.

    Args:
        agent_id: Agent identifier
        intervention_type: Type of intervention (replan, refresh_context, reflect)
        latency_seconds: Intervention latency in seconds
    """
    _get_ace_interventions_total().labels(
        agent_id=agent_id, type=intervention_type
    ).inc()
    _get_ace_intervention_latency_seconds().labels(
        type=intervention_type
    ).observe(latency_seconds)


def record_ace_metric_computation(
    operation: str,
    duration_seconds: float,
) -> None:
    """
    Record ACE metric computation duration.

    Args:
        operation: Operation type (baseline_update, error_detection, etc.)
        duration_seconds: Computation duration in seconds
    """
    _get_ace_metric_computation_duration_seconds().labels(
        operation=operation
    ).observe(duration_seconds)


def record_ace_mem_query(
    query_type: str,
    duration_seconds: float,
) -> None:
    """
    Record ACE-MEM coordination query.

    Args:
        query_type: Type of query (error_context, intervention_history, etc.)
        duration_seconds: Query duration in seconds
    """
    _get_ace_mem_query_duration_seconds().labels(
        query_type=query_type
    ).observe(duration_seconds)


# ============================================================================
# Module-level attribute access for lazy-initialized metrics
# ============================================================================


def __getattr__(name: str):
    """Module-level attribute access for lazy-initialized metrics."""
    if name == "ace_performance_updates_total":
        return _get_ace_performance_updates_total()
    elif name == "ace_errors_total":
        return _get_ace_errors_total()
    elif name == "ace_interventions_total":
        return _get_ace_interventions_total()
    elif name == "ace_baseline_deviation":
        return _get_ace_baseline_deviation()
    elif name == "ace_error_rate":
        return _get_ace_error_rate()
    elif name == "ace_intervention_effectiveness":
        return _get_ace_intervention_effectiveness()
    elif name == "ace_context_staleness":
        return _get_ace_context_staleness()
    elif name == "ace_metric_computation_duration_seconds":
        return _get_ace_metric_computation_duration_seconds()
    elif name == "ace_intervention_latency_seconds":
        return _get_ace_intervention_latency_seconds()
    elif name == "ace_mem_query_duration_seconds":
        return _get_ace_mem_query_duration_seconds()
    elif name == "ace_stage_duration_seconds":
        return _get_ace_stage_duration_seconds()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
