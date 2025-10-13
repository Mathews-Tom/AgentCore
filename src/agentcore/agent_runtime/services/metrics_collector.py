"""
Metrics collection service for agent runtime monitoring.

This module provides comprehensive metrics collection using Prometheus,
including execution metrics, resource usage, philosophy-specific metrics,
and custom metric definitions.
"""

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
)

from ..models.agent_config import AgentPhilosophy

logger = structlog.get_logger()


class MetricType(str, Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricsCollector:
    """Collects and exposes metrics for agent runtime monitoring."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus registry (creates new if None)
        """
        self._registry = registry or CollectorRegistry()
        self._custom_metrics: dict[str, Any] = {}

        # Initialize core metrics
        self._init_agent_metrics()
        self._init_resource_metrics()
        self._init_performance_metrics()
        self._init_philosophy_metrics()
        self._init_tool_metrics()
        self._init_error_metrics()

        # Metric snapshots for time-series analysis
        self._metric_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._history_max_size = 1000

        logger.info("metrics_collector_initialized", registry_id=id(self._registry))

    def _init_agent_metrics(self) -> None:
        """Initialize agent lifecycle metrics."""
        # Agent counts by status
        self.agents_total = Counter(
            "agentcore_agents_total",
            "Total number of agents created",
            ["philosophy", "status"],
            registry=self._registry,
        )

        self.agents_active = Gauge(
            "agentcore_agents_active",
            "Number of currently active agents",
            ["philosophy"],
            registry=self._registry,
        )

        # Agent lifecycle durations
        self.agent_initialization_duration = Histogram(
            "agentcore_agent_initialization_seconds",
            "Time spent initializing agents",
            ["philosophy"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )

        self.agent_execution_duration = Histogram(
            "agentcore_agent_execution_seconds",
            "Total agent execution time",
            ["philosophy"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            registry=self._registry,
        )

        # Agent state transitions
        self.agent_state_transitions = Counter(
            "agentcore_agent_state_transitions_total",
            "Total agent state transitions",
            ["from_state", "to_state", "philosophy"],
            registry=self._registry,
        )

    def _init_resource_metrics(self) -> None:
        """Initialize resource usage metrics."""
        # CPU usage
        self.agent_cpu_usage = Gauge(
            "agentcore_agent_cpu_percent",
            "Agent CPU usage percentage",
            ["agent_id", "philosophy"],
            registry=self._registry,
        )

        # Memory usage
        self.agent_memory_usage = Gauge(
            "agentcore_agent_memory_mb",
            "Agent memory usage in MB",
            ["agent_id", "philosophy"],
            registry=self._registry,
        )

        # Container metrics
        self.container_creation_duration = Histogram(
            "agentcore_container_creation_seconds",
            "Container creation time",
            ["philosophy", "warm_start"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            registry=self._registry,
        )

        # System resource utilization
        self.system_cpu_usage = Gauge(
            "agentcore_system_cpu_percent",
            "System-wide CPU usage",
            registry=self._registry,
        )

        self.system_memory_usage = Gauge(
            "agentcore_system_memory_percent",
            "System-wide memory usage percentage",
            registry=self._registry,
        )

        self.system_memory_available = Gauge(
            "agentcore_system_memory_available_mb",
            "Available system memory in MB",
            registry=self._registry,
        )

    def _init_performance_metrics(self) -> None:
        """Initialize performance optimization metrics."""
        # Cache performance
        self.cache_hits = Counter(
            "agentcore_cache_hits_total",
            "Total cache hits",
            ["cache_type"],
            registry=self._registry,
        )

        self.cache_misses = Counter(
            "agentcore_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
            registry=self._registry,
        )

        self.cache_size = Gauge(
            "agentcore_cache_size",
            "Current cache size",
            ["cache_type"],
            registry=self._registry,
        )

        # Container pool metrics
        self.pool_size = Gauge(
            "agentcore_container_pool_size",
            "Container pool size",
            ["philosophy"],
            registry=self._registry,
        )

        self.warm_starts = Counter(
            "agentcore_warm_starts_total",
            "Total warm container starts",
            ["philosophy"],
            registry=self._registry,
        )

        self.cold_starts = Counter(
            "agentcore_cold_starts_total",
            "Total cold container starts",
            ["philosophy"],
            registry=self._registry,
        )

        # Garbage collection
        self.gc_collections = Counter(
            "agentcore_gc_collections_total",
            "Total garbage collection runs",
            registry=self._registry,
        )

        self.memory_released = Counter(
            "agentcore_memory_released_mb_total",
            "Total memory released by GC in MB",
            registry=self._registry,
        )

    def _init_philosophy_metrics(self) -> None:
        """Initialize philosophy-specific metrics."""
        # ReAct metrics
        self.react_iterations = Histogram(
            "agentcore_react_iterations",
            "Number of ReAct iterations",
            ["agent_id"],
            buckets=(1, 3, 5, 10, 15, 20, 30, 50),
            registry=self._registry,
        )

        # Chain-of-Thought metrics
        self.cot_steps = Histogram(
            "agentcore_cot_steps",
            "Number of Chain-of-Thought reasoning steps",
            ["agent_id"],
            buckets=(1, 3, 5, 10, 15, 20),
            registry=self._registry,
        )

        # Multi-Agent metrics
        self.multi_agent_messages = Counter(
            "agentcore_multi_agent_messages_total",
            "Total inter-agent messages",
            ["message_type"],
            registry=self._registry,
        )

        self.multi_agent_consensus = Counter(
            "agentcore_multi_agent_consensus_total",
            "Total consensus operations",
            ["result"],
            registry=self._registry,
        )

        # Autonomous agent metrics
        self.autonomous_goals = Counter(
            "agentcore_autonomous_goals_total",
            "Total autonomous agent goals",
            ["status", "priority"],
            registry=self._registry,
        )

        self.autonomous_decisions = Counter(
            "agentcore_autonomous_decisions_total",
            "Total autonomous agent decisions",
            ["agent_id"],
            registry=self._registry,
        )

    def _init_tool_metrics(self) -> None:
        """Initialize tool execution metrics."""
        self.tool_executions = Counter(
            "agentcore_tool_executions_total",
            "Total tool executions",
            ["tool_id", "status"],
            registry=self._registry,
        )

        self.tool_execution_duration = Histogram(
            "agentcore_tool_execution_seconds",
            "Tool execution duration",
            ["tool_id"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            registry=self._registry,
        )

        self.tool_errors = Counter(
            "agentcore_tool_errors_total",
            "Total tool execution errors",
            ["tool_id", "error_type"],
            registry=self._registry,
        )

    def _init_error_metrics(self) -> None:
        """Initialize error and failure metrics."""
        self.errors_total = Counter(
            "agentcore_errors_total",
            "Total errors",
            ["error_type", "component"],
            registry=self._registry,
        )

        self.agent_failures = Counter(
            "agentcore_agent_failures_total",
            "Total agent failures",
            ["philosophy", "failure_reason"],
            registry=self._registry,
        )

        # Runtime info
        self.runtime_info = Info(
            "agentcore_runtime",
            "Agent runtime information",
            registry=self._registry,
        )

    def record_agent_created(
        self,
        philosophy: AgentPhilosophy,
        status: str = "initializing",
    ) -> None:
        """
        Record agent creation.

        Args:
            philosophy: Agent philosophy type
            status: Initial agent status
        """
        self.agents_total.labels(philosophy=philosophy.value, status=status).inc()
        self.agents_active.labels(philosophy=philosophy.value).inc()

    def record_agent_completed(
        self,
        philosophy: AgentPhilosophy,
        status: str,
    ) -> None:
        """
        Record agent completion.

        Args:
            philosophy: Agent philosophy type
            status: Final agent status (completed, failed, terminated)
        """
        self.agents_active.labels(philosophy=philosophy.value).dec()
        self.agents_total.labels(philosophy=philosophy.value, status=status).inc()

    def record_agent_initialization(
        self,
        philosophy: AgentPhilosophy,
        duration_seconds: float,
    ) -> None:
        """
        Record agent initialization time.

        Args:
            philosophy: Agent philosophy type
            duration_seconds: Initialization duration
        """
        self.agent_initialization_duration.labels(
            philosophy=philosophy.value
        ).observe(duration_seconds)

    def record_agent_execution(
        self,
        philosophy: AgentPhilosophy,
        duration_seconds: float,
    ) -> None:
        """
        Record agent execution time.

        Args:
            philosophy: Agent philosophy type
            duration_seconds: Execution duration
        """
        self.agent_execution_duration.labels(philosophy=philosophy.value).observe(
            duration_seconds
        )

    def record_state_transition(
        self,
        from_state: str,
        to_state: str,
        philosophy: AgentPhilosophy,
    ) -> None:
        """
        Record agent state transition.

        Args:
            from_state: Previous state
            to_state: New state
            philosophy: Agent philosophy type
        """
        self.agent_state_transitions.labels(
            from_state=from_state,
            to_state=to_state,
            philosophy=philosophy.value,
        ).inc()

    def update_resource_usage(
        self,
        agent_id: str,
        philosophy: AgentPhilosophy,
        cpu_percent: float,
        memory_mb: float,
    ) -> None:
        """
        Update agent resource usage.

        Args:
            agent_id: Agent identifier
            philosophy: Agent philosophy type
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
        """
        self.agent_cpu_usage.labels(
            agent_id=agent_id, philosophy=philosophy.value
        ).set(cpu_percent)

        self.agent_memory_usage.labels(
            agent_id=agent_id, philosophy=philosophy.value
        ).set(memory_mb)

    def record_container_creation(
        self,
        philosophy: AgentPhilosophy,
        duration_seconds: float,
        warm_start: bool,
    ) -> None:
        """
        Record container creation.

        Args:
            philosophy: Agent philosophy type
            duration_seconds: Creation duration
            warm_start: Whether this was a warm start
        """
        self.container_creation_duration.labels(
            philosophy=philosophy.value,
            warm_start=str(warm_start),
        ).observe(duration_seconds)

        if warm_start:
            self.warm_starts.labels(philosophy=philosophy.value).inc()
        else:
            self.cold_starts.labels(philosophy=philosophy.value).inc()

    def update_system_resources(
        self,
        cpu_percent: float,
        memory_percent: float,
        memory_available_mb: float,
    ) -> None:
        """
        Update system resource metrics.

        Args:
            cpu_percent: System CPU usage
            memory_percent: System memory usage percentage
            memory_available_mb: Available memory in MB
        """
        self.system_cpu_usage.set(cpu_percent)
        self.system_memory_usage.set(memory_percent)
        self.system_memory_available.set(memory_available_mb)

    def record_cache_access(
        self,
        cache_type: str,
        hit: bool,
        current_size: int,
    ) -> None:
        """
        Record cache access.

        Args:
            cache_type: Type of cache
            hit: Whether it was a hit
            current_size: Current cache size
        """
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()

        self.cache_size.labels(cache_type=cache_type).set(current_size)

    def record_tool_execution(
        self,
        tool_id: str,
        duration_seconds: float,
        status: str = "success",
        error_type: str | None = None,
    ) -> None:
        """
        Record tool execution.

        Args:
            tool_id: Tool identifier
            duration_seconds: Execution duration
            status: Execution status
            error_type: Error type if failed
        """
        self.tool_executions.labels(tool_id=tool_id, status=status).inc()
        self.tool_execution_duration.labels(tool_id=tool_id).observe(duration_seconds)

        if error_type:
            self.tool_errors.labels(tool_id=tool_id, error_type=error_type).inc()

    def record_error(
        self,
        error_type: str,
        component: str,
    ) -> None:
        """
        Record error occurrence.

        Args:
            error_type: Type of error
            component: Component where error occurred
        """
        self.errors_total.labels(error_type=error_type, component=component).inc()

    def record_agent_failure(
        self,
        philosophy: AgentPhilosophy,
        failure_reason: str,
    ) -> None:
        """
        Record agent failure.

        Args:
            philosophy: Agent philosophy type
            failure_reason: Reason for failure
        """
        self.agent_failures.labels(
            philosophy=philosophy.value,
            failure_reason=failure_reason,
        ).inc()

    def record_react_iterations(self, agent_id: str, iterations: int) -> None:
        """
        Record ReAct iterations.

        Args:
            agent_id: Agent identifier
            iterations: Number of iterations
        """
        self.react_iterations.labels(agent_id=agent_id).observe(iterations)

    def record_cot_steps(self, agent_id: str, steps: int) -> None:
        """
        Record Chain-of-Thought steps.

        Args:
            agent_id: Agent identifier
            steps: Number of reasoning steps
        """
        self.cot_steps.labels(agent_id=agent_id).observe(steps)

    def record_multi_agent_message(self, message_type: str) -> None:
        """
        Record multi-agent message.

        Args:
            message_type: Type of message (direct, broadcast, etc.)
        """
        self.multi_agent_messages.labels(message_type=message_type).inc()

    def record_consensus(self, result: str) -> None:
        """
        Record consensus operation.

        Args:
            result: Consensus result (reached, failed, etc.)
        """
        self.multi_agent_consensus.labels(result=result).inc()

    def record_autonomous_goal(self, status: str, priority: str) -> None:
        """
        Record autonomous agent goal.

        Args:
            status: Goal status
            priority: Goal priority
        """
        self.autonomous_goals.labels(status=status, priority=priority).inc()

    def record_autonomous_decision(self, agent_id: str) -> None:
        """
        Record autonomous agent decision.

        Args:
            agent_id: Agent identifier
        """
        self.autonomous_decisions.labels(agent_id=agent_id).inc()

    def record_gc_collection(self, memory_released_mb: float) -> None:
        """
        Record garbage collection.

        Args:
            memory_released_mb: Memory released in MB
        """
        self.gc_collections.inc()
        self.memory_released.inc(memory_released_mb)

    def set_runtime_info(self, info: dict[str, str]) -> None:
        """
        Set runtime information.

        Args:
            info: Runtime information dictionary
        """
        self.runtime_info.info(info)

    def create_custom_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        labels: list[str] | None = None,
    ) -> Any:
        """
        Create custom metric.

        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            labels: Metric labels

        Returns:
            Created metric object
        """
        labels = labels or []

        metric_map = {
            MetricType.COUNTER: Counter,
            MetricType.GAUGE: Gauge,
            MetricType.HISTOGRAM: Histogram,
            MetricType.SUMMARY: Summary,
        }

        metric_class = metric_map[metric_type]
        metric = metric_class(
            f"agentcore_{name}",
            description,
            labels,
            registry=self._registry,
        )

        self._custom_metrics[name] = metric
        logger.info("custom_metric_created", name=name, type=metric_type.value)

        return metric

    def get_custom_metric(self, name: str) -> Any | None:
        """
        Get custom metric by name.

        Args:
            name: Metric name

        Returns:
            Metric object or None
        """
        return self._custom_metrics.get(name)

    def snapshot_metrics(self) -> dict[str, Any]:
        """
        Create snapshot of current metrics.

        Returns:
            Dictionary with metric snapshots
        """
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "agents": {
                    "active": sum(
                        self.agents_active.labels(philosophy=p.value)._value.get()
                        for p in AgentPhilosophy
                    ),
                },
                "system": {
                    "cpu_percent": self.system_cpu_usage._value.get(),
                    "memory_percent": self.system_memory_usage._value.get(),
                    "memory_available_mb": self.system_memory_available._value.get(),
                },
            },
        }

        # Record in history
        self._metric_history["snapshots"].append(snapshot)
        if len(self._metric_history["snapshots"]) > self._history_max_size:
            self._metric_history["snapshots"].pop(0)

        return snapshot

    def get_metric_history(
        self,
        metric_name: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get metric history.

        Args:
            metric_name: Specific metric name (all if None)
            limit: Number of recent snapshots to return

        Returns:
            List of metric snapshots
        """
        if metric_name:
            history = self._metric_history.get(metric_name, [])
        else:
            history = self._metric_history.get("snapshots", [])

        if limit:
            return history[-limit:]

        return history

    def get_registry(self) -> CollectorRegistry:
        """
        Get Prometheus registry.

        Returns:
            Prometheus collector registry
        """
        return self._registry


# Global metrics collector instance
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector
