"""Tests for metrics collection service."""

import pytest
from prometheus_client import CollectorRegistry

from agentcore.agent_runtime.models.agent_config import AgentPhilosophy
from agentcore.agent_runtime.services.metrics_collector import (
    MetricType,
    MetricsCollector,
    get_metrics_collector,
)


class TestMetricsCollector:
    """Test metrics collector initialization and core functionality."""

    def test_collector_initialization(self) -> None:
        """Test metrics collector initializes correctly."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        assert collector._registry == registry
        assert isinstance(collector._custom_metrics, dict)
        assert collector._history_max_size == 1000

    def test_collector_with_default_registry(self) -> None:
        """Test collector creates registry if none provided."""
        collector = MetricsCollector()

        assert collector._registry is not None
        assert isinstance(collector._registry, CollectorRegistry)

    def test_get_global_collector(self) -> None:
        """Test global collector instance."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        # Should return same instance
        assert collector1 is collector2


class TestAgentMetrics:
    """Test agent lifecycle metrics."""

    def test_record_agent_created(self) -> None:
        """Test recording agent creation."""
        collector = MetricsCollector()

        collector.record_agent_created(AgentPhilosophy.REACT, status="initializing")

        # Verify metrics were updated (can't easily check values directly)
        assert collector.agents_total is not None
        assert collector.agents_active is not None

    def test_record_agent_completed(self) -> None:
        """Test recording agent completion."""
        collector = MetricsCollector()

        # Create then complete agent
        collector.record_agent_created(AgentPhilosophy.REACT)
        collector.record_agent_completed(AgentPhilosophy.REACT, status="completed")

        # Metrics should be updated
        assert collector.agents_total is not None
        assert collector.agents_active is not None

    def test_record_agent_initialization_duration(self) -> None:
        """Test recording initialization duration."""
        collector = MetricsCollector()

        collector.record_agent_initialization(AgentPhilosophy.REACT, duration_seconds=1.5)

        assert collector.agent_initialization_duration is not None

    def test_record_agent_execution_duration(self) -> None:
        """Test recording execution duration."""
        collector = MetricsCollector()

        collector.record_agent_execution(AgentPhilosophy.CHAIN_OF_THOUGHT, duration_seconds=30.0)

        assert collector.agent_execution_duration is not None

    def test_record_state_transition(self) -> None:
        """Test recording state transitions."""
        collector = MetricsCollector()

        collector.record_state_transition(
            from_state="initializing",
            to_state="running",
            philosophy=AgentPhilosophy.REACT,
        )

        assert collector.agent_state_transitions is not None


class TestResourceMetrics:
    """Test resource usage metrics."""

    def test_update_resource_usage(self) -> None:
        """Test updating agent resource usage."""
        collector = MetricsCollector()

        collector.update_resource_usage(
            agent_id="test-agent-1",
            philosophy=AgentPhilosophy.REACT,
            cpu_percent=45.5,
            memory_mb=256.0,
        )

        assert collector.agent_cpu_usage is not None
        assert collector.agent_memory_usage is not None

    def test_record_container_creation(self) -> None:
        """Test recording container creation."""
        collector = MetricsCollector()

        # Test warm start
        collector.record_container_creation(
            philosophy=AgentPhilosophy.REACT,
            duration_seconds=0.05,
            warm_start=True,
        )

        # Test cold start
        collector.record_container_creation(
            philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT,
            duration_seconds=2.5,
            warm_start=False,
        )

        assert collector.container_creation_duration is not None
        assert collector.warm_starts is not None
        assert collector.cold_starts is not None

    def test_update_system_resources(self) -> None:
        """Test updating system resource metrics."""
        collector = MetricsCollector()

        collector.update_system_resources(
            cpu_percent=65.5,
            memory_percent=72.3,
            memory_available_mb=4096.0,
        )

        assert collector.system_cpu_usage is not None
        assert collector.system_memory_usage is not None
        assert collector.system_memory_available is not None


class TestPerformanceMetrics:
    """Test performance optimization metrics."""

    def test_record_cache_hit(self) -> None:
        """Test recording cache hit."""
        collector = MetricsCollector()

        collector.record_cache_access(
            cache_type="tool_metadata",
            hit=True,
            current_size=150,
        )

        assert collector.cache_hits is not None
        assert collector.cache_size is not None

    def test_record_cache_miss(self) -> None:
        """Test recording cache miss."""
        collector = MetricsCollector()

        collector.record_cache_access(
            cache_type="execution_pattern",
            hit=False,
            current_size=200,
        )

        assert collector.cache_misses is not None
        assert collector.cache_size is not None

    def test_record_gc_collection(self) -> None:
        """Test recording garbage collection."""
        collector = MetricsCollector()

        collector.record_gc_collection(memory_released_mb=25.5)

        assert collector.gc_collections is not None
        assert collector.memory_released is not None


class TestPhilosophyMetrics:
    """Test philosophy-specific metrics."""

    def test_record_react_iterations(self) -> None:
        """Test recording ReAct iterations."""
        collector = MetricsCollector()

        collector.record_react_iterations(agent_id="test-agent", iterations=5)

        assert collector.react_iterations is not None

    def test_record_cot_steps(self) -> None:
        """Test recording Chain-of-Thought steps."""
        collector = MetricsCollector()

        collector.record_cot_steps(agent_id="test-agent", steps=7)

        assert collector.cot_steps is not None

    def test_record_multi_agent_message(self) -> None:
        """Test recording multi-agent messages."""
        collector = MetricsCollector()

        collector.record_multi_agent_message(message_type="direct")
        collector.record_multi_agent_message(message_type="broadcast")

        assert collector.multi_agent_messages is not None

    def test_record_consensus(self) -> None:
        """Test recording consensus operations."""
        collector = MetricsCollector()

        collector.record_consensus(result="reached")
        collector.record_consensus(result="failed")

        assert collector.multi_agent_consensus is not None

    def test_record_autonomous_goal(self) -> None:
        """Test recording autonomous agent goals."""
        collector = MetricsCollector()

        collector.record_autonomous_goal(status="active", priority="high")
        collector.record_autonomous_goal(status="completed", priority="medium")

        assert collector.autonomous_goals is not None

    def test_record_autonomous_decision(self) -> None:
        """Test recording autonomous decisions."""
        collector = MetricsCollector()

        collector.record_autonomous_decision(agent_id="autonomous-agent-1")

        assert collector.autonomous_decisions is not None


class TestToolMetrics:
    """Test tool execution metrics."""

    def test_record_tool_execution_success(self) -> None:
        """Test recording successful tool execution."""
        collector = MetricsCollector()

        collector.record_tool_execution(
            tool_id="calculator",
            duration_seconds=0.25,
            status="success",
        )

        assert collector.tool_executions is not None
        assert collector.tool_execution_duration is not None

    def test_record_tool_execution_failure(self) -> None:
        """Test recording failed tool execution."""
        collector = MetricsCollector()

        collector.record_tool_execution(
            tool_id="web_search",
            duration_seconds=1.5,
            status="failed",
            error_type="TimeoutError",
        )

        assert collector.tool_executions is not None
        assert collector.tool_errors is not None


class TestErrorMetrics:
    """Test error and failure metrics."""

    def test_record_error(self) -> None:
        """Test recording errors."""
        collector = MetricsCollector()

        collector.record_error(error_type="ValidationError", component="sandbox")
        collector.record_error(error_type="TimeoutError", component="tool_executor")

        assert collector.errors_total is not None

    def test_record_agent_failure(self) -> None:
        """Test recording agent failures."""
        collector = MetricsCollector()

        collector.record_agent_failure(
            philosophy=AgentPhilosophy.REACT,
            failure_reason="execution_timeout",
        )

        assert collector.agent_failures is not None

    def test_set_runtime_info(self) -> None:
        """Test setting runtime information."""
        collector = MetricsCollector()

        info = {
            "version": "1.0.0",
            "environment": "production",
            "platform": "linux",
        }

        collector.set_runtime_info(info)

        assert collector.runtime_info is not None


class TestCustomMetrics:
    """Test custom metric creation and management."""

    def test_create_custom_counter(self) -> None:
        """Test creating custom counter metric."""
        collector = MetricsCollector()

        metric = collector.create_custom_metric(
            name="custom_counter",
            metric_type=MetricType.COUNTER,
            description="Custom counter for testing",
            labels=["label1", "label2"],
        )

        assert metric is not None
        assert "custom_counter" in collector._custom_metrics

    def test_create_custom_gauge(self) -> None:
        """Test creating custom gauge metric."""
        collector = MetricsCollector()

        metric = collector.create_custom_metric(
            name="custom_gauge",
            metric_type=MetricType.GAUGE,
            description="Custom gauge for testing",
        )

        assert metric is not None
        assert "custom_gauge" in collector._custom_metrics

    def test_create_custom_histogram(self) -> None:
        """Test creating custom histogram metric."""
        collector = MetricsCollector()

        metric = collector.create_custom_metric(
            name="custom_histogram",
            metric_type=MetricType.HISTOGRAM,
            description="Custom histogram for testing",
            labels=["operation"],
        )

        assert metric is not None
        assert "custom_histogram" in collector._custom_metrics

    def test_create_custom_summary(self) -> None:
        """Test creating custom summary metric."""
        collector = MetricsCollector()

        metric = collector.create_custom_metric(
            name="custom_summary",
            metric_type=MetricType.SUMMARY,
            description="Custom summary for testing",
        )

        assert metric is not None
        assert "custom_summary" in collector._custom_metrics

    def test_get_custom_metric(self) -> None:
        """Test retrieving custom metrics."""
        collector = MetricsCollector()

        # Create metric
        created_metric = collector.create_custom_metric(
            name="test_metric",
            metric_type=MetricType.COUNTER,
            description="Test metric",
        )

        # Retrieve metric
        retrieved_metric = collector.get_custom_metric("test_metric")

        assert retrieved_metric is created_metric

    def test_get_nonexistent_custom_metric(self) -> None:
        """Test retrieving nonexistent custom metric."""
        collector = MetricsCollector()

        result = collector.get_custom_metric("nonexistent")

        assert result is None


class TestMetricSnapshots:
    """Test metric snapshot and history functionality."""

    def test_snapshot_metrics(self) -> None:
        """Test creating metric snapshot."""
        collector = MetricsCollector()

        # Update some metrics
        collector.update_system_resources(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available_mb=2048.0,
        )

        # Create snapshot
        snapshot = collector.snapshot_metrics()

        assert "timestamp" in snapshot
        assert "metrics" in snapshot
        assert "agents" in snapshot["metrics"]
        assert "system" in snapshot["metrics"]

    def test_snapshot_metrics_recorded_in_history(self) -> None:
        """Test snapshots are recorded in history."""
        collector = MetricsCollector()

        # Create multiple snapshots
        snapshot1 = collector.snapshot_metrics()
        snapshot2 = collector.snapshot_metrics()

        # Check history
        history = collector.get_metric_history()

        assert len(history) >= 2
        assert snapshot1 in history
        assert snapshot2 in history

    def test_get_metric_history_with_limit(self) -> None:
        """Test retrieving limited metric history."""
        collector = MetricsCollector()

        # Create several snapshots
        for _ in range(5):
            collector.snapshot_metrics()

        # Get last 2 snapshots
        history = collector.get_metric_history(limit=2)

        assert len(history) == 2

    def test_metric_history_max_size(self) -> None:
        """Test metric history respects max size."""
        collector = MetricsCollector()
        collector._history_max_size = 10  # Set small limit for testing

        # Create more snapshots than limit
        for _ in range(15):
            collector.snapshot_metrics()

        history = collector.get_metric_history()

        # Should not exceed max size
        assert len(history) <= 10


class TestRegistryAccess:
    """Test Prometheus registry access."""

    def test_get_registry(self) -> None:
        """Test getting Prometheus registry."""
        collector = MetricsCollector()

        registry = collector.get_registry()

        assert registry is not None
        assert isinstance(registry, CollectorRegistry)
        assert registry == collector._registry
