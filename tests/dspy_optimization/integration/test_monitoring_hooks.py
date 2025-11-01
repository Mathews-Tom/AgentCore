"""Tests for Real-time Optimization Monitoring Hooks."""

from __future__ import annotations

import pytest

from agentcore.dspy_optimization.integration.monitoring_hooks import (
    MonitoringEvent,
    MonitoringEventType,
    OptimizationMonitor,
)
from agentcore.dspy_optimization.models import (
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)


@pytest.fixture
def monitor() -> OptimizationMonitor:
    """Create optimization monitor for testing."""
    return OptimizationMonitor()


@pytest.fixture
def sample_metrics() -> PerformanceMetrics:
    """Create sample performance metrics."""
    return PerformanceMetrics(
        success_rate=0.9,
        avg_cost_per_task=0.01,
        avg_latency_ms=150,
        quality_score=0.85,
    )


def test_register_callback(monitor: OptimizationMonitor) -> None:
    """Test registering monitoring callback."""
    callback_invoked = False

    def callback(event: MonitoringEvent) -> None:
        nonlocal callback_invoked
        callback_invoked = True

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_STARTED, callback)

    event = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )

    monitor.emit_event(event)

    assert callback_invoked is True


def test_unregister_callback(monitor: OptimizationMonitor) -> None:
    """Test unregistering monitoring callback."""
    callback_count = 0

    def callback(event: MonitoringEvent) -> None:
        nonlocal callback_count
        callback_count += 1

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_STARTED, callback)
    monitor.unregister_callback(MonitoringEventType.OPTIMIZATION_STARTED, callback)

    event = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )

    monitor.emit_event(event)

    assert callback_count == 0


def test_emit_event(monitor: OptimizationMonitor, sample_metrics: PerformanceMetrics) -> None:
    """Test emitting monitoring event."""
    received_events: list[MonitoringEvent] = []

    def callback(event: MonitoringEvent) -> None:
        received_events.append(event)

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_STARTED, callback)

    event = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
        metrics=sample_metrics,
    )

    monitor.emit_event(event)

    assert len(received_events) == 1
    assert received_events[0].agent_id == "agent-123"
    assert received_events[0].optimization_id == "opt-001"


def test_emit_event_stores_history(monitor: OptimizationMonitor) -> None:
    """Test that emitted events are stored in history."""
    event = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )

    monitor.emit_event(event)

    history = monitor.get_event_history()
    assert len(history) == 1
    assert history[0].agent_id == "agent-123"


def test_on_optimization_started(monitor: OptimizationMonitor, sample_metrics: PerformanceMetrics) -> None:
    """Test optimization started event."""
    received_events: list[MonitoringEvent] = []

    def callback(event: MonitoringEvent) -> None:
        received_events.append(event)

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_STARTED, callback)

    monitor.on_optimization_started(
        agent_id="agent-123",
        optimization_id="opt-001",
        baseline_metrics=sample_metrics,
        objectives={"test": "value"},
    )

    assert len(received_events) == 1
    assert received_events[0].event_type == MonitoringEventType.OPTIMIZATION_STARTED
    assert received_events[0].metrics == sample_metrics


def test_on_optimization_iteration(monitor: OptimizationMonitor, sample_metrics: PerformanceMetrics) -> None:
    """Test optimization iteration event."""
    received_events: list[MonitoringEvent] = []

    def callback(event: MonitoringEvent) -> None:
        received_events.append(event)

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_ITERATION, callback)

    monitor.on_optimization_iteration(
        agent_id="agent-123",
        optimization_id="opt-001",
        iteration=5,
        current_metrics=sample_metrics,
        iteration_data={"improvement": 0.05},
    )

    assert len(received_events) == 1
    assert received_events[0].data["iteration"] == 5


def test_on_optimization_completed(monitor: OptimizationMonitor) -> None:
    """Test optimization completed event."""
    received_events: list[MonitoringEvent] = []

    def callback(event: MonitoringEvent) -> None:
        received_events.append(event)

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_COMPLETED, callback)

    result = OptimizationResult(
        status=OptimizationStatus.COMPLETED,
        improvement_percentage=15.5,
    )

    monitor.on_optimization_completed(
        agent_id="agent-123",
        optimization_id="opt-001",
        result=result,
    )

    assert len(received_events) == 1
    assert received_events[0].event_type == MonitoringEventType.OPTIMIZATION_COMPLETED


def test_on_optimization_failed(monitor: OptimizationMonitor) -> None:
    """Test optimization failed event."""
    received_events: list[MonitoringEvent] = []

    def callback(event: MonitoringEvent) -> None:
        received_events.append(event)

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_FAILED, callback)

    monitor.on_optimization_failed(
        agent_id="agent-123",
        optimization_id="opt-001",
        error="Test error",
    )

    assert len(received_events) == 1
    assert received_events[0].data["error"] == "Test error"


def test_on_performance_improved(monitor: OptimizationMonitor, sample_metrics: PerformanceMetrics) -> None:
    """Test performance improvement event."""
    received_events: list[MonitoringEvent] = []

    def callback(event: MonitoringEvent) -> None:
        received_events.append(event)

    monitor.register_callback(MonitoringEventType.PERFORMANCE_IMPROVED, callback)

    improved_metrics = PerformanceMetrics(
        success_rate=0.95,
        avg_cost_per_task=0.008,
        avg_latency_ms=120,
        quality_score=0.9,
    )

    monitor.on_performance_improved(
        agent_id="agent-123",
        optimization_id="opt-001",
        baseline=sample_metrics,
        improved=improved_metrics,
        improvement_percentage=10.5,
    )

    assert len(received_events) == 1
    assert received_events[0].data["improvement_percentage"] == 10.5


def test_on_performance_degraded(monitor: OptimizationMonitor, sample_metrics: PerformanceMetrics) -> None:
    """Test performance degradation event."""
    received_events: list[MonitoringEvent] = []

    def callback(event: MonitoringEvent) -> None:
        received_events.append(event)

    monitor.register_callback(MonitoringEventType.PERFORMANCE_DEGRADED, callback)

    degraded_metrics = PerformanceMetrics(
        success_rate=0.85,
        avg_cost_per_task=0.015,
        avg_latency_ms=200,
        quality_score=0.75,
    )

    monitor.on_performance_degraded(
        agent_id="agent-123",
        optimization_id="opt-001",
        baseline=sample_metrics,
        degraded=degraded_metrics,
        degradation_percentage=8.5,
    )

    assert len(received_events) == 1
    assert received_events[0].event_type == MonitoringEventType.PERFORMANCE_DEGRADED


def test_get_active_optimizations(monitor: OptimizationMonitor, sample_metrics: PerformanceMetrics) -> None:
    """Test getting active optimizations."""
    monitor.on_optimization_started(
        agent_id="agent-123",
        optimization_id="opt-001",
        baseline_metrics=sample_metrics,
        objectives={},
    )

    active = monitor.get_active_optimizations()

    assert len(active) == 1
    assert active[0]["agent_id"] == "agent-123"


def test_get_optimization_status(monitor: OptimizationMonitor, sample_metrics: PerformanceMetrics) -> None:
    """Test getting optimization status."""
    monitor.on_optimization_started(
        agent_id="agent-123",
        optimization_id="opt-001",
        baseline_metrics=sample_metrics,
        objectives={},
    )

    status = monitor.get_optimization_status("opt-001")

    assert status is not None
    assert status["agent_id"] == "agent-123"
    assert status["baseline_metrics"] == sample_metrics


def test_get_optimization_status_not_found(monitor: OptimizationMonitor) -> None:
    """Test getting status for non-existent optimization."""
    status = monitor.get_optimization_status("non-existent")
    assert status is None


def test_get_event_history(monitor: OptimizationMonitor) -> None:
    """Test getting event history."""
    event1 = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )
    event2 = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_COMPLETED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )

    monitor.emit_event(event1)
    monitor.emit_event(event2)

    history = monitor.get_event_history()

    assert len(history) == 2


def test_get_event_history_filtered_by_agent(monitor: OptimizationMonitor) -> None:
    """Test getting event history filtered by agent ID."""
    event1 = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )
    event2 = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-456",
        optimization_id="opt-002",
    )

    monitor.emit_event(event1)
    monitor.emit_event(event2)

    history = monitor.get_event_history(agent_id="agent-123")

    assert len(history) == 1
    assert history[0].agent_id == "agent-123"


def test_get_event_history_filtered_by_type(monitor: OptimizationMonitor) -> None:
    """Test getting event history filtered by event type."""
    event1 = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )
    event2 = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_COMPLETED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )

    monitor.emit_event(event1)
    monitor.emit_event(event2)

    history = monitor.get_event_history(event_type=MonitoringEventType.OPTIMIZATION_STARTED)

    assert len(history) == 1
    assert history[0].event_type == MonitoringEventType.OPTIMIZATION_STARTED


def test_get_event_history_with_limit(monitor: OptimizationMonitor) -> None:
    """Test getting event history with limit."""
    for i in range(10):
        event = MonitoringEvent(
            event_type=MonitoringEventType.OPTIMIZATION_ITERATION,
            agent_id="agent-123",
            optimization_id="opt-001",
        )
        monitor.emit_event(event)

    history = monitor.get_event_history(limit=5)

    assert len(history) == 5


def test_clear_history(monitor: OptimizationMonitor) -> None:
    """Test clearing event history."""
    event = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )

    monitor.emit_event(event)
    monitor.clear_history()

    history = monitor.get_event_history()
    assert len(history) == 0


def test_callback_error_handling(monitor: OptimizationMonitor) -> None:
    """Test that callback errors don't break event emission."""

    def failing_callback(event: MonitoringEvent) -> None:
        raise ValueError("Test error")

    successful_count = 0

    def successful_callback(event: MonitoringEvent) -> None:
        nonlocal successful_count
        successful_count += 1

    monitor.register_callback(MonitoringEventType.OPTIMIZATION_STARTED, failing_callback)
    monitor.register_callback(MonitoringEventType.OPTIMIZATION_STARTED, successful_callback)

    event = MonitoringEvent(
        event_type=MonitoringEventType.OPTIMIZATION_STARTED,
        agent_id="agent-123",
        optimization_id="opt-001",
    )

    # Should not raise exception
    monitor.emit_event(event)

    # Successful callback should still be invoked
    assert successful_count == 1


def test_history_size_limit(monitor: OptimizationMonitor) -> None:
    """Test that event history respects size limit."""
    # Default max_history_size is 1000
    for i in range(1100):
        event = MonitoringEvent(
            event_type=MonitoringEventType.OPTIMIZATION_ITERATION,
            agent_id="agent-123",
            optimization_id="opt-001",
        )
        monitor.emit_event(event)

    history = monitor.get_event_history(limit=2000)
    assert len(history) == 1000  # Should be capped at max_history_size
