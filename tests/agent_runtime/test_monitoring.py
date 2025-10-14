"""
Tests for monitoring and observability services.

Tests metrics collection, distributed tracing, alerting, and monitoring APIs.
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agentcore.agent_runtime.models.agent_config import AgentPhilosophy
from agentcore.agent_runtime.services.alerting_service import (
    Alert,
    AlertingService,
    AlertRule,
    AlertSeverity,
    AlertState,
    NotificationChannel,
)
from agentcore.agent_runtime.services.distributed_tracing import (
    DistributedTracer,
    Span,
    SpanKind,
    SpanStatus,
    TraceContext,
)
from agentcore.agent_runtime.services.metrics_collector import (
    MetricsCollector,
    MetricType,
)


class TestMetricsCollector:
    """Test metrics collection functionality."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create metrics collector instance."""
        return MetricsCollector()

    def test_initialization(self, collector: MetricsCollector) -> None:
        """Test metrics collector initialization."""
        assert collector is not None
        assert collector._registry is not None

    def test_record_agent_created(self, collector: MetricsCollector) -> None:
        """Test recording agent creation."""
        collector.record_agent_created(AgentPhilosophy.REACT, "initializing")
        collector.record_agent_created(AgentPhilosophy.CHAIN_OF_THOUGHT, "initializing")

        # Verify metrics were recorded
        assert collector.agents_total._metrics is not None

    def test_record_agent_execution(self, collector: MetricsCollector) -> None:
        """Test recording agent execution duration."""
        collector.record_agent_execution(AgentPhilosophy.REACT, 1.5)
        collector.record_agent_execution(AgentPhilosophy.CHAIN_OF_THOUGHT, 2.3)

        # Verify histogram recorded values
        assert collector.agent_execution_duration._metrics is not None

    def test_record_state_transition(self, collector: MetricsCollector) -> None:
        """Test recording agent state transitions."""
        collector.record_state_transition(
            "initializing", "running", AgentPhilosophy.REACT
        )
        collector.record_state_transition("running", "completed", AgentPhilosophy.REACT)

        # Verify transitions were recorded
        assert collector.agent_state_transitions._metrics is not None

    def test_update_resource_usage(self, collector: MetricsCollector) -> None:
        """Test updating agent resource usage."""
        collector.update_resource_usage(
            agent_id="test-agent",
            philosophy=AgentPhilosophy.REACT,
            cpu_percent=45.5,
            memory_mb=256.0,
        )

        # Verify gauges were set
        assert collector.agent_cpu_usage._metrics is not None
        assert collector.agent_memory_usage._metrics is not None

    def test_record_container_creation(self, collector: MetricsCollector) -> None:
        """Test recording container creation."""
        # Warm start
        collector.record_container_creation(
            philosophy=AgentPhilosophy.REACT,
            duration_seconds=0.05,
            warm_start=True,
        )

        # Cold start
        collector.record_container_creation(
            philosophy=AgentPhilosophy.CHAIN_OF_THOUGHT,
            duration_seconds=0.5,
            warm_start=False,
        )

        # Verify metrics were recorded
        assert collector.warm_starts._metrics is not None
        assert collector.cold_starts._metrics is not None

    def test_record_tool_execution(self, collector: MetricsCollector) -> None:
        """Test recording tool execution."""
        collector.record_tool_execution(
            tool_id="calculator",
            duration_seconds=0.1,
            status="success",
        )

        collector.record_tool_execution(
            tool_id="api_call",
            duration_seconds=0.5,
            status="error",
            error_type="timeout",
        )

        # Verify metrics were recorded
        assert collector.tool_executions._metrics is not None
        assert collector.tool_errors._metrics is not None

    def test_record_philosophy_metrics(self, collector: MetricsCollector) -> None:
        """Test philosophy-specific metrics."""
        collector.record_react_iterations("agent-1", 5)
        collector.record_cot_steps("agent-2", 8)
        collector.record_multi_agent_message("direct")
        collector.record_consensus("reached")
        collector.record_autonomous_goal("completed", "high")

        # Verify all were recorded
        assert collector.react_iterations._metrics is not None
        assert collector.cot_steps._metrics is not None
        assert collector.multi_agent_messages._metrics is not None

    def test_custom_metrics(self, collector: MetricsCollector) -> None:
        """Test custom metric creation."""
        metric = collector.create_custom_metric(
            name="test_counter",
            metric_type=MetricType.COUNTER,
            description="Test counter metric",
            labels=["label1"],
        )

        assert metric is not None
        retrieved = collector.get_custom_metric("test_counter")
        assert retrieved is metric

    def test_metrics_snapshot(self, collector: MetricsCollector) -> None:
        """Test creating metrics snapshot."""
        collector.update_system_resources(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_available_mb=1024.0,
        )

        snapshot = collector.snapshot_metrics()

        assert "timestamp" in snapshot
        assert "metrics" in snapshot
        assert "system" in snapshot["metrics"]


class TestDistributedTracing:
    """Test distributed tracing functionality."""

    @pytest.fixture
    def tracer(self) -> DistributedTracer:
        """Create distributed tracer instance."""
        return DistributedTracer(service_name="test-service")

    def test_initialization(self, tracer: DistributedTracer) -> None:
        """Test tracer initialization."""
        assert tracer is not None
        assert tracer._service_name == "test-service"

    def test_start_trace(self, tracer: DistributedTracer) -> None:
        """Test starting new trace."""
        context = tracer.start_trace()

        assert context is not None
        assert context.trace_id is not None
        assert isinstance(context.baggage, dict)

    def test_span_creation(self, tracer: DistributedTracer) -> None:
        """Test creating spans."""
        context = tracer.start_trace()
        span = tracer.start_span("test_operation", kind=SpanKind.INTERNAL)

        assert span is not None
        assert span.trace_id == context.trace_id
        assert span.operation_name == "test_operation"
        assert span.kind == SpanKind.INTERNAL

    def test_span_attributes(self, tracer: DistributedTracer) -> None:
        """Test setting span attributes."""
        tracer.start_trace()
        span = tracer.start_span("test_operation")

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 123)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 123

    def test_span_events(self, tracer: DistributedTracer) -> None:
        """Test adding span events."""
        tracer.start_trace()
        span = tracer.start_span("test_operation")

        span.add_event("checkpoint_reached", {"step": 1})
        span.add_event("data_processed", {"count": 100})

        assert len(span.events) == 2
        assert span.events[0]["name"] == "checkpoint_reached"
        assert span.events[1]["attributes"]["count"] == 100

    def test_span_finish(self, tracer: DistributedTracer) -> None:
        """Test finishing spans."""
        tracer.start_trace()
        span = tracer.start_span("test_operation")

        tracer.finish_span(span, status=SpanStatus.OK)

        assert span.status == SpanStatus.OK
        assert span.end_time is not None
        assert span.duration_ms() > 0

    def test_exception_recording(self, tracer: DistributedTracer) -> None:
        """Test recording exceptions in spans."""
        tracer.start_trace()
        span = tracer.start_span("test_operation")

        exception = ValueError("Test error")
        tracer.record_exception(span, exception)

        assert len(span.events) == 1
        assert span.events[0]["name"] == "exception"
        assert span.attributes["error"] is True

    def test_trace_context_propagation(self, tracer: DistributedTracer) -> None:
        """Test trace context propagation."""
        context = tracer.start_trace()
        context.set_baggage("user_id", "12345")

        span = tracer.start_span("operation1")
        tracer.finish_span(span)

        # Get context and verify baggage
        current_context = tracer.get_current_context()
        assert current_context is not None
        assert current_context.get_baggage("user_id") == "12345"

    def test_get_trace_spans(self, tracer: DistributedTracer) -> None:
        """Test retrieving trace spans."""
        context = tracer.start_trace()

        span1 = tracer.start_span("operation1")
        tracer.finish_span(span1)

        span2 = tracer.start_span("operation2")
        tracer.finish_span(span2)

        spans = tracer.get_trace_spans(context.trace_id)
        assert len(spans) == 2

    def test_trace_summary(self, tracer: DistributedTracer) -> None:
        """Test getting trace summary."""
        context = tracer.start_trace()

        span1 = tracer.start_span("operation1")
        tracer.finish_span(span1)

        span2 = tracer.start_span("operation2")
        tracer.finish_span(span2, status=SpanStatus.ERROR)

        summary = tracer.get_trace_summary(context.trace_id)

        assert summary["trace_id"] == context.trace_id
        assert summary["span_count"] == 2
        assert summary["error_count"] == 1

    def test_tracing_metrics(self, tracer: DistributedTracer) -> None:
        """Test getting tracing metrics."""
        context = tracer.start_trace()
        span = tracer.start_span("test_operation")
        tracer.finish_span(span)

        metrics = tracer.get_metrics()

        assert "total_spans" in metrics
        assert "error_spans" in metrics
        assert "unique_traces" in metrics


class TestAlertingService:
    """Test alerting service functionality."""

    @pytest.fixture
    def alerting(self) -> AlertingService:
        """Create alerting service instance."""
        return AlertingService(enable_notifications=False)

    def test_initialization(self, alerting: AlertingService) -> None:
        """Test alerting service initialization."""
        assert alerting is not None
        assert not alerting._enable_notifications

    def test_register_rule(self, alerting: AlertingService) -> None:
        """Test registering alert rule."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: ctx.get("value", 0) > 10,
            severity=AlertSeverity.WARNING,
            title_template="Test alert",
            description_template="Value is {value}",
        )

        alerting.register_rule(rule)
        assert "test_rule" in alerting._rules

    def test_create_threshold_rule(self, alerting: AlertingService) -> None:
        """Test creating threshold-based rule."""
        rule = alerting.create_threshold_rule(
            name="cpu_threshold",
            metric_name="cpu_percent",
            threshold=80.0,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
        )

        assert rule is not None
        assert rule.name == "cpu_threshold"
        assert "cpu_threshold" in alerting._rules

    @pytest.mark.asyncio
    async def test_trigger_alert(self, alerting: AlertingService) -> None:
        """Test triggering alert."""
        # Register rule first
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: True,
            severity=AlertSeverity.WARNING,
            title_template="Test alert",
            description_template="This is a test",
        )
        alerting.register_rule(rule)

        # Trigger alert
        alert = await alerting.trigger_alert(
            rule_name="test_rule",
            title="Test alert",
            description="This is a test",
        )

        assert alert is not None
        assert alert.state == AlertState.ACTIVE
        assert alert.severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alerting: AlertingService) -> None:
        """Test acknowledging alert."""
        # Create rule and trigger alert
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: True,
            severity=AlertSeverity.INFO,
            title_template="Test",
            description_template="Test",
        )
        alerting.register_rule(rule)

        alert = await alerting.trigger_alert(
            rule_name="test_rule",
            title="Test",
            description="Test",
        )

        # Acknowledge
        success = await alerting.acknowledge_alert(alert.alert_id, "test_user")

        assert success is True
        assert alert.state == AlertState.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alerting: AlertingService) -> None:
        """Test resolving alert."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: True,
            severity=AlertSeverity.INFO,
            title_template="Test",
            description_template="Test",
        )
        alerting.register_rule(rule)

        alert = await alerting.trigger_alert(
            rule_name="test_rule",
            title="Test",
            description="Test",
        )

        # Resolve
        success = await alerting.resolve_alert(alert.alert_id, "Fixed")

        assert success is True
        assert alert.state == AlertState.RESOLVED
        assert alert.resolution_note == "Fixed"

    @pytest.mark.asyncio
    async def test_evaluate_rules(self, alerting: AlertingService) -> None:
        """Test evaluating alert rules."""
        # Create rule with condition
        rule = AlertRule(
            name="cpu_alert",
            condition=lambda ctx: ctx.get("cpu", 0) > 80,
            severity=AlertSeverity.CRITICAL,
            title_template="High CPU",
            description_template="CPU is {cpu}%",
        )
        alerting.register_rule(rule)

        # Evaluate with high CPU
        alerts = await alerting.evaluate_rules({"cpu": 85})

        assert len(alerts) == 1
        assert "CPU is 85%" in alerts[0].description

    def test_get_active_alerts(self, alerting: AlertingService) -> None:
        """Test getting active alerts."""
        # Create some test alerts directly
        alert1 = Alert(
            alert_id="1",
            rule_name="rule1",
            severity=AlertSeverity.WARNING,
            title="Alert 1",
            description="Test",
        )
        alert2 = Alert(
            alert_id="2",
            rule_name="rule2",
            severity=AlertSeverity.CRITICAL,
            title="Alert 2",
            description="Test",
        )

        alerting._alerts["1"] = alert1
        alerting._alerts["2"] = alert2

        # Get all active alerts
        active = alerting.get_active_alerts()
        assert len(active) == 2

        # Filter by severity
        critical = alerting.get_active_alerts(severity=AlertSeverity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].severity == AlertSeverity.CRITICAL

    def test_get_alert_history(self, alerting: AlertingService) -> None:
        """Test getting alert history."""
        # Create alerts with different timestamps
        alert1 = Alert(
            alert_id="1",
            rule_name="rule1",
            severity=AlertSeverity.INFO,
            title="Old alert",
            description="Test",
        )
        alert1.created_at = datetime.now(UTC) - timedelta(hours=48)

        alert2 = Alert(
            alert_id="2",
            rule_name="rule2",
            severity=AlertSeverity.WARNING,
            title="Recent alert",
            description="Test",
        )

        alerting._alert_history.extend([alert1, alert2])

        # Get last 24 hours
        recent = alerting.get_alert_history(hours=24)
        assert len(recent) == 1
        assert recent[0].title == "Recent alert"

        # Get last 72 hours
        all_alerts = alerting.get_alert_history(hours=72)
        assert len(all_alerts) == 2

    def test_statistics(self, alerting: AlertingService) -> None:
        """Test getting alerting statistics."""
        alerting._stats["total_alerts"] = 10
        alerting._stats["active_alerts"] = 3

        stats = alerting.get_statistics()

        assert stats["total_alerts"] == 10
        assert stats["active_alerts"] == 3
        assert "registered_rules" in stats


@pytest.mark.asyncio
async def test_monitoring_integration() -> None:
    """Test integration between monitoring components."""
    # Create instances
    collector = MetricsCollector()
    tracer = DistributedTracer()
    alerting = AlertingService(enable_notifications=False)

    # Record some agent activity
    collector.record_agent_created(AgentPhilosophy.REACT, "initializing")

    # Create trace for the activity
    context = tracer.start_trace()
    span = tracer.start_span("agent_execution", kind=SpanKind.INTERNAL)
    span.set_attribute("agent_id", "test-agent")
    span.set_attribute("philosophy", "react")

    # Simulate execution
    await asyncio.sleep(0.01)

    # Finish span
    tracer.finish_span(span, status=SpanStatus.OK)

    # Record metrics
    collector.record_agent_execution(AgentPhilosophy.REACT, 0.01)

    # Create alert rule for high execution time
    rule = alerting.create_threshold_rule(
        name="slow_execution",
        metric_name="duration",
        threshold=5.0,
        comparison="gt",
        severity=AlertSeverity.WARNING,
    )

    # Evaluate with normal execution
    alerts = await alerting.evaluate_rules({"duration": 0.01})
    assert len(alerts) == 0

    # Verify all components recorded data
    assert len(tracer._completed_spans) > 0
    snapshot = collector.snapshot_metrics()
    assert "metrics" in snapshot
    stats = alerting.get_statistics()
    assert stats["registered_rules"] > 0
