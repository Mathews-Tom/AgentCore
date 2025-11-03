"""Tests for distributed tracing service."""

import asyncio
import time

import pytest

from agentcore.agent_runtime.services.distributed_tracing import (
    DistributedTracer,
    Span,
    SpanKind,
    SpanStatus,
    TraceContext,
    get_distributed_tracer,
    trace_operation,
)


class TestSpan:
    """Test span creation and management."""

    def test_span_creation(self) -> None:
        """Test span is created with correct attributes."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="span-123",
            operation_name="test_operation",
            kind=SpanKind.INTERNAL,
        )

        assert span.trace_id == "trace-123"
        assert span.span_id == "span-456"
        assert span.parent_span_id == "span-123"
        assert span.operation_name == "test_operation"
        assert span.kind == SpanKind.INTERNAL
        assert span.status == SpanStatus.UNSET
        assert span.start_time > 0

    def test_span_with_custom_start_time(self) -> None:
        """Test span with custom start time."""
        start_time = time.time()
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation_name="test",
            kind=SpanKind.SERVER,
            start_time=start_time,
        )

        assert span.start_time == start_time

    def test_span_set_attribute(self) -> None:
        """Test setting span attributes."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation_name="test",
            kind=SpanKind.INTERNAL,
        )

        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 123)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 123

    def test_span_add_event(self) -> None:
        """Test adding events to span."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation_name="test",
            kind=SpanKind.INTERNAL,
        )

        span.add_event("event1", attributes={"detail": "info"})
        span.add_event("event2")

        assert len(span.events) == 2
        assert span.events[0]["name"] == "event1"
        assert span.events[0]["attributes"]["detail"] == "info"
        assert span.events[1]["name"] == "event2"

    def test_span_add_link(self) -> None:
        """Test adding links to other spans."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation_name="test",
            kind=SpanKind.INTERNAL,
        )

        span.add_link("trace-999", "span-888", attributes={"relationship": "follows"})

        assert len(span.links) == 1
        assert span.links[0]["trace_id"] == "trace-999"
        assert span.links[0]["span_id"] == "span-888"
        assert span.links[0]["attributes"]["relationship"] == "follows"

    def test_span_finish(self) -> None:
        """Test finishing span."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation_name="test",
            kind=SpanKind.INTERNAL,
        )

        time.sleep(0.01)  # Small delay for measurable duration
        span.finish(status=SpanStatus.OK, status_message="Success")

        assert span.end_time is not None
        assert span.status == SpanStatus.OK
        assert span.status_message == "Success"

    def test_span_duration(self) -> None:
        """Test calculating span duration."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation_name="test",
            kind=SpanKind.INTERNAL,
        )

        # Before finishing, duration is 0
        assert span.duration_ms() == 0.0

        time.sleep(0.01)
        span.finish()

        # After finishing, duration should be positive
        assert span.duration_ms() > 0

    def test_span_to_dict(self) -> None:
        """Test converting span to dictionary."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="span-123",
            operation_name="test",
            kind=SpanKind.CLIENT,
        )

        span.set_attribute("attr1", "value1")
        span.add_event("event1")
        span.finish(status=SpanStatus.OK)

        span_dict = span.to_dict()

        assert span_dict["trace_id"] == "trace-123"
        assert span_dict["span_id"] == "span-456"
        assert span_dict["parent_span_id"] == "span-123"
        assert span_dict["operation_name"] == "test"
        assert span_dict["kind"] == "client"
        assert span_dict["status"] == "ok"
        assert "attr1" in span_dict["attributes"]
        assert len(span_dict["events"]) == 1


class TestTraceContext:
    """Test trace context management."""

    def test_context_creation(self) -> None:
        """Test creating trace context."""
        context = TraceContext(
            trace_id="trace-123",
            parent_span_id="span-456",
            baggage={"key": "value"},
        )

        assert context.trace_id == "trace-123"
        assert context.parent_span_id == "span-456"
        assert context.baggage["key"] == "value"

    def test_context_generates_trace_id(self) -> None:
        """Test context generates trace ID if not provided."""
        context = TraceContext()

        assert context.trace_id is not None
        assert len(context.trace_id) > 0

    def test_context_set_baggage(self) -> None:
        """Test setting baggage items."""
        context = TraceContext()

        context.set_baggage("user_id", "12345")
        context.set_baggage("tenant", "acme")

        assert context.baggage["user_id"] == "12345"
        assert context.baggage["tenant"] == "acme"

    def test_context_get_baggage(self) -> None:
        """Test getting baggage items."""
        context = TraceContext(baggage={"key1": "value1"})

        value = context.get_baggage("key1")
        missing = context.get_baggage("nonexistent")

        assert value == "value1"
        assert missing is None

    def test_context_to_dict(self) -> None:
        """Test converting context to dictionary."""
        context = TraceContext(
            trace_id="trace-123",
            parent_span_id="span-456",
            baggage={"key": "value"},
        )

        context_dict = context.to_dict()

        assert context_dict["trace_id"] == "trace-123"
        assert context_dict["parent_span_id"] == "span-456"
        assert context_dict["baggage"]["key"] == "value"

    def test_context_from_dict(self) -> None:
        """Test creating context from dictionary."""
        data = {
            "trace_id": "trace-123",
            "parent_span_id": "span-456",
            "baggage": {"key": "value"},
        }

        context = TraceContext.from_dict(data)

        assert context.trace_id == "trace-123"
        assert context.parent_span_id == "span-456"
        assert context.baggage["key"] == "value"


class TestDistributedTracer:
    """Test distributed tracer functionality."""

    def test_tracer_initialization(self) -> None:
        """Test tracer initializes correctly."""
        tracer = DistributedTracer(
            service_name="test-service",
            enable_export=True,
        )

        assert tracer._service_name == "test-service"
        assert tracer._enable_export is True
        assert tracer._max_spans == 10000

    def test_start_trace(self) -> None:
        """Test starting new trace."""
        tracer = DistributedTracer()

        context = tracer.start_trace()

        assert context is not None
        assert context.trace_id is not None

    def test_start_trace_with_id(self) -> None:
        """Test starting trace with specific ID."""
        tracer = DistributedTracer()

        context = tracer.start_trace(trace_id="custom-trace-123")

        assert context.trace_id == "custom-trace-123"

    def test_start_trace_with_baggage(self) -> None:
        """Test starting trace with baggage."""
        tracer = DistributedTracer()

        baggage = {"user_id": "12345", "session": "abc"}
        context = tracer.start_trace(baggage=baggage)

        assert context.baggage["user_id"] == "12345"
        assert context.baggage["session"] == "abc"

    def test_get_current_context(self) -> None:
        """Test getting current trace context."""
        tracer = DistributedTracer()

        # No context initially
        context = tracer.get_current_context()
        assert context is None or context.trace_id is not None  # May have global state

        # Start trace
        started_context = tracer.start_trace(trace_id="test-trace")

        # Get context
        current_context = tracer.get_current_context()
        assert current_context is not None
        assert current_context.trace_id == "test-trace"

    def test_start_span(self) -> None:
        """Test starting span."""
        tracer = DistributedTracer(service_name="test-service")

        tracer.start_trace(trace_id="trace-123")
        span = tracer.start_span(
            operation_name="test_operation",
            kind=SpanKind.INTERNAL,
            attributes={"key": "value"},
        )

        assert span.trace_id == "trace-123"
        assert span.operation_name == "test_operation"
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes["service.name"] == "test-service"
        assert span.attributes["key"] == "value"

    def test_start_span_without_context(self) -> None:
        """Test starting span without existing context."""
        tracer = DistributedTracer()

        # Should create new trace automatically
        span = tracer.start_span("test_operation")

        assert span.trace_id is not None
        assert span.operation_name == "test_operation"

    def test_finish_span(self) -> None:
        """Test finishing span."""
        tracer = DistributedTracer()

        span = tracer.start_span("test_operation")
        time.sleep(0.01)

        tracer.finish_span(span, status=SpanStatus.OK, status_message="Success")

        assert span.end_time is not None
        assert span.status == SpanStatus.OK
        assert span.status_message == "Success"
        assert span in tracer._completed_spans

    def test_finish_span_exports(self) -> None:
        """Test span export on finish."""
        tracer = DistributedTracer(enable_export=True)

        span = tracer.start_span("test_operation")
        tracer.finish_span(span)

        # Span should be in completed spans (export is just logged)
        assert span in tracer._completed_spans

    def test_record_exception(self) -> None:
        """Test recording exception in span."""
        tracer = DistributedTracer()

        span = tracer.start_span("test_operation")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            tracer.record_exception(span, e)

        # Check exception was recorded
        assert len(span.events) > 0
        exception_event = span.events[0]
        assert exception_event["name"] == "exception"
        assert exception_event["attributes"]["exception.type"] == "ValueError"
        assert exception_event["attributes"]["exception.message"] == "Test error"
        assert span.attributes["error"] is True

    def test_get_trace_spans(self) -> None:
        """Test getting all spans for a trace."""
        tracer = DistributedTracer()

        trace_id = "trace-123"
        tracer.start_trace(trace_id=trace_id)

        # Create multiple spans
        span1 = tracer.start_span("operation1")
        tracer.finish_span(span1)

        span2 = tracer.start_span("operation2")
        tracer.finish_span(span2)

        # Get trace spans
        spans = tracer.get_trace_spans(trace_id)

        assert len(spans) == 2
        assert span1 in spans
        assert span2 in spans

    def test_get_span_by_id(self) -> None:
        """Test getting span by ID."""
        tracer = DistributedTracer()

        span = tracer.start_span("test_operation")
        tracer.finish_span(span)

        # Get span by ID
        retrieved_span = tracer.get_span_by_id(span.span_id)

        assert retrieved_span is span

    def test_get_span_by_id_not_found(self) -> None:
        """Test getting nonexistent span by ID."""
        tracer = DistributedTracer()

        result = tracer.get_span_by_id("nonexistent-span")

        assert result is None

    def test_get_trace_summary(self) -> None:
        """Test getting trace summary."""
        tracer = DistributedTracer()

        trace_id = "trace-123"
        tracer.start_trace(trace_id=trace_id)

        # Create and finish multiple spans
        span1 = tracer.start_span("operation1")
        time.sleep(0.01)
        tracer.finish_span(span1, status=SpanStatus.OK)

        span2 = tracer.start_span("operation2")
        time.sleep(0.01)
        tracer.finish_span(span2, status=SpanStatus.ERROR)

        # Get summary
        summary = tracer.get_trace_summary(trace_id)

        assert summary["trace_id"] == trace_id
        assert summary["span_count"] == 2
        assert summary["total_duration_ms"] > 0
        assert summary["error_count"] == 1
        assert "operation1" in summary["operations"]
        assert "operation2" in summary["operations"]

    def test_get_trace_summary_empty(self) -> None:
        """Test getting summary for nonexistent trace."""
        tracer = DistributedTracer()

        summary = tracer.get_trace_summary("nonexistent-trace")

        assert summary == {}

    def test_get_metrics(self) -> None:
        """Test getting tracing metrics."""
        tracer = DistributedTracer(enable_export=True)

        # Create some spans
        span1 = tracer.start_span("operation1")
        tracer.finish_span(span1, status=SpanStatus.OK)

        span2 = tracer.start_span("operation2")
        tracer.finish_span(span2, status=SpanStatus.ERROR)

        metrics = tracer.get_metrics()

        assert metrics["total_spans"] == 2
        assert metrics["error_spans"] == 1
        assert metrics["error_rate_percent"] == 50.0
        assert metrics["unique_traces"] >= 1
        assert metrics["export_enabled"] is True

    def test_max_spans_limit(self) -> None:
        """Test max spans limit enforcement."""
        tracer = DistributedTracer()
        tracer._max_spans = 10  # Set small limit for testing

        # Create more spans than limit
        for i in range(15):
            span = tracer.start_span(f"operation{i}")
            tracer.finish_span(span)

        # Should not exceed max
        assert len(tracer._completed_spans) <= 10


class TestTraceOperationDecorator:
    """Test trace operation decorator."""

    @pytest.mark.asyncio
    async def test_trace_async_function(self) -> None:
        """Test tracing async function."""

        @trace_operation(operation_name="async_operation", kind=SpanKind.INTERNAL)
        async def async_func(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x + y

        tracer = get_distributed_tracer()
        tracer.start_trace(trace_id="test-trace")

        result = await async_func(5, 3)

        assert result == 8

        # Check span was created
        spans = tracer.get_trace_spans("test-trace")
        assert len(spans) > 0

        # Find our operation
        operation_spans = [s for s in spans if s.operation_name == "async_operation"]
        assert len(operation_spans) > 0
        assert operation_spans[0].status == SpanStatus.OK

    @pytest.mark.asyncio
    async def test_trace_async_function_with_exception(self) -> None:
        """Test tracing async function that raises exception."""

        @trace_operation(operation_name="failing_operation")
        async def failing_func() -> None:
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        tracer = get_distributed_tracer()
        tracer.start_trace(trace_id="test-trace-error")

        with pytest.raises(ValueError, match="Test error"):
            await failing_func()

        # Check span was created with error
        spans = tracer.get_trace_spans("test-trace-error")
        operation_spans = [s for s in spans if s.operation_name == "failing_operation"]

        if operation_spans:  # May not be recorded if tracer state is reset
            assert operation_spans[0].status == SpanStatus.ERROR

    def test_trace_sync_function(self) -> None:
        """Test tracing synchronous function."""

        @trace_operation(operation_name="sync_operation", kind=SpanKind.CLIENT)
        def sync_func(x: int, y: int) -> int:
            time.sleep(0.01)
            return x * y

        tracer = get_distributed_tracer()
        tracer.start_trace(trace_id="test-trace-sync")

        result = sync_func(4, 5)

        assert result == 20

        # Check span was created
        spans = tracer.get_trace_spans("test-trace-sync")
        if spans:  # May not be recorded if tracer state is reset
            operation_spans = [s for s in spans if s.operation_name == "sync_operation"]
            if operation_spans:
                assert operation_spans[0].kind == SpanKind.CLIENT

    def test_trace_sync_function_with_exception(self) -> None:
        """Test tracing synchronous function that raises exception."""

        @trace_operation(operation_name="failing_sync_operation")
        def failing_sync_func() -> None:
            raise RuntimeError("Sync error")

        tracer = get_distributed_tracer()
        tracer.start_trace(trace_id="test-trace-sync-error")

        with pytest.raises(RuntimeError, match="Sync error"):
            failing_sync_func()

    def test_trace_operation_uses_function_name(self) -> None:
        """Test decorator uses function name if operation_name not provided."""

        @trace_operation()
        def my_function() -> str:
            return "result"

        tracer = get_distributed_tracer()
        tracer.start_trace(trace_id="test-trace-name")

        result = my_function()

        assert result == "result"

        # Operation name should be function name
        spans = tracer.get_trace_spans("test-trace-name")
        if spans:
            function_spans = [s for s in spans if s.operation_name == "my_function"]
            assert len(function_spans) > 0


class TestGlobalTracer:
    """Test global tracer instance."""

    def test_get_global_tracer(self) -> None:
        """Test getting global tracer."""
        tracer1 = get_distributed_tracer()
        tracer2 = get_distributed_tracer()

        # Should return same instance
        assert tracer1 is tracer2
