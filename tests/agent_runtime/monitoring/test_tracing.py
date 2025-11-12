"""Tests for OpenTelemetry distributed tracing integration."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from agentcore.agent_runtime.monitoring.tracing import (
    add_span_attributes,
    add_span_event,
    configure_tracing,
    get_span_id,
    get_trace_id,
    get_tracer,
    record_exception,
)


class InMemorySpanExporter(SpanExporter):
    """In-memory span exporter for testing."""

    def __init__(self):
        self.spans: list[ReadableSpan] = []

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        """Export spans to memory."""
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the exporter."""
        return True

    def get_finished_spans(self) -> list[ReadableSpan]:
        """Get all finished spans."""
        return self.spans

    def clear(self) -> None:
        """Clear all spans."""
        self.spans.clear()


@pytest.fixture(scope="session")
def setup_tracing():
    """Set up in-memory tracing for testing (session-scoped to avoid re-configuration)."""
    # Create in-memory exporter to capture spans
    exporter = InMemorySpanExporter()

    # Configure tracing with in-memory exporter
    provider = configure_tracing(
        service_name="test-service",
        service_version="1.0.0",
        sample_rate=1.0,
        enable_console_export=False,
    )

    # Add in-memory span processor for testing
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    return exporter


@pytest.fixture(autouse=True)
def clear_spans(setup_tracing):
    """Clear spans before each test."""
    setup_tracing.clear()
    yield
    setup_tracing.clear()


def test_configure_tracing():
    """Test tracing configuration."""
    # Note: Can't reconfigure tracer provider once set globally,
    # so this test just verifies the function returns a TracerProvider
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)


def test_get_tracer():
    """Test tracer instance retrieval."""
    tracer = get_tracer("test.module")

    assert tracer is not None
    assert isinstance(tracer, trace.Tracer)


def test_span_attributes(setup_tracing):
    """Test adding custom attributes to spans."""
    exporter = setup_tracing
    tracer = get_tracer("test.attributes")

    with tracer.start_as_current_span("test_span") as span:
        add_span_attributes(
            tool_id="test_tool",
            user_id="user-123",
            custom_value=42,
            is_test=True,
        )

    # Verify span was recorded
    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    # Verify attributes
    span = spans[0]
    assert span.attributes["tool_id"] == "test_tool"
    assert span.attributes["user_id"] == "user-123"
    assert span.attributes["custom_value"] == 42
    assert span.attributes["is_test"] is True


def test_span_events(setup_tracing):
    """Test adding events to spans."""
    exporter = setup_tracing
    tracer = get_tracer("test.events")

    with tracer.start_as_current_span("test_span"):
        add_span_event("operation_started", step=1)
        add_span_event("operation_completed", step=2, duration_ms=100)

    # Verify span was recorded
    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    # Verify events
    span = spans[0]
    events = span.events
    assert len(events) == 2

    assert events[0].name == "operation_started"
    assert events[0].attributes["step"] == 1

    assert events[1].name == "operation_completed"
    assert events[1].attributes["step"] == 2
    assert events[1].attributes["duration_ms"] == 100


def test_record_exception(setup_tracing):
    """Test recording exceptions in spans."""
    exporter = setup_tracing
    tracer = get_tracer("test.exceptions")

    with tracer.start_as_current_span("test_span"):
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            record_exception(e)

    # Verify span was recorded
    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    # Verify exception was recorded
    span = spans[0]
    assert len(span.events) == 1

    event = span.events[0]
    assert event.name == "exception"
    assert "exception.type" in event.attributes
    assert "exception.message" in event.attributes
    assert event.attributes["exception.message"] == "Test error message"

    # Verify span status
    assert span.status.status_code == trace.StatusCode.ERROR
    assert "Test error message" in span.status.description


def test_get_trace_id(setup_tracing):
    """Test retrieving trace ID from current span."""
    tracer = get_tracer("test.trace_id")

    # No active span should return empty string
    assert get_trace_id() == ""

    # With active span should return trace ID
    with tracer.start_as_current_span("test_span"):
        trace_id = get_trace_id()
        assert trace_id != ""
        assert len(trace_id) == 32  # Hex string of 16-byte trace ID


def test_get_span_id(setup_tracing):
    """Test retrieving span ID from current span."""
    tracer = get_tracer("test.span_id")

    # No active span should return empty string
    assert get_span_id() == ""

    # With active span should return span ID
    with tracer.start_as_current_span("test_span"):
        span_id = get_span_id()
        assert span_id != ""
        assert len(span_id) == 16  # Hex string of 8-byte span ID


def test_nested_spans(setup_tracing):
    """Test nested span creation and context propagation."""
    exporter = setup_tracing
    tracer = get_tracer("test.nested")

    with tracer.start_as_current_span("parent_span") as parent:
        parent_trace_id = get_trace_id()
        add_span_attributes(level="parent")

        with tracer.start_as_current_span("child_span") as child:
            child_trace_id = get_trace_id()
            add_span_attributes(level="child")

            # Both spans should share the same trace ID
            assert parent_trace_id == child_trace_id

    # Verify both spans were recorded
    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Verify parent-child relationship
    child_span = spans[0]
    parent_span = spans[1]

    assert child_span.attributes["level"] == "child"
    assert parent_span.attributes["level"] == "parent"
    assert child_span.parent.span_id == parent_span.context.span_id


def test_span_without_recording(setup_tracing):
    """Test that operations on spans work correctly."""
    # Create a span
    tracer = get_tracer("test.non_recording")

    with tracer.start_as_current_span("test_span"):
        # These should not fail
        add_span_attributes(test_attr="value")
        add_span_event("test_event")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            record_exception(e)

    # Verify span was recorded
    exporter = setup_tracing
    spans = exporter.get_finished_spans()
    assert len(spans) == 1


@pytest.mark.asyncio
async def test_tracing_with_async_operations(setup_tracing):
    """Test tracing integration with async operations."""
    import asyncio

    exporter = setup_tracing
    tracer = get_tracer("test.async")

    async def async_operation(op_name: str, delay: float):
        with tracer.start_as_current_span(f"async_{op_name}"):
            add_span_event("operation_started", operation_name=op_name)
            await asyncio.sleep(delay)
            add_span_event("operation_completed", operation_name=op_name)

    # Run multiple async operations
    await asyncio.gather(
        async_operation("op1", 0.01),
        async_operation("op2", 0.01),
        async_operation("op3", 0.01),
    )

    # Verify all spans were recorded
    spans = exporter.get_finished_spans()
    assert len(spans) == 3

    # Verify each span has the expected events
    for span in spans:
        assert len(span.events) == 2
        assert span.events[0].name == "operation_started"
        assert span.events[1].name == "operation_completed"
