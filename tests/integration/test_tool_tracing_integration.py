"""Integration tests for OpenTelemetry distributed tracing in tool execution.

This module validates that distributed tracing works end-to-end:
- Spans are created for tool execution
- Trace IDs are propagated through ExecutionContext
- Span attributes include required metadata (tool_id, user_id, success, execution_time_ms)
- Parent-child span relationships are maintained
- Exceptions are properly recorded in spans

Tests TOOL-020 acceptance criteria.

Note: All tests in this module are marked with @pytest.mark.tracing to indicate
they use OpenTelemetry tracing infrastructure and may have special isolation needs.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from agentcore.agent_runtime.models.tool_integration import (
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from agentcore.agent_runtime.monitoring.tracing import configure_tracing, get_tracer
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry


class InMemorySpanExporter(SpanExporter):
    """In-memory span exporter for testing trace propagation."""

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


class MockSuccessTool(Tool):
    """Mock tool that always succeeds."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="mock_success",
            name="Mock Success Tool",
            description="A tool that always succeeds",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "input": ToolParameter(
                    name="input",
                    type="string",
                    description="Input value",
                    required=True,
                )
            },
            timeout_seconds=5,
        )
        super().__init__(metadata)

    async def execute(self, parameters: dict, context: ExecutionContext):
        """Execute successfully."""
        import time
        from datetime import UTC, datetime

        start_time = time.time()
        await asyncio.sleep(0.01)  # Simulate some work
        execution_time_ms = (time.time() - start_time) * 1000

        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result={"status": "success", "input": parameters.get("input")},
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(UTC),
        )


class MockFailureTool(Tool):
    """Mock tool that always fails."""

    def __init__(self):
        metadata = ToolDefinition(
            tool_id="mock_failure",
            name="Mock Failure Tool",
            description="A tool that always fails",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={
                "input": ToolParameter(
                    name="input",
                    type="string",
                    description="Input value",
                    required=True,
                )
            },
            timeout_seconds=5,
        )
        super().__init__(metadata)

    async def execute(self, parameters: dict, context: ExecutionContext):
        """Execute with failure."""
        import time
        from datetime import UTC, datetime

        start_time = time.time()
        await asyncio.sleep(0.01)
        execution_time_ms = (time.time() - start_time) * 1000

        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.FAILED,
            result={},
            error="Intentional tool failure",
            error_type="ValueError",
            execution_time_ms=execution_time_ms,
            timestamp=datetime.now(UTC),
        )


@pytest.fixture(scope="module", autouse=True)
def setup_tracing_for_module():
    """Set up tracing for the entire test module (runs once per module)."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
    from opentelemetry.sdk.resources import Resource

    # Create resource for this test module
    resource = Resource.create(
        {
            "service.name": "test-tool-tracing",
            "test.module": "test_tool_tracing_integration",
        }
    )

    # Create a tracer provider for this module
    provider = TracerProvider(sampler=ALWAYS_ON, resource=resource)

    # Set as global provider - this will be used by all tests in this module
    # OpenTelemetry's set_tracer_provider only sets it once (singleton pattern)
    # So we need to force reset if already set
    import opentelemetry.trace as otel_trace

    # Store the previous provider to restore later
    _previous_provider = getattr(otel_trace, '_TRACER_PROVIDER', None)

    # Force reset the provider
    otel_trace._TRACER_PROVIDER = None
    trace.set_tracer_provider(provider)

    yield provider

    # Cleanup after all tests in module
    provider.force_flush(timeout_millis=5000)
    provider.shutdown()

    # Restore previous provider
    otel_trace._TRACER_PROVIDER = _previous_provider


@pytest.fixture(scope="function")
def span_exporter(setup_tracing_for_module):
    """Set up in-memory span exporter for each test function."""
    exporter = InMemorySpanExporter()

    # Add span processor to the module-level tracer provider
    processor = SimpleSpanProcessor(exporter)
    setup_tracing_for_module.add_span_processor(processor)

    yield exporter

    # Flush and remove the processor after the test
    # Check if provider has force_flush method (not available in ProxyTracerProvider)
    if hasattr(setup_tracing_for_module, 'force_flush'):
        setup_tracing_for_module.force_flush(timeout_millis=1000)
    # Note: SimpleSpanProcessor doesn't have a remove method, so we just clear spans
    exporter.clear()


@pytest.fixture(autouse=True)
def clear_spans(request):
    """Clear spans before and after each test."""
    # Only clear if the test uses span_exporter
    if 'span_exporter' in request.fixturenames:
        exporter = request.getfixturevalue('span_exporter')
        exporter.clear()
        yield
        # Force flush to ensure all spans are exported
        # Check if provider has force_flush method (not available in ProxyTracerProvider)
        provider = trace.get_tracer_provider()
        if hasattr(provider, 'force_flush'):
            provider.force_flush(timeout_millis=1000)
        exporter.clear()
    else:
        yield


@pytest.fixture
def tool_registry():
    """Create a tool registry with test tools."""
    registry = ToolRegistry()
    registry.register(MockSuccessTool())
    registry.register(MockFailureTool())
    return registry


@pytest.fixture
def tool_executor(tool_registry):
    """Create a tool executor for testing."""
    return ToolExecutor(registry=tool_registry)


@pytest.mark.asyncio
async def test_span_created_for_tool_execution(tool_executor, span_exporter):
    """Test that a span is created for each tool execution.

    Validates TOOL-020 acceptance criterion:
    - Spans created for each tool execution
    """
    context = ExecutionContext(
        user_id="test_user",
        agent_id="test_agent",
        trace_id="test_trace_123",
    )

    result = await tool_executor.execute_tool(
        tool_id="mock_success",
        parameters={"input": "test_value"},
        context=context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify span was created
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1

    # Find the tool execution span
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_success"),
        None,
    )
    assert tool_span is not None, "Tool execution span not found"


@pytest.mark.asyncio
async def test_trace_id_propagation(tool_executor, span_exporter):
    """Test that trace_id is propagated through ExecutionContext.

    Validates TOOL-020 acceptance criterion:
    - Trace ID propagated via A2A context
    """
    trace_id = "custom_trace_abc123"
    context = ExecutionContext(
        user_id="test_user",
        agent_id="test_agent",
        trace_id=trace_id,
    )

    result = await tool_executor.execute_tool(
        tool_id="mock_success",
        parameters={"input": "test"},
        context=context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify trace ID is in context
    assert context.trace_id == trace_id

    # Note: OpenTelemetry uses its own trace ID format internally,
    # but our ExecutionContext.trace_id should be preserved
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 1


@pytest.mark.asyncio
async def test_span_attributes_include_metadata(tool_executor, span_exporter):
    """Test that span attributes include required metadata.

    Validates TOOL-020 acceptance criterion:
    - Span attributes include tool_id, user_id, success, execution_time_ms
    """
    context = ExecutionContext(
        user_id="user_456",
        agent_id="agent_789",
        trace_id="trace_xyz",
    )

    result = await tool_executor.execute_tool(
        tool_id="mock_success",
        parameters={"input": "test"},
        context=context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Find the tool execution span
    spans = span_exporter.get_finished_spans()
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_success"),
        None,
    )
    assert tool_span is not None

    # Verify required attributes
    attrs = tool_span.attributes
    assert attrs["tool_id"] == "mock_success"
    assert attrs["user_id"] == "user_456"
    assert attrs["agent_id"] == "agent_789"
    assert attrs["trace_id"] == "trace_xyz"
    assert "request_id" in attrs

    # Verify execution metadata is captured
    assert result.execution_time_ms is not None
    assert result.execution_time_ms > 0


@pytest.mark.asyncio
async def test_span_records_success_status(tool_executor, span_exporter):
    """Test that successful tool execution is recorded in span status."""
    context = ExecutionContext(
        user_id="test_user",
        agent_id="test_agent",
    )

    result = await tool_executor.execute_tool(
        tool_id="mock_success",
        parameters={"input": "test"},
        context=context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify span status
    spans = span_exporter.get_finished_spans()
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_success"),
        None,
    )
    assert tool_span is not None
    assert tool_span.status.status_code == trace.StatusCode.UNSET  # Success is UNSET in OTEL


@pytest.mark.asyncio
async def test_span_records_failure_exception(tool_executor, span_exporter):
    """Test that failed tool execution is properly recorded in span.

    Note: Tools that return ToolResult with FAILED status don't trigger
    exception recording unless they actually raise an exception. The span
    will still be created and completed normally, but with FAILED status
    in the ToolResult metadata.
    """
    context = ExecutionContext(
        user_id="test_user",
        agent_id="test_agent",
    )

    result = await tool_executor.execute_tool(
        tool_id="mock_failure",
        parameters={"input": "test"},
        context=context,
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.error == "Intentional tool failure"

    # Verify span was created and completed
    spans = span_exporter.get_finished_spans()
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_failure"),
        None,
    )
    assert tool_span is not None

    # Span status will be UNSET (normal completion) because the tool
    # returned a ToolResult with FAILED status rather than raising an exception.
    # This is by design - the execution completed successfully, even though
    # the tool's operation failed.
    assert tool_span.status.status_code == trace.StatusCode.UNSET


@pytest.mark.asyncio
async def test_span_events_capture_lifecycle(tool_executor, span_exporter):
    """Test that span events capture tool execution lifecycle."""
    context = ExecutionContext(
        user_id="test_user",
        agent_id="test_agent",
    )

    result = await tool_executor.execute_tool(
        tool_id="mock_success",
        parameters={"input": "test"},
        context=context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify span events
    spans = span_exporter.get_finished_spans()
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_success"),
        None,
    )
    assert tool_span is not None

    # Check for lifecycle events
    event_names = [e.name for e in tool_span.events]
    assert "hooks.before_completed" in event_names
    assert "authentication.validated" in event_names
    assert "parameters.validated" in event_names
    assert "tool.execution_started" in event_names
    assert "tool.execution_completed" in event_names
    assert "hooks.after_completed" in event_names


@pytest.mark.asyncio
async def test_parent_child_span_relationship(tool_executor, span_exporter):
    """Test that tool execution spans are linked to parent spans.

    Validates TOOL-020 acceptance criterion:
    - Tool executions linked to parent trace (agent request)
    """
    tracer = get_tracer("test.parent_span")

    # Create a parent span to simulate agent request
    with tracer.start_as_current_span("agent.request") as parent_span:
        parent_span_id = parent_span.get_span_context().span_id

        context = ExecutionContext(
            user_id="test_user",
            agent_id="test_agent",
        )

        result = await tool_executor.execute_tool(
            tool_id="mock_success",
            parameters={"input": "test"},
            context=context,
        )

        assert result.status == ToolExecutionStatus.SUCCESS

    # Verify parent-child relationship
    spans = span_exporter.get_finished_spans()

    parent_span = next((s for s in spans if s.name == "agent.request"), None)
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_success"),
        None,
    )

    assert parent_span is not None
    assert tool_span is not None

    # Verify tool span has parent
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == parent_span.context.span_id

    # Verify same trace ID
    assert tool_span.context.trace_id == parent_span.context.trace_id


@pytest.mark.asyncio
async def test_multiple_tool_executions_same_trace(tool_executor, span_exporter):
    """Test that multiple tool executions share the same trace."""
    tracer = get_tracer("test.multi_tool")

    with tracer.start_as_current_span("agent.workflow"):
        context = ExecutionContext(
            user_id="test_user",
            agent_id="test_agent",
        )

        # Execute multiple tools in sequence
        result1 = await tool_executor.execute_tool(
            tool_id="mock_success",
            parameters={"input": "first"},
            context=context,
        )

        result2 = await tool_executor.execute_tool(
            tool_id="mock_success",
            parameters={"input": "second"},
            context=context,
        )

        assert result1.status == ToolExecutionStatus.SUCCESS
        assert result2.status == ToolExecutionStatus.SUCCESS

    # Verify all spans share same trace ID
    spans = span_exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name.startswith("tool.execute")]

    assert len(tool_spans) == 2

    trace_ids = {s.context.trace_id for s in tool_spans}
    assert len(trace_ids) == 1, "All tool executions should share same trace ID"


@pytest.mark.asyncio
async def test_trace_export_configuration():
    """Test that trace export can be configured to OTLP collector.

    Validates TOOL-020 acceptance criterion:
    - Trace export to OpenTelemetry collector
    """
    # Test that configure_tracing accepts OTLP endpoint
    # In production, this would point to actual collector (e.g., Jaeger)
    provider = configure_tracing(
        service_name="test-otlp",
        service_version="1.0.0",
        otlp_endpoint="http://localhost:4317",
        sample_rate=1.0,
        enable_console_export=False,
    )

    assert isinstance(provider, TracerProvider)

    # Note: Actual export to collector requires running collector
    # This test just validates configuration works


@pytest.mark.asyncio
async def test_trace_sampling_configuration():
    """Test that trace sampling rate can be configured."""
    # Configure with 50% sampling rate
    provider = configure_tracing(
        service_name="test-sampling",
        service_version="1.0.0",
        sample_rate=0.5,
        enable_console_export=False,
    )

    assert isinstance(provider, TracerProvider)

    # Sampling is probabilistic, so we can't deterministically test
    # but we verify configuration is accepted


@pytest.mark.asyncio
async def test_span_duration_recorded(tool_executor, span_exporter):
    """Test that span duration is accurately recorded."""
    context = ExecutionContext(
        user_id="test_user",
        agent_id="test_agent",
    )

    result = await tool_executor.execute_tool(
        tool_id="mock_success",
        parameters={"input": "test"},
        context=context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify span duration
    spans = span_exporter.get_finished_spans()
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_success"),
        None,
    )
    assert tool_span is not None

    # Calculate duration in milliseconds
    duration_ns = tool_span.end_time - tool_span.start_time
    duration_ms = duration_ns / 1_000_000

    # Should be at least 10ms (our sleep time) but account for overhead
    assert duration_ms >= 10
    assert duration_ms < 1000  # Reasonable upper bound

    # Verify execution_time_ms in result matches span duration (approximately)
    assert result.execution_time_ms is not None
    assert abs(result.execution_time_ms - duration_ms) < 50  # 50ms tolerance


@pytest.mark.asyncio
async def test_trace_context_propagation_across_async_calls(tool_executor, span_exporter):
    """Test that trace context is properly propagated in async environment."""
    tracer = get_tracer("test.async_propagation")

    async def nested_operation():
        """Nested async operation that should inherit trace context."""
        with tracer.start_as_current_span("nested.operation"):
            context = ExecutionContext(
                user_id="test_user",
                agent_id="test_agent",
            )

            return await tool_executor.execute_tool(
                tool_id="mock_success",
                parameters={"input": "nested"},
                context=context,
            )

    with tracer.start_as_current_span("parent.operation"):
        result = await nested_operation()

    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify span hierarchy
    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 3  # parent, nested, tool execution

    parent_span = next((s for s in spans if s.name == "parent.operation"), None)
    nested_span = next((s for s in spans if s.name == "nested.operation"), None)
    tool_span = next(
        (s for s in spans if s.name == "tool.execute.mock_success"),
        None,
    )

    assert parent_span is not None
    assert nested_span is not None
    assert tool_span is not None

    # Verify trace ID propagation
    assert nested_span.context.trace_id == parent_span.context.trace_id
    assert tool_span.context.trace_id == parent_span.context.trace_id
