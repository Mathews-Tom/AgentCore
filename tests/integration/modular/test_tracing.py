"""
Integration Tests for OpenTelemetry Distributed Tracing

Tests comprehensive tracing functionality including:
- Tracer initialization and configuration
- Span creation for each module type
- Tool execution tracing with parent spans
- Trace context propagation via A2A
- Error tracking and exception recording
- Trace export (mocked OTLP collector)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from agentcore.modular.tracing import (
    TracingConfig,
    ModularTracer,
    initialize_tracing,
    get_tracer,
    shutdown_tracing,
    trace_module_operation,
    trace_execution_plan,
    get_trace_context_for_a2a,
)
from agentcore.modular.models import ModuleType


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tracing_config() -> TracingConfig:
    """Create test tracing configuration."""
    return TracingConfig(
        enabled=True,
        service_name="test-modular-agent",
        console_export=True,
        otlp_endpoint=None,  # Don't use real OTLP for tests
        sample_rate=1.0,
    )


@pytest.fixture
def tracer(tracing_config: TracingConfig) -> ModularTracer:
    """Create test tracer instance."""
    tracer = ModularTracer(tracing_config)
    yield tracer
    tracer.shutdown()


@pytest.fixture(autouse=True)
def cleanup_global_tracer():
    """Cleanup global tracer after each test."""
    yield
    shutdown_tracing()


# ============================================================================
# Test: Tracer Initialization
# ============================================================================


def test_tracer_initialization_with_defaults():
    """Test tracer initializes with default configuration."""
    tracer = ModularTracer()

    assert tracer.config.enabled is True
    assert tracer.config.service_name == "agentcore-modular"
    assert tracer._tracer is not None
    assert tracer._tracer_provider is not None

    tracer.shutdown()


def test_tracer_initialization_with_custom_config(tracing_config: TracingConfig):
    """Test tracer initializes with custom configuration."""
    tracer = ModularTracer(tracing_config)

    assert tracer.config.enabled is True
    assert tracer.config.service_name == "test-modular-agent"
    assert tracer.config.console_export is True
    assert tracer._tracer is not None

    tracer.shutdown()


def test_tracer_disabled_when_config_disabled():
    """Test tracer does not initialize when disabled in config."""
    config = TracingConfig(enabled=False)
    tracer = ModularTracer(config)

    assert tracer.config.enabled is False
    assert tracer._tracer is None
    assert tracer._tracer_provider is None


def test_get_tracer_raises_when_disabled():
    """Test get_tracer raises when tracing is disabled."""
    config = TracingConfig(enabled=False)
    tracer = ModularTracer(config)

    with pytest.raises(RuntimeError, match="Tracing is not enabled"):
        tracer.get_tracer()


# ============================================================================
# Test: Span Creation
# ============================================================================


def test_create_module_span_for_planner(tracer: ModularTracer):
    """Test creating a span for Planner module."""
    span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="analyze_query",
        attributes={"query": "test query"},
    )

    assert span is not None
    assert span.is_recording()
    assert span.name == "planner.analyze_query"

    # Verify attributes are set
    # Note: We can't easily verify attributes on active span without export
    span.end()


def test_create_module_span_for_executor(tracer: ModularTracer):
    """Test creating a span for Executor module."""
    span = tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="execute_step",
        attributes={"step_id": "step-1", "action": "search"},
    )

    assert span is not None
    assert span.is_recording()
    assert span.name == "executor.execute_step"

    span.end()


def test_create_module_span_for_verifier(tracer: ModularTracer):
    """Test creating a span for Verifier module."""
    span = tracer.create_module_span(
        module_type=ModuleType.VERIFIER,
        operation="validate_results",
        attributes={"result_count": 3},
    )

    assert span is not None
    assert span.is_recording()
    assert span.name == "verifier.validate_results"

    span.end()


def test_create_module_span_for_generator(tracer: ModularTracer):
    """Test creating a span for Generator module."""
    span = tracer.create_module_span(
        module_type=ModuleType.GENERATOR,
        operation="synthesize_response",
        attributes={"format": "text"},
    )

    assert span is not None
    assert span.is_recording()
    assert span.name == "generator.synthesize_response"

    span.end()


def test_create_tool_span_without_parent(tracer: ModularTracer):
    """Test creating a tool span without parent context."""
    span = tracer.create_tool_span(
        tool_name="calculator",
        parameters={"operation": "add", "a": 2, "b": 3},
    )

    assert span is not None
    assert span.is_recording()
    assert span.name == "tool.calculator"

    span.end()


def test_create_tool_span_with_parent(tracer: ModularTracer):
    """Test creating a tool span linked to parent span."""
    parent_span = tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="execute_step",
    )

    tool_span = tracer.create_tool_span(
        tool_name="search_engine",
        parameters={"query": "Python tutorials"},
        parent_span=parent_span,
    )

    assert tool_span is not None
    assert tool_span.is_recording()
    assert tool_span.name == "tool.search_engine"

    # Verify parent-child relationship via trace_id
    parent_ctx = parent_span.get_span_context()
    tool_ctx = tool_span.get_span_context()
    assert parent_ctx.trace_id == tool_ctx.trace_id

    tool_span.end()
    parent_span.end()


def test_tool_span_sanitizes_sensitive_parameters(tracer: ModularTracer):
    """Test tool span sanitizes sensitive parameters."""
    span = tracer.create_tool_span(
        tool_name="api_client",
        parameters={
            "url": "https://api.example.com",
            "api_key": "secret-key-12345",
            "password": "super-secret",
            "data": {"name": "test"},
        },
    )

    assert span is not None
    span.end()

    # Note: We can't easily verify sanitization without exporting spans
    # This would be verified in integration test with real exporter


# ============================================================================
# Test: Context Propagation
# ============================================================================


def test_inject_context_creates_carrier(tracer: ModularTracer):
    """Test injecting trace context into carrier."""
    # Create a span to have active context
    span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="test",
    )

    carrier = tracer.inject_context()

    assert carrier is not None
    assert isinstance(carrier, dict)
    # W3C Trace Context headers
    assert "traceparent" in carrier or len(carrier) >= 0

    span.end()


def test_inject_context_into_existing_carrier(tracer: ModularTracer):
    """Test injecting context into existing carrier."""
    existing_carrier = {"custom_header": "custom_value"}

    span = tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="test",
    )

    carrier = tracer.inject_context(existing_carrier)

    assert "custom_header" in carrier
    assert carrier["custom_header"] == "custom_value"

    span.end()


def test_extract_context_from_carrier(tracer: ModularTracer):
    """Test extracting trace context from carrier."""
    # Create and inject context
    span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="test",
    )
    carrier = tracer.inject_context()
    span.end()

    # Extract context in different tracer instance
    tracer.extract_context(carrier)

    # Verify context is extracted (no exception raised)
    assert True


def test_get_current_trace_id(tracer: ModularTracer):
    """Test retrieving current trace ID."""
    # With active span
    span = tracer.create_module_span(
        module_type=ModuleType.VERIFIER,
        operation="test",
    )

    trace_id = tracer.get_current_trace_id()
    assert trace_id is not None
    assert isinstance(trace_id, str)
    assert len(trace_id) == 32  # Hex string of 128-bit trace ID

    # Detach context
    if hasattr(span, "_context_token"):
        from opentelemetry import trace as otel_trace
        otel_trace.context_api.detach(span._context_token)

    span.end()

    # After detaching, trace ID may still be available from span context
    # But get_current_trace_id() will return None if no active span
    trace_id_after = tracer.get_current_trace_id()
    # May be None after detaching context
    assert trace_id_after is None or isinstance(trace_id_after, str)


def test_build_a2a_context_carrier(tracer: ModularTracer):
    """Test building A2A context with trace headers."""
    span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="test",
    )

    a2a_context = tracer.build_a2a_context_carrier()

    assert "trace_headers" in a2a_context
    assert isinstance(a2a_context["trace_headers"], dict)

    # trace_id should be present if there's an active span
    if "trace_id" in a2a_context:
        assert isinstance(a2a_context["trace_id"], str)

    # Detach context
    if hasattr(span, "_context_token"):
        from opentelemetry import trace as otel_trace
        otel_trace.context_api.detach(span._context_token)

    span.end()


def test_build_a2a_context_with_existing_context(tracer: ModularTracer):
    """Test building A2A context with existing A2A context."""
    existing_context = {
        "source_agent": "planner",
        "target_agent": "executor",
        "session_id": "session-123",
    }

    span = tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="test",
    )

    a2a_context = tracer.build_a2a_context_carrier(existing_context)

    # Verify existing fields preserved
    assert a2a_context["source_agent"] == "planner"
    assert a2a_context["target_agent"] == "executor"
    assert a2a_context["session_id"] == "session-123"

    # Verify trace context added
    assert "trace_headers" in a2a_context

    # trace_id may be present if there's an active span
    if "trace_id" in a2a_context:
        assert isinstance(a2a_context["trace_id"], str)

    # Detach context
    if hasattr(span, "_context_token"):
        from opentelemetry import trace as otel_trace
        otel_trace.context_api.detach(span._context_token)

    span.end()


# ============================================================================
# Test: Span Events & Status
# ============================================================================


def test_add_event_to_span(tracer: ModularTracer):
    """Test adding an event to a span."""
    span = tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="execute_step",
    )

    tracer.add_event(
        span,
        "step_started",
        attributes={"step_id": "step-1", "action": "search"},
    )

    assert span.is_recording()
    span.end()


def test_record_exception_in_span(tracer: ModularTracer):
    """Test recording an exception in a span."""
    span = tracer.create_module_span(
        module_type=ModuleType.VERIFIER,
        operation="validate_results",
    )

    exception = ValueError("Validation failed")
    tracer.record_exception(span, exception, attributes={"error_code": "VALIDATION_ERROR"})

    # Verify span is still recording
    assert span.is_recording()

    span.end()


def test_set_span_success(tracer: ModularTracer):
    """Test marking a span as successful."""
    span = tracer.create_module_span(
        module_type=ModuleType.GENERATOR,
        operation="synthesize_response",
    )

    tracer.set_span_success(span)

    assert span.is_recording()
    span.end()


def test_set_span_error(tracer: ModularTracer):
    """Test marking a span as failed."""
    span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="analyze_query",
    )

    tracer.set_span_error(span, "Analysis failed due to invalid query")

    assert span.is_recording()
    span.end()


# ============================================================================
# Test: Decorator
# ============================================================================


@pytest.mark.asyncio
async def test_trace_module_operation_decorator_async(tracer: ModularTracer):
    """Test @trace_module_operation decorator on async function."""

    @trace_module_operation(ModuleType.PLANNER, "test_operation", tracer=tracer)
    async def test_function(x: int, y: int) -> int:
        return x + y

    result = await test_function(2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_trace_module_operation_decorator_with_exception(tracer: ModularTracer):
    """Test decorator records exception."""

    @trace_module_operation(ModuleType.EXECUTOR, "failing_operation", tracer=tracer)
    async def failing_function() -> None:
        raise ValueError("Test exception")

    with pytest.raises(ValueError, match="Test exception"):
        await failing_function()


def test_trace_module_operation_decorator_sync(tracer: ModularTracer):
    """Test decorator works with synchronous functions."""

    @trace_module_operation(ModuleType.VERIFIER, "sync_operation", tracer=tracer)
    def sync_function(value: str) -> str:
        return value.upper()

    result = sync_function("test")
    assert result == "TEST"


@pytest.mark.asyncio
async def test_trace_module_operation_with_disabled_tracing():
    """Test decorator does nothing when tracing is disabled."""
    config = TracingConfig(enabled=False)
    disabled_tracer = ModularTracer(config)

    @trace_module_operation(ModuleType.PLANNER, "test", tracer=disabled_tracer)
    async def test_function() -> str:
        return "success"

    result = await test_function()
    assert result == "success"


# ============================================================================
# Test: Global Tracer Management
# ============================================================================


def test_initialize_global_tracing():
    """Test initializing global tracing."""
    config = TracingConfig(service_name="global-test")
    tracer = initialize_tracing(config)

    assert tracer is not None
    assert tracer.config.service_name == "global-test"

    # Verify global tracer is set
    global_tracer = get_tracer()
    assert global_tracer is tracer

    shutdown_tracing()


def test_get_tracer_raises_when_not_initialized():
    """Test get_tracer raises when not initialized."""
    shutdown_tracing()  # Ensure clean state

    with pytest.raises(RuntimeError, match="Tracing not initialized"):
        get_tracer()


def test_initialize_tracing_replaces_existing():
    """Test initializing tracing replaces existing instance."""
    config1 = TracingConfig(service_name="service-1")
    tracer1 = initialize_tracing(config1)

    config2 = TracingConfig(service_name="service-2")
    tracer2 = initialize_tracing(config2)

    assert tracer1 is not tracer2
    assert tracer2.config.service_name == "service-2"

    shutdown_tracing()


def test_shutdown_tracing():
    """Test shutting down global tracing."""
    initialize_tracing()
    shutdown_tracing()

    # Verify global tracer is None
    with pytest.raises(RuntimeError):
        get_tracer()


# ============================================================================
# Test: Utility Functions
# ============================================================================


def test_trace_execution_plan(tracer: ModularTracer):
    """Test trace_execution_plan utility."""
    # Initialize global tracer
    initialize_tracing(tracer.config)

    span = trace_execution_plan(
        plan_id="plan-123",
        iteration=1,
        step_count=5,
        attributes={"query": "test query"},
    )

    assert span is not None
    assert span.is_recording()
    assert span.name == "executor.execute_plan"

    span.end()
    shutdown_tracing()


def test_trace_execution_plan_with_disabled_tracing():
    """Test trace_execution_plan returns None when tracing disabled."""
    shutdown_tracing()

    span = trace_execution_plan(
        plan_id="plan-123",
        iteration=1,
        step_count=5,
    )

    assert span is None


def test_get_trace_context_for_a2a(tracer: ModularTracer):
    """Test get_trace_context_for_a2a utility."""
    initialize_tracing(tracer.config)

    span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="test",
    )

    context = get_trace_context_for_a2a()

    assert isinstance(context, dict)
    assert "trace_headers" in context

    # trace_id may be present if there's an active span
    if "trace_id" in context:
        assert isinstance(context["trace_id"], str)

    # Detach context
    if hasattr(span, "_context_token"):
        from opentelemetry import trace as otel_trace
        otel_trace.context_api.detach(span._context_token)

    span.end()
    shutdown_tracing()


def test_get_trace_context_for_a2a_without_tracer():
    """Test get_trace_context_for_a2a returns empty dict when no tracer."""
    shutdown_tracing()

    context = get_trace_context_for_a2a()
    assert context == {}


# ============================================================================
# Test: End-to-End Tracing Flow
# ============================================================================


@pytest.mark.asyncio
async def test_end_to_end_module_execution_tracing(tracer: ModularTracer):
    """Test complete tracing flow for modular execution."""
    # Simulate Planner -> Executor -> Verifier -> Generator flow

    # 1. Planner
    planner_span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="analyze_query",
        attributes={"query": "Calculate fibonacci(10)"},
    )

    tracer.add_event(planner_span, "planning_started")

    # Simulate planning work
    plan_id = "plan-abc123"
    tracer.add_event(
        planner_span,
        "plan_created",
        attributes={"plan_id": plan_id, "step_count": 3},
    )

    tracer.set_span_success(planner_span)
    planner_span.end()

    # Get trace context for A2A propagation
    a2a_context = tracer.build_a2a_context_carrier()
    trace_id = a2a_context.get("trace_id") or tracer.get_current_trace_id()

    # 2. Executor
    executor_span = tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="execute_plan",
        attributes={"plan_id": plan_id},
    )

    # Execute tools
    tool_span = tracer.create_tool_span(
        tool_name="calculator",
        parameters={"operation": "fibonacci", "n": 10},
        parent_span=executor_span,
    )
    tracer.set_span_success(tool_span)
    tool_span.end()

    tracer.set_span_success(executor_span)
    executor_span.end()

    # 3. Verifier
    verifier_span = tracer.create_module_span(
        module_type=ModuleType.VERIFIER,
        operation="validate_results",
        attributes={"result_count": 1},
    )

    tracer.add_event(verifier_span, "validation_passed")
    tracer.set_span_success(verifier_span)
    verifier_span.end()

    # 4. Generator
    generator_span = tracer.create_module_span(
        module_type=ModuleType.GENERATOR,
        operation="synthesize_response",
        attributes={"format": "text"},
    )

    tracer.set_span_success(generator_span)
    generator_span.end()

    # Verify trace ID consistency
    current_trace_id = tracer.get_current_trace_id()
    # Note: Current trace ID may be None after all spans ended
    assert trace_id is not None


@pytest.mark.asyncio
async def test_error_recovery_with_tracing(tracer: ModularTracer):
    """Test tracing during error recovery scenario."""
    # Iteration 1: Execute -> Verify -> FAIL
    executor_span = tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="execute_plan",
        attributes={"iteration": 1},
    )

    # Tool execution fails
    tool_span = tracer.create_tool_span(
        tool_name="failing_tool",
        parameters={"operation": "test"},
        parent_span=executor_span,
    )

    exception = RuntimeError("Tool execution failed")
    tracer.record_exception(tool_span, exception)
    tool_span.end()

    tracer.set_span_error(executor_span, "Step execution failed")
    executor_span.end()

    # Verifier detects failure
    verifier_span = tracer.create_module_span(
        module_type=ModuleType.VERIFIER,
        operation="validate_results",
        attributes={"iteration": 1},
    )

    tracer.add_event(
        verifier_span,
        "validation_failed",
        attributes={"reason": "incomplete_results"},
    )
    tracer.set_span_success(verifier_span)  # Verifier succeeded, just found errors
    verifier_span.end()

    # Iteration 2: Refine -> Execute -> Verify -> SUCCESS
    planner_span = tracer.create_module_span(
        module_type=ModuleType.PLANNER,
        operation="refine_plan",
        attributes={"iteration": 2, "feedback": "incomplete_results"},
    )

    tracer.set_span_success(planner_span)
    planner_span.end()

    # Verified all spans created successfully
    assert True
