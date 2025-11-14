"""OpenTelemetry Distributed Tracing for Agent Runtime.

Provides distributed tracing integration for tool execution and agent operations.
Implements comprehensive observability as per FR-3.5 in the tool integration spec.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: trace.Tracer | None = None
_tracer_provider: TracerProvider | None = None


def configure_tracing(
    service_name: str = "agentcore-runtime",
    service_version: str = "1.0.0",
    otlp_endpoint: str | None = None,
    sample_rate: float = 1.0,
    enable_console_export: bool = False,
) -> TracerProvider:
    """Configure OpenTelemetry distributed tracing for agent runtime.

    Sets up the tracer provider with appropriate exporters and sampling.
    Should be called during application startup.

    Args:
        service_name: Name of the service for tracing
        service_version: Version of the service
        otlp_endpoint: OTLP endpoint for trace export (e.g., "http://jaeger:4317")
        sample_rate: Sampling rate for traces (0.0 to 1.0, default 1.0 = all traces)
        enable_console_export: Enable console span exporter for debugging

    Returns:
        Configured TracerProvider instance

    Example:
        >>> configure_tracing(
        ...     service_name="agentcore-runtime",
        ...     service_version="1.0.0",
        ...     otlp_endpoint="http://localhost:4317",
        ...     sample_rate=0.1
        ... )
    """
    global _tracer_provider

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "telemetry.sdk.language": "python",
            "telemetry.sdk.name": "opentelemetry",
            "deployment.environment": "production",  # TODO: Make configurable
        }
    )

    # Create tracer provider with sampling
    _tracer_provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(sample_rate),
    )

    # Add OTLP exporter if endpoint is provided
    if otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"Configured OTLP trace exporter to {otlp_endpoint}")
        except Exception as e:
            logger.error(f"Failed to configure OTLP exporter: {e}")

    # Add console exporter for debugging
    if enable_console_export:
        console_exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
        logger.info("Enabled console span exporter for debugging")

    # Set global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    logger.info(
        f"Configured distributed tracing for {service_name} "
        f"(version={service_version}, sample_rate={sample_rate})"
    )

    return _tracer_provider


def get_tracer(name: str = "agentcore.agent_runtime") -> trace.Tracer:
    """Get a tracer instance for creating spans.

    Args:
        name: Name of the tracer (typically module or component name)

    Returns:
        Tracer instance for creating spans

    Example:
        >>> tracer = get_tracer("agentcore.agent_runtime.tools")
        >>> with tracer.start_as_current_span("execute_tool"):
        ...     # Tool execution logic
        ...     pass
    """
    # Always get tracer from global provider (may be default or configured)
    return trace.get_tracer(name)


def add_span_attributes(**attributes: str | int | float | bool) -> None:
    """Add custom attributes to the current span.

    Useful for adding business-specific metadata to traces such as
    tool_id, agent_id, user_id, etc.

    Args:
        **attributes: Key-value pairs to add as span attributes

    Example:
        >>> add_span_attributes(
        ...     tool_id="google_search",
        ...     agent_id="agent-123",
        ...     user_id="user-456"
        ... )
    """
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, **attributes: str | int | float | bool) -> None:
    """Add an event to the current span.

    Events represent significant points in execution such as
    "parameter_validation_started", "tool_invoked", "retry_attempt", etc.

    Args:
        name: Name of the event
        **attributes: Event attributes

    Example:
        >>> add_span_event("retry_attempt", attempt=2, delay_ms=1000)
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, attributes=attributes)


def record_exception(exception: Exception) -> None:
    """Record an exception in the current span.

    Captures the exception details and marks the span as errored.
    Should be called when catching exceptions that affect the operation.

    Args:
        exception: Exception to record

    Example:
        >>> try:
        ...     execute_risky_tool()
        ... except ToolExecutionError as e:
        ...     record_exception(e)
        ...     raise
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.record_exception(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))


def get_trace_id() -> str:
    """Get the trace ID for the current span.

    Returns:
        Trace ID as a hexadecimal string, or empty string if no active span

    Example:
        >>> trace_id = get_trace_id()
        >>> logger.info(f"Processing tool execution with trace_id={trace_id}")
    """
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return ""


def get_span_id() -> str:
    """Get the span ID for the current span.

    Returns:
        Span ID as a hexadecimal string, or empty string if no active span

    Example:
        >>> span_id = get_span_id()
        >>> logger.debug(f"Current span_id={span_id}")
    """
    span = trace.get_current_span()
    if span.is_recording():
        ctx = span.get_span_context()
        if ctx.is_valid:
            return format(ctx.span_id, "016x")
    return ""
