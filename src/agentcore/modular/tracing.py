"""
OpenTelemetry Distributed Tracing for Modular Agent Core

Provides comprehensive distributed tracing infrastructure for tracking execution
flow across all modules (Planner, Executor, Verifier, Generator) with span
creation, context propagation, and trace export.

Key Features:
- Automatic span creation for each module execution
- Trace context propagation via A2A context
- Tool execution linking to parent trace
- Trace export to OTLP collectors (Jaeger, Zipkin, etc.)
- Span attributes and events for rich context
- Error tracking and exception recording
"""

from __future__ import annotations

import contextvars
import functools
from typing import Any, Callable, TypeVar, ParamSpec
from datetime import datetime, timezone

import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode, Span, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel, Field

from agentcore.modular.models import ModuleType

logger = structlog.get_logger()

# Type variables for decorators
P = ParamSpec("P")
T = TypeVar("T")

# Context variable for current trace context
_trace_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "trace_context", default={}
)


# ============================================================================
# Configuration Models
# ============================================================================


class TracingConfig(BaseModel):
    """Configuration for OpenTelemetry tracing."""

    enabled: bool = Field(default=True, description="Enable/disable tracing")
    service_name: str = Field(
        default="agentcore-modular", description="Service name for traces"
    )
    otlp_endpoint: str | None = Field(
        None, description="OTLP exporter endpoint (e.g., http://localhost:4317)"
    )
    console_export: bool = Field(
        default=False, description="Export traces to console for debugging"
    )
    sample_rate: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Trace sampling rate (0.0-1.0)"
    )
    batch_export: bool = Field(
        default=True, description="Use batch span processor (vs simple)"
    )
    max_export_batch_size: int = Field(
        default=512, description="Maximum batch size for export"
    )
    export_timeout_ms: int = Field(
        default=30000, description="Export timeout in milliseconds"
    )


# ============================================================================
# Tracer Provider Setup
# ============================================================================


class ModularTracer:
    """
    OpenTelemetry tracer for modular agent system.

    Manages tracer initialization, span creation, context propagation,
    and trace export to OTLP collectors.
    """

    def __init__(self, config: TracingConfig | None = None) -> None:
        """
        Initialize the modular tracer.

        Args:
            config: Tracing configuration (None for defaults)
        """
        self.config = config or TracingConfig()
        self._tracer_provider: TracerProvider | None = None
        self._tracer: trace.Tracer | None = None
        self._propagator = TraceContextTextMapPropagator()

        if self.config.enabled:
            self._setup_tracer()

        logger.info(
            "ModularTracer initialized",
            enabled=self.config.enabled,
            service_name=self.config.service_name,
            otlp_endpoint=self.config.otlp_endpoint,
        )

    def _setup_tracer(self) -> None:
        """Set up OpenTelemetry tracer provider and exporters."""
        # Create resource with service name
        resource = Resource(attributes={SERVICE_NAME: self.config.service_name})

        # Create tracer provider
        self._tracer_provider = TracerProvider(resource=resource)

        # Add exporters
        if self.config.otlp_endpoint:
            # OTLP exporter for Jaeger, Zipkin, etc.
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            if self.config.batch_export:
                processor = BatchSpanProcessor(
                    otlp_exporter,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.export_timeout_ms,
                )
            else:
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor

                processor = SimpleSpanProcessor(otlp_exporter)
            self._tracer_provider.add_span_processor(processor)
            logger.info("OTLP exporter configured", endpoint=self.config.otlp_endpoint)

        if self.config.console_export:
            # Console exporter for debugging
            console_exporter = ConsoleSpanExporter()
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            processor = SimpleSpanProcessor(console_exporter)
            self._tracer_provider.add_span_processor(processor)
            logger.info("Console exporter configured")

        # Set global tracer provider
        trace.set_tracer_provider(self._tracer_provider)

        # Get tracer for this module
        self._tracer = trace.get_tracer(
            instrumenting_module_name="agentcore.modular",
            instrumenting_library_version="1.0.0",
        )

    def get_tracer(self) -> trace.Tracer:
        """
        Get the OpenTelemetry tracer.

        Returns:
            Tracer instance

        Raises:
            RuntimeError: If tracing is disabled
        """
        if not self.config.enabled or not self._tracer:
            raise RuntimeError("Tracing is not enabled")
        return self._tracer

    def shutdown(self) -> None:
        """Shutdown tracer provider and flush pending spans."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
            logger.info("Tracer provider shutdown")

    # ========================================================================
    # Span Creation
    # ========================================================================

    def create_module_span(
        self,
        module_type: ModuleType,
        operation: str,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Create a span for a module operation.

        Args:
            module_type: Type of module (Planner, Executor, Verifier, Generator)
            operation: Operation name (e.g., "analyze_query", "execute_step")
            attributes: Additional span attributes

        Returns:
            Active span that must be ended by caller

        Raises:
            RuntimeError: If tracing is disabled
        """
        if not self.config.enabled or not self._tracer:
            raise RuntimeError("Tracing is not enabled")

        span_name = f"{module_type.value}.{operation}"

        span = self._tracer.start_span(
            name=span_name,
            kind=SpanKind.INTERNAL,
        )

        # Activate span in context
        token = trace.context_api.attach(trace.set_span_in_context(span))

        # Store token on span for cleanup (custom attribute)
        # Note: This is a workaround since we can't easily pass token back
        setattr(span, "_context_token", token)

        # Set standard attributes
        span.set_attribute("module.type", module_type.value)
        span.set_attribute("module.operation", operation)
        span.set_attribute("service.name", self.config.service_name)

        # Set custom attributes
        if attributes:
            for key, value in attributes.items():
                self._set_span_attribute(span, key, value)

        return span

    def create_tool_span(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        parent_span: Span | None = None,
    ) -> Span:
        """
        Create a span for tool execution linked to parent trace.

        Args:
            tool_name: Name of the tool being invoked
            parameters: Tool parameters
            parent_span: Parent span (None to use current context)

        Returns:
            Active span

        Raises:
            RuntimeError: If tracing is disabled
        """
        if not self.config.enabled or not self._tracer:
            raise RuntimeError("Tracing is not enabled")

        span_name = f"tool.{tool_name}"

        # Create span with parent context
        if parent_span:
            ctx = trace.set_span_in_context(parent_span)
            span = self._tracer.start_span(
                name=span_name,
                kind=SpanKind.CLIENT,
                context=ctx,
            )
        else:
            span = self._tracer.start_span(
                name=span_name,
                kind=SpanKind.CLIENT,
            )

        # Set attributes
        span.set_attribute("tool.name", tool_name)
        if parameters:
            # Sanitize parameters (remove sensitive data)
            safe_params = self._sanitize_parameters(parameters)
            span.set_attribute("tool.parameters", str(safe_params))

        return span

    def _set_span_attribute(self, span: Span, key: str, value: Any) -> None:
        """
        Set span attribute with type handling.

        Args:
            span: Span to set attribute on
            key: Attribute key
            value: Attribute value
        """
        # OpenTelemetry supports: str, bool, int, float, or sequences thereof
        if value is None:
            return
        elif isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        elif isinstance(value, (list, tuple)):
            # Convert to list of primitives
            primitive_list = [
                v for v in value if isinstance(v, (str, bool, int, float))
            ]
            if primitive_list:
                span.set_attribute(key, primitive_list)
        else:
            # Convert to string for complex types
            span.set_attribute(key, str(value))

    def _sanitize_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize parameters to remove sensitive data.

        Args:
            params: Parameters dictionary

        Returns:
            Sanitized parameters
        """
        sensitive_keys = {"api_key", "secret", "password", "token", "credential"}
        sanitized = {}

        for key, value in params.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized

    # ========================================================================
    # Context Propagation
    # ========================================================================

    def inject_context(self, carrier: dict[str, str] | None = None) -> dict[str, str]:
        """
        Inject trace context into carrier for propagation.

        Args:
            carrier: Carrier dictionary (None to create new)

        Returns:
            Carrier with injected trace context
        """
        if carrier is None:
            carrier = {}

        if self.config.enabled:
            self._propagator.inject(carrier)

        return carrier

    def extract_context(self, carrier: dict[str, str]) -> None:
        """
        Extract trace context from carrier and set in current context.

        Args:
            carrier: Carrier with trace context
        """
        if self.config.enabled:
            ctx = self._propagator.extract(carrier)
            # Store in context variable for access
            _trace_context.set(carrier)

    def get_current_trace_id(self) -> str | None:
        """
        Get current trace ID from active span.

        Returns:
            Trace ID or None if no active span
        """
        if not self.config.enabled:
            return None

        current_span = trace.get_current_span()
        if current_span:
            span_context = current_span.get_span_context()
            if span_context.is_valid:
                trace_id = span_context.trace_id
                # Convert to hex string
                return f"{trace_id:032x}"

        return None

    def build_a2a_context_carrier(
        self, a2a_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Build A2A context with trace propagation headers.

        Args:
            a2a_context: Existing A2A context (None to create new)

        Returns:
            A2A context with trace headers
        """
        if a2a_context is None:
            a2a_context = {}

        # Inject trace context
        trace_carrier = self.inject_context()
        a2a_context["trace_headers"] = trace_carrier

        # Add trace ID for easy reference
        trace_id = self.get_current_trace_id()
        if trace_id:
            a2a_context["trace_id"] = trace_id

        return a2a_context

    # ========================================================================
    # Span Events & Status
    # ========================================================================

    def add_event(
        self,
        span: Span,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an event to a span.

        Args:
            span: Span to add event to
            name: Event name
            attributes: Event attributes
        """
        event_attrs = {}
        if attributes:
            for key, value in attributes.items():
                if isinstance(value, (str, bool, int, float)):
                    event_attrs[key] = value
                else:
                    event_attrs[key] = str(value)

        span.add_event(name, attributes=event_attrs)

    def record_exception(
        self,
        span: Span,
        exception: Exception,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Record an exception in a span.

        Args:
            span: Span to record exception in
            exception: Exception to record
            attributes: Additional attributes
        """
        span.record_exception(exception, attributes=attributes)
        span.set_status(Status(StatusCode.ERROR, str(exception)))

        logger.error(
            "Exception recorded in span",
            exception=str(exception),
            exception_type=type(exception).__name__,
            span_name=span.name if hasattr(span, "name") else "unknown",
        )

    def set_span_success(self, span: Span) -> None:
        """
        Mark a span as successful.

        Args:
            span: Span to mark as successful
        """
        span.set_status(Status(StatusCode.OK))

    def set_span_error(self, span: Span, error_message: str) -> None:
        """
        Mark a span as failed.

        Args:
            span: Span to mark as failed
            error_message: Error message
        """
        span.set_status(Status(StatusCode.ERROR, error_message))


# ============================================================================
# Decorator for Automatic Tracing
# ============================================================================


def trace_module_operation(
    module_type: ModuleType,
    operation: str | None = None,
    tracer: ModularTracer | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to automatically trace module operations.

    Args:
        module_type: Type of module
        operation: Operation name (None to use function name)
        tracer: Tracer instance (None to use global)

    Returns:
        Decorated function

    Example:
        >>> @trace_module_operation(ModuleType.PLANNER, "analyze_query")
        ... async def analyze_query(self, query: str) -> Plan:
        ...     return await self._do_analysis(query)
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get tracer
            active_tracer = tracer or _get_global_tracer()
            if not active_tracer or not active_tracer.config.enabled:
                return await func(*args, **kwargs)  # type: ignore

            # Determine operation name
            op_name = operation or func.__name__

            # Extract attributes from kwargs
            attributes = {
                "function": func.__name__,
                "module": func.__module__,
            }

            # Start span
            span = active_tracer.create_module_span(
                module_type=module_type,
                operation=op_name,
                attributes=attributes,
            )

            try:
                # Execute function
                result = await func(*args, **kwargs)  # type: ignore

                # Mark success
                active_tracer.set_span_success(span)
                return result

            except Exception as e:
                # Record exception
                active_tracer.record_exception(span, e)
                raise

            finally:
                # End span
                span.end()

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Get tracer
            active_tracer = tracer or _get_global_tracer()
            if not active_tracer or not active_tracer.config.enabled:
                return func(*args, **kwargs)

            # Determine operation name
            op_name = operation or func.__name__

            # Extract attributes
            attributes = {
                "function": func.__name__,
                "module": func.__module__,
            }

            # Start span
            span = active_tracer.create_module_span(
                module_type=module_type,
                operation=op_name,
                attributes=attributes,
            )

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Mark success
                active_tracer.set_span_success(span)
                return result

            except Exception as e:
                # Record exception
                active_tracer.record_exception(span, e)
                raise

            finally:
                # End span
                span.end()

        # Return appropriate wrapper based on function type
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


# ============================================================================
# Global Tracer Management
# ============================================================================

_global_tracer: ModularTracer | None = None


def initialize_tracing(config: TracingConfig | None = None) -> ModularTracer:
    """
    Initialize global tracing infrastructure.

    Args:
        config: Tracing configuration (None for defaults)

    Returns:
        Initialized tracer instance
    """
    global _global_tracer

    if _global_tracer:
        logger.warning("Tracing already initialized, shutting down previous instance")
        _global_tracer.shutdown()

    _global_tracer = ModularTracer(config)
    logger.info("Global tracing initialized")

    return _global_tracer


def get_tracer() -> ModularTracer:
    """
    Get the global tracer instance.

    Returns:
        Global tracer

    Raises:
        RuntimeError: If tracing not initialized
    """
    if not _global_tracer:
        raise RuntimeError("Tracing not initialized. Call initialize_tracing() first.")
    return _global_tracer


def _get_global_tracer() -> ModularTracer | None:
    """Get global tracer without raising exception."""
    return _global_tracer


def shutdown_tracing() -> None:
    """Shutdown global tracing infrastructure."""
    global _global_tracer

    if _global_tracer:
        _global_tracer.shutdown()
        _global_tracer = None
        logger.info("Global tracing shutdown")


# ============================================================================
# Utility Functions
# ============================================================================


def trace_execution_plan(
    plan_id: str,
    iteration: int,
    step_count: int,
    attributes: dict[str, Any] | None = None,
) -> Span | None:
    """
    Create a span for execution plan processing.

    Args:
        plan_id: Plan identifier
        iteration: Current iteration number
        step_count: Number of steps in plan
        attributes: Additional attributes

    Returns:
        Active span or None if tracing disabled
    """
    tracer = _get_global_tracer()
    if not tracer or not tracer.config.enabled:
        return None

    span_attrs = {
        "plan.id": plan_id,
        "plan.iteration": iteration,
        "plan.step_count": step_count,
    }
    if attributes:
        span_attrs.update(attributes)

    return tracer.create_module_span(
        module_type=ModuleType.EXECUTOR,
        operation="execute_plan",
        attributes=span_attrs,
    )


def get_trace_context_for_a2a() -> dict[str, Any]:
    """
    Get trace context formatted for A2A protocol propagation.

    Returns:
        Dictionary with trace context for A2A
    """
    tracer = _get_global_tracer()
    if not tracer:
        return {}

    return tracer.build_a2a_context_carrier()
