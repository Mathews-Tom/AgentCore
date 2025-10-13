"""
Distributed tracing service for agent runtime using OpenTelemetry.

This module provides end-to-end tracing for agent execution, tool calls,
and cross-service communication with context propagation and span management.
"""

import functools
import time
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from typing import Any, ParamSpec, TypeVar
from uuid import uuid4

import structlog

logger = structlog.get_logger()

# Context variable for trace context
_trace_context: ContextVar[dict[str, Any]] = ContextVar(
    "trace_context",
    default={},
)


class SpanKind(str, Enum):
    """Types of spans in distributed tracing."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span completion status."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


class Span:
    """Represents a single span in a distributed trace."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        operation_name: str,
        kind: SpanKind,
        start_time: float | None = None,
    ) -> None:
        """
        Initialize span.

        Args:
            trace_id: Trace identifier
            span_id: Span identifier
            parent_span_id: Parent span ID
            operation_name: Name of the operation
            kind: Span kind
            start_time: Start timestamp (uses current time if None)
        """
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.kind = kind
        self.start_time = start_time or time.time()
        self.end_time: float | None = None
        self.status = SpanStatus.UNSET
        self.status_message: str | None = None
        self.attributes: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.links: list[dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set span attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """
        Add event to span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        }
        self.events.append(event)

    def add_link(
        self,
        trace_id: str,
        span_id: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Add link to another span.

        Args:
            trace_id: Linked trace ID
            span_id: Linked span ID
            attributes: Link attributes
        """
        link = {
            "trace_id": trace_id,
            "span_id": span_id,
            "attributes": attributes or {},
        }
        self.links.append(link)

    def finish(
        self,
        status: SpanStatus = SpanStatus.OK,
        status_message: str | None = None,
    ) -> None:
        """
        Finish span.

        Args:
            status: Span status
            status_message: Status message
        """
        self.end_time = time.time()
        self.status = status
        self.status_message = status_message

    def duration_ms(self) -> float:
        """
        Get span duration in milliseconds.

        Returns:
            Duration in milliseconds
        """
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """
        Convert span to dictionary.

        Returns:
            Span data as dictionary
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
            "links": self.links,
        }


class TraceContext:
    """Manages trace context for distributed tracing."""

    def __init__(
        self,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        baggage: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize trace context.

        Args:
            trace_id: Trace identifier (generates new if None)
            parent_span_id: Parent span ID
            baggage: Context baggage (key-value pairs)
        """
        self.trace_id = trace_id or str(uuid4())
        self.parent_span_id = parent_span_id
        self.baggage = baggage or {}
        self.active_span: Span | None = None

    def set_baggage(self, key: str, value: Any) -> None:
        """
        Set baggage item.

        Args:
            key: Baggage key
            value: Baggage value
        """
        self.baggage[key] = value

    def get_baggage(self, key: str) -> Any | None:
        """
        Get baggage item.

        Args:
            key: Baggage key

        Returns:
            Baggage value or None
        """
        return self.baggage.get(key)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context to dictionary for serialization.

        Returns:
            Context data as dictionary
        """
        return {
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceContext":
        """
        Create context from dictionary.

        Args:
            data: Context data dictionary

        Returns:
            TraceContext instance
        """
        return cls(
            trace_id=data.get("trace_id"),
            parent_span_id=data.get("parent_span_id"),
            baggage=data.get("baggage", {}),
        )


class DistributedTracer:
    """Main distributed tracing service."""

    def __init__(
        self,
        service_name: str = "agent-runtime",
        enable_export: bool = True,
    ) -> None:
        """
        Initialize distributed tracer.

        Args:
            service_name: Name of the service
            enable_export: Enable span export to backend
        """
        self._service_name = service_name
        self._enable_export = enable_export
        self._completed_spans: list[Span] = []
        self._max_spans = 10000

        logger.info(
            "distributed_tracer_initialized",
            service_name=service_name,
            export_enabled=enable_export,
        )

    def start_trace(
        self,
        trace_id: str | None = None,
        baggage: dict[str, Any] | None = None,
    ) -> TraceContext:
        """
        Start new trace.

        Args:
            trace_id: Trace identifier (generates new if None)
            baggage: Initial context baggage

        Returns:
            New trace context
        """
        context = TraceContext(trace_id=trace_id, baggage=baggage)
        _trace_context.set(context.to_dict())

        logger.info("trace_started", trace_id=context.trace_id)
        return context

    def get_current_context(self) -> TraceContext | None:
        """
        Get current trace context.

        Returns:
            Current trace context or None
        """
        context_data = _trace_context.get()
        if context_data:
            return TraceContext.from_dict(context_data)
        return None

    def start_span(
        self,
        operation_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """
        Start new span.

        Args:
            operation_name: Name of the operation
            kind: Span kind
            attributes: Initial span attributes

        Returns:
            New span instance
        """
        context = self.get_current_context()
        if not context:
            context = self.start_trace()

        # Create span
        span = Span(
            trace_id=context.trace_id,
            span_id=str(uuid4()),
            parent_span_id=context.parent_span_id,
            operation_name=operation_name,
            kind=kind,
        )

        # Set default attributes
        span.set_attribute("service.name", self._service_name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # Update context with current span
        context.active_span = span
        context.parent_span_id = span.span_id
        _trace_context.set(context.to_dict())

        logger.debug(
            "span_started",
            trace_id=span.trace_id,
            span_id=span.span_id,
            operation=operation_name,
        )

        return span

    def finish_span(
        self,
        span: Span,
        status: SpanStatus = SpanStatus.OK,
        status_message: str | None = None,
    ) -> None:
        """
        Finish span.

        Args:
            span: Span to finish
            status: Span status
            status_message: Status message
        """
        span.finish(status=status, status_message=status_message)

        # Store completed span
        self._completed_spans.append(span)
        if len(self._completed_spans) > self._max_spans:
            self._completed_spans.pop(0)

        # Export span if enabled
        if self._enable_export:
            self._export_span(span)

        logger.debug(
            "span_finished",
            trace_id=span.trace_id,
            span_id=span.span_id,
            duration_ms=span.duration_ms(),
            status=status.value,
        )

    def record_exception(
        self,
        span: Span,
        exception: Exception,
    ) -> None:
        """
        Record exception in span.

        Args:
            span: Span to record exception in
            exception: Exception that occurred
        """
        span.add_event(
            "exception",
            attributes={
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
                "exception.stacktrace": "",  # Could add traceback here
            },
        )
        span.set_attribute("error", True)

    def _export_span(self, span: Span) -> None:
        """
        Export span to backend.

        Args:
            span: Span to export
        """
        # In production, this would send to Jaeger, Zipkin, etc.
        # For now, we just log it
        logger.info(
            "span_exported",
            trace_id=span.trace_id,
            span_id=span.span_id,
            operation=span.operation_name,
            duration_ms=span.duration_ms(),
        )

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """
        Get all spans for a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            List of spans in the trace
        """
        return [s for s in self._completed_spans if s.trace_id == trace_id]

    def get_span_by_id(self, span_id: str) -> Span | None:
        """
        Get span by ID.

        Args:
            span_id: Span identifier

        Returns:
            Span or None if not found
        """
        for span in self._completed_spans:
            if span.span_id == span_id:
                return span
        return None

    def get_trace_summary(self, trace_id: str) -> dict[str, Any]:
        """
        Get summary of a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            Trace summary dictionary
        """
        spans = self.get_trace_spans(trace_id)

        if not spans:
            return {}

        total_duration = sum(s.duration_ms() for s in spans)
        error_count = sum(1 for s in spans if s.status == SpanStatus.ERROR)

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "start_time": min(s.start_time for s in spans),
            "end_time": max(s.end_time for s in spans if s.end_time),
            "operations": list({s.operation_name for s in spans}),
        }

    def get_metrics(self) -> dict[str, Any]:
        """
        Get tracing metrics.

        Returns:
            Dictionary with tracing statistics
        """
        total_spans = len(self._completed_spans)
        error_spans = sum(
            1 for s in self._completed_spans if s.status == SpanStatus.ERROR
        )

        trace_ids = {s.trace_id for s in self._completed_spans}

        return {
            "total_spans": total_spans,
            "error_spans": error_spans,
            "error_rate_percent": (error_spans / total_spans * 100)
            if total_spans > 0
            else 0,
            "unique_traces": len(trace_ids),
            "export_enabled": self._enable_export,
        }


# Type variables for decorator
P = ParamSpec("P")
T = TypeVar("T")


def trace_operation(
    operation_name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to trace function execution.

    Args:
        operation_name: Name of operation (uses function name if None)
        kind: Span kind
        attributes: Initial span attributes

    Returns:
        Decorated function
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_distributed_tracer()
            op_name = operation_name or func.__name__

            span = tracer.start_span(op_name, kind=kind, attributes=attributes)

            try:
                result = await func(*args, **kwargs)
                tracer.finish_span(span, status=SpanStatus.OK)
                return result
            except Exception as e:
                tracer.record_exception(span, e)
                tracer.finish_span(
                    span,
                    status=SpanStatus.ERROR,
                    status_message=str(e),
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            tracer = get_distributed_tracer()
            op_name = operation_name or func.__name__

            span = tracer.start_span(op_name, kind=kind, attributes=attributes)

            try:
                result = func(*args, **kwargs)
                tracer.finish_span(span, status=SpanStatus.OK)
                return result
            except Exception as e:
                tracer.record_exception(span, e)
                tracer.finish_span(
                    span,
                    status=SpanStatus.ERROR,
                    status_message=str(e),
                )
                raise

        # Return appropriate wrapper based on function type
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


# Global tracer instance
_global_tracer: DistributedTracer | None = None


def get_distributed_tracer() -> DistributedTracer:
    """Get global distributed tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DistributedTracer()
    return _global_tracer
