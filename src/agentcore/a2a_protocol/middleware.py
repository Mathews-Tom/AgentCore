"""
A2A Protocol Layer Middleware

Cross-cutting concerns including logging, metrics, and request handling.
"""

import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram

from agentcore.a2a_protocol.config import settings

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

ACTIVE_CONNECTIONS = Gauge(
    "websocket_connections_active", "Number of active WebSocket connections"
)

logger = structlog.get_logger()


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application."""

    @app.middleware("http")
    async def logging_middleware(request: Request, call_next: Callable) -> Response:
        """Add structured logging and request tracing."""
        # Generate trace ID for request correlation
        trace_id = str(uuid.uuid4())

        # Setup structured logging context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            url=str(request.url),
            user_agent=request.headers.get("user-agent"),
        )

        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Calculate request duration
            duration = time.time() - start_time

            # Log request completion
            logger.info(
                "Request completed",
                status_code=response.status_code,
                duration=duration,
            )

            # Add trace ID to response headers
            response.headers["X-Trace-ID"] = trace_id

            return response

        except Exception as exc:
            duration = time.time() - start_time

            logger.error(
                "Request failed",
                error=str(exc),
                duration=duration,
                exc_info=True,
            )
            raise

    if settings.ENABLE_METRICS:

        @app.middleware("http")
        async def metrics_middleware(request: Request, call_next: Callable) -> Response:
            """Collect Prometheus metrics."""
            start_time = time.time()

            try:
                response = await call_next(request)

                # Record metrics
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                ).inc()

                REQUEST_DURATION.labels(
                    method=request.method, endpoint=request.url.path
                ).observe(time.time() - start_time)

                return response

            except Exception as exc:
                # Record error metrics
                REQUEST_COUNT.labels(
                    method=request.method, endpoint=request.url.path, status_code=500
                ).inc()

                REQUEST_DURATION.labels(
                    method=request.method, endpoint=request.url.path
                ).observe(time.time() - start_time)

                raise
