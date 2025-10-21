"""
Logging Middleware

Request/response logging with structured logging and distributed tracing.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable

import structlog
from fastapi import Request, Response

logger = structlog.get_logger()


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    Add structured logging and request tracing.

    Generates trace ID for request correlation and logs request/response details.
    """
    # Generate trace ID for request correlation
    trace_id = str(uuid.uuid4())

    # Setup structured logging context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        trace_id=trace_id,
        method=request.method,
        url=str(request.url),
        client_host=request.client.host if request.client else None,
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
        response.headers["X-Request-ID"] = trace_id

        return response

    except Exception as exc:
        duration = time.time() - start_time

        logger.error(
            "Request failed",
            error=str(exc),
            error_type=type(exc).__name__,
            duration=duration,
            exc_info=True,
        )
        raise
