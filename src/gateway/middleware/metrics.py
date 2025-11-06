"""
Metrics Middleware

Prometheus metrics collection for monitoring and observability.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastapi import Request, Response

# Import metrics from central location to avoid duplication
from gateway.monitoring.metrics import (
    ACTIVE_REQUESTS,
    REQUEST_COUNT,
    REQUEST_DURATION,
)


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """
    Collect Prometheus metrics for requests.

    Tracks request counts, durations, and active requests.
    """
    ACTIVE_REQUESTS.inc()
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

    finally:
        ACTIVE_REQUESTS.dec()
