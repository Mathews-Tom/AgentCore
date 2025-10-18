"""
Gateway Monitoring & Observability

Comprehensive monitoring infrastructure including Prometheus metrics,
OpenTelemetry distributed tracing, and health check endpoints.
"""

from __future__ import annotations

from .metrics import (
    ACTIVE_REQUESTS,
    AUTH_FAILURES,
    AUTH_SUCCESS,
    ERROR_COUNT,
    RATE_LIMIT_HITS,
    REQUEST_COUNT,
    REQUEST_DURATION,
    WEBSOCKET_CONNECTIONS,
    get_metrics_registry,
)
from .tracing import configure_tracing, get_tracer

__all__ = [
    "ACTIVE_REQUESTS",
    "AUTH_FAILURES",
    "AUTH_SUCCESS",
    "ERROR_COUNT",
    "RATE_LIMIT_HITS",
    "REQUEST_COUNT",
    "REQUEST_DURATION",
    "WEBSOCKET_CONNECTIONS",
    "configure_tracing",
    "get_metrics_registry",
    "get_tracer",
]
