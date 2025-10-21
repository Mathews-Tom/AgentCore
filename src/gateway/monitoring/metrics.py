"""
Prometheus Metrics Collection

Comprehensive metrics for gateway monitoring and observability including:
- HTTP request/response metrics
- Authentication and authorization metrics
- Rate limiting and DDoS protection metrics
- WebSocket connection metrics
- Error tracking and latency monitoring
"""

from __future__ import annotations

import os
import sys
from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
)

# Create a test-safe registry if in test mode
_is_test_mode = "PYTEST_CURRENT_TEST" in os.environ

# Use custom registry for tests to avoid duplication issues
if _is_test_mode or "__gateway_metrics_cache__" in sys.modules:
    # Use a custom registry for tests
    if "__gateway_metrics_registry__" not in sys.modules:
        _test_registry = CollectorRegistry()
        sys.modules["__gateway_metrics_registry__"] = type(sys)(
            "__gateway_metrics_registry__"
        )
        sys.modules["__gateway_metrics_registry__"].registry = _test_registry
    _REGISTRY = sys.modules["__gateway_metrics_registry__"].registry
else:
    _REGISTRY = REGISTRY

# Use global cache from conftest if available (for tests), otherwise create local cache
if "__gateway_metrics_cache__" in sys.modules:
    _metrics_cache: dict[str, Any] = sys.modules["__gateway_metrics_cache__"].cache
else:
    _metrics_cache: dict[str, Any] = {}


def clear_metrics_cache() -> None:
    """Clear the metrics cache. Useful for testing."""
    _metrics_cache.clear()


def _get_or_create_info(name: str, documentation: str) -> Info:
    """Get existing Info metric or create new one, handling duplicates."""
    if name in _metrics_cache:
        return _metrics_cache[name]

    try:
        metric = Info(name, documentation, registry=_REGISTRY)
        _metrics_cache[name] = metric
        return metric
    except ValueError:
        # Metric already registered, retrieve it from registry
        for collector in _REGISTRY._collector_to_names.keys():
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = collector
                return collector
        raise


def _get_or_create_counter(
    name: str, documentation: str, labelnames: list[str] | None = None
) -> Counter:
    """Get existing Counter metric or create new one, handling duplicates."""
    if name in _metrics_cache:
        return _metrics_cache[name]

    try:
        metric = Counter(name, documentation, labelnames or [], registry=_REGISTRY)
        _metrics_cache[name] = metric
        return metric
    except ValueError:
        # Metric already registered, retrieve it from registry
        for collector in _REGISTRY._collector_to_names.keys():
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = collector
                return collector
        raise


def _get_or_create_gauge(
    name: str, documentation: str, labelnames: list[str] | None = None
) -> Gauge:
    """Get existing Gauge metric or create new one, handling duplicates."""
    if name in _metrics_cache:
        return _metrics_cache[name]

    try:
        metric = Gauge(name, documentation, labelnames or [], registry=_REGISTRY)
        _metrics_cache[name] = metric
        return metric
    except ValueError:
        # Metric already registered, retrieve it from registry
        for collector in _REGISTRY._collector_to_names.keys():
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = collector
                return collector
        raise


def _get_or_create_histogram(
    name: str,
    documentation: str,
    labelnames: list[str] | None = None,
    buckets: tuple[float, ...] | None = None,
) -> Histogram:
    """Get existing Histogram metric or create new one, handling duplicates."""
    if name in _metrics_cache:
        return _metrics_cache[name]

    try:
        kwargs = {"buckets": buckets} if buckets else {}
        metric = Histogram(
            name, documentation, labelnames or [], registry=_REGISTRY, **kwargs
        )
        _metrics_cache[name] = metric
        return metric
    except ValueError:
        # Metric already registered, retrieve it from registry
        for collector in _REGISTRY._collector_to_names.keys():
            if hasattr(collector, "_name") and collector._name == name:
                _metrics_cache[name] = collector
                return collector
        raise


# Gateway information
GATEWAY_INFO = _get_or_create_info(
    "gateway_info",
    "Gateway service information",
)

# HTTP Request Metrics
REQUEST_COUNT = _get_or_create_counter(
    "gateway_http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_DURATION = _get_or_create_histogram(
    "gateway_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    ),
)

ACTIVE_REQUESTS = _get_or_create_gauge(
    "gateway_http_requests_active",
    "Number of active HTTP requests",
)

# Error Tracking
ERROR_COUNT = _get_or_create_counter(
    "gateway_errors_total",
    "Total number of errors",
    ["error_type", "method", "path"],
)

# Authentication Metrics
AUTH_SUCCESS = _get_or_create_counter(
    "gateway_auth_success_total",
    "Total number of successful authentication attempts",
    ["auth_method"],
)

AUTH_FAILURES = _get_or_create_counter(
    "gateway_auth_failures_total",
    "Total number of failed authentication attempts",
    ["auth_method", "failure_reason"],
)

# Authorization Metrics
AUTHZ_SUCCESS = _get_or_create_counter(
    "gateway_authz_success_total",
    "Total number of successful authorization checks",
    ["resource", "action"],
)

AUTHZ_FAILURES = _get_or_create_counter(
    "gateway_authz_failures_total",
    "Total number of failed authorization checks",
    ["resource", "action", "failure_reason"],
)

# Rate Limiting Metrics
RATE_LIMIT_HITS = _get_or_create_counter(
    "gateway_rate_limit_hits_total",
    "Total number of rate limit hits",
    ["limit_type", "identifier"],
)

RATE_LIMIT_REMAINING = _get_or_create_gauge(
    "gateway_rate_limit_remaining",
    "Remaining rate limit quota",
    ["limit_type", "identifier"],
)

# DDoS Protection Metrics
DDOS_BLOCKS = _get_or_create_counter(
    "gateway_ddos_blocks_total",
    "Total number of DDoS protection blocks",
    ["block_type", "reason"],
)

DDOS_SUSPICIOUS_PATTERNS = _get_or_create_counter(
    "gateway_ddos_suspicious_patterns_total",
    "Total number of suspicious traffic patterns detected",
    ["pattern_type"],
)

# WebSocket Connection Metrics
WEBSOCKET_CONNECTIONS = _get_or_create_gauge(
    "gateway_websocket_connections_active",
    "Number of active WebSocket connections",
)

WEBSOCKET_MESSAGES_SENT = _get_or_create_counter(
    "gateway_websocket_messages_sent_total",
    "Total number of WebSocket messages sent",
    ["message_type"],
)

WEBSOCKET_MESSAGES_RECEIVED = _get_or_create_counter(
    "gateway_websocket_messages_received_total",
    "Total number of WebSocket messages received",
    ["message_type"],
)

WEBSOCKET_ERRORS = _get_or_create_counter(
    "gateway_websocket_errors_total",
    "Total number of WebSocket errors",
    ["error_type"],
)

# Backend Service Metrics
BACKEND_REQUEST_COUNT = _get_or_create_counter(
    "gateway_backend_requests_total",
    "Total number of requests to backend services",
    ["service", "method", "status_code"],
)

BACKEND_REQUEST_DURATION = _get_or_create_histogram(
    "gateway_backend_request_duration_seconds",
    "Backend service request duration in seconds",
    ["service", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

BACKEND_ERRORS = _get_or_create_counter(
    "gateway_backend_errors_total",
    "Total number of backend service errors",
    ["service", "error_type"],
)

# Circuit Breaker Metrics
CIRCUIT_BREAKER_STATE = _get_or_create_gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service"],
)

CIRCUIT_BREAKER_TRIPS = _get_or_create_counter(
    "gateway_circuit_breaker_trips_total",
    "Total number of circuit breaker trips",
    ["service", "reason"],
)

# Cache Metrics
CACHE_HITS = _get_or_create_counter(
    "gateway_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
)

CACHE_MISSES = _get_or_create_counter(
    "gateway_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
)

CACHE_SIZE = _get_or_create_gauge(
    "gateway_cache_size_bytes",
    "Current cache size in bytes",
    ["cache_type"],
)

# Request Size Metrics
REQUEST_SIZE = _get_or_create_histogram(
    "gateway_http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "path"],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
)

RESPONSE_SIZE = _get_or_create_histogram(
    "gateway_http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "path"],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
)

# Compression Metrics
COMPRESSION_RATIO = _get_or_create_histogram(
    "gateway_compression_ratio",
    "Response compression ratio",
    ["content_type"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

COMPRESSION_SAVINGS = _get_or_create_counter(
    "gateway_compression_savings_bytes_total",
    "Total bytes saved through compression",
    ["content_type"],
)

# TLS Metrics
TLS_HANDSHAKES = _get_or_create_counter(
    "gateway_tls_handshakes_total",
    "Total number of TLS handshakes",
    ["tls_version", "cipher_suite"],
)

TLS_ERRORS = _get_or_create_counter(
    "gateway_tls_errors_total",
    "Total number of TLS errors",
    ["error_type"],
)

# Session Metrics
ACTIVE_SESSIONS = _get_or_create_gauge(
    "gateway_sessions_active",
    "Number of active user sessions",
)

SESSION_DURATION = _get_or_create_histogram(
    "gateway_session_duration_seconds",
    "User session duration in seconds",
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400),
)


def get_metrics_registry() -> CollectorRegistry:
    """
    Get the Prometheus metrics registry.

    Returns:
        The Prometheus registry containing all metrics (custom for tests, global otherwise)
    """
    return _REGISTRY


def set_gateway_info(name: str, version: str, **extra_labels: str) -> None:
    """
    Set gateway service information.

    Args:
        name: Gateway service name
        version: Gateway version
        **extra_labels: Additional labels to include
    """
    info_dict: dict[str, Any] = {
        "name": name,
        "version": version,
    }
    info_dict.update(extra_labels)
    GATEWAY_INFO.info(info_dict)
