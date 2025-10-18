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

from typing import Any

from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
)

# Gateway information
GATEWAY_INFO = Info(
    "gateway_info",
    "Gateway service information",
)

# HTTP Request Metrics
REQUEST_COUNT = Counter(
    "gateway_http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status_code"],
)

REQUEST_DURATION = Histogram(
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

ACTIVE_REQUESTS = Gauge(
    "gateway_http_requests_active",
    "Number of active HTTP requests",
)

# Error Tracking
ERROR_COUNT = Counter(
    "gateway_errors_total",
    "Total number of errors",
    ["error_type", "method", "path"],
)

# Authentication Metrics
AUTH_SUCCESS = Counter(
    "gateway_auth_success_total",
    "Total number of successful authentication attempts",
    ["auth_method"],
)

AUTH_FAILURES = Counter(
    "gateway_auth_failures_total",
    "Total number of failed authentication attempts",
    ["auth_method", "failure_reason"],
)

# Authorization Metrics
AUTHZ_SUCCESS = Counter(
    "gateway_authz_success_total",
    "Total number of successful authorization checks",
    ["resource", "action"],
)

AUTHZ_FAILURES = Counter(
    "gateway_authz_failures_total",
    "Total number of failed authorization checks",
    ["resource", "action", "failure_reason"],
)

# Rate Limiting Metrics
RATE_LIMIT_HITS = Counter(
    "gateway_rate_limit_hits_total",
    "Total number of rate limit hits",
    ["limit_type", "identifier"],
)

RATE_LIMIT_REMAINING = Gauge(
    "gateway_rate_limit_remaining",
    "Remaining rate limit quota",
    ["limit_type", "identifier"],
)

# DDoS Protection Metrics
DDOS_BLOCKS = Counter(
    "gateway_ddos_blocks_total",
    "Total number of DDoS protection blocks",
    ["block_type", "reason"],
)

DDOS_SUSPICIOUS_PATTERNS = Counter(
    "gateway_ddos_suspicious_patterns_total",
    "Total number of suspicious traffic patterns detected",
    ["pattern_type"],
)

# WebSocket Connection Metrics
WEBSOCKET_CONNECTIONS = Gauge(
    "gateway_websocket_connections_active",
    "Number of active WebSocket connections",
)

WEBSOCKET_MESSAGES_SENT = Counter(
    "gateway_websocket_messages_sent_total",
    "Total number of WebSocket messages sent",
    ["message_type"],
)

WEBSOCKET_MESSAGES_RECEIVED = Counter(
    "gateway_websocket_messages_received_total",
    "Total number of WebSocket messages received",
    ["message_type"],
)

WEBSOCKET_ERRORS = Counter(
    "gateway_websocket_errors_total",
    "Total number of WebSocket errors",
    ["error_type"],
)

# Backend Service Metrics
BACKEND_REQUEST_COUNT = Counter(
    "gateway_backend_requests_total",
    "Total number of requests to backend services",
    ["service", "method", "status_code"],
)

BACKEND_REQUEST_DURATION = Histogram(
    "gateway_backend_request_duration_seconds",
    "Backend service request duration in seconds",
    ["service", "method"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

BACKEND_ERRORS = Counter(
    "gateway_backend_errors_total",
    "Total number of backend service errors",
    ["service", "error_type"],
)

# Circuit Breaker Metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service"],
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "gateway_circuit_breaker_trips_total",
    "Total number of circuit breaker trips",
    ["service", "reason"],
)

# Cache Metrics
CACHE_HITS = Counter(
    "gateway_cache_hits_total",
    "Total number of cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "gateway_cache_misses_total",
    "Total number of cache misses",
    ["cache_type"],
)

CACHE_SIZE = Gauge(
    "gateway_cache_size_bytes",
    "Current cache size in bytes",
    ["cache_type"],
)

# Request Size Metrics
REQUEST_SIZE = Histogram(
    "gateway_http_request_size_bytes",
    "HTTP request size in bytes",
    ["method", "path"],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
)

RESPONSE_SIZE = Histogram(
    "gateway_http_response_size_bytes",
    "HTTP response size in bytes",
    ["method", "path"],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
)

# Compression Metrics
COMPRESSION_RATIO = Histogram(
    "gateway_compression_ratio",
    "Response compression ratio",
    ["content_type"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

COMPRESSION_SAVINGS = Counter(
    "gateway_compression_savings_bytes_total",
    "Total bytes saved through compression",
    ["content_type"],
)

# TLS Metrics
TLS_HANDSHAKES = Counter(
    "gateway_tls_handshakes_total",
    "Total number of TLS handshakes",
    ["tls_version", "cipher_suite"],
)

TLS_ERRORS = Counter(
    "gateway_tls_errors_total",
    "Total number of TLS errors",
    ["error_type"],
)

# Session Metrics
ACTIVE_SESSIONS = Gauge(
    "gateway_sessions_active",
    "Number of active user sessions",
)

SESSION_DURATION = Histogram(
    "gateway_session_duration_seconds",
    "User session duration in seconds",
    buckets=(60, 300, 600, 1800, 3600, 7200, 14400, 28800, 86400),
)


def get_metrics_registry() -> CollectorRegistry:
    """
    Get the Prometheus metrics registry.

    Returns:
        The global Prometheus registry containing all metrics
    """
    return REGISTRY


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
