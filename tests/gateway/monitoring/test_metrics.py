"""Tests for Prometheus metrics collection."""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from gateway.monitoring.metrics import (
    ACTIVE_REQUESTS,
    AUTH_FAILURES,
    AUTH_SUCCESS,
    ERROR_COUNT,
    RATE_LIMIT_HITS,
    REQUEST_COUNT,
    REQUEST_DURATION,
    WEBSOCKET_CONNECTIONS,
    get_metrics_registry,
    set_gateway_info)


class TestMetrics:
    """Test Prometheus metrics functionality."""

    def test_get_metrics_registry(self) -> None:
        """Test getting the metrics registry."""
        from prometheus_client import CollectorRegistry

        registry = get_metrics_registry()
        assert isinstance(registry, CollectorRegistry)

    def test_set_gateway_info(self) -> None:
        """Test setting gateway information."""
        set_gateway_info(
            name="test-gateway",
            version="1.0.0",
            environment="test")
        # If no exception is raised, the test passes

    def test_request_count_metric(self) -> None:
        """Test HTTP request counter metric."""
        # Record some requests
        REQUEST_COUNT.labels(method="GET", path="/api/test", status_code="200").inc()
        REQUEST_COUNT.labels(method="POST", path="/api/test", status_code="201").inc()

        # Verify metric exists and has correct labels
        metric_value = REQUEST_COUNT.labels(
            method="GET", path="/api/test", status_code="200"
        )._value._value
        assert metric_value >= 1

    def test_request_duration_metric(self) -> None:
        """Test HTTP request duration histogram."""
        # Record some request durations
        REQUEST_DURATION.labels(method="GET", path="/api/test").observe(0.05)
        REQUEST_DURATION.labels(method="GET", path="/api/test").observe(0.1)
        REQUEST_DURATION.labels(method="POST", path="/api/test").observe(0.2)

        # Verify metric exists
        metric = REQUEST_DURATION.labels(method="GET", path="/api/test")
        assert metric is not None

    def test_active_requests_gauge(self) -> None:
        """Test active requests gauge."""
        # Increment active requests
        ACTIVE_REQUESTS.inc()
        assert ACTIVE_REQUESTS._value._value >= 1

        # Decrement active requests
        ACTIVE_REQUESTS.dec()

    def test_error_count_metric(self) -> None:
        """Test error counter metric."""
        ERROR_COUNT.labels(
            error_type="ValidationError",
            method="POST",
            path="/api/test"
        ).inc()

        metric_value = ERROR_COUNT.labels(
            error_type="ValidationError",
            method="POST",
            path="/api/test"
        )._value._value
        assert metric_value >= 1

    def test_auth_success_metric(self) -> None:
        """Test authentication success counter."""
        AUTH_SUCCESS.labels(auth_method="jwt").inc()
        AUTH_SUCCESS.labels(auth_method="oauth").inc()

        metric_value = AUTH_SUCCESS.labels(auth_method="jwt")._value._value
        assert metric_value >= 1

    def test_auth_failures_metric(self) -> None:
        """Test authentication failure counter."""
        AUTH_FAILURES.labels(
            auth_method="jwt",
            failure_reason="invalid_token"
        ).inc()

        metric_value = AUTH_FAILURES.labels(
            auth_method="jwt",
            failure_reason="invalid_token"
        )._value._value
        assert metric_value >= 1

    def test_rate_limit_hits_metric(self) -> None:
        """Test rate limit hits counter."""
        RATE_LIMIT_HITS.labels(
            limit_type="client_ip",
            identifier="192.168.1.1"
        ).inc()

        metric_value = RATE_LIMIT_HITS.labels(
            limit_type="client_ip",
            identifier="192.168.1.1"
        )._value._value
        assert metric_value >= 1

    def test_websocket_connections_gauge(self) -> None:
        """Test WebSocket connections gauge."""
        # Simulate connection opening
        WEBSOCKET_CONNECTIONS.inc()
        assert WEBSOCKET_CONNECTIONS._value._value >= 1

        # Simulate connection closing
        WEBSOCKET_CONNECTIONS.dec()

    def test_metric_labels(self) -> None:
        """Test that metrics accept expected labels."""
        # Test various label combinations
        REQUEST_COUNT.labels(
            method="GET",
            path="/health",
            status_code="200"
        ).inc()

        REQUEST_DURATION.labels(
            method="POST",
            path="/api/v1/agents"
        ).observe(0.15)

        ERROR_COUNT.labels(
            error_type="TimeoutError",
            method="GET",
            path="/api/v1/agents"
        ).inc()

        # If no exceptions raised, labels are valid
        assert True
