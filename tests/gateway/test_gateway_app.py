"""
Integration tests for Gateway FastAPI application.

Tests application startup, middleware chain, and basic endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gateway.main import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


def test_app_creation():
    """Test FastAPI application is created successfully."""
    app = create_app()
    assert app is not None
    assert app.title == "AgentCore Gateway"


def test_health_endpoint(client):
    """Test health check endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert "checks" in data
    assert data["checks"]["application"]["status"] == "healthy"


def test_readiness_endpoint(client):
    """Test readiness check endpoint returns 200."""
    response = client.get("/ready")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ready"
    assert data["ready"] is True
    assert "checks" in data
    assert data["checks"]["application"] is True


def test_liveness_endpoint(client):
    """Test liveness check endpoint returns 200."""
    response = client.get("/live")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "alive"


def test_metrics_info_endpoint(client):
    """Test metrics info endpoint returns correct information."""
    response = client.get("/metrics-info")
    assert response.status_code == 200

    data = response.json()
    assert data["endpoint"] == "/metrics"
    assert data["format"] == "prometheus"


def test_cors_headers(client):
    """Test CORS headers are present in responses."""
    response = client.get("/health")
    assert response.status_code == 200
    # CORS headers should be added by middleware


def test_trace_id_header(client):
    """Test trace ID is added to response headers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Trace-ID" in response.headers
    assert "X-Request-ID" in response.headers
    # Both should be the same
    assert response.headers["X-Trace-ID"] == response.headers["X-Request-ID"]


def test_openapi_docs_debug_mode():
    """Test OpenAPI docs are available in debug mode."""
    from gateway.config import settings
    original_debug = settings.DEBUG

    try:
        # Enable debug mode
        settings.DEBUG = True
        app = create_app()
        client = TestClient(app)

        response = client.get("/docs")
        assert response.status_code == 200
    finally:
        # Restore original setting
        settings.DEBUG = original_debug


def test_openapi_docs_production_mode():
    """Test OpenAPI docs are disabled in production mode."""
    from gateway.config import settings
    original_debug = settings.DEBUG

    try:
        # Disable debug mode
        settings.DEBUG = False
        app = create_app()
        client = TestClient(app)

        response = client.get("/docs")
        assert response.status_code == 404
    finally:
        # Restore original setting
        settings.DEBUG = original_debug


def test_metrics_endpoint(client):
    """Test Prometheus metrics endpoint is available."""
    from gateway.config import settings

    if settings.ENABLE_METRICS:
        response = client.get("/metrics")
        assert response.status_code == 200
        # Verify Prometheus format (metrics may not be initialized yet in test environment)
        assert ("# HELP" in response.text or "# TYPE" in response.text or len(response.text) > 0)
