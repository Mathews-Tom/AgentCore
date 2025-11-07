"""
Tests for Gateway middleware components.

Tests logging, CORS, and metrics middleware functionality.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from gateway.middleware.logging import logging_middleware
from gateway.middleware.cors import setup_cors
from gateway.middleware.metrics import (
    metrics_middleware,
    REQUEST_COUNT,
    REQUEST_DURATION,
    ACTIVE_REQUESTS)


@pytest.fixture
def app_with_middleware():
    """Create test app with middleware."""
    app = FastAPI()

    @app.middleware("http")
    async def add_logging(request: Request, call_next):
        return await logging_middleware(request, call_next)

    @app.middleware("http")
    async def add_metrics(request: Request, call_next):
        return await metrics_middleware(request, call_next)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")

    return app


def test_logging_middleware_adds_trace_id(app_with_middleware):
    """Test logging middleware adds trace ID to response."""
    client = TestClient(app_with_middleware)
    response = client.get("/test")

    assert response.status_code == 200
    assert "X-Trace-ID" in response.headers
    assert "X-Request-ID" in response.headers


def test_logging_middleware_handles_errors(app_with_middleware):
    """Test logging middleware handles errors properly."""
    client = TestClient(app_with_middleware)

    with pytest.raises(ValueError):
        client.get("/error")


def test_metrics_middleware_counts_requests(app_with_middleware):
    """Test metrics middleware counts requests."""
    client = TestClient(app_with_middleware)

    # Get initial count
    initial_count = REQUEST_COUNT.labels(
        method="GET",
        path="/test",
        status_code=200
    )._value.get()

    # Make request
    response = client.get("/test")
    assert response.status_code == 200

    # Check count increased
    final_count = REQUEST_COUNT.labels(
        method="GET",
        path="/test",
        status_code=200
    )._value.get()

    assert final_count > initial_count


def test_cors_middleware_setup():
    """Test CORS middleware is configured correctly."""
    app = FastAPI()
    setup_cors(app)

    # Check middleware is added
    assert len(app.user_middleware) > 0


def test_active_requests_gauge():
    """Test active requests gauge tracks concurrent requests."""
    initial_active = ACTIVE_REQUESTS._value.get()

    app = FastAPI()

    @app.middleware("http")
    async def add_metrics(request: Request, call_next):
        return await metrics_middleware(request, call_next)

    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200
    # After request completes, gauge should return to initial value
    final_active = ACTIVE_REQUESTS._value.get()
    assert final_active == initial_active
