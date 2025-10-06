"""Tests for Agent Runtime main application."""

import pytest
from fastapi.testclient import TestClient

from agentcore.agent_runtime.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "agent-runtime"


def test_readiness_check(client: TestClient) -> None:
    """Test readiness check endpoint."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["service"] == "agent-runtime"


def test_root_endpoint(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "AgentCore Agent Runtime Layer"
    assert data["version"] == "0.1.0"
    assert data["status"] == "operational"


def test_metrics_endpoint(client: TestClient) -> None:
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    # Prometheus metrics should be in text format
    assert "text/plain" in response.headers.get("content-type", "")
