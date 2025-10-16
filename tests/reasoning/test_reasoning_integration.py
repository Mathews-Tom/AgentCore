"""
Integration test for Reasoning JSON-RPC method registration.

Tests that reasoning.bounded_context method is properly registered
and accessible via the JSON-RPC endpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentcore.a2a_protocol.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_reasoning_method_registered_in_rpc_methods(client):
    """Test that reasoning.bounded_context appears in rpc.methods introspection."""
    response = client.post(
        "/api/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "rpc.methods",
            "id": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert "result" in data
    assert "methods" in data["result"]

    # Check reasoning.bounded_context is in the list
    methods = data["result"]["methods"]
    assert "reasoning.bounded_context" in methods


def test_reasoning_method_callable(client):
    """Test that reasoning.bounded_context method can be called (validates registration)."""
    # Call with minimal params to verify method is callable
    # Will fail due to missing LLM config, but proves method is registered
    response = client.post(
        "/api/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "reasoning.bounded_context",
            "params": {
                "query": "Test query",
            },
            "id": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should have either result or error, but not both
    # Error expected due to LLM configuration, but method should be found
    assert "id" in data
    assert data["id"] == 1

    # Method should be found (not -32601 METHOD_NOT_FOUND)
    if "error" in data:
        # Should not be "Method not found" error
        assert data["error"]["code"] != -32601


def test_reasoning_method_parameter_validation(client):
    """Test parameter validation for reasoning.bounded_context method."""
    # Call with invalid params (empty query)
    response = client.post(
        "/api/v1/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "reasoning.bounded_context",
            "params": {},  # Missing required query param
            "id": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    # Should return error due to missing required parameter
    assert "error" in data
    # ValueErrors raised by handler are mapped to INTERNAL_ERROR (-32603)
    # Should not be METHOD_NOT_FOUND (-32601) which proves method is registered
    assert data["error"]["code"] != -32601
    assert data["error"]["code"] == -32603  # Internal error for handler exceptions


def test_app_starts_with_reasoning_method():
    """Test that app starts successfully with reasoning method registered."""
    # This test just verifies the app object is created correctly
    assert app is not None
    assert hasattr(app, "routes")

    # Verify JSON-RPC route exists
    routes = [route.path for route in app.routes]
    assert "/api/v1/jsonrpc" in routes
