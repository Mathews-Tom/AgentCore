"""
Tests for JSON-RPC 2.0 implementation.

Validates JSON-RPC request/response handling, batch processing,
and A2A protocol extensions.
"""

import pytest
from fastapi.testclient import TestClient

from agentcore.a2a_protocol.main import app
from agentcore.a2a_protocol.models.jsonrpc import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    JsonRpcErrorCode,
    JsonRpcBatchRequest,
    A2AContext,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestJsonRpcModels:
    """Test JSON-RPC model validation."""

    def test_valid_request(self):
        """Test valid JSON-RPC request creation."""
        request = JsonRpcRequest(
            method="test.method",
            params={"arg": "value"},
            id=1
        )
        assert request.method == "test.method"
        assert request.params == {"arg": "value"}
        assert request.id == 1
        assert not request.is_notification

    def test_notification_request(self):
        """Test notification request (no ID)."""
        request = JsonRpcRequest(
            method="test.method",
            params={"arg": "value"}
        )
        assert request.id is None
        assert request.is_notification

    def test_valid_response(self):
        """Test valid JSON-RPC response creation."""
        response = JsonRpcResponse(
            result={"success": True},
            id=1
        )
        assert response.result == {"success": True}
        assert response.error is None
        assert response.id == 1

    def test_error_response(self):
        """Test JSON-RPC error response."""
        error = JsonRpcError(
            code=JsonRpcErrorCode.METHOD_NOT_FOUND.value,
            message="Method not found"
        )
        response = JsonRpcResponse(
            error=error,
            id=1
        )
        assert response.error is not None
        assert response.result is None

    def test_a2a_context(self):
        """Test A2A protocol context."""
        context = A2AContext(
            source_agent="agent_1",
            target_agent="agent_2",
            timestamp="2024-01-01T00:00:00Z"
        )
        assert context.source_agent == "agent_1"
        assert context.target_agent == "agent_2"
        assert context.trace_id is not None

    def test_batch_request(self):
        """Test batch request validation."""
        requests = [
            JsonRpcRequest(method="test.method1", id=1),
            JsonRpcRequest(method="test.method2", id=2)
        ]
        batch = JsonRpcBatchRequest(requests=requests)
        assert len(batch.requests) == 2


class TestJsonRpcApi:
    """Test JSON-RPC API endpoints."""

    def test_ping_method(self, client):
        """Test built-in ping method."""
        response = client.post(
            "/api/v1/jsonrpc",
            json={"jsonrpc": "2.0", "method": "rpc.ping", "id": 1}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert "result" in data
        assert data["result"]["pong"] is True

    def test_version_method(self, client):
        """Test built-in version method."""
        response = client.post(
            "/api/v1/jsonrpc",
            json={"jsonrpc": "2.0", "method": "rpc.version", "id": 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        assert "result" in data
        assert "agentcore_version" in data["result"]
        assert "jsonrpc_version" in data["result"]
        assert data["result"]["jsonrpc_version"] == "2.0"

    def test_methods_method(self, client):
        """Test built-in methods listing."""
        response = client.post(
            "/api/v1/jsonrpc",
            json={"jsonrpc": "2.0", "method": "rpc.methods", "id": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 3
        assert "methods" in data["result"]
        methods = data["result"]["methods"]
        assert "rpc.ping" in methods
        assert "rpc.version" in methods
        assert "rpc.methods" in methods

    def test_notification_request(self, client):
        """Test notification handling (no response expected)."""
        response = client.post(
            "/api/v1/jsonrpc",
            json={"jsonrpc": "2.0", "method": "rpc.ping"}
        )
        assert response.status_code == 204  # No content

    def test_batch_request(self, client):
        """Test batch request processing."""
        batch_request = [
            {"jsonrpc": "2.0", "method": "rpc.ping", "id": 1},
            {"jsonrpc": "2.0", "method": "rpc.version", "id": 2}
        ]
        response = client.post("/api/v1/jsonrpc", json=batch_request)
        assert response.status_code == 200
        data = response.json()
        assert "responses" in data
        responses = data["responses"]
        assert len(responses) == 2

        # Check first response (ping)
        ping_response = next(r for r in responses if r["id"] == 1)
        assert ping_response["result"]["pong"] is True

        # Check second response (version)
        version_response = next(r for r in responses if r["id"] == 2)
        assert "agentcore_version" in version_response["result"]

    def test_invalid_method(self, client):
        """Test handling of invalid method."""
        response = client.post(
            "/api/v1/jsonrpc",
            json={"jsonrpc": "2.0", "method": "invalid.method", "id": 4}
        )
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == JsonRpcErrorCode.INTERNAL_ERROR.value

    def test_invalid_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/api/v1/jsonrpc",
            content="{invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # FastAPI validation error

    def test_method_list_endpoint(self, client):
        """Test the methods listing endpoint."""
        response = client.get("/api/v1/jsonrpc/methods")
        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        methods = data["methods"]
        assert "rpc.ping" in methods

    def test_ping_endpoint(self, client):
        """Test the ping endpoint."""
        response = client.post("/api/v1/jsonrpc/ping")
        assert response.status_code == 200
        data = response.json()
        assert data["pong"] is True
        assert "timestamp" in data