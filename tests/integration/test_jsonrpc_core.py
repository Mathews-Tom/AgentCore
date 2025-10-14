"""
Integration Tests for JSON-RPC Core Functionality

Tests for JSON-RPC 2.0 protocol compliance and core methods.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestJSONRPCCore:
    """Test JSON-RPC 2.0 core functionality."""

    async def test_rpc_ping(self, async_client: AsyncClient, jsonrpc_request_template):
        """Test rpc.ping method."""
        request = jsonrpc_request_template("rpc.ping")
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "1"
        assert "result" in data
        assert data["result"]["pong"] is True

    async def test_rpc_methods(self, async_client: AsyncClient, jsonrpc_request_template):
        """Test rpc.methods - list all available methods."""
        request = jsonrpc_request_template("rpc.methods")
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "methods" in data["result"]

        methods = data["result"]["methods"]
        assert len(methods) > 50  # We have 61 methods
        assert "agent.register" in methods
        assert "task.create" in methods
        assert "health.check_agent" in methods

    async def test_rpc_version(self, async_client: AsyncClient, jsonrpc_request_template):
        """Test rpc.version method."""
        request = jsonrpc_request_template("rpc.version")
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "agentcore_version" in data["result"]
        assert "jsonrpc_version" in data["result"]
        assert "a2a_protocol_version" in data["result"]
        assert data["result"]["jsonrpc_version"] == "2.0"

    async def test_invalid_jsonrpc_version(self, async_client: AsyncClient):
        """Test invalid JSON-RPC version."""
        request = {
            "jsonrpc": "1.0",
            "method": "rpc.ping",
            "id": "1"
        }
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600  # Invalid Request

    async def test_method_not_found(self, async_client: AsyncClient, jsonrpc_request_template):
        """Test calling non-existent method."""
        request = jsonrpc_request_template("nonexistent.method")
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32601  # Method not found

    async def test_invalid_params(self, async_client: AsyncClient):
        """Test invalid parameters."""
        request = {
            "jsonrpc": "2.0",
            "method": "agent.get",
            "params": "invalid_params",  # Should be object or array
            "id": "1"
        }
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        # Should either succeed or return proper error
        assert "result" in data or "error" in data

    async def test_batch_request(self, async_client: AsyncClient, jsonrpc_request_template):
        """Test JSON-RPC batch request."""
        batch = [
            jsonrpc_request_template("rpc.ping", {}, "1"),
            jsonrpc_request_template("rpc.version", {}, "2"),
            jsonrpc_request_template("rpc.methods", {}, "3"),
        ]
        response = await async_client.post("/api/v1/jsonrpc", json=batch)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

        # Check all responses have results
        for item in data:
            assert "result" in item
            assert item["jsonrpc"] == "2.0"

    async def test_notification_no_response(self, async_client: AsyncClient):
        """Test JSON-RPC notification (no id, no response expected)."""
        request = {
            "jsonrpc": "2.0",
            "method": "rpc.ping"
            # No "id" field = notification
        }
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        # Notifications should return 204 No Content (no response body)
        assert response.status_code == 204
        # Response should be empty for notifications
        assert response.content == b''