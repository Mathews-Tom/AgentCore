"""Unit tests for JSON-RPC 2.0 protocol models.

These tests verify Pydantic model validation and JSON-RPC 2.0 compliance.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcore_cli.protocol.models import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    A2AContext,
)


class TestJsonRpcError:
    """Tests for JsonRpcError model."""

    def test_error_with_code_and_message(self) -> None:
        """Test error with code and message."""
        error = JsonRpcError(code=-32600, message="Invalid Request")

        assert error.code == -32600
        assert error.message == "Invalid Request"
        assert error.data is None

    def test_error_with_data(self) -> None:
        """Test error with additional data."""
        error = JsonRpcError(
            code=-32602,
            message="Invalid params",
            data={"param": "name", "reason": "required"},
        )

        assert error.code == -32602
        assert error.data["param"] == "name"
        assert error.data["reason"] == "required"

    def test_error_serialization(self) -> None:
        """Test error serializes to dict correctly."""
        error = JsonRpcError(code=-32601, message="Method not found")
        data = error.model_dump()

        assert data["code"] == -32601
        assert data["message"] == "Method not found"
        assert data["data"] is None


class TestJsonRpcRequest:
    """Tests for JsonRpcRequest model."""

    def test_request_minimal(self) -> None:
        """Test minimal request with method only."""
        request = JsonRpcRequest(method="ping")

        assert request.jsonrpc == "2.0"
        assert request.method == "ping"
        assert request.params == {}
        assert request.id is None

    def test_request_with_params(self) -> None:
        """Test request with parameters."""
        request = JsonRpcRequest(
            method="agent.register",
            params={"name": "test-agent", "capabilities": ["python"]},
            id=1,
        )

        assert request.method == "agent.register"
        assert request.params["name"] == "test-agent"
        assert request.params["capabilities"] == ["python"]
        assert request.id == 1

    def test_request_params_is_dict(self) -> None:
        """Test params must be dictionary (not array)."""
        # Valid: params as dict
        request = JsonRpcRequest(method="test", params={"key": "value"})
        assert isinstance(request.params, dict)

    def test_request_serialization(self) -> None:
        """Test request serializes to JSON-RPC 2.0 format."""
        request = JsonRpcRequest(
            method="agent.list",
            params={"limit": 10},
            id=42,
        )
        data = request.model_dump()

        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "agent.list"
        assert data["params"] == {"limit": 10}
        assert data["id"] == 42

    def test_request_deserialization(self) -> None:
        """Test request can be created from dict."""
        data = {
            "jsonrpc": "2.0",
            "method": "agent.info",
            "params": {"agent_id": "agent-001"},
            "id": 1,
        }
        request = JsonRpcRequest(**data)

        assert request.method == "agent.info"
        assert request.params["agent_id"] == "agent-001"

    def test_request_id_can_be_string(self) -> None:
        """Test request ID can be string."""
        request = JsonRpcRequest(method="test", id="request-uuid-123")

        assert request.id == "request-uuid-123"

    def test_request_id_can_be_int(self) -> None:
        """Test request ID can be integer."""
        request = JsonRpcRequest(method="test", id=999)

        assert request.id == 999

    def test_request_notification_has_no_id(self) -> None:
        """Test notification (no id field)."""
        request = JsonRpcRequest(method="agent.heartbeat")

        assert request.id is None


class TestJsonRpcResponse:
    """Tests for JsonRpcResponse model."""

    def test_response_with_result(self) -> None:
        """Test successful response with result."""
        response = JsonRpcResponse(
            result={"agent_id": "agent-001", "status": "active"},
            id=1,
        )

        assert response.jsonrpc == "2.0"
        assert response.result["agent_id"] == "agent-001"
        assert response.error is None
        assert response.id == 1

    def test_response_with_error(self) -> None:
        """Test error response."""
        response = JsonRpcResponse(
            error=JsonRpcError(code=-32601, message="Method not found"),
            id=1,
        )

        assert response.result is None
        assert response.error is not None
        assert response.error.code == -32601
        assert response.id == 1

    def test_response_serialization(self) -> None:
        """Test response serializes correctly."""
        response = JsonRpcResponse(
            result={"count": 5},
            id=10,
        )
        data = response.model_dump()

        assert data["jsonrpc"] == "2.0"
        assert data["result"] == {"count": 5}
        assert data["id"] == 10

    def test_response_deserialization(self) -> None:
        """Test response can be created from dict."""
        data = {
            "jsonrpc": "2.0",
            "result": {"task_id": "task-001"},
            "id": 5,
        }
        response = JsonRpcResponse(**data)

        assert response.result["task_id"] == "task-001"
        assert response.id == 5

    def test_response_with_null_result(self) -> None:
        """Test response with null result is valid."""
        response = JsonRpcResponse(result=None, id=1)

        assert response.result is None
        assert response.error is None


class TestA2AContext:
    """Tests for A2A context model."""

    def test_context_minimal(self) -> None:
        """Test minimal A2A context."""
        context = A2AContext(
            trace_id="trace-123",
            source_agent="agent-001",
        )

        assert context.trace_id == "trace-123"
        assert context.source_agent == "agent-001"
        assert context.target_agent is None
        assert context.session_id is None
        assert context.timestamp is None

    def test_context_full(self) -> None:
        """Test complete A2A context."""
        context = A2AContext(
            trace_id="trace-456",
            source_agent="agent-001",
            target_agent="agent-002",
            session_id="session-789",
            timestamp="2025-10-22T12:00:00Z",
        )

        assert context.trace_id == "trace-456"
        assert context.source_agent == "agent-001"
        assert context.target_agent == "agent-002"
        assert context.session_id == "session-789"
        assert context.timestamp == "2025-10-22T12:00:00Z"

    def test_context_serialization(self) -> None:
        """Test A2A context serializes correctly."""
        context = A2AContext(
            trace_id="trace-xyz",
            source_agent="cli",
            target_agent="server",
        )
        data = context.model_dump(exclude_none=True)

        assert data["trace_id"] == "trace-xyz"
        assert data["source_agent"] == "cli"
        assert data["target_agent"] == "server"
        assert "session_id" not in data  # Excluded because None
        assert "timestamp" not in data  # Excluded because None


class TestJsonRpcCompliance:
    """Tests for JSON-RPC 2.0 specification compliance."""

    def test_request_has_params_wrapper(self) -> None:
        """CRITICAL: Test params are wrapped in object, not sent as flat dict."""
        request = JsonRpcRequest(
            method="agent.register",
            params={"name": "test", "capabilities": ["python"]},
        )

        # Serialize to dict (what gets sent over wire)
        data = request.model_dump()

        # CRITICAL: params must be an object, not flattened
        assert "params" in data
        assert isinstance(data["params"], dict)
        assert data["params"]["name"] == "test"
        assert data["params"]["capabilities"] == ["python"]

        # Ensure params are NOT flattened to top level
        assert "name" not in data  # Should be in params, not top level
        assert "capabilities" not in data  # Should be in params, not top level

    def test_request_always_has_jsonrpc_field(self) -> None:
        """Test every request has jsonrpc: '2.0' field."""
        request = JsonRpcRequest(method="test")
        data = request.model_dump()

        assert "jsonrpc" in data
        assert data["jsonrpc"] == "2.0"

    def test_response_always_has_jsonrpc_field(self) -> None:
        """Test every response has jsonrpc: '2.0' field."""
        response = JsonRpcResponse(result={}, id=1)
        data = response.model_dump()

        assert "jsonrpc" in data
        assert data["jsonrpc"] == "2.0"

    def test_notification_has_no_id(self) -> None:
        """Test notifications omit id field."""
        request = JsonRpcRequest(method="notify")
        data = request.model_dump(exclude_none=True)

        # Notification should not have id field when serialized
        assert request.id is None
