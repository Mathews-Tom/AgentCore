"""Unit tests for JSON-RPC client.

These tests verify the JsonRpcClient enforces protocol compliance and
properly handles requests/responses.
"""

from __future__ import annotations

from unittest.mock import Mock, patch
import pytest

from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.protocol.models import A2AContext
from agentcore_cli.protocol.exceptions import (
    JsonRpcProtocolError,
    MethodNotFoundError,
    InvalidParamsError,
    InternalError,
    ParseError,
)
from agentcore_cli.transport.http import HttpTransport


class TestJsonRpcClientInit:
    """Tests for JsonRpcClient initialization."""

    def test_init_with_transport(self) -> None:
        """Test client initialization with transport."""
        transport = Mock(spec=HttpTransport)
        client = JsonRpcClient(transport)

        assert client.transport is transport
        assert client.auth_token is None
        assert client.endpoint == "/api/v1/jsonrpc"
        assert client.request_id == 0

    def test_init_with_auth_token(self) -> None:
        """Test client initialization with auth token."""
        transport = Mock(spec=HttpTransport)
        client = JsonRpcClient(transport, auth_token="jwt-token-123")

        assert client.auth_token == "jwt-token-123"

    def test_init_with_custom_endpoint(self) -> None:
        """Test client initialization with custom endpoint."""
        transport = Mock(spec=HttpTransport)
        client = JsonRpcClient(transport, endpoint="/custom/rpc")

        assert client.endpoint == "/custom/rpc"


class TestJsonRpcClientCall:
    """Tests for JsonRpcClient.call() method."""

    def test_call_success(self) -> None:
        """Test successful JSON-RPC call."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {
            "jsonrpc": "2.0",
            "result": {"agent_id": "agent-001"},
            "id": 1,
        }

        client = JsonRpcClient(transport)
        result = client.call("agent.register", {"name": "test-agent"})

        assert result == {"agent_id": "agent-001"}
        transport.post.assert_called_once()

    def test_call_builds_proper_params_wrapper(self) -> None:
        """CRITICAL: Test call wraps params in object."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {
            "jsonrpc": "2.0",
            "result": {},
            "id": 1,
        }

        client = JsonRpcClient(transport)
        client.call("agent.register", {"name": "test", "capabilities": ["python"]})

        # Verify the request sent to transport
        call_args = transport.post.call_args
        request_data = call_args.args[1]  # Second arg is data

        # CRITICAL: params must be wrapped
        assert "params" in request_data
        assert isinstance(request_data["params"], dict)
        assert request_data["params"]["name"] == "test"
        assert request_data["params"]["capabilities"] == ["python"]

        # Ensure NOT flattened
        assert "name" not in request_data
        assert "capabilities" not in request_data

    def test_call_includes_jsonrpc_version(self) -> None:
        """Test call includes jsonrpc: '2.0' field."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}

        client = JsonRpcClient(transport)
        client.call("test.method")

        call_args = transport.post.call_args
        request_data = call_args.args[1]

        assert request_data["jsonrpc"] == "2.0"

    def test_call_auto_increments_id(self) -> None:
        """Test request IDs auto-increment."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}

        client = JsonRpcClient(transport)
        client.call("method1")
        client.call("method2")
        client.call("method3")

        # Verify IDs incremented
        calls = transport.post.call_args_list
        assert calls[0].args[1]["id"] == 1
        assert calls[1].args[1]["id"] == 2
        assert calls[2].args[1]["id"] == 3

    def test_call_with_auth_token(self) -> None:
        """Test call includes auth token in headers."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}

        client = JsonRpcClient(transport, auth_token="jwt-123")
        client.call("test.method")

        call_args = transport.post.call_args
        headers = call_args.kwargs["headers"]

        assert headers["Authorization"] == "Bearer jwt-123"

    def test_call_with_a2a_context(self) -> None:
        """Test call injects A2A context."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}

        client = JsonRpcClient(transport)
        context = A2AContext(
            trace_id="trace-123",
            source_agent="cli",
            session_id="session-456",
        )
        client.call("test.method", a2a_context=context)

        call_args = transport.post.call_args
        request_data = call_args.args[1]

        # Verify A2A context in params
        assert "a2a_context" in request_data["params"]
        assert request_data["params"]["a2a_context"]["trace_id"] == "trace-123"
        assert request_data["params"]["a2a_context"]["source_agent"] == "cli"

    def test_call_with_empty_params(self) -> None:
        """Test call with no params sends empty object."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}

        client = JsonRpcClient(transport)
        client.call("test.method")

        call_args = transport.post.call_args
        request_data = call_args.args[1]

        assert request_data["params"] == {}

    def test_call_raises_method_not_found(self) -> None:
        """Test call raises MethodNotFoundError on -32601."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32601,
                "message": "Method not found",
                "data": {"method": "unknown.method"},
            },
            "id": 1,
        }

        client = JsonRpcClient(transport)

        with pytest.raises(MethodNotFoundError) as exc_info:
            client.call("unknown.method")

        assert exc_info.value.code == -32601

    def test_call_raises_invalid_params(self) -> None:
        """Test call raises InvalidParamsError on -32602."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32602,
                "message": "Invalid params",
            },
            "id": 1,
        }

        client = JsonRpcClient(transport)

        with pytest.raises(InvalidParamsError) as exc_info:
            client.call("test.method")

        assert exc_info.value.code == -32602

    def test_call_raises_internal_error(self) -> None:
        """Test call raises InternalError on -32603."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": "Internal error",
            },
            "id": 1,
        }

        client = JsonRpcClient(transport)

        with pytest.raises(InternalError) as exc_info:
            client.call("test.method")

        assert exc_info.value.code == -32603


class TestJsonRpcClientBatchCall:
    """Tests for JsonRpcClient.batch_call() method."""

    def test_batch_call_success(self) -> None:
        """Test successful batch call."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = [
            {"jsonrpc": "2.0", "result": {"count": 5}, "id": 1},
            {"jsonrpc": "2.0", "result": {"count": 10}, "id": 2},
        ]

        client = JsonRpcClient(transport)
        results = client.batch_call([
            ("agent.list", {"limit": 5}),
            ("task.list", {"limit": 10}),
        ])

        assert len(results) == 2
        assert results[0] == {"count": 5}
        assert results[1] == {"count": 10}

    def test_batch_call_sends_array(self) -> None:
        """Test batch call sends array of requests."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = [
            {"jsonrpc": "2.0", "result": {}, "id": 1},
            {"jsonrpc": "2.0", "result": {}, "id": 2},
        ]

        client = JsonRpcClient(transport)
        client.batch_call([
            ("method1", {}),
            ("method2", {}),
        ])

        call_args = transport.post.call_args
        request_data = call_args.args[1]

        assert isinstance(request_data, list)
        assert len(request_data) == 2
        assert request_data[0]["method"] == "method1"
        assert request_data[1]["method"] == "method2"

    def test_batch_call_increments_ids(self) -> None:
        """Test batch call uses unique IDs for each request."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = [
            {"jsonrpc": "2.0", "result": {}, "id": 1},
            {"jsonrpc": "2.0", "result": {}, "id": 2},
            {"jsonrpc": "2.0", "result": {}, "id": 3},
        ]

        client = JsonRpcClient(transport)
        client.batch_call([
            ("method1", {}),
            ("method2", {}),
            ("method3", {}),
        ])

        call_args = transport.post.call_args
        request_data = call_args.args[1]

        assert request_data[0]["id"] == 1
        assert request_data[1]["id"] == 2
        assert request_data[2]["id"] == 3


class TestJsonRpcClientNotify:
    """Tests for JsonRpcClient.notify() method."""

    def test_notify_sends_request_without_id(self) -> None:
        """Test notify sends request without id field."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = None  # No response expected

        client = JsonRpcClient(transport)
        client.notify("agent.heartbeat", {"agent_id": "agent-001"})

        call_args = transport.post.call_args
        request_data = call_args.args[1]

        # Notification must NOT have id field
        assert "id" not in request_data
        assert request_data["method"] == "agent.heartbeat"
        assert request_data["params"]["agent_id"] == "agent-001"

    def test_notify_has_params_wrapper(self) -> None:
        """Test notify properly wraps params."""
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = None

        client = JsonRpcClient(transport)
        client.notify("test", {"key": "value"})

        call_args = transport.post.call_args
        request_data = call_args.args[1]

        assert "params" in request_data
        assert request_data["params"]["key"] == "value"


class TestJsonRpcClientIntegration:
    """Integration tests for JsonRpcClient with real transport mock."""

    def test_end_to_end_call_flow(self) -> None:
        """Test complete call flow from request to response."""
        # Create mock transport
        transport = Mock(spec=HttpTransport)
        transport.post.return_value = {
            "jsonrpc": "2.0",
            "result": {
                "agent_id": "agent-001",
                "name": "test-agent",
                "status": "active",
            },
            "id": 1,
        }

        # Create client and make call
        client = JsonRpcClient(transport, auth_token="jwt-token")
        result = client.call(
            "agent.register",
            {"name": "test-agent", "capabilities": ["python", "analysis"]},
        )

        # Verify result
        assert result["agent_id"] == "agent-001"
        assert result["name"] == "test-agent"

        # Verify transport was called correctly
        transport.post.assert_called_once()
        call_args = transport.post.call_args

        # Verify endpoint
        assert call_args.args[0] == "/api/v1/jsonrpc"

        # Verify request structure
        request = call_args.args[1]
        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "agent.register"
        assert request["params"]["name"] == "test-agent"
        assert request["params"]["capabilities"] == ["python", "analysis"]

        # Verify headers
        headers = call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer jwt-token"
