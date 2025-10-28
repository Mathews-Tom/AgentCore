"""Unit tests for JSON-RPC client."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import requests

from agentcore_cli.client import AgentCoreClient
from agentcore_cli.exceptions import (
    ApiError,
    AuthenticationError,
    ConnectionError,
    JsonRpcError,
    TimeoutError)


class TestAgentCoreClient:
    """Test suite for AgentCoreClient."""

    def test_init_default_params(self) -> None:
        """Test client initialization with default parameters."""
        client = AgentCoreClient("http://localhost:8001")

        assert client.api_url == "http://localhost:8001/api/v1/jsonrpc"
        assert client.timeout == 30
        assert client.verify_ssl is True
        assert client.auth_token is None
        assert client.request_id == 0

    def test_init_custom_params(self) -> None:
        """Test client initialization with custom parameters."""
        client = AgentCoreClient(
            api_url="https://api.example.com",
            timeout=60,
            retries=5,
            verify_ssl=False,
            auth_token="test-token")

        assert client.api_url == "https://api.example.com/api/v1/jsonrpc"
        assert client.timeout == 60
        assert client.verify_ssl is False
        assert client.auth_token == "test-token"

    def test_init_strips_trailing_slash(self) -> None:
        """Test that trailing slash is stripped from API URL."""
        client = AgentCoreClient("http://localhost:8001/")
        assert client.api_url == "http://localhost:8001/api/v1/jsonrpc"

    @patch("agentcore_cli.client.requests.Session.post")
    def test_successful_call(self, mock_post: Mock) -> None:
        """Test successful JSON-RPC call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"agent_id": "agent-123", "status": "active"},
            "id": 1,
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")
        result = client.call("agent.register", {"name": "test-agent"})

        assert result == {"agent_id": "agent-123", "status": "active"}
        assert client.request_id == 1

        # Verify request payload
        call_args = mock_post.call_args
        assert call_args[1]["json"] == {
            "jsonrpc": "2.0",
            "method": "agent.register",
            "params": {"name": "test-agent"},
            "id": 1,
        }

    @patch("agentcore_cli.client.requests.Session.post")
    def test_call_with_auth_token(self, mock_post: Mock) -> None:
        """Test call with authentication token."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {},
            "id": 1,
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient(
            "http://localhost:8001",
            auth_token="test-token-123")
        client.call("agent.list")

        # Verify authorization header
        call_args = mock_post.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-token-123"

    @patch("agentcore_cli.client.requests.Session.post")
    def test_call_without_params(self, mock_post: Mock) -> None:
        """Test call without parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {"agents": []},
            "id": 1,
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")
        result = client.call("agent.list")

        assert result == {"agents": []}

        # Verify empty params dict is sent
        call_args = mock_post.call_args
        assert call_args[1]["json"]["params"] == {}

    @patch("agentcore_cli.client.requests.Session.post")
    def test_call_increments_request_id(self, mock_post: Mock) -> None:
        """Test that request ID increments with each call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": {},
            "id": 1,
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        client.call("method1")
        assert client.request_id == 1

        client.call("method2")
        assert client.request_id == 2

        client.call("method3")
        assert client.request_id == 3

    @patch("agentcore_cli.client.requests.Session.post")
    def test_jsonrpc_error(self, mock_post: Mock) -> None:
        """Test JSON-RPC error response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32600,
                "message": "Invalid Request",
                "data": {"details": "Missing required field"},
            },
            "id": 1,
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(JsonRpcError) as exc_info:
            client.call("agent.register")

        error = exc_info.value
        assert error.code == -32600
        assert error.message == "Invalid Request"
        assert error.data == {"details": "Missing required field"}
        assert "JSON-RPC Error -32600" in str(error)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_authentication_error(self, mock_post: Mock) -> None:
        """Test 401 authentication error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Unauthorized"}
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(AuthenticationError) as exc_info:
            client.call("agent.list")

        assert "Authentication failed" in str(exc_info.value)
        assert "AGENTCORE_TOKEN" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_http_error(self, mock_post: Mock) -> None:
        """Test HTTP error response."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ApiError) as exc_info:
            client.call("agent.list")

        error = exc_info.value
        assert error.status_code == 500
        assert "500" in str(error)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_timeout_error(self, mock_post: Mock) -> None:
        """Test request timeout."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

        client = AgentCoreClient("http://localhost:8001", timeout=10)

        with pytest.raises(TimeoutError) as exc_info:
            client.call("agent.list")

        assert "timed out after 10s" in str(exc_info.value)
        assert "AGENTCORE_API_TIMEOUT" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_connection_error(self, mock_post: Mock) -> None:
        """Test connection error."""
        mock_post.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ConnectionError) as exc_info:
            client.call("agent.list")

        error_msg = str(exc_info.value)
        assert "Cannot connect" in error_msg
        assert "http://localhost:8001/api/v1/jsonrpc" in error_msg
        assert "agentcore config show" in error_msg

    @patch("agentcore_cli.client.requests.Session.post")
    def test_invalid_json_response(self, mock_post: Mock) -> None:
        """Test invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Invalid JSON", "", 0
        )
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ApiError) as exc_info:
            client.call("agent.list")

        assert "Invalid JSON response" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_batch_call_success(self, mock_post: Mock) -> None:
        """Test successful batch call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"jsonrpc": "2.0", "result": {"agent_id": "agent-1"}, "id": 1},
            {"jsonrpc": "2.0", "result": {"agent_id": "agent-2"}, "id": 2},
            {"jsonrpc": "2.0", "result": {"task_id": "task-1"}, "id": 3},
        ]
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")
        results = client.batch_call([
            ("agent.register", {"name": "agent1"}),
            ("agent.register", {"name": "agent2"}),
            ("task.create", {"type": "test"}),
        ])

        assert len(results) == 3
        assert results[0] == {"agent_id": "agent-1"}
        assert results[1] == {"agent_id": "agent-2"}
        assert results[2] == {"task_id": "task-1"}

    @patch("agentcore_cli.client.requests.Session.post")
    def test_batch_call_with_error(self, mock_post: Mock) -> None:
        """Test batch call with one error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"jsonrpc": "2.0", "result": {"agent_id": "agent-1"}, "id": 1},
            {
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid params"},
                "id": 2,
            },
        ]
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(JsonRpcError) as exc_info:
            client.batch_call([
                ("agent.register", {"name": "agent1"}),
                ("agent.register", None),  # Invalid params
            ])

        assert exc_info.value.code == -32600

    def test_context_manager(self) -> None:
        """Test client as context manager."""
        with AgentCoreClient("http://localhost:8001") as client:
            assert client.session is not None

        # Session should be closed after context
        # We can't directly test if session is closed, but we can verify
        # the context manager works without errors

    @patch("agentcore_cli.client.requests.Session.close")
    def test_close(self, mock_close: Mock) -> None:
        """Test client close method."""
        client = AgentCoreClient("http://localhost:8001")
        client.close()

        mock_close.assert_called_once()

    @patch("agentcore_cli.client.requests.Session.post")
    def test_get_error_message_from_json(self, mock_post: Mock) -> None:
        """Test extracting error message from JSON response."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {"message": "Validation failed"}
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ApiError) as exc_info:
            client.call("agent.register")

        assert "Validation failed" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_get_error_message_from_detail(self, mock_post: Mock) -> None:
        """Test extracting error message from detail field."""
        mock_response = Mock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"detail": "Unprocessable Entity"}
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ApiError) as exc_info:
            client.call("agent.register")

        assert "Unprocessable Entity" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_ssl_verification_disabled(self, mock_post: Mock) -> None:
        """Test SSL verification can be disabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"jsonrpc": "2.0", "result": {}, "id": 1}
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001", verify_ssl=False)
        client.call("test.method")

        # Verify SSL verification is disabled
        call_args = mock_post.call_args
        assert call_args[1]["verify"] is False

    def test_request_headers(self) -> None:
        """Test request headers are set correctly."""
        with patch("agentcore_cli.client.requests.Session.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "jsonrpc": "2.0",
                "result": {},
                "id": 1,
            }
            mock_post.return_value = mock_response

            client = AgentCoreClient("http://localhost:8001")
            client.call("test.method")

            call_args = mock_post.call_args
            headers = call_args[1]["headers"]

            assert headers["Content-Type"] == "application/json"
            assert headers["Accept"] == "application/json"


    @patch("agentcore_cli.client.requests.Session.post")
    def test_batch_call_timeout(self, mock_post: Mock) -> None:
        """Test batch call timeout."""
        mock_post.side_effect = requests.exceptions.Timeout("Batch timeout")

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(TimeoutError) as exc_info:
            client.batch_call([("method1", {}), ("method2", {})])

        assert "Batch request timed out" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_batch_call_connection_error(self, mock_post: Mock) -> None:
        """Test batch call connection error."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection lost")

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ConnectionError) as exc_info:
            client.batch_call([("method1", {})])

        assert "Cannot connect" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_batch_call_request_exception(self, mock_post: Mock) -> None:
        """Test batch call with generic request exception."""
        mock_post.side_effect = requests.exceptions.RequestException("Generic error")

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ConnectionError) as exc_info:
            client.batch_call([("method1", {})])

        assert "Batch request failed" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_batch_call_http_error(self, mock_post: Mock) -> None:
        """Test batch call HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.reason = "Internal Server Error"
        mock_response.json.side_effect = Exception("Invalid JSON")
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ApiError) as exc_info:
            client.batch_call([("method1", {})])

        assert exc_info.value.status_code == 500

    @patch("agentcore_cli.client.requests.Session.post")
    def test_get_error_message_with_message_field(self, mock_post: Mock) -> None:
        """Test extracting error from message field."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "Bad request error"}
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ApiError) as exc_info:
            client.call("test.method")

        assert "Bad request error" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_get_error_message_fallback_to_reason(self, mock_post: Mock) -> None:
        """Test fallback to HTTP reason when no JSON error."""
        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.reason = "Service Unavailable"
        mock_response.json.side_effect = Exception("Not JSON")
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ApiError) as exc_info:
            client.call("test.method")

        assert "Service Unavailable" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_generic_request_exception(self, mock_post: Mock) -> None:
        """Test generic request exception handling."""
        mock_post.side_effect = requests.exceptions.RequestException("Generic error")

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(ConnectionError) as exc_info:
            client.call("test.method")

        assert "Request failed" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_jsonrpc_error_non_dict(self, mock_post: Mock) -> None:
        """Test JSON-RPC error with non-dict error value."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "error": "Simple error string",
            "id": 1,
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")

        with pytest.raises(JsonRpcError) as exc_info:
            client.call("test.method")

        assert "Simple error string" in str(exc_info.value)

    @patch("agentcore_cli.client.requests.Session.post")
    def test_result_non_dict(self, mock_post: Mock) -> None:
        """Test result value that is not a dict."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jsonrpc": "2.0",
            "result": "string result",
            "id": 1,
        }
        mock_post.return_value = mock_response

        client = AgentCoreClient("http://localhost:8001")
        result = client.call("test.method")

        # Should return empty dict for non-dict results
        assert result == {}


class TestJsonRpcError:
    """Test suite for JsonRpcError exception."""

    def test_jsonrpc_error_init(self) -> None:
        """Test JsonRpcError initialization."""
        error_data = {
            "code": -32601,
            "message": "Method not found",
            "data": {"method": "unknown.method"},
        }

        error = JsonRpcError(error_data)

        assert error.code == -32601
        assert error.message == "Method not found"
        assert error.data == {"method": "unknown.method"}

    def test_jsonrpc_error_invalid_types(self) -> None:
        """Test JsonRpcError with invalid type values."""
        error_data = {
            "code": "invalid",  # Should default to -32603
            "message": 12345,  # Should default to "Unknown error"
        }

        error = JsonRpcError(error_data)

        assert error.code == -32603
        assert error.message == "Unknown error"

    def test_jsonrpc_error_str_without_data(self) -> None:
        """Test string representation without data."""
        error = JsonRpcError({"code": -32600, "message": "Invalid Request"})

        error_str = str(error)
        assert "JSON-RPC Error -32600" in error_str
        assert "Invalid Request" in error_str

    def test_jsonrpc_error_str_with_data(self) -> None:
        """Test string representation with data."""
        error = JsonRpcError({
            "code": -32600,
            "message": "Invalid Request",
            "data": {"field": "name"},
        })

        error_str = str(error)
        assert "JSON-RPC Error -32600" in error_str
        assert "Invalid Request" in error_str
        assert "field" in error_str

    def test_jsonrpc_error_default_values(self) -> None:
        """Test default values for missing fields."""
        error = JsonRpcError({})

        assert error.code == -32603
        assert error.message == "Unknown error"
        assert error.data is None
