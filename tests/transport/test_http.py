"""Unit tests for HTTP transport layer.

These tests verify the HTTP transport layer behavior in isolation using
mocks for the requests library. No actual network calls are made.
"""

from __future__ import annotations

from unittest.mock import Mock, patch
import pytest
import requests

from agentcore_cli.transport.http import HttpTransport
from agentcore_cli.transport.exceptions import (
    HttpError,
    NetworkError,
    TimeoutError,
    TransportError,
)


class TestHttpTransportInit:
    """Tests for HttpTransport initialization."""

    def test_init_with_defaults(self) -> None:
        """Test HttpTransport initialization with default parameters."""
        transport = HttpTransport("http://localhost:8001")

        assert transport.base_url == "http://localhost:8001"
        assert transport.timeout == 30
        assert transport.verify_ssl is True
        assert transport.session is not None

    def test_init_with_custom_params(self) -> None:
        """Test HttpTransport initialization with custom parameters."""
        transport = HttpTransport(
            base_url="https://api.example.com",
            timeout=60,
            retries=5,
            verify_ssl=False,
        )

        assert transport.base_url == "https://api.example.com"
        assert transport.timeout == 60
        assert transport.verify_ssl is False

    def test_init_strips_trailing_slash(self) -> None:
        """Test base_url trailing slash is removed."""
        transport = HttpTransport("http://localhost:8001/")

        assert transport.base_url == "http://localhost:8001"

    def test_init_empty_base_url_raises_error(self) -> None:
        """Test initialization fails with empty base_url."""
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            HttpTransport("")


class TestHttpTransportPost:
    """Tests for HttpTransport.post() method."""

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_success(self, mock_post: Mock) -> None:
        """Test successful POST request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_post.return_value = mock_response

        transport = HttpTransport("http://localhost:8001")
        result = transport.post(
            "/api/v1/jsonrpc",
            {"jsonrpc": "2.0", "method": "ping", "id": 1},
        )

        assert result == {"result": "success"}
        mock_post.assert_called_once()

        # Verify call arguments
        call_args = mock_post.call_args
        assert call_args.args[0] == "http://localhost:8001/api/v1/jsonrpc"
        assert call_args.kwargs["json"] == {"jsonrpc": "2.0", "method": "ping", "id": 1}
        assert call_args.kwargs["timeout"] == 30
        assert call_args.kwargs["verify"] is True

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_with_custom_headers(self, mock_post: Mock) -> None:
        """Test POST request with custom headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        transport = HttpTransport("http://localhost:8001")
        transport.post(
            "/api/v1/jsonrpc",
            {},
            headers={"Authorization": "Bearer token123"},
        )

        call_args = mock_post.call_args
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer token123"
        assert call_args.kwargs["headers"]["Content-Type"] == "application/json"

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_http_error_400(self, mock_post: Mock) -> None:
        """Test POST raises HttpError on 400 status code."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        transport = HttpTransport("http://localhost:8001")

        with pytest.raises(HttpError) as exc_info:
            transport.post("/api/v1/jsonrpc", {})

        assert exc_info.value.status_code == 400
        assert "400" in str(exc_info.value)

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_http_error_500(self, mock_post: Mock) -> None:
        """Test POST raises HttpError on 500 status code."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        transport = HttpTransport("http://localhost:8001")

        with pytest.raises(HttpError) as exc_info:
            transport.post("/api/v1/jsonrpc", {})

        assert exc_info.value.status_code == 500

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_timeout_error(self, mock_post: Mock) -> None:
        """Test POST raises TimeoutError on request timeout."""
        mock_post.side_effect = requests.exceptions.Timeout("Connection timeout")

        transport = HttpTransport("http://localhost:8001", timeout=5)

        with pytest.raises(TimeoutError) as exc_info:
            transport.post("/api/v1/jsonrpc", {})

        assert "timed out" in str(exc_info.value).lower()
        assert exc_info.value.cause is not None

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_connection_error(self, mock_post: Mock) -> None:
        """Test POST raises NetworkError on connection failure."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        transport = HttpTransport("http://localhost:8001")

        with pytest.raises(NetworkError) as exc_info:
            transport.post("/api/v1/jsonrpc", {})

        assert "connection failed" in str(exc_info.value).lower()
        assert exc_info.value.cause is not None

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_invalid_json_response(self, mock_post: Mock) -> None:
        """Test POST raises TransportError on invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_post.return_value = mock_response

        transport = HttpTransport("http://localhost:8001")

        with pytest.raises(TransportError) as exc_info:
            transport.post("/api/v1/jsonrpc", {})

        assert "invalid json" in str(exc_info.value).lower()
        assert exc_info.value.cause is not None

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_post_generic_request_exception(self, mock_post: Mock) -> None:
        """Test POST raises TransportError on generic request exception."""
        mock_post.side_effect = requests.exceptions.RequestException("Unknown error")

        transport = HttpTransport("http://localhost:8001")

        with pytest.raises(TransportError) as exc_info:
            transport.post("/api/v1/jsonrpc", {})

        assert "transport error" in str(exc_info.value).lower()


class TestHttpTransportRetryLogic:
    """Tests for HTTP transport retry logic configuration."""

    def test_retry_strategy_configured_correctly(self) -> None:
        """Test retry strategy is configured with correct parameters."""
        transport = HttpTransport("http://localhost:8001", retries=3)

        # Get adapter
        adapter = transport.session.get_adapter("http://")

        # Verify retry configuration
        assert adapter.max_retries is not None
        assert adapter.max_retries.total == 3
        assert adapter.max_retries.backoff_factor == 1
        assert 429 in adapter.max_retries.status_forcelist
        assert 500 in adapter.max_retries.status_forcelist
        assert 502 in adapter.max_retries.status_forcelist
        assert 503 in adapter.max_retries.status_forcelist
        assert 504 in adapter.max_retries.status_forcelist

    def test_retry_backoff_factor(self) -> None:
        """Test exponential backoff is configured (1s, 2s, 4s, 8s)."""
        transport = HttpTransport("http://localhost:8001", retries=3)

        adapter = transport.session.get_adapter("http://")
        # Backoff factor of 1 gives: 1s, 2s, 4s, 8s progression
        assert adapter.max_retries.backoff_factor == 1


class TestHttpTransportContextManager:
    """Tests for HttpTransport context manager protocol."""

    def test_context_manager_enter_returns_self(self) -> None:
        """Test context manager __enter__ returns self."""
        transport = HttpTransport("http://localhost:8001")

        with transport as t:
            assert t is transport

    @patch("agentcore_cli.transport.http.requests.Session.close")
    def test_context_manager_exit_closes_session(self, mock_close: Mock) -> None:
        """Test context manager __exit__ closes session."""
        transport = HttpTransport("http://localhost:8001")

        with transport:
            pass

        mock_close.assert_called_once()

    @patch("agentcore_cli.transport.http.requests.Session.close")
    def test_close_method(self, mock_close: Mock) -> None:
        """Test close() method closes session."""
        transport = HttpTransport("http://localhost:8001")
        transport.close()

        mock_close.assert_called_once()


class TestHttpTransportSSLVerification:
    """Tests for SSL/TLS verification configuration."""

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_ssl_verification_enabled(self, mock_post: Mock) -> None:
        """Test SSL verification is enabled by default."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        transport = HttpTransport("https://api.example.com")
        transport.post("/api/v1/jsonrpc", {})

        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is True

    @patch("agentcore_cli.transport.http.requests.Session.post")
    def test_ssl_verification_disabled(self, mock_post: Mock) -> None:
        """Test SSL verification can be disabled."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        transport = HttpTransport("https://api.example.com", verify_ssl=False)
        transport.post("/api/v1/jsonrpc", {})

        call_args = mock_post.call_args
        assert call_args.kwargs["verify"] is False


class TestHttpTransportConnectionPooling:
    """Tests for connection pooling configuration."""

    def test_session_has_adapter_configured(self) -> None:
        """Test session has HTTP adapter with connection pooling."""
        transport = HttpTransport("http://localhost:8001")

        # Verify adapter is configured for HTTP and HTTPS
        assert "http://" in transport.session.adapters
        assert "https://" in transport.session.adapters

        # Verify connection pool settings
        http_adapter = transport.session.get_adapter("http://")
        assert http_adapter.max_retries is not None
        assert http_adapter._pool_connections == 10
        assert http_adapter._pool_maxsize == 10
