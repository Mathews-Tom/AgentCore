"""
Comprehensive tests for A2A Client.

Tests cover agent registration, task lifecycle, health reporting, and error handling.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx

from agentcore.agent_runtime.services.a2a_client import (
    A2AClient,
    A2AClientError,
    A2AConnectionError,
    A2ARegistrationError,
)
from agentcore.agent_runtime.models.agent_config import AgentConfig, AgentPhilosophy
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcResponse, JsonRpcError


@pytest.fixture
def agent_config():
    """Create sample agent config."""
    return AgentConfig(
        agent_id="test-agent-1",
        philosophy=AgentPhilosophy.AUTONOMOUS
    )


@pytest.fixture
def mock_http_client():
    """Create mock HTTP client."""
    client = MagicMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client


# ==================== Client Initialization Tests ====================


@pytest.mark.asyncio
async def test_client_context_manager():
    """Test client as async context manager."""
    client = A2AClient(base_url="http://localhost:8001")

    async with client as c:
        assert c._client is not None
        assert isinstance(c._client, httpx.AsyncClient)

    # Client should be closed after context
    assert client._client is None


@pytest.mark.asyncio
async def test_client_initialization_parameters():
    """Test client initialization with custom parameters."""
    client = A2AClient(
        base_url="http://custom-server:9000",
        timeout=60.0
    )

    assert client._base_url == "http://custom-server:9000"
    assert client._jsonrpc_url == "http://custom-server:9000/api/v1/jsonrpc"
    assert client._timeout == 60.0


@pytest.mark.asyncio
async def test_client_base_url_trailing_slash():
    """Test that trailing slash is removed from base URL."""
    client = A2AClient(base_url="http://localhost:8001/")
    assert client._base_url == "http://localhost:8001"


# ==================== JSON-RPC Call Tests ====================


@pytest.mark.asyncio
async def test_jsonrpc_call_not_initialized():
    """Test JSON-RPC call without initialization."""
    client = A2AClient()

    with pytest.raises(A2AConnectionError, match="Client not initialized"):
        await client._call_jsonrpc("test.method")


@pytest.mark.asyncio
async def test_jsonrpc_call_success(mock_http_client):
    """Test successful JSON-RPC call."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"status": "success"}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client._call_jsonrpc("test.method", {"param": "value"})

    assert result == {"status": "success"}
    mock_http_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_jsonrpc_call_with_error(mock_http_client):
    """Test JSON-RPC call that returns error."""
    # Mock error response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "error": {
            "code": -32600,
            "message": "Invalid request"
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    with pytest.raises(A2AClientError, match="JSON-RPC error: Invalid request"):
        await client._call_jsonrpc("test.method")


@pytest.mark.asyncio
async def test_jsonrpc_call_http_error(mock_http_client):
    """Test JSON-RPC call with HTTP error."""
    mock_http_client.post.side_effect = httpx.HTTPError("Connection failed")

    client = A2AClient()
    client._client = mock_http_client

    with pytest.raises(A2AConnectionError, match="HTTP error"):
        await client._call_jsonrpc("test.method")


@pytest.mark.asyncio
async def test_jsonrpc_call_unexpected_error(mock_http_client):
    """Test JSON-RPC call with unexpected error."""
    mock_http_client.post.side_effect = Exception("Unexpected")

    client = A2AClient()
    client._client = mock_http_client

    with pytest.raises(A2AClientError, match="Unexpected error"):
        await client._call_jsonrpc("test.method")


# ==================== Agent Registration Tests ====================


@pytest.mark.asyncio
async def test_register_agent_success(agent_config, mock_http_client):
    """Test successful agent registration."""
    # Mock successful registration response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {
            "agent_id": "test-agent-1",
            "status": "registered"
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    agent_id = await client.register_agent(agent_config)

    assert agent_id == "test-agent-1"
    mock_http_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_register_agent_failure(agent_config, mock_http_client):
    """Test agent registration failure."""
    mock_http_client.post.side_effect = httpx.HTTPError("Connection failed")

    client = A2AClient()
    client._client = mock_http_client

    with pytest.raises(A2ARegistrationError, match="Failed to register agent"):
        await client.register_agent(agent_config)


# ==================== Agent Unregistration Tests ====================


@pytest.mark.asyncio
async def test_unregister_agent_success(mock_http_client):
    """Test successful agent unregistration."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.unregister_agent("test-agent-1")

    assert result is True


@pytest.mark.asyncio
async def test_unregister_agent_no_success_field(mock_http_client):
    """Test unregistration with missing success field."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.unregister_agent("test-agent-1")

    assert result is False


# ==================== Status Update Tests ====================


@pytest.mark.asyncio
async def test_update_agent_status_success(mock_http_client):
    """Test successful status update."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.update_agent_status("test-agent-1", "active", {"load": 0.5})

    assert result is True


@pytest.mark.asyncio
async def test_update_agent_status_without_metadata(mock_http_client):
    """Test status update without metadata."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.update_agent_status("test-agent-1", "idle")

    assert result is True
    # Verify metadata was sent as empty dict
    call_params = mock_http_client.post.call_args[1]["json"]["params"]
    assert call_params["metadata"] == {}


# ==================== Health Reporting Tests ====================


@pytest.mark.asyncio
async def test_report_health_success(mock_http_client):
    """Test successful health report."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.report_health(
        "test-agent-1",
        "healthy",
        {"cpu": 50, "memory": 70}
    )

    assert result is True


@pytest.mark.asyncio
async def test_report_health_default_values(mock_http_client):
    """Test health report with default values."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.report_health("test-agent-1")

    assert result is True
    # Verify defaults were used
    call_params = mock_http_client.post.call_args[1]["json"]["params"]
    assert call_params["status"] == "healthy"
    assert call_params["metrics"] == {}


# ==================== Task Lifecycle Tests ====================


@pytest.mark.asyncio
async def test_accept_task_success(mock_http_client):
    """Test successful task acceptance."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.accept_task("task-1", "agent-1")

    assert result is True


@pytest.mark.asyncio
async def test_start_task_success(mock_http_client):
    """Test successful task start."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.start_task("task-1", "agent-1")

    assert result is True


@pytest.mark.asyncio
async def test_complete_task_success(mock_http_client):
    """Test successful task completion."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.complete_task("task-1", "agent-1", {"output": "result"})

    assert result is True


@pytest.mark.asyncio
async def test_fail_task_success(mock_http_client):
    """Test successful task failure reporting."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": {"success": True}
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.fail_task("task-1", "agent-1", "Task failed due to error")

    assert result is True


# ==================== Ping Tests ====================


@pytest.mark.asyncio
async def test_ping_success(mock_http_client):
    """Test successful ping."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": "pong"
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.ping()

    assert result is True


@pytest.mark.asyncio
async def test_ping_failure(mock_http_client):
    """Test ping failure."""
    mock_http_client.post.side_effect = httpx.HTTPError("Connection failed")

    client = A2AClient()
    client._client = mock_http_client

    result = await client.ping()

    assert result is False


@pytest.mark.asyncio
async def test_ping_wrong_response(mock_http_client):
    """Test ping with wrong response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "id": "123",
        "result": "invalid"  # Not "pong"
    }
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    client = A2AClient()
    client._client = mock_http_client

    result = await client.ping()

    assert result is False


# ==================== Integration Tests ====================


@pytest.mark.asyncio
async def test_full_client_lifecycle(agent_config):
    """Test full client lifecycle with context manager."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client.post = AsyncMock()
        mock_client.aclose = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock responses
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        # Registration response
        mock_response.json.side_effect = [
            {
                "jsonrpc": "2.0",
                "id": "1",
                "result": {"agent_id": "test-agent-1"}
            },
            {
                "jsonrpc": "2.0",
                "id": "2",
                "result": {"success": True}
            }
        ]

        async with A2AClient() as client:
            # Register agent
            agent_id = await client.register_agent(agent_config)
            assert agent_id == "test-agent-1"

            # Unregister agent
            result = await client.unregister_agent(agent_id)
            assert result is True

        # Verify client was closed
        mock_client.aclose.assert_called_once()


# ==================== Exception Classes Tests ====================


def test_exception_hierarchy():
    """Test exception class hierarchy."""
    assert issubclass(A2AConnectionError, A2AClientError)
    assert issubclass(A2ARegistrationError, A2AClientError)

    # Test instantiation
    conn_err = A2AConnectionError("Connection failed")
    assert str(conn_err) == "Connection failed"

    reg_err = A2ARegistrationError("Registration failed")
    assert str(reg_err) == "Registration failed"
