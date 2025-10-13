"""Tests for agent API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from agentcore.agent_runtime.main import app
from agentcore.agent_runtime.models.agent_config import AgentPhilosophy


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


def test_create_agent_endpoint(client: TestClient) -> None:
    """Test agent creation endpoint."""
    with patch("agentcore.agent_runtime.routers.agents._lifecycle_manager") as mock_lifecycle:
        # Mock the lifecycle manager
        mock_state = MagicMock()
        mock_state.agent_id = "test-agent-001"
        mock_state.container_id = "test-container-id"
        mock_state.status = "initializing"

        mock_lifecycle.create_agent = AsyncMock(return_value=mock_state)

        response = client.post(
            "/api/v1/agents",
            json={
                "config": {
                    "agent_id": "test-agent-001",
                    "philosophy": "react",
                }
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "test-agent-001"
        assert data["container_id"] == "test-container-id"
        assert data["status"] == "initializing"


def test_start_agent_endpoint(client: TestClient) -> None:
    """Test agent start endpoint."""
    with patch("agentcore.agent_runtime.routers.agents._lifecycle_manager") as mock_lifecycle:
        mock_lifecycle.start_agent = AsyncMock()

        response = client.post("/api/v1/agents/test-agent-001/start")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["agent_id"] == "test-agent-001"


def test_pause_agent_endpoint(client: TestClient) -> None:
    """Test agent pause endpoint."""
    with patch("agentcore.agent_runtime.routers.agents._lifecycle_manager") as mock_lifecycle:
        mock_lifecycle.pause_agent = AsyncMock()

        response = client.post("/api/v1/agents/test-agent-001/pause")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"


def test_terminate_agent_endpoint(client: TestClient) -> None:
    """Test agent termination endpoint."""
    with patch("agentcore.agent_runtime.routers.agents._lifecycle_manager") as mock_lifecycle:
        mock_lifecycle.terminate_agent = AsyncMock()

        response = client.delete("/api/v1/agents/test-agent-001")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "terminated"


def test_get_agent_status_endpoint(client: TestClient) -> None:
    """Test get agent status endpoint."""
    with patch("agentcore.agent_runtime.routers.agents._lifecycle_manager") as mock_lifecycle:
        from agentcore.agent_runtime.models.agent_state import AgentExecutionState
        from datetime import datetime

        mock_state = AgentExecutionState(
            agent_id="test-agent-001",
            status="running",
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )
        mock_lifecycle.get_agent_status = AsyncMock(return_value=mock_state)

        response = client.get("/api/v1/agents/test-agent-001/status")

        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test-agent-001"
        assert data["status"] == "running"


def test_list_agents_endpoint(client: TestClient) -> None:
    """Test list agents endpoint."""
    with patch("agentcore.agent_runtime.routers.agents._lifecycle_manager") as mock_lifecycle:
        from agentcore.agent_runtime.models.agent_state import AgentExecutionState
        from datetime import datetime

        mock_states = [
            AgentExecutionState(
                agent_id=f"test-agent-{i}",
                status="running",
                created_at=datetime.now(),
                last_updated=datetime.now(),
            )
            for i in range(3)
        ]
        mock_lifecycle.list_agents = AsyncMock(return_value=mock_states)

        response = client.get("/api/v1/agents")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
