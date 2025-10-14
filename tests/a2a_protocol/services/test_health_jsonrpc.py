"""
Unit tests for Health Monitoring JSON-RPC Service.

Tests for health monitoring and service discovery JSON-RPC method handlers
covering agent health checks, discovery, and capability listing.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.a2a_protocol.database.models import AgentDB
from agentcore.a2a_protocol.models.agent import AgentStatus
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest


class TestCheckAgent:
    """Test health.check_agent JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_check_agent_healthy(self, mock_monitor):
        """Test checking healthy agent."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_check_agent

        mock_monitor.check_agent_health = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.check_agent",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_check_agent(request)

        assert result["success"] is True
        assert result["agent_id"] == "agent-123"
        assert result["is_healthy"] is True
        assert "checked_at" in result

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_check_agent_unhealthy(self, mock_monitor):
        """Test checking unhealthy agent."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_check_agent

        mock_monitor.check_agent_health = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.check_agent",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_check_agent(request)

        assert result["is_healthy"] is False

    @pytest.mark.asyncio
    async def test_check_agent_missing_params(self):
        """Test checking agent with missing parameters."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_check_agent

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.check_agent",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_check_agent(request)


class TestCheckAll:
    """Test health.check_all JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_check_all_agents(self, mock_monitor):
        """Test checking all agents."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_check_all

        results = {
            "agent-1": True,
            "agent-2": True,
            "agent-3": False,
        }
        mock_monitor.check_all_agents = AsyncMock(return_value=results)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.check_all",
            params={},
            id="1",
        )

        result = await handle_check_all(request)

        assert result["success"] is True
        assert result["total_agents"] == 3
        assert result["healthy_count"] == 2
        assert result["unhealthy_count"] == 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_check_all_agents_empty(self, mock_monitor):
        """Test checking all agents when no agents exist."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_check_all

        mock_monitor.check_all_agents = AsyncMock(return_value={})

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.check_all",
            params={},
            id="1",
        )

        result = await handle_check_all(request)

        assert result["total_agents"] == 0
        assert result["healthy_count"] == 0


class TestHealthHistory:
    """Test health.get_history JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_get_history_success(self, mock_monitor):
        """Test getting agent health history."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_history

        history = [
            {"timestamp": datetime.now(UTC).isoformat(), "healthy": True},
            {"timestamp": datetime.now(UTC).isoformat(), "healthy": True},
        ]
        mock_monitor.get_agent_health_history = AsyncMock(return_value=history)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.get_history",
            params={"agent_id": "agent-123", "limit": 10},
            id="1",
        )

        result = await handle_get_history(request)

        assert result["success"] is True
        assert result["agent_id"] == "agent-123"
        assert result["count"] == 2

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_get_history_default_limit(self, mock_monitor):
        """Test getting health history with default limit."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_history

        mock_monitor.get_agent_health_history = AsyncMock(return_value=[])

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.get_history",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_get_history(request)

        mock_monitor.get_agent_health_history.assert_called_once_with("agent-123", 10)

    @pytest.mark.asyncio
    async def test_get_history_missing_params(self):
        """Test getting history with missing parameters."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_history

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.get_history",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_get_history(request)


class TestUnhealthyAgents:
    """Test health.get_unhealthy JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_get_unhealthy_agents(self, mock_monitor):
        """Test getting unhealthy agents."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_unhealthy

        unhealthy = ["agent-1", "agent-3"]
        mock_monitor.get_unhealthy_agents = AsyncMock(return_value=unhealthy)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.get_unhealthy",
            params={},
            id="1",
        )

        result = await handle_get_unhealthy(request)

        assert result["success"] is True
        assert result["count"] == 2
        assert result["unhealthy_agents"] == unhealthy

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_get_unhealthy_agents_empty(self, mock_monitor):
        """Test getting unhealthy agents when all are healthy."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_unhealthy

        mock_monitor.get_unhealthy_agents = AsyncMock(return_value=[])

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.get_unhealthy",
            params={},
            id="1",
        )

        result = await handle_get_unhealthy(request)

        assert result["count"] == 0


class TestHealthStats:
    """Test health.get_stats JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.health_monitor")
    async def test_get_health_stats(self, mock_monitor):
        """Test getting health monitoring statistics."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_stats

        stats = {
            "total_agents": 10,
            "healthy_agents": 8,
            "unhealthy_agents": 2,
            "average_response_time_ms": 50,
        }
        mock_monitor.get_statistics = Mock(return_value=stats)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="health.get_stats",
            params={},
            id="1",
        )

        result = await handle_get_stats(request)

        assert result["success"] is True
        assert result["stats"] == stats


class TestDiscoveryFindAgents:
    """Test discovery.find_agents JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.get_session")
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.AgentRepository")
    async def test_find_agents_all(self, mock_repo, mock_session):
        """Test finding all agents."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_find_agents

        # Mock agents
        agent1 = AgentDB(
            id="agent-1",
            name="Agent 1",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            capabilities=["capability-1"],
            requirements={},
            endpoint="http://agent1.example.com",
            current_load=5,
            max_load=10,
            last_seen=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        agent2 = AgentDB(
            id="agent-2",
            name="Agent 2",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            capabilities=["capability-2"],
            requirements={},
            endpoint="http://agent2.example.com",
            current_load=3,
            max_load=10,
            last_seen=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_session.return_value.__aenter__.return_value = Mock()
        mock_repo.get_all = AsyncMock(return_value=[agent1, agent2])

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="discovery.find_agents",
            params={},
            id="1",
        )

        result = await handle_find_agents(request)

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["agents"]) == 2

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.get_session")
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.AgentRepository")
    async def test_find_agents_by_capabilities(self, mock_repo, mock_session):
        """Test finding agents by capabilities."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_find_agents

        agent = AgentDB(
            id="agent-1",
            name="Agent 1",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            capabilities=["capability-1"],
            requirements={},
            endpoint="http://agent1.example.com",
            current_load=5,
            max_load=10,
            last_seen=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_session.return_value.__aenter__.return_value = Mock()
        mock_repo.get_by_capabilities = AsyncMock(return_value=[agent])

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="discovery.find_agents",
            params={"capabilities": ["capability-1"], "status": "active"},
            id="1",
        )

        result = await handle_find_agents(request)

        assert result["count"] == 1


class TestDiscoveryGetAgent:
    """Test discovery.get_agent JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.get_session")
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.AgentRepository")
    async def test_get_agent_success(self, mock_repo, mock_session):
        """Test getting agent details."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_agent

        agent = AgentDB(
            id="agent-123",
            name="Test Agent",
            version="1.0.0",
            description="Test agent description",
            status=AgentStatus.ACTIVE,
            capabilities=["capability-1", "capability-2"],
            requirements={"min_memory": "1GB"},
            endpoint="http://agent.example.com",
            current_load=5,
            max_load=10,
            last_seen=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_session.return_value.__aenter__.return_value = Mock()
        mock_repo.get_by_id = AsyncMock(return_value=agent)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="discovery.get_agent",
            params={"agent_id": "agent-123"},
            id="1",
        )

        result = await handle_get_agent(request)

        assert result["success"] is True
        assert result["agent"]["agent_id"] == "agent-123"
        assert result["agent"]["name"] == "Test Agent"
        assert len(result["agent"]["capabilities"]) == 2

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.get_session")
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.AgentRepository")
    async def test_get_agent_not_found(self, mock_repo, mock_session):
        """Test getting non-existent agent."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_agent

        mock_session.return_value.__aenter__.return_value = Mock()
        mock_repo.get_by_id = AsyncMock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="discovery.get_agent",
            params={"agent_id": "nonexistent"},
            id="1",
        )

        with pytest.raises(ValueError, match="Agent not found"):
            await handle_get_agent(request)

    @pytest.mark.asyncio
    async def test_get_agent_missing_params(self):
        """Test getting agent with missing parameters."""
        from agentcore.a2a_protocol.services.health_jsonrpc import handle_get_agent

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="discovery.get_agent",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_get_agent(request)


class TestDiscoveryListCapabilities:
    """Test discovery.list_capabilities JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.get_session")
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.AgentRepository")
    async def test_list_capabilities(self, mock_repo, mock_session):
        """Test listing all capabilities."""
        from agentcore.a2a_protocol.services.health_jsonrpc import (
            handle_list_capabilities,
        )

        agent1 = AgentDB(
            id="agent-1",
            name="Agent 1",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            capabilities=["capability-1", "capability-2"],
            requirements={},
            endpoint="http://agent1.example.com",
            current_load=5,
            max_load=10,
            last_seen=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        agent2 = AgentDB(
            id="agent-2",
            name="Agent 2",
            version="1.0.0",
            status=AgentStatus.ACTIVE,
            capabilities=["capability-2", "capability-3"],
            requirements={},
            endpoint="http://agent2.example.com",
            current_load=3,
            max_load=10,
            last_seen=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        mock_session.return_value.__aenter__.return_value = Mock()
        mock_repo.get_all = AsyncMock(return_value=[agent1, agent2])

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="discovery.list_capabilities",
            params={},
            id="1",
        )

        result = await handle_list_capabilities(request)

        assert result["success"] is True
        assert result["count"] == 3  # capability-1, capability-2, capability-3
        assert "capability-1" in result["capabilities"]
        assert "capability-2" in result["capabilities"]
        assert "capability-3" in result["capabilities"]

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.get_session")
    @patch("agentcore.a2a_protocol.services.health_jsonrpc.AgentRepository")
    async def test_list_capabilities_empty(self, mock_repo, mock_session):
        """Test listing capabilities when no agents exist."""
        from agentcore.a2a_protocol.services.health_jsonrpc import (
            handle_list_capabilities,
        )

        mock_session.return_value.__aenter__.return_value = Mock()
        mock_repo.get_all = AsyncMock(return_value=[])

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="discovery.list_capabilities",
            params={},
            id="1",
        )

        result = await handle_list_capabilities(request)

        assert result["count"] == 0
        assert result["capabilities"] == []
