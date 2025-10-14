"""
Integration Tests for Agent Lifecycle

Tests for agent registration, discovery, and management.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestAgentLifecycle:
    """Test complete agent lifecycle."""

    async def test_agent_register(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card
    ):
        """Test agent registration."""
        request = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        response = await async_client.post("/api/v1/jsonrpc", json=request)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["agent_id"] == sample_agent_card["agent_id"]
        assert data["result"]["status"] == "registered"
        assert "discovery_url" in data["result"]

    async def test_agent_get(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card
    ):
        """Test retrieving agent details."""
        # First register agent
        register_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=register_req)

        # Then retrieve it
        get_req = jsonrpc_request_template("agent.get", {
            "agent_id": sample_agent_card["agent_id"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=get_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["agent"]["agent_id"] == sample_agent_card["agent_id"]
        assert data["result"]["agent"]["agent_name"] == sample_agent_card["agent_name"]

    async def test_agent_discover_by_capability(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card
    ):
        """Test agent discovery by capability."""
        # Register agent
        register_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=register_req)

        # Discover by capability
        discover_req = jsonrpc_request_template("agent.discover", {
            "capabilities": ["text-generation"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=discover_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["count"] >= 1
        # Check our agent is in the results
        agent_ids = [a["agent_id"] for a in data["result"]["agents"]]
        assert sample_agent_card["agent_id"] in agent_ids

    async def test_agent_list(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card
    ):
        """Test listing all agents."""
        # Register agent
        register_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=register_req)

        # List agents
        list_req = jsonrpc_request_template("agent.list")
        response = await async_client.post("/api/v1/jsonrpc", json=list_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["count"] >= 1

    async def test_agent_update_status(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card
    ):
        """Test updating agent status."""
        # Register agent
        register_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=register_req)

        # Update status to maintenance
        update_req = jsonrpc_request_template("agent.update_status", {
            "agent_id": sample_agent_card["agent_id"],
            "status": "maintenance"
        })
        response = await async_client.post("/api/v1/jsonrpc", json=update_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True
        assert data["result"]["new_status"] == "maintenance"

    async def test_agent_ping(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card
    ):
        """Test agent ping (heartbeat)."""
        # Register agent
        register_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=register_req)

        # Ping agent
        ping_req = jsonrpc_request_template("agent.ping", {
            "agent_id": sample_agent_card["agent_id"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=ping_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True

    async def test_agent_unregister(
        self,
        async_client: AsyncClient,
        jsonrpc_request_template,
        sample_agent_card
    ):
        """Test agent unregistration."""
        # Register agent
        register_req = jsonrpc_request_template("agent.register", {
            "agent_card": sample_agent_card
        })
        await async_client.post("/api/v1/jsonrpc", json=register_req)

        # Unregister agent
        unregister_req = jsonrpc_request_template("agent.unregister", {
            "agent_id": sample_agent_card["agent_id"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=unregister_req)

        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert data["result"]["success"] is True

        # Verify agent is gone
        get_req = jsonrpc_request_template("agent.get", {
            "agent_id": sample_agent_card["agent_id"]
        })
        response = await async_client.post("/api/v1/jsonrpc", json=get_req)
        data = response.json()
        # Should return error or not found
        assert "error" in data or (
            "result" in data and data["result"].get("agent") is None
        )