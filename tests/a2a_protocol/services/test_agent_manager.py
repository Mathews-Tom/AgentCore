"""
Comprehensive tests for AgentManager service.

Tests cover agent registration, discovery, lifecycle management, and cleanup.
"""

from datetime import UTC, datetime, timedelta

import pytest

from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCapability,
    AgentCard,
    AgentDiscoveryQuery,
    AgentEndpoint,
    AgentMetadata,
    AgentRegistrationRequest,
    AgentStatus,
    AuthenticationType,
    EndpointType)
from agentcore.a2a_protocol.services.agent_manager import AgentManager, agent_manager


@pytest.fixture
def manager():
    """Create a fresh AgentManager instance."""
    return AgentManager()


@pytest.fixture
def sample_agent_card():
    """Create a sample agent card for testing."""
    return AgentCard(
        agent_id="test-agent-1",
        agent_name="Test Agent",
        agent_version="1.0.0",
        status=AgentStatus.ACTIVE,
        description="Test agent for unit testing",
        endpoints=[AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)],
        capabilities=[
            AgentCapability(name="text.generation", version="1.0"),
            AgentCapability(name="text.analysis", version="1.0"),
        ],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE, required=False
        ),
        metadata=AgentMetadata(tags=["ai", "nlp"], category="language-processing"))


@pytest.fixture
def another_agent_card():
    """Create another sample agent card for testing."""
    return AgentCard(
        agent_id="test-agent-2",
        agent_name="Another Agent",
        agent_version="2.0.0",
        status=AgentStatus.ACTIVE,
        endpoints=[AgentEndpoint(url="http://localhost:8002", type=EndpointType.HTTP)],
        capabilities=[AgentCapability(name="image.generation", version="1.0")],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE, required=False
        ),
        metadata=AgentMetadata(tags=["ai", "image"], category="image-processing"))


# ==================== Agent Registration Tests ====================


@pytest.mark.asyncio
async def test_register_agent_success(manager, sample_agent_card):
    """Test successful agent registration."""
    request = AgentRegistrationRequest(
        agent_card=sample_agent_card, override_existing=False
    )

    response = await manager.register_agent(request)

    assert response.agent_id == "test-agent-1"
    assert response.status == "registered"
    assert response.discovery_url == "/.well-known/agents/test-agent-1"
    assert response.message == "Agent registered successfully"


@pytest.mark.asyncio
async def test_register_agent_duplicate_without_override(manager, sample_agent_card):
    """Test that duplicate registration fails without override flag."""
    request = AgentRegistrationRequest(
        agent_card=sample_agent_card, override_existing=False
    )

    # First registration succeeds
    await manager.register_agent(request)

    # Second registration fails
    with pytest.raises(ValueError, match="already registered"):
        await manager.register_agent(request)


@pytest.mark.asyncio
async def test_register_agent_override_existing(manager, sample_agent_card):
    """Test updating existing agent with override flag."""
    request = AgentRegistrationRequest(
        agent_card=sample_agent_card, override_existing=False
    )

    # First registration
    response1 = await manager.register_agent(request)
    first_created_at = sample_agent_card.created_at

    # Update with override
    sample_agent_card.description = "Updated description"
    request_override = AgentRegistrationRequest(
        agent_card=sample_agent_card, override_existing=True
    )
    response2 = await manager.register_agent(request_override)

    assert response2.agent_id == "test-agent-1"
    assert response2.status == "registered"

    # Verify agent was updated
    agent = await manager.get_agent("test-agent-1")
    assert agent.description == "Updated description"
    assert agent.created_at == first_created_at  # Created timestamp preserved


@pytest.mark.asyncio
async def test_register_agent_duplicate_capabilities(manager, sample_agent_card):
    """Test that agents with duplicate capability names fail validation."""
    sample_agent_card.capabilities = [
        AgentCapability(name="text.generation", version="1.0"),
        AgentCapability(name="text.generation", version="2.0"),  # Duplicate name
    ]

    request = AgentRegistrationRequest(
        agent_card=sample_agent_card, override_existing=False
    )

    with pytest.raises(ValueError, match="unique names"):
        await manager.register_agent(request)


# ==================== Agent Retrieval Tests ====================


@pytest.mark.asyncio
async def test_get_agent_success(manager, sample_agent_card):
    """Test retrieving an existing agent."""
    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await manager.register_agent(request)

    agent = await manager.get_agent("test-agent-1")

    assert agent is not None
    assert agent.agent_id == "test-agent-1"
    assert agent.agent_name == "Test Agent"


@pytest.mark.asyncio
async def test_get_agent_not_found(manager):
    """Test retrieving a non-existent agent."""
    agent = await manager.get_agent("nonexistent-agent")
    assert agent is None


@pytest.mark.asyncio
async def test_get_agent_summary_success(manager, sample_agent_card):
    """Test retrieving agent discovery summary."""
    request = AgentRegistrationRequest(agent_card=sample_agent_card)
    await manager.register_agent(request)

    summary = await manager.get_agent_summary("test-agent-1")

    assert summary is not None
    assert summary["agent_id"] == "test-agent-1"
    assert summary["agent_name"] == "Test Agent"
    assert "text.generation" in summary["capabilities"]
    assert "text.analysis" in summary["capabilities"]


@pytest.mark.asyncio
async def test_get_agent_summary_not_found(manager):
    """Test retrieving summary for non-existent agent."""
    summary = await manager.get_agent_summary("nonexistent-agent")
    assert summary is None


# ==================== Agent Discovery Tests ====================


@pytest.mark.asyncio
async def test_discover_all_agents(manager, sample_agent_card, another_agent_card):
    """Test discovering all agents without filters."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    query = AgentDiscoveryQuery()
    response = await manager.discover_agents(query)

    assert response.total_count == 2
    assert len(response.agents) == 2
    assert response.has_more is False


@pytest.mark.asyncio
async def test_discover_agents_by_capability(
    manager, sample_agent_card, another_agent_card
):
    """Test discovering agents by capability."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    query = AgentDiscoveryQuery(capabilities=["text.generation"])
    response = await manager.discover_agents(query)

    assert response.total_count == 1
    assert len(response.agents) == 1
    assert response.agents[0]["agent_id"] == "test-agent-1"


@pytest.mark.asyncio
async def test_discover_agents_by_multiple_capabilities(
    manager, sample_agent_card, another_agent_card
):
    """Test discovering agents with multiple capabilities (AND logic)."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    query = AgentDiscoveryQuery(capabilities=["text.generation", "text.analysis"])
    response = await manager.discover_agents(query)

    assert response.total_count == 1
    assert response.agents[0]["agent_id"] == "test-agent-1"


@pytest.mark.asyncio
async def test_discover_agents_by_status(
    manager, sample_agent_card, another_agent_card
):
    """Test discovering agents by status."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))

    # Register second agent as inactive
    another_agent_card.status = AgentStatus.INACTIVE
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    query = AgentDiscoveryQuery(status=AgentStatus.ACTIVE)
    response = await manager.discover_agents(query)

    assert response.total_count == 1
    assert response.agents[0]["agent_id"] == "test-agent-1"


@pytest.mark.asyncio
async def test_discover_agents_by_tags(manager, sample_agent_card, another_agent_card):
    """Test discovering agents by tags."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    query = AgentDiscoveryQuery(tags=["nlp"])
    response = await manager.discover_agents(query)

    assert response.total_count == 1
    assert response.agents[0]["agent_id"] == "test-agent-1"


@pytest.mark.asyncio
async def test_discover_agents_by_category(
    manager, sample_agent_card, another_agent_card
):
    """Test discovering agents by category."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    query = AgentDiscoveryQuery(category="image-processing")
    response = await manager.discover_agents(query)

    assert response.total_count == 1
    assert response.agents[0]["agent_id"] == "test-agent-2"


@pytest.mark.asyncio
async def test_discover_agents_by_name_pattern(
    manager, sample_agent_card, another_agent_card
):
    """Test discovering agents by name pattern (regex)."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    query = AgentDiscoveryQuery(name_pattern="Test.*")
    response = await manager.discover_agents(query)

    assert response.total_count == 1
    assert response.agents[0]["agent_id"] == "test-agent-1"


@pytest.mark.asyncio
async def test_discover_agents_invalid_regex(manager, sample_agent_card):
    """Test discovering agents with invalid regex pattern (should handle gracefully)."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))

    query = AgentDiscoveryQuery(name_pattern="[invalid(regex")
    response = await manager.discover_agents(query)

    # Invalid regex is ignored, returns all agents
    assert response.total_count == 1


@pytest.mark.asyncio
async def test_discover_agents_pagination(
    manager, sample_agent_card, another_agent_card
):
    """Test agent discovery with pagination."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    # Get first page (limit 1)
    query = AgentDiscoveryQuery(limit=1, offset=0)
    response = await manager.discover_agents(query)

    assert response.total_count == 2
    assert len(response.agents) == 1
    assert response.has_more is True

    # Get second page
    query2 = AgentDiscoveryQuery(limit=1, offset=1)
    response2 = await manager.discover_agents(query2)

    assert response2.total_count == 2
    assert len(response2.agents) == 1
    assert response2.has_more is False


# ==================== Agent Unregistration Tests ====================


@pytest.mark.asyncio
async def test_unregister_agent_success(manager, sample_agent_card):
    """Test successful agent unregistration."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))

    result = await manager.unregister_agent("test-agent-1")

    assert result is True

    # Verify agent is removed
    agent = await manager.get_agent("test-agent-1")
    assert agent is None


@pytest.mark.asyncio
async def test_unregister_agent_not_found(manager):
    """Test unregistering a non-existent agent."""
    result = await manager.unregister_agent("nonexistent-agent")
    assert result is False


# ==================== Agent Status Updates Tests ====================


@pytest.mark.asyncio
async def test_update_agent_status_success(manager, sample_agent_card):
    """Test updating agent status."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))

    result = await manager.update_agent_status("test-agent-1", AgentStatus.MAINTENANCE)

    assert result is True

    # Verify status was updated
    agent = await manager.get_agent("test-agent-1")
    assert agent.status == AgentStatus.MAINTENANCE


@pytest.mark.asyncio
async def test_update_agent_status_not_found(manager):
    """Test updating status for non-existent agent."""
    result = await manager.update_agent_status("nonexistent-agent", AgentStatus.ACTIVE)
    assert result is False


@pytest.mark.asyncio
async def test_ping_agent_success(manager, sample_agent_card):
    """Test updating agent last_seen timestamp."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))

    # Get initial last_seen
    agent_before = await manager.get_agent("test-agent-1")
    last_seen_before = agent_before.last_seen

    # Wait a tiny bit and ping
    import asyncio

    await asyncio.sleep(0.01)

    result = await manager.ping_agent("test-agent-1")

    assert result is True

    # Verify last_seen was updated
    agent_after = await manager.get_agent("test-agent-1")
    assert agent_after.last_seen > last_seen_before


@pytest.mark.asyncio
async def test_ping_agent_not_found(manager):
    """Test pinging non-existent agent."""
    result = await manager.ping_agent("nonexistent-agent")
    assert result is False


# ==================== Agent Listing & Counts Tests ====================


@pytest.mark.asyncio
async def test_list_all_agents(manager, sample_agent_card, another_agent_card):
    """Test listing all agents."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    agents = await manager.list_all_agents()

    assert len(agents) == 2
    assert any(a["agent_id"] == "test-agent-1" for a in agents)
    assert any(a["agent_id"] == "test-agent-2" for a in agents)


@pytest.mark.asyncio
async def test_get_agent_count(manager, sample_agent_card, another_agent_card):
    """Test getting agent count."""
    assert await manager.get_agent_count() == 0

    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    assert await manager.get_agent_count() == 1

    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )
    assert await manager.get_agent_count() == 2


@pytest.mark.asyncio
async def test_get_capabilities_index(manager, sample_agent_card, another_agent_card):
    """Test getting capabilities index with counts."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    capabilities = await manager.get_capabilities_index()

    assert capabilities["text.generation"] == 1
    assert capabilities["text.analysis"] == 1
    assert capabilities["image.generation"] == 1


# ==================== Agent Cleanup Tests ====================


@pytest.mark.asyncio
async def test_cleanup_inactive_agents(manager, sample_agent_card, another_agent_card):
    """Test cleanup of inactive agents."""
    # Register two agents
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))
    await manager.register_agent(
        AgentRegistrationRequest(agent_card=another_agent_card)
    )

    # Manually set one agent's last_seen to old timestamp
    agent = await manager.get_agent("test-agent-1")
    agent.last_seen = datetime.now(UTC) - timedelta(hours=48)

    # Cleanup agents inactive for more than 24 hours
    removed_count = await manager.cleanup_inactive_agents(max_inactive_hours=24)

    assert removed_count == 1

    # Verify the old agent was removed
    assert await manager.get_agent("test-agent-1") is None
    assert await manager.get_agent("test-agent-2") is not None


@pytest.mark.asyncio
async def test_cleanup_no_inactive_agents(manager, sample_agent_card):
    """Test cleanup when all agents are active."""
    await manager.register_agent(AgentRegistrationRequest(agent_card=sample_agent_card))

    removed_count = await manager.cleanup_inactive_agents(max_inactive_hours=24)

    assert removed_count == 0
    assert await manager.get_agent("test-agent-1") is not None


# ==================== Global Instance Test ====================


@pytest.mark.asyncio
async def test_global_agent_manager_instance():
    """Test that global agent_manager instance exists and works."""
    assert agent_manager is not None
    assert isinstance(agent_manager, AgentManager)
