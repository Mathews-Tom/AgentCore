"""Tests for agent discovery with reasoning capabilities filtering (BCR-018)."""

from __future__ import annotations

import pytest

from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCard,
    AgentDiscoveryQuery,
    AgentEndpoint,
    AgentRegistrationRequest,
    AuthenticationType,
    EndpointType,
)
from agentcore.a2a_protocol.services.agent_manager import AgentManager


class TestAgentDiscoveryReasoning:
    """Test suite for agent discovery with reasoning filtering."""

    @pytest.fixture
    def agent_manager(self) -> AgentManager:
        """Create agent manager instance."""
        return AgentManager()

    @pytest.fixture
    async def populated_agent_manager(
        self, agent_manager: AgentManager
    ) -> AgentManager:
        """Create agent manager with test agents."""
        # Agent 1: No reasoning support
        agent1 = AgentCard(
            agent_id="agent-1",
            agent_name="Basic Agent",
            endpoints=[
                AgentEndpoint(url="https://agent1.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
        )

        # Agent 2: Reasoning support, no config
        agent2 = AgentCard(
            agent_id="agent-2",
            agent_name="Reasoning Agent",
            endpoints=[
                AgentEndpoint(url="https://agent2.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        # Agent 3: Reasoning support with config
        agent3 = AgentCard(
            agent_id="agent-3",
            agent_name="Advanced Reasoning Agent",
            endpoints=[
                AgentEndpoint(url="https://agent3.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config={"max_iterations": 10, "chunk_size": 16384},
        )

        # Agent 4: No reasoning support
        agent4 = AgentCard(
            agent_id="agent-4",
            agent_name="Another Basic Agent",
            endpoints=[
                AgentEndpoint(url="https://agent4.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
        )

        # Register all agents
        for agent in [agent1, agent2, agent3, agent4]:
            await agent_manager.register_agent(
                AgentRegistrationRequest(agent_card=agent, override_existing=False)
            )

        return agent_manager

    async def test_discover_all_agents_no_filter(
        self, populated_agent_manager: AgentManager
    ) -> None:
        """Test discovering all agents without reasoning filter."""
        query = AgentDiscoveryQuery()

        response = await populated_agent_manager.discover_agents(query)

        assert response.total_count == 4
        assert len(response.agents) == 4

    async def test_discover_agents_with_reasoning_true(
        self, populated_agent_manager: AgentManager
    ) -> None:
        """Test discovering only agents with reasoning support."""
        query = AgentDiscoveryQuery(has_bounded_reasoning=True)

        response = await populated_agent_manager.discover_agents(query)

        assert response.total_count == 2
        assert len(response.agents) == 2

        # Verify all returned agents support reasoning
        for agent in response.agents:
            assert agent["supports_bounded_reasoning"] is True

        # Verify specific agents
        agent_ids = {agent["agent_id"] for agent in response.agents}
        assert agent_ids == {"agent-2", "agent-3"}

    async def test_discover_agents_with_reasoning_false(
        self, populated_agent_manager: AgentManager
    ) -> None:
        """Test discovering only agents without reasoning support."""
        query = AgentDiscoveryQuery(has_bounded_reasoning=False)

        response = await populated_agent_manager.discover_agents(query)

        assert response.total_count == 2
        assert len(response.agents) == 2

        # Verify all returned agents don't support reasoning
        for agent in response.agents:
            assert agent["supports_bounded_reasoning"] is False

        # Verify specific agents
        agent_ids = {agent["agent_id"] for agent in response.agents}
        assert agent_ids == {"agent-1", "agent-4"}

    async def test_discover_agents_reasoning_filter_none(
        self, populated_agent_manager: AgentManager
    ) -> None:
        """Test that None filter returns all agents."""
        query = AgentDiscoveryQuery(has_bounded_reasoning=None)

        response = await populated_agent_manager.discover_agents(query)

        assert response.total_count == 4
        assert len(response.agents) == 4

    async def test_discover_agents_with_pagination(
        self, populated_agent_manager: AgentManager
    ) -> None:
        """Test reasoning filter with pagination."""
        query = AgentDiscoveryQuery(has_bounded_reasoning=True, limit=1, offset=0)

        response = await populated_agent_manager.discover_agents(query)

        assert response.total_count == 2
        assert len(response.agents) == 1
        assert response.has_more is True

        # Get second page
        query2 = AgentDiscoveryQuery(has_bounded_reasoning=True, limit=1, offset=1)
        response2 = await populated_agent_manager.discover_agents(query2)

        assert response2.total_count == 2
        assert len(response2.agents) == 1
        assert response2.has_more is False

        # Verify we got different agents
        assert response.agents[0]["agent_id"] != response2.agents[0]["agent_id"]

    async def test_discover_agents_combined_filters(
        self, agent_manager: AgentManager
    ) -> None:
        """Test reasoning filter combined with other filters."""
        # Create agents with tags and reasoning support
        agent1 = AgentCard(
            agent_id="agent-tagged-1",
            agent_name="Tagged Reasoning Agent",
            endpoints=[
                AgentEndpoint(url="https://agent1.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            metadata={
                "tags": ["production", "ml"],
                "category": None,
                "license": None,
                "documentation_url": None,
                "source_code_url": None,
                "support_contact": None,
            },
        )

        agent2 = AgentCard(
            agent_id="agent-tagged-2",
            agent_name="Tagged Basic Agent",
            endpoints=[
                AgentEndpoint(url="https://agent2.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
            metadata={
                "tags": ["production", "ml"],
                "category": None,
                "license": None,
                "documentation_url": None,
                "source_code_url": None,
                "support_contact": None,
            },
        )

        agent3 = AgentCard(
            agent_id="agent-tagged-3",
            agent_name="Tagged Reasoning Agent 2",
            endpoints=[
                AgentEndpoint(url="https://agent3.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            metadata={
                "tags": ["staging"],
                "category": None,
                "license": None,
                "documentation_url": None,
                "source_code_url": None,
                "support_contact": None,
            },
        )

        # Register agents
        for agent in [agent1, agent2, agent3]:
            await agent_manager.register_agent(
                AgentRegistrationRequest(agent_card=agent, override_existing=False)
            )

        # Query: reasoning=True AND tags=["production"]
        query = AgentDiscoveryQuery(
            has_bounded_reasoning=True, tags=["production", "ml"]
        )

        response = await agent_manager.discover_agents(query)

        assert response.total_count == 1
        assert len(response.agents) == 1
        assert response.agents[0]["agent_id"] == "agent-tagged-1"

    async def test_discover_agents_name_pattern_with_reasoning(
        self, populated_agent_manager: AgentManager
    ) -> None:
        """Test reasoning filter combined with name pattern."""
        query = AgentDiscoveryQuery(
            has_bounded_reasoning=True, name_pattern="Advanced"
        )

        response = await populated_agent_manager.discover_agents(query)

        assert response.total_count == 1
        assert len(response.agents) == 1
        assert response.agents[0]["agent_id"] == "agent-3"
        assert response.agents[0]["agent_name"] == "Advanced Reasoning Agent"

    async def test_discover_agents_empty_result(
        self, agent_manager: AgentManager
    ) -> None:
        """Test discovery with reasoning filter that matches no agents."""
        # Register only non-reasoning agents
        agent = AgentCard(
            agent_id="basic-agent",
            agent_name="Basic Agent",
            endpoints=[
                AgentEndpoint(url="https://agent.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
        )

        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent, override_existing=False)
        )

        # Query for reasoning agents
        query = AgentDiscoveryQuery(has_bounded_reasoning=True)

        response = await agent_manager.discover_agents(query)

        assert response.total_count == 0
        assert len(response.agents) == 0
        assert response.has_more is False

    async def test_discovery_summary_includes_reasoning_flag(
        self, populated_agent_manager: AgentManager
    ) -> None:
        """Test that discovery results include supports_bounded_reasoning field."""
        query = AgentDiscoveryQuery()

        response = await populated_agent_manager.discover_agents(query)

        # Verify all results have the reasoning field
        for agent in response.agents:
            assert "supports_bounded_reasoning" in agent
            assert isinstance(agent["supports_bounded_reasoning"], bool)

    async def test_query_serialization_with_reasoning(self) -> None:
        """Test that AgentDiscoveryQuery serializes correctly with reasoning filter."""
        query = AgentDiscoveryQuery(
            has_bounded_reasoning=True, capabilities=["chat"], limit=10
        )

        # Test model_dump
        data = query.model_dump()
        assert data["has_bounded_reasoning"] is True
        assert data["capabilities"] == ["chat"]
        assert data["limit"] == 10

        # Test reconstruction
        query2 = AgentDiscoveryQuery(**data)
        assert query2.has_bounded_reasoning is True
        assert query2.capabilities == ["chat"]
