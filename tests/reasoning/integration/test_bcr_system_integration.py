"""
Phase 3 System Integration Tests for Bounded Context Reasoning (BCR-020).

Tests full system integration including:
- Agent registration with reasoning capabilities
- Discovery filtering by reasoning support
- Message routing to reasoning agents
- Authentication and authorization
"""

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
from agentcore.a2a_protocol.models.security import Permission, Role
from agentcore.a2a_protocol.services.agent_manager import agent_manager
from agentcore.a2a_protocol.services.message_router import (
    MessageRouter,
    RoutingStrategy,
)
from agentcore.a2a_protocol.services.security_service import security_service


class TestBCRSystemIntegration:
    """Full system integration tests for bounded reasoning."""

    @pytest.fixture
    def message_router(self) -> MessageRouter:
        """Create message router instance."""
        return MessageRouter()

    @pytest.fixture
    async def setup_system(self) -> dict[str, str]:
        """Setup system with agents and generate auth tokens."""
        # Clear existing agents
        agent_manager._agents.clear()

        # Create test agents with various reasoning capabilities
        agents = [
            {
                "id": "agent-basic",
                "name": "Basic Agent",
                "reasoning": False,
                "config": None,
            },
            {
                "id": "agent-reasoning-simple",
                "name": "Simple Reasoning Agent",
                "reasoning": True,
                "config": None,
            },
            {
                "id": "agent-reasoning-advanced",
                "name": "Advanced Reasoning Agent",
                "reasoning": True,
                "config": {
                    "max_iterations": 10,
                    "chunk_size": 16384,
                    "carryover_size": 8192,
                    "temperature": 0.8,
                },
            },
        ]

        for agent_data in agents:
            agent = AgentCard(
                agent_id=agent_data["id"],
                agent_name=agent_data["name"],
                endpoints=[
                    AgentEndpoint(
                        url=f"https://{agent_data['id']}.com/api",
                        type=EndpointType.HTTPS,
                    )
                ],
                authentication=AgentAuthentication(type=AuthenticationType.JWT),
                supports_bounded_reasoning=agent_data["reasoning"],
                reasoning_config=agent_data["config"],
            )

            await agent_manager.register_agent(
                AgentRegistrationRequest(agent_card=agent, override_existing=False)
            )

        # Generate auth tokens
        admin_token = security_service.generate_token(
            subject="admin-user", role=Role.ADMIN, agent_id=None
        )

        agent_token = security_service.generate_token(
            subject="agent-test", role=Role.AGENT, agent_id="agent-test"
        )

        return {
            "admin_token": admin_token,
            "agent_token": agent_token,
        }

    async def test_agent_registration_with_reasoning_full_flow(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test complete agent registration flow with reasoning capabilities."""
        # Register new agent with reasoning
        new_agent = AgentCard(
            agent_id="test-reasoning-full",
            agent_name="Test Reasoning Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://test-reasoning.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.JWT),
            supports_bounded_reasoning=True,
            reasoning_config={
                "max_iterations": 5,
                "chunk_size": 8192,
                "carryover_size": 4096,
            },
        )

        request = AgentRegistrationRequest(
            agent_card=new_agent, override_existing=False
        )

        response = await agent_manager.register_agent(request)

        assert response.status == "registered"
        assert response.agent_id == "test-reasoning-full"

        # Verify agent was registered correctly
        registered_agent = await agent_manager.get_agent("test-reasoning-full")
        assert registered_agent is not None
        assert registered_agent.supports_bounded_reasoning is True
        assert registered_agent.reasoning_config["max_iterations"] == 5

    async def test_agent_registration_validation_errors(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test agent registration rejects invalid reasoning configs."""
        # Invalid max_iterations
        invalid_agent = AgentCard(
            agent_id="test-invalid",
            agent_name="Invalid Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://test-invalid.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.JWT),
            reasoning_config={"max_iterations": 100},  # > 50
        )

        request = AgentRegistrationRequest(
            agent_card=invalid_agent, override_existing=False
        )

        with pytest.raises(ValueError, match="max_iterations must be between 1 and 50"):
            await agent_manager.register_agent(request)

    async def test_discovery_filter_reasoning_agents(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test agent discovery filtering by reasoning support."""
        # Discover only reasoning agents
        query = AgentDiscoveryQuery(has_bounded_reasoning=True)
        response = await agent_manager.discover_agents(query)

        assert response.total_count == 2
        reasoning_agent_ids = {agent["agent_id"] for agent in response.agents}
        assert reasoning_agent_ids == {
            "agent-reasoning-simple",
            "agent-reasoning-advanced",
        }

        # Verify all returned agents support reasoning
        for agent in response.agents:
            assert agent["supports_bounded_reasoning"] is True

    async def test_discovery_filter_non_reasoning_agents(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test agent discovery filtering for non-reasoning agents."""
        query = AgentDiscoveryQuery(has_bounded_reasoning=False)
        response = await agent_manager.discover_agents(query)

        assert response.total_count == 1
        assert response.agents[0]["agent_id"] == "agent-basic"
        assert response.agents[0]["supports_bounded_reasoning"] is False

    async def test_discovery_no_filter_returns_all(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test discovery without reasoning filter returns all agents."""
        query = AgentDiscoveryQuery()
        response = await agent_manager.discover_agents(query)

        assert response.total_count == 3

    async def test_message_routing_prefers_reasoning_agents(
        self, setup_system: dict[str, str], message_router: MessageRouter
    ) -> None:
        """Test message routing prioritizes reasoning-capable agents."""
        # Test prioritization directly without creating envelope
        candidates = ["agent-basic", "agent-reasoning-simple", "agent-reasoning-advanced"]

        # Test prioritization directly
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        # Reasoning agents should come first
        assert prioritized[0] in ["agent-reasoning-simple", "agent-reasoning-advanced"]
        assert prioritized[1] in ["agent-reasoning-simple", "agent-reasoning-advanced"]
        assert prioritized[2] == "agent-basic"

    async def test_routing_strategy_with_reasoning_priority(
        self, setup_system: dict[str, str], message_router: MessageRouter
    ) -> None:
        """Test various routing strategies respect reasoning prioritization."""
        candidates = ["agent-basic", "agent-reasoning-simple", "agent-reasoning-advanced"]

        # Test CAPABILITY_MATCH
        selected = await message_router._select_agent(
            candidates, RoutingStrategy.CAPABILITY_MATCH
        )
        assert selected in ["agent-reasoning-simple", "agent-reasoning-advanced"]

        # Test ROUND_ROBIN
        selected = await message_router._select_agent(
            candidates, RoutingStrategy.ROUND_ROBIN
        )
        assert selected in ["agent-reasoning-simple", "agent-reasoning-advanced"]

    async def test_end_to_end_registration_discovery_routing(
        self, setup_system: dict[str, str], message_router: MessageRouter
    ) -> None:
        """Test complete end-to-end flow: register -> discover -> route."""
        # 1. Register new reasoning agent
        new_agent = AgentCard(
            agent_id="e2e-reasoning",
            agent_name="E2E Reasoning Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://e2e-reasoning.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.JWT),
            supports_bounded_reasoning=True,
            reasoning_config={"max_iterations": 7, "temperature": 0.6},
        )

        registration_response = await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=new_agent, override_existing=False)
        )

        assert registration_response.status == "registered"

        # 2. Discover reasoning agents
        query = AgentDiscoveryQuery(has_bounded_reasoning=True)
        discovery_response = await agent_manager.discover_agents(query)

        # Should now include the new agent (total 3)
        assert discovery_response.total_count == 3
        agent_ids = {agent["agent_id"] for agent in discovery_response.agents}
        assert "e2e-reasoning" in agent_ids

        # 3. Route message and verify reasoning agent is selected
        candidates = ["agent-basic", "e2e-reasoning"]
        selected = await message_router._select_agent(
            candidates, RoutingStrategy.CAPABILITY_MATCH
        )

        # Should select reasoning agent
        assert selected == "e2e-reasoning"

    async def test_discovery_summary_includes_reasoning_metadata(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test that discovery summaries include reasoning support information."""
        # Get specific agent
        agent_summary = await agent_manager.get_agent_summary("agent-reasoning-advanced")

        assert agent_summary is not None
        assert "supports_bounded_reasoning" in agent_summary
        assert agent_summary["supports_bounded_reasoning"] is True
        assert agent_summary["agent_id"] == "agent-reasoning-advanced"

    async def test_multi_criteria_discovery(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test discovery with multiple criteria including reasoning."""
        # Add agents with tags
        tagged_reasoning_agent = AgentCard(
            agent_id="tagged-reasoning",
            agent_name="Tagged Reasoning Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://tagged.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.JWT),
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

        await agent_manager.register_agent(
            AgentRegistrationRequest(
                agent_card=tagged_reasoning_agent, override_existing=False
            )
        )

        # Query with both reasoning and tags filters
        query = AgentDiscoveryQuery(has_bounded_reasoning=True, tags=["production", "ml"])
        response = await agent_manager.discover_agents(query)

        assert response.total_count == 1
        assert response.agents[0]["agent_id"] == "tagged-reasoning"

    async def test_reasoning_config_validation_during_registration(
        self, setup_system: dict[str, str]
    ) -> None:
        """Test that reasoning config is validated during registration."""
        # Valid config at boundaries
        valid_agent = AgentCard(
            agent_id="boundary-test",
            agent_name="Boundary Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://boundary.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.JWT),
            reasoning_config={
                "max_iterations": 50,
                "chunk_size": 32768,
                "carryover_size": 16384,
                "temperature": 2.0,
            },
        )

        response = await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=valid_agent, override_existing=False)
        )

        assert response.status == "registered"

    async def test_system_handles_mixed_agent_population(
        self, setup_system: dict[str, str], message_router: MessageRouter
    ) -> None:
        """Test system handles mixed population of reasoning and non-reasoning agents."""
        # Verify current population
        all_agents = await agent_manager.list_all_agents()
        reasoning_count = sum(
            1 for agent in all_agents if agent.get("supports_bounded_reasoning", False)
        )
        non_reasoning_count = len(all_agents) - reasoning_count

        assert reasoning_count == 2  # simple and advanced
        assert non_reasoning_count == 1  # basic

        # Test routing works with mixed population
        candidates = [agent["agent_id"] for agent in all_agents]
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        # Verify reasoning agents come first
        first_two = prioritized[:2]
        last_one = prioritized[2:]

        # First two should be reasoning agents
        for agent_id in first_two:
            agent = await agent_manager.get_agent(agent_id)
            assert agent.supports_bounded_reasoning is True

        # Last one should be non-reasoning
        for agent_id in last_one:
            agent = await agent_manager.get_agent(agent_id)
            assert agent.supports_bounded_reasoning is False
