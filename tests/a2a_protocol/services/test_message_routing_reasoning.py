"""Tests for message routing with reasoning agent prioritization (BCR-019)."""

from __future__ import annotations

import pytest

from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCard,
    AgentEndpoint,
    AgentRegistrationRequest,
    AuthenticationType,
    EndpointType,
)
from agentcore.a2a_protocol.models.jsonrpc import MessageEnvelope
from agentcore.a2a_protocol.services.agent_manager import agent_manager
from agentcore.a2a_protocol.services.message_router import (
    MessageRouter,
    RoutingStrategy,
)


class TestMessageRoutingReasoning:
    """Test suite for message routing with reasoning prioritization."""

    @pytest.fixture
    def message_router(self) -> MessageRouter:
        """Create message router instance."""
        return MessageRouter()

    @pytest.fixture
    async def populated_agent_manager(self) -> None:
        """Create agent manager with test agents."""
        # Clear existing agents
        agent_manager._agents.clear()

        # Agent 1: No reasoning support
        agent1 = AgentCard(
            agent_id="agent-basic-1",
            agent_name="Basic Agent 1",
            endpoints=[
                AgentEndpoint(url="https://agent1.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
        )

        # Agent 2: Reasoning support
        agent2 = AgentCard(
            agent_id="agent-reasoning-1",
            agent_name="Reasoning Agent 1",
            endpoints=[
                AgentEndpoint(url="https://agent2.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config={"max_iterations": 10, "chunk_size": 16384},
        )

        # Agent 3: Reasoning support
        agent3 = AgentCard(
            agent_id="agent-reasoning-2",
            agent_name="Reasoning Agent 2",
            endpoints=[
                AgentEndpoint(url="https://agent3.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        # Agent 4: No reasoning support
        agent4 = AgentCard(
            agent_id="agent-basic-2",
            agent_name="Basic Agent 2",
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

    async def test_prioritize_reasoning_agents_all_reasoning(
        self, message_router: MessageRouter
    ) -> None:
        """Test prioritization when all candidates support reasoning."""
        # Setup agents
        agent1 = AgentCard(
            agent_id="reasoning-a",
            agent_name="Reasoning A",
            endpoints=[
                AgentEndpoint(url="https://a.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        agent2 = AgentCard(
            agent_id="reasoning-b",
            agent_name="Reasoning B",
            endpoints=[
                AgentEndpoint(url="https://b.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent1, override_existing=True)
        )
        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent2, override_existing=True)
        )

        candidates = ["reasoning-a", "reasoning-b"]
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        # All agents support reasoning, order should be preserved
        assert len(prioritized) == 2
        assert set(prioritized) == {"reasoning-a", "reasoning-b"}

    async def test_prioritize_reasoning_agents_mixed(
        self, message_router: MessageRouter
    ) -> None:
        """Test prioritization with mixed reasoning/non-reasoning agents."""
        # Setup agents
        basic_agent = AgentCard(
            agent_id="basic",
            agent_name="Basic Agent",
            endpoints=[
                AgentEndpoint(url="https://basic.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
        )

        reasoning_agent = AgentCard(
            agent_id="reasoning",
            agent_name="Reasoning Agent",
            endpoints=[
                AgentEndpoint(url="https://reasoning.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=basic_agent, override_existing=True)
        )
        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=reasoning_agent, override_existing=True)
        )

        candidates = ["basic", "reasoning"]
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        # Reasoning agent should come first
        assert len(prioritized) == 2
        assert prioritized[0] == "reasoning"
        assert prioritized[1] == "basic"

    async def test_prioritize_reasoning_agents_none_reasoning(
        self, message_router: MessageRouter
    ) -> None:
        """Test prioritization when no candidates support reasoning."""
        # Setup agents
        agent1 = AgentCard(
            agent_id="basic-1",
            agent_name="Basic Agent 1",
            endpoints=[
                AgentEndpoint(url="https://agent1.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
        )

        agent2 = AgentCard(
            agent_id="basic-2",
            agent_name="Basic Agent 2",
            endpoints=[
                AgentEndpoint(url="https://agent2.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
        )

        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent1, override_existing=True)
        )
        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent2, override_existing=True)
        )

        candidates = ["basic-1", "basic-2"]
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        # No reasoning agents, order should be preserved
        assert len(prioritized) == 2
        assert set(prioritized) == {"basic-1", "basic-2"}

    async def test_select_agent_prefers_reasoning(
        self, message_router: MessageRouter
    ) -> None:
        """Test that agent selection prefers reasoning agents."""
        # Setup agents
        basic_agent = AgentCard(
            agent_id="select-basic",
            agent_name="Basic Agent",
            endpoints=[
                AgentEndpoint(url="https://basic.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
        )

        reasoning_agent = AgentCard(
            agent_id="select-reasoning",
            agent_name="Reasoning Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://reasoning.com/api", type=EndpointType.HTTPS
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=basic_agent, override_existing=True)
        )
        await agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=reasoning_agent, override_existing=True)
        )

        # Test with CAPABILITY_MATCH strategy (default)
        candidates = ["select-basic", "select-reasoning"]
        selected = await message_router._select_agent(
            candidates, RoutingStrategy.CAPABILITY_MATCH
        )

        # Should select reasoning agent
        assert selected == "select-reasoning"

    async def test_select_agent_round_robin_with_reasoning_priority(
        self, message_router: MessageRouter
    ) -> None:
        """Test round-robin selection respects reasoning prioritization."""
        # Setup agents
        basic1 = AgentCard(
            agent_id="rr-basic-1",
            agent_name="Basic 1",
            endpoints=[
                AgentEndpoint(url="https://basic1.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
        )

        basic2 = AgentCard(
            agent_id="rr-basic-2",
            agent_name="Basic 2",
            endpoints=[
                AgentEndpoint(url="https://basic2.com/api", type=EndpointType.HTTPS)
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
        )

        reasoning1 = AgentCard(
            agent_id="rr-reasoning-1",
            agent_name="Reasoning 1",
            endpoints=[
                AgentEndpoint(
                    url="https://reasoning1.com/api", type=EndpointType.HTTPS
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        reasoning2 = AgentCard(
            agent_id="rr-reasoning-2",
            agent_name="Reasoning 2",
            endpoints=[
                AgentEndpoint(
                    url="https://reasoning2.com/api", type=EndpointType.HTTPS
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        for agent in [basic1, basic2, reasoning1, reasoning2]:
            await agent_manager.register_agent(
                AgentRegistrationRequest(agent_card=agent, override_existing=True)
            )

        candidates = ["rr-basic-1", "rr-basic-2", "rr-reasoning-1", "rr-reasoning-2"]

        # First call should select first reasoning agent
        selected1 = await message_router._select_agent(
            candidates, RoutingStrategy.ROUND_ROBIN
        )
        assert selected1 in ["rr-reasoning-1", "rr-reasoning-2"]

        # Second call should rotate through reasoning agents first
        selected2 = await message_router._select_agent(
            candidates, RoutingStrategy.ROUND_ROBIN
        )
        assert selected2 in ["rr-reasoning-1", "rr-reasoning-2"]

    async def test_select_agent_least_loaded_with_reasoning_priority(
        self, message_router: MessageRouter
    ) -> None:
        """Test least-loaded selection with reasoning prioritization ordering."""
        # Setup multiple reasoning agents with different loads
        reasoning1 = AgentCard(
            agent_id="ll-reasoning-1",
            agent_name="Reasoning 1",
            endpoints=[
                AgentEndpoint(
                    url="https://reasoning1.com/api", type=EndpointType.HTTPS
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        reasoning2 = AgentCard(
            agent_id="ll-reasoning-2",
            agent_name="Reasoning 2",
            endpoints=[
                AgentEndpoint(
                    url="https://reasoning2.com/api", type=EndpointType.HTTPS
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        reasoning3 = AgentCard(
            agent_id="ll-reasoning-3",
            agent_name="Reasoning 3",
            endpoints=[
                AgentEndpoint(
                    url="https://reasoning3.com/api", type=EndpointType.HTTPS
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        for agent in [reasoning1, reasoning2, reasoning3]:
            await agent_manager.register_agent(
                AgentRegistrationRequest(agent_card=agent, override_existing=True)
            )

        # Set loads: reasoning-1 has lowest load
        message_router._agent_load["ll-reasoning-1"] = 2
        message_router._agent_load["ll-reasoning-2"] = 5
        message_router._agent_load["ll-reasoning-3"] = 8

        candidates = ["ll-reasoning-1", "ll-reasoning-2", "ll-reasoning-3"]
        selected = await message_router._select_agent(
            candidates, RoutingStrategy.LEAST_LOADED
        )

        # Should select reasoning-1 (lowest load among reasoning agents)
        assert selected == "ll-reasoning-1"

    async def test_prioritize_empty_candidates(
        self, message_router: MessageRouter
    ) -> None:
        """Test prioritization with empty candidate list."""
        candidates = []
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        assert prioritized == []

    async def test_prioritize_nonexistent_agent(
        self, message_router: MessageRouter
    ) -> None:
        """Test prioritization handles nonexistent agents gracefully."""
        candidates = ["nonexistent-agent"]
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        # Nonexistent agents should be treated as non-reasoning
        assert len(prioritized) == 1
        assert prioritized[0] == "nonexistent-agent"

    async def test_prioritize_maintains_order_within_groups(
        self, message_router: MessageRouter
    ) -> None:
        """Test that prioritization maintains order within reasoning/non-reasoning groups."""
        # Setup agents
        agents = [
            ("r1", True),
            ("b1", False),
            ("r2", True),
            ("b2", False),
            ("r3", True),
        ]

        for agent_id, reasoning in agents:
            agent = AgentCard(
                agent_id=agent_id,
                agent_name=f"Agent {agent_id}",
                endpoints=[
                    AgentEndpoint(
                        url=f"https://{agent_id}.com/api", type=EndpointType.HTTPS
                    )
                ],
                authentication=AgentAuthentication(type=AuthenticationType.NONE),
                supports_bounded_reasoning=reasoning,
            )
            await agent_manager.register_agent(
                AgentRegistrationRequest(agent_card=agent, override_existing=True)
            )

        candidates = ["r1", "b1", "r2", "b2", "r3"]
        prioritized = await message_router._prioritize_reasoning_agents(candidates)

        # All reasoning agents should come first, preserving their original order
        # Then non-reasoning agents in their original order
        assert prioritized == ["r1", "r2", "r3", "b1", "b2"]
