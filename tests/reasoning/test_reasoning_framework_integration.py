"""
Integration tests for Reasoning Strategy Framework.

Tests the complete system across all three phases:
- Phase 1: Strategy Framework Core (protocol, registry, selector)
- Phase 2: Unified JSON-RPC API (reasoning.execute)
- Phase 3: Agent Integration (capability advertisement, discovery)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCapability,
    AgentCard,
    AgentDiscoveryQuery,
    AgentEndpoint,
    AgentRegistrationRequest,
    AgentStatus,
    AuthenticationType,
    EndpointType,
)
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.agent_manager import AgentManager
from agentcore.reasoning.models.reasoning_models import (
    ReasoningMetrics,
    ReasoningResult,
)
from agentcore.reasoning.services.reasoning_execute_jsonrpc import (
    handle_reasoning_execute,
)
from agentcore.reasoning.services.strategy_registry import registry
from agentcore.reasoning.services.strategy_selector import StrategySelector


class MockReasoningStrategy:
    """Mock reasoning strategy for testing."""

    def __init__(self, name: str = "mock_strategy", version: str = "1.0.0"):
        self._name = name
        self._version = version

    async def execute(self, query: str, **kwargs: Any) -> ReasoningResult:
        """Mock execute implementation."""
        return ReasoningResult(
            answer=f"Mock answer from {self.name}: {query}",
            strategy_used=self.name,
            metrics=ReasoningMetrics(
                total_tokens=100,
                execution_time_ms=1000,
                strategy_specific={
                    "mock": True,
                    "total_iterations": 1,
                    "compute_savings_pct": 0.0,
                },
            ),
            trace=[{"step": 0, "action": "mock"}],
        )

    def get_config_schema(self) -> dict[str, Any]:
        """Mock config schema."""
        return {"type": "object", "properties": {}}

    def get_capabilities(self) -> list[str]:
        """Mock capabilities."""
        return [f"reasoning.strategy.{self.name}"]

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name

    @property
    def version(self) -> str:
        """Strategy version."""
        return self._version


class TestReasoningFrameworkIntegration:
    """Integration tests for the complete reasoning framework."""

    def setup_method(self):
        """Set up each test with clean state."""
        # Clear registry
        registry.clear()

        # Register mock strategies
        self.bounded_context_strategy = MockReasoningStrategy(
            name="bounded_context"
        )
        self.cot_strategy = MockReasoningStrategy(name="chain_of_thought")
        self.react_strategy = MockReasoningStrategy(name="react")

        registry.register(self.bounded_context_strategy)
        registry.register(self.cot_strategy)
        registry.register(self.react_strategy)

        # Create agent manager
        self.agent_manager = AgentManager()

    def teardown_method(self):
        """Clean up after each test."""
        registry.clear()

    # ========================================================================
    # Phase 3: Agent Registration with Reasoning Capabilities
    # ========================================================================

    @pytest.mark.asyncio
    async def test_register_agent_with_reasoning_capabilities(self):
        """Test registering an agent with reasoning strategy capabilities."""
        agent_card = AgentCard(
            agent_name="reasoning_agent",
            capabilities=[
                AgentCapability(
                    name="reasoning.strategy.bounded_context",
                    version="1.0.0",
                    description="Bounded context reasoning",
                    parameters={
                        "chunk_size": 8192,
                        "carryover_size": 4096,
                        "max_iterations": 10,
                    },
                ),
                AgentCapability(
                    name="reasoning.strategy.chain_of_thought",
                    version="1.0.0",
                    description="Chain of thought reasoning",
                ),
            ],
            endpoints=[
                AgentEndpoint(
                    url="http://localhost:8001",
                    type=EndpointType.HTTP,
                )
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        request = AgentRegistrationRequest(agent_card=agent_card)
        response = await self.agent_manager.register_agent(request)

        assert response.status == "registered"
        assert response.agent_id == agent_card.agent_id

        # Verify agent was stored
        stored_agent = await self.agent_manager.get_agent(agent_card.agent_id)
        assert stored_agent is not None
        assert len(stored_agent.get_reasoning_strategies()) == 2
        assert "bounded_context" in stored_agent.get_reasoning_strategies()
        assert "chain_of_thought" in stored_agent.get_reasoning_strategies()

    @pytest.mark.asyncio
    async def test_register_agent_validates_reasoning_strategy_params(self):
        """Test that agent registration validates reasoning strategy parameters."""
        # Invalid chunk_size (too small)
        agent_card = AgentCard(
            agent_name="invalid_agent",
            capabilities=[
                AgentCapability(
                    name="reasoning.strategy.bounded_context",
                    parameters={
                        "chunk_size": 500,  # Too small (min 1024)
                    },
                )
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        request = AgentRegistrationRequest(agent_card=agent_card)

        with pytest.raises(ValueError) as exc_info:
            await self.agent_manager.register_agent(request)

        assert "chunk_size" in str(exc_info.value).lower()
        assert "1024" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_register_agent_validates_strategy_name_format(self):
        """Test that agent registration validates strategy name format."""
        # Invalid strategy name (contains uppercase)
        agent_card = AgentCard(
            agent_name="invalid_agent",
            capabilities=[
                AgentCapability(
                    name="reasoning.strategy.BoundedContext",  # Invalid: uppercase
                )
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        request = AgentRegistrationRequest(agent_card=agent_card)

        with pytest.raises(ValueError) as exc_info:
            await self.agent_manager.register_agent(request)

        assert "invalid reasoning strategy name" in str(exc_info.value).lower()

    # ========================================================================
    # Phase 3: Agent Discovery by Reasoning Strategies
    # ========================================================================

    @pytest.mark.asyncio
    async def test_discover_agents_by_reasoning_strategy(self):
        """Test discovering agents by reasoning strategy."""
        # Register two agents
        agent1 = AgentCard(
            agent_name="agent_with_bounded_context",
            capabilities=[
                AgentCapability(name="reasoning.strategy.bounded_context")
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        agent2 = AgentCard(
            agent_name="agent_with_cot",
            capabilities=[
                AgentCapability(name="reasoning.strategy.chain_of_thought")
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8002", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent1)
        )
        await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent2)
        )

        # Discover agents with bounded_context
        query = AgentDiscoveryQuery(reasoning_strategies=["bounded_context"])
        response = await self.agent_manager.discover_agents(query)

        assert response.total_count == 1
        assert len(response.agents) == 1
        assert response.agents[0]["agent_name"] == "agent_with_bounded_context"

    @pytest.mark.asyncio
    async def test_discover_agents_by_multiple_strategies(self):
        """Test discovering agents that support multiple strategies (AND logic)."""
        # Register agent with both strategies
        agent1 = AgentCard(
            agent_name="agent_with_both",
            capabilities=[
                AgentCapability(name="reasoning.strategy.bounded_context"),
                AgentCapability(name="reasoning.strategy.chain_of_thought"),
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        # Register agent with only one strategy
        agent2 = AgentCard(
            agent_name="agent_with_one",
            capabilities=[
                AgentCapability(name="reasoning.strategy.bounded_context")
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8002", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent1)
        )
        await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent2)
        )

        # Discover agents with BOTH strategies (AND logic)
        query = AgentDiscoveryQuery(
            reasoning_strategies=["bounded_context", "chain_of_thought"]
        )
        response = await self.agent_manager.discover_agents(query)

        # Only agent1 should match (has both)
        assert response.total_count == 1
        assert response.agents[0]["agent_name"] == "agent_with_both"

    @pytest.mark.asyncio
    async def test_discover_agents_includes_reasoning_strategies_in_summary(self):
        """Test that discovery summary includes reasoning_strategies field."""
        agent = AgentCard(
            agent_name="test_agent",
            capabilities=[
                AgentCapability(name="reasoning.strategy.bounded_context"),
                AgentCapability(name="reasoning.strategy.react"),
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent)
        )

        summary = await self.agent_manager.get_agent_summary(agent.agent_id)

        assert "reasoning_strategies" in summary
        assert "bounded_context" in summary["reasoning_strategies"]
        assert "react" in summary["reasoning_strategies"]
        assert len(summary["reasoning_strategies"]) == 2

    # ========================================================================
    # Phase 2 & 3: reasoning.execute with Capability-Based Routing
    # ========================================================================

    @pytest.mark.asyncio
    async def test_reasoning_execute_with_explicit_strategy(self):
        """Test reasoning.execute with explicitly requested strategy."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "What is 2+2?",
                "strategy": "bounded_context",
            },
            id="test-1",
        )

        result = await handle_reasoning_execute(request)

        assert "answer" in result
        assert "bounded_context" in result["answer"]
        assert result["strategy_used"] == "bounded_context"
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_reasoning_execute_with_agent_capabilities(self):
        """Test reasoning.execute with capability-based strategy inference."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Solve this problem",
                "agent_capabilities": [
                    "reasoning.strategy.chain_of_thought",
                    "other_capability",
                ],
            },
            id="test-2",
        )

        result = await handle_reasoning_execute(request)

        assert "answer" in result
        assert result["strategy_used"] == "chain_of_thought"

    @pytest.mark.asyncio
    async def test_reasoning_execute_with_strategy_config(self):
        """Test reasoning.execute with strategy-specific configuration."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Complex reasoning task",
                "strategy": "bounded_context",
                "strategy_config": {
                    "chunk_size": 16384,
                    "max_iterations": 10,
                },
            },
            id="test-3",
        )

        result = await handle_reasoning_execute(request)

        assert "answer" in result
        assert result["strategy_used"] == "bounded_context"

    # ========================================================================
    # Phase 1 & 2: Strategy Selection Logic
    # ========================================================================

    def test_strategy_selector_with_agent_capabilities(self):
        """Test StrategySelector with agent capabilities inference."""
        selector = StrategySelector(registry, default_strategy=None)

        # Should infer strategy from capabilities
        selected = selector.select(
            request_strategy=None,
            agent_strategy=None,
            agent_capabilities=["reasoning.strategy.react", "other_cap"],
        )

        assert selected == "react"

    def test_strategy_selector_precedence(self):
        """Test StrategySelector precedence: request > agent > default."""
        selector = StrategySelector(registry, default_strategy="react")

        # Request overrides all
        selected = selector.select(
            request_strategy="bounded_context",
            agent_strategy="chain_of_thought",
            agent_capabilities=["reasoning.strategy.react"],
        )
        assert selected == "bounded_context"

        # Agent strategy overrides default and capabilities
        selected = selector.select(
            request_strategy=None,
            agent_strategy="chain_of_thought",
            agent_capabilities=["reasoning.strategy.react"],
        )
        assert selected == "chain_of_thought"

        # Capabilities override default
        selected = selector.select(
            request_strategy=None,
            agent_strategy=None,
            agent_capabilities=["reasoning.strategy.bounded_context"],
        )
        assert selected == "bounded_context"

        # Falls back to default
        selected = selector.select(
            request_strategy=None,
            agent_strategy=None,
            agent_capabilities=None,
        )
        assert selected == "react"

    # ========================================================================
    # Backward Compatibility Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_backward_compatibility_supports_bounded_reasoning(self):
        """Test backward compatibility with deprecated supports_bounded_reasoning field."""
        # Old-style agent registration
        agent = AgentCard(
            agent_name="legacy_agent",
            supports_bounded_reasoning=True,
            reasoning_config={
                "chunk_size": 8192,
                "max_iterations": 5,
            },
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        request = AgentRegistrationRequest(agent_card=agent)
        response = await self.agent_manager.register_agent(request)

        assert response.status == "registered"

        # Old-style discovery query
        query = AgentDiscoveryQuery(has_bounded_reasoning=True)
        disc_response = await self.agent_manager.discover_agents(query)

        assert disc_response.total_count == 1
        assert disc_response.agents[0]["agent_name"] == "legacy_agent"

    @pytest.mark.asyncio
    async def test_backward_compatibility_mixed_old_and_new(self):
        """Test that old and new approaches work together."""
        # Old-style agent
        old_agent = AgentCard(
            agent_name="old_agent",
            supports_bounded_reasoning=True,
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        # New-style agent
        new_agent = AgentCard(
            agent_name="new_agent",
            capabilities=[
                AgentCapability(name="reasoning.strategy.bounded_context")
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8002", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=old_agent)
        )
        await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=new_agent)
        )

        # Old-style query should find both
        query = AgentDiscoveryQuery(has_bounded_reasoning=True)
        response = await self.agent_manager.discover_agents(query)

        assert response.total_count == 2

        # New-style query should only find new agent
        query = AgentDiscoveryQuery(reasoning_strategies=["bounded_context"])
        response = await self.agent_manager.discover_agents(query)

        assert response.total_count == 1
        assert response.agents[0]["agent_name"] == "new_agent"

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_reasoning_execute_with_nonexistent_strategy(self):
        """Test reasoning.execute with non-existent strategy returns error."""
        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Test query",
                "strategy": "nonexistent_strategy",
            },
            id="test-error-1",
        )

        result = await handle_reasoning_execute(request)

        # Result could be dict or JsonRpcResponse
        if hasattr(result, 'error'):
            # It's a JsonRpcResponse object
            assert result.error is not None
        else:
            # It's a dict
            assert "error" in result

    @pytest.mark.asyncio
    async def test_agent_registration_with_invalid_carryover_size(self):
        """Test that invalid carryover_size >= chunk_size is rejected."""
        agent = AgentCard(
            agent_name="invalid_agent",
            capabilities=[
                AgentCapability(
                    name="reasoning.strategy.bounded_context",
                    parameters={
                        "chunk_size": 8192,
                        "carryover_size": 8192,  # Invalid: must be < chunk_size
                    },
                )
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        request = AgentRegistrationRequest(agent_card=agent)

        with pytest.raises(ValueError) as exc_info:
            await self.agent_manager.register_agent(request)

        assert "carryover_size" in str(exc_info.value).lower()
        assert "chunk_size" in str(exc_info.value).lower()

    # ========================================================================
    # End-to-End Workflow Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_end_to_end_agent_registration_discovery_execution(self):
        """Test complete workflow: register → discover → execute."""
        # 1. Register agent with reasoning capabilities
        agent = AgentCard(
            agent_name="e2e_agent",
            capabilities=[
                AgentCapability(
                    name="reasoning.strategy.bounded_context",
                    parameters={
                        "chunk_size": 8192,
                        "max_iterations": 10,
                    },
                ),
                AgentCapability(name="reasoning.strategy.chain_of_thought"),
            ],
            endpoints=[
                AgentEndpoint(url="http://localhost:8001", type=EndpointType.HTTP)
            ],
            authentication=AgentAuthentication(
                type=AuthenticationType.NONE, required=False
            ),
        )

        reg_response = await self.agent_manager.register_agent(
            AgentRegistrationRequest(agent_card=agent)
        )
        assert reg_response.status == "registered"

        # 2. Discover agent by reasoning strategy
        query = AgentDiscoveryQuery(reasoning_strategies=["bounded_context"])
        disc_response = await self.agent_manager.discover_agents(query)

        assert disc_response.total_count == 1
        discovered_agent = disc_response.agents[0]
        assert "bounded_context" in discovered_agent["reasoning_strategies"]

        # 3. Execute reasoning with discovered agent's capabilities
        # Note: capabilities in summary are strings, not dicts
        agent_capabilities = discovered_agent["capabilities"]

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="reasoning.execute",
            params={
                "query": "Solve complex problem",
                "agent_capabilities": agent_capabilities,  # Already a list of strings
            },
            id="e2e-1",
        )

        exec_result = await handle_reasoning_execute(request)

        assert "answer" in exec_result
        assert exec_result["strategy_used"] in ["bounded_context", "chain_of_thought"]
        assert "metrics" in exec_result
