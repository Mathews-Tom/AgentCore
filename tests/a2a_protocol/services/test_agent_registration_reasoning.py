"""Tests for agent registration validation with reasoning capabilities (BCR-017)."""

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
from agentcore.a2a_protocol.services.agent_manager import AgentManager


class TestAgentRegistrationReasoningValidation:
    """Test suite for agent registration reasoning validation."""

    @pytest.fixture
    def agent_manager(self) -> AgentManager:
        """Create agent manager instance."""
        return AgentManager()

    @pytest.fixture
    def basic_agent_card(self) -> AgentCard:
        """Create basic agent card."""
        return AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
        )

    async def test_register_agent_without_reasoning(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent without reasoning capabilities."""
        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        response = await agent_manager.register_agent(request)

        assert response.status == "registered"
        assert response.agent_id == basic_agent_card.agent_id

        # Verify agent was registered
        agent = await agent_manager.get_agent(basic_agent_card.agent_id)
        assert agent is not None
        assert agent.supports_bounded_reasoning is False
        assert agent.reasoning_config is None

    async def test_register_agent_with_reasoning_support(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with reasoning support enabled."""
        basic_agent_card.supports_bounded_reasoning = True

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        response = await agent_manager.register_agent(request)

        assert response.status == "registered"

        agent = await agent_manager.get_agent(basic_agent_card.agent_id)
        assert agent.supports_bounded_reasoning is True
        assert agent.reasoning_config is None

    async def test_register_agent_with_valid_reasoning_config(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with valid reasoning configuration."""
        basic_agent_card.supports_bounded_reasoning = True
        basic_agent_card.reasoning_config = {
            "max_iterations": 10,
            "chunk_size": 16384,
            "carryover_size": 8192,
            "temperature": 0.8,
        }

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        response = await agent_manager.register_agent(request)

        assert response.status == "registered"

        agent = await agent_manager.get_agent(basic_agent_card.agent_id)
        assert agent.reasoning_config["max_iterations"] == 10
        assert agent.reasoning_config["chunk_size"] == 16384
        assert agent.reasoning_config["carryover_size"] == 8192
        assert agent.reasoning_config["temperature"] == 0.8

    async def test_register_agent_with_partial_reasoning_config(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with partial reasoning configuration."""
        basic_agent_card.reasoning_config = {
            "max_iterations": 7,
            "temperature": 0.5,
        }

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        response = await agent_manager.register_agent(request)

        assert response.status == "registered"

        agent = await agent_manager.get_agent(basic_agent_card.agent_id)
        assert agent.reasoning_config["max_iterations"] == 7
        assert agent.reasoning_config["temperature"] == 0.5

    async def test_register_agent_invalid_max_iterations_low(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with max_iterations below minimum."""
        basic_agent_card.reasoning_config = {"max_iterations": 0}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="max_iterations must be between 1 and 50"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_max_iterations_high(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with max_iterations above maximum."""
        basic_agent_card.reasoning_config = {"max_iterations": 51}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="max_iterations must be between 1 and 50"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_chunk_size_low(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with chunk_size below minimum."""
        basic_agent_card.reasoning_config = {"chunk_size": 1023}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="chunk_size must be between 1024 and 32768"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_chunk_size_high(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with chunk_size above maximum."""
        basic_agent_card.reasoning_config = {"chunk_size": 32769}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="chunk_size must be between 1024 and 32768"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_carryover_size_low(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with carryover_size below minimum."""
        basic_agent_card.reasoning_config = {"carryover_size": 511}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="carryover_size must be between 512 and 16384"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_carryover_size_high(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with carryover_size above maximum."""
        basic_agent_card.reasoning_config = {"carryover_size": 16385}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="carryover_size must be between 512 and 16384"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_temperature_low(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with temperature below minimum."""
        basic_agent_card.reasoning_config = {"temperature": -0.1}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="temperature must be between 0.0 and 2.0"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_temperature_high(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with temperature above maximum."""
        basic_agent_card.reasoning_config = {"temperature": 2.1}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="temperature must be between 0.0 and 2.0"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_carryover_exceeds_chunk_size(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with carryover_size >= chunk_size."""
        basic_agent_card.reasoning_config = {
            "chunk_size": 4096,
            "carryover_size": 4096,
        }

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(
            ValueError, match="carryover_size must be less than chunk_size"
        ):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_max_iterations_type(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with non-integer max_iterations."""
        basic_agent_card.reasoning_config = {"max_iterations": "10"}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(ValueError, match="max_iterations must be an integer"):
            await agent_manager.register_agent(request)

    async def test_register_agent_invalid_temperature_type(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with non-numeric temperature."""
        basic_agent_card.reasoning_config = {"temperature": "0.7"}

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        with pytest.raises(ValueError, match="temperature must be a number"):
            await agent_manager.register_agent(request)

    async def test_register_agent_boundary_values(
        self, agent_manager: AgentManager, basic_agent_card: AgentCard
    ) -> None:
        """Test registering agent with boundary values for reasoning config."""
        basic_agent_card.reasoning_config = {
            "max_iterations": 1,
            "chunk_size": 1024,
            "carryover_size": 512,
            "temperature": 0.0,
        }

        request = AgentRegistrationRequest(
            agent_card=basic_agent_card,
            override_existing=False,
        )

        response = await agent_manager.register_agent(request)

        assert response.status == "registered"

        # Test upper boundaries
        basic_agent_card2 = AgentCard(
            agent_name="Test Agent 2",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            reasoning_config={
                "max_iterations": 50,
                "chunk_size": 32768,
                "carryover_size": 16384,
                "temperature": 2.0,
            },
        )

        request2 = AgentRegistrationRequest(
            agent_card=basic_agent_card2,
            override_existing=False,
        )

        response2 = await agent_manager.register_agent(request2)

        assert response2.status == "registered"
