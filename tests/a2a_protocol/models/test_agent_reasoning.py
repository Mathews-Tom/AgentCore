"""Tests for AgentCard bounded reasoning capabilities (BCR-016)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCard,
    AgentEndpoint,
    AgentStatus,
    AuthenticationType,
    EndpointType,
)


class TestAgentCardReasoningSupport:
    """Test suite for AgentCard bounded reasoning fields."""

    def test_agent_card_default_reasoning_support(self) -> None:
        """Test that supports_bounded_reasoning defaults to False."""
        agent = AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
        )

        assert agent.supports_bounded_reasoning is False
        assert agent.reasoning_config is None

    def test_agent_card_with_reasoning_support(self) -> None:
        """Test creating agent with reasoning support enabled."""
        agent = AgentCard(
            agent_name="Reasoning Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
        )

        assert agent.supports_bounded_reasoning is True
        assert agent.reasoning_config is None

    def test_agent_card_with_reasoning_config(self) -> None:
        """Test creating agent with custom reasoning configuration."""
        reasoning_config = {
            "max_iterations": 10,
            "chunk_size": 16384,
            "carryover_size": 8192,
            "temperature": 0.8,
        }

        agent = AgentCard(
            agent_name="Reasoning Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config=reasoning_config,
        )

        assert agent.supports_bounded_reasoning is True
        assert agent.reasoning_config == reasoning_config
        assert agent.reasoning_config["max_iterations"] == 10
        assert agent.reasoning_config["chunk_size"] == 16384
        assert agent.reasoning_config["carryover_size"] == 8192
        assert agent.reasoning_config["temperature"] == 0.8

    def test_agent_card_reasoning_config_without_support(self) -> None:
        """Test that reasoning_config can be set even if supports_bounded_reasoning is False."""
        # This is valid - agent may declare config but not be actively supporting it
        reasoning_config = {"max_iterations": 5}

        agent = AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=False,
            reasoning_config=reasoning_config,
        )

        assert agent.supports_bounded_reasoning is False
        assert agent.reasoning_config == reasoning_config

    def test_agent_card_reasoning_config_empty_dict(self) -> None:
        """Test creating agent with empty reasoning config."""
        agent = AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config={},
        )

        assert agent.supports_bounded_reasoning is True
        assert agent.reasoning_config == {}

    def test_agent_card_reasoning_config_partial(self) -> None:
        """Test creating agent with partial reasoning configuration."""
        reasoning_config = {
            "max_iterations": 7,
            "temperature": 0.5,
        }

        agent = AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config=reasoning_config,
        )

        assert agent.supports_bounded_reasoning is True
        assert agent.reasoning_config["max_iterations"] == 7
        assert agent.reasoning_config["temperature"] == 0.5
        assert "chunk_size" not in agent.reasoning_config
        assert "carryover_size" not in agent.reasoning_config

    def test_discovery_summary_includes_reasoning_support(self) -> None:
        """Test that discovery summary includes supports_bounded_reasoning field."""
        agent = AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config={"max_iterations": 5},
        )

        summary = agent.to_discovery_summary()

        assert "supports_bounded_reasoning" in summary
        assert summary["supports_bounded_reasoning"] is True

    def test_discovery_summary_reasoning_false(self) -> None:
        """Test discovery summary when reasoning is not supported."""
        agent = AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
        )

        summary = agent.to_discovery_summary()

        assert "supports_bounded_reasoning" in summary
        assert summary["supports_bounded_reasoning"] is False

    def test_agent_card_serialization_with_reasoning(self) -> None:
        """Test that agent card with reasoning fields serializes correctly."""
        reasoning_config = {
            "max_iterations": 10,
            "chunk_size": 16384,
            "temperature": 0.7,
        }

        agent = AgentCard(
            agent_name="Reasoning Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config=reasoning_config,
        )

        # Test model_dump (Pydantic v2)
        data = agent.model_dump()
        assert data["supports_bounded_reasoning"] is True
        assert data["reasoning_config"] == reasoning_config

        # Test JSON serialization
        json_str = agent.model_dump_json()
        assert "supports_bounded_reasoning" in json_str
        assert "reasoning_config" in json_str

    def test_agent_card_deserialization_with_reasoning(self) -> None:
        """Test that agent card with reasoning fields deserializes correctly."""
        data = {
            "agent_name": "Reasoning Agent",
            "endpoints": [
                {
                    "url": "https://example.com/api",
                    "type": "https",
                }
            ],
            "authentication": {"type": "none"},
            "supports_bounded_reasoning": True,
            "reasoning_config": {
                "max_iterations": 10,
                "chunk_size": 16384,
            },
        }

        agent = AgentCard(**data)

        assert agent.supports_bounded_reasoning is True
        assert agent.reasoning_config["max_iterations"] == 10
        assert agent.reasoning_config["chunk_size"] == 16384

    def test_agent_card_reasoning_config_complex_types(self) -> None:
        """Test reasoning config with various value types."""
        reasoning_config = {
            "max_iterations": 10,
            "chunk_size": 16384,
            "carryover_size": 8192,
            "temperature": 0.7,
            "custom_setting": True,
            "model_name": "gpt-5",
            "parameters": {"top_p": 0.9, "frequency_penalty": 0.1},
        }

        agent = AgentCard(
            agent_name="Test Agent",
            endpoints=[
                AgentEndpoint(
                    url="https://example.com/api",
                    type=EndpointType.HTTPS,
                )
            ],
            authentication=AgentAuthentication(type=AuthenticationType.NONE),
            supports_bounded_reasoning=True,
            reasoning_config=reasoning_config,
        )

        assert agent.reasoning_config == reasoning_config
        assert agent.reasoning_config["custom_setting"] is True
        assert agent.reasoning_config["model_name"] == "gpt-5"
        assert agent.reasoning_config["parameters"]["top_p"] == 0.9
