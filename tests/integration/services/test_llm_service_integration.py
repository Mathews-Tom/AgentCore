"""Integration tests for LLM Service provider registry with real API clients.

This module tests the ProviderRegistry with actual provider instances.
Tests are skipped if API keys are not configured in the environment.

Tests cover:
- Registry initialization with all 3 providers
- Provider instance creation and caching
- Real provider clients are returned (not mocks)
"""

from __future__ import annotations

import os

import pytest

from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI
from agentcore.a2a_protocol.services.llm_service import ProviderRegistry


class TestProviderRegistryIntegration:
    """Integration tests for ProviderRegistry with real provider instances."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not configured",
    )
    def test_registry_creates_real_openai_client(self) -> None:
        """Test registry creates actual OpenAI client instance."""
        registry = ProviderRegistry()

        client = registry.get_provider_for_model("gpt-4.1-mini")

        # Verify it's a real OpenAI client, not a mock
        assert isinstance(client, LLMClientOpenAI)
        assert hasattr(client, "client")  # Has AsyncOpenAI instance

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Anthropic API key not configured",
    )
    def test_registry_creates_real_anthropic_client(self) -> None:
        """Test registry creates actual Anthropic client instance."""
        registry = ProviderRegistry()

        client = registry.get_provider_for_model("claude-3-5-haiku-20241022")

        # Verify it's a real Anthropic client, not a mock
        assert isinstance(client, LLMClientAnthropic)
        assert hasattr(client, "client")  # Has AsyncAnthropic instance

    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="Google API key not configured",
    )
    def test_registry_creates_real_gemini_client(self) -> None:
        """Test registry creates actual Gemini client instance."""
        registry = ProviderRegistry()

        client = registry.get_provider_for_model("gemini-1.5-flash")

        # Verify it's a real Gemini client, not a mock
        assert isinstance(client, LLMClientGemini)
        assert hasattr(client, "timeout")
        assert hasattr(client, "max_retries")

    @pytest.mark.skipif(
        not all(
            [
                os.getenv("OPENAI_API_KEY"),
                os.getenv("ANTHROPIC_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
            ]
        ),
        reason="All API keys not configured",
    )
    def test_registry_with_all_providers(self) -> None:
        """Test registry can manage all three providers simultaneously."""
        registry = ProviderRegistry(timeout=90.0, max_retries=5)

        # Request model from each provider
        openai_client = registry.get_provider_for_model("gpt-4.1-mini")
        anthropic_client = registry.get_provider_for_model("claude-3-5-haiku-20241022")
        gemini_client = registry.get_provider_for_model("gemini-1.5-flash")

        # Verify all are real instances
        assert isinstance(openai_client, LLMClientOpenAI)
        assert isinstance(anthropic_client, LLMClientAnthropic)
        assert isinstance(gemini_client, LLMClientGemini)

        # Verify singleton - same instances returned on subsequent requests
        assert openai_client is registry.get_provider_for_model("gpt-5-mini")
        assert anthropic_client is registry.get_provider_for_model("claude-3-opus")
        assert gemini_client is registry.get_provider_for_model("gemini-1.5-pro")

        # Verify configuration propagated to clients
        assert openai_client.timeout == 90.0
        assert openai_client.max_retries == 5
        assert anthropic_client.timeout == 90.0
        assert anthropic_client.max_retries == 5
        assert gemini_client.timeout == 90.0
        assert gemini_client.max_retries == 5
