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

from agentcore.a2a_protocol.models.llm import LLMRequest, LLMResponse
from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI
from agentcore.a2a_protocol.services.llm_service import LLMService, ProviderRegistry


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


class TestLLMServiceIntegration:
    """Integration tests for LLMService with real provider APIs.

    These tests make actual API calls to verify end-to-end functionality.
    Tests are skipped if API keys are not configured.
    """

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not configured",
    )
    async def test_complete_with_openai(self) -> None:
        """Test complete() with real OpenAI API call."""
        service = LLMService(timeout=30.0, max_retries=2)

        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else"}],
            max_tokens=10,
            temperature=0.0,
            trace_id="integration-test-001",
            source_agent="test-agent",
        )

        response = await service.complete(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content
        assert "test" in response.content.lower()
        assert response.provider == "openai"
        assert response.model == "gpt-4.1-mini"
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.latency_ms > 0
        assert response.trace_id == "integration-test-001"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Anthropic API key not configured",
    )
    async def test_complete_with_anthropic(self) -> None:
        """Test complete() with real Anthropic API call."""
        service = LLMService(timeout=30.0)

        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Respond with 'hello' only"}],
            max_tokens=20,
            temperature=0.0,
            trace_id="integration-test-002",
        )

        response = await service.complete(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.provider == "anthropic"
        assert response.model == "claude-3-5-haiku-20241022"
        assert response.usage.total_tokens > 0
        assert response.latency_ms > 0
        assert response.trace_id == "integration-test-002"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("GOOGLE_API_KEY"),
        reason="Google API key not configured",
    )
    async def test_complete_with_gemini(self) -> None:
        """Test complete() with real Gemini API call."""
        service = LLMService()

        request = LLMRequest(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say 'hi'"}],
            max_tokens=10,
            temperature=0.0,
            trace_id="integration-test-003",
        )

        response = await service.complete(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content
        assert response.provider == "gemini"
        assert response.model == "gemini-1.5-flash"
        assert response.usage.total_tokens > 0
        assert response.latency_ms > 0
        assert response.trace_id == "integration-test-003"

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not configured",
    )
    async def test_stream_with_openai(self) -> None:
        """Test stream() with real OpenAI streaming API call."""
        service = LLMService()

        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Count from 1 to 3"}],
            max_tokens=20,
            temperature=0.0,
            stream=True,
            trace_id="integration-test-stream-001",
        )

        tokens = []
        async for token in service.stream(request):
            tokens.append(token)
            assert isinstance(token, str)

        # Verify we received tokens
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0

    @pytest.mark.asyncio
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
    async def test_service_with_multiple_providers(self) -> None:
        """Test LLMService can handle requests to all three providers."""
        service = LLMService(timeout=30.0, max_retries=2)

        # OpenAI request
        openai_request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say 'openai'"}],
            max_tokens=10,
            temperature=0.0,
        )
        openai_response = await service.complete(openai_request)
        assert openai_response.provider == "openai"

        # Anthropic request
        anthropic_request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Say 'anthropic'"}],
            max_tokens=10,
            temperature=0.0,
        )
        anthropic_response = await service.complete(anthropic_request)
        assert anthropic_response.provider == "anthropic"

        # Gemini request
        gemini_request = LLMRequest(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say 'gemini'"}],
            max_tokens=10,
            temperature=0.0,
        )
        gemini_response = await service.complete(gemini_request)
        assert gemini_response.provider == "gemini"

        # Verify all responses are distinct and valid
        assert openai_response.content != anthropic_response.content or openai_response.content != gemini_response.content

    @pytest.mark.asyncio
    async def test_global_singleton_instance(self) -> None:
        """Test that global llm_service singleton is functional."""
        from agentcore.a2a_protocol.services.llm_service import llm_service

        # Verify singleton exists and has correct type
        assert isinstance(llm_service, LLMService)
        assert hasattr(llm_service, "complete")
        assert hasattr(llm_service, "stream")
        assert hasattr(llm_service, "registry")
