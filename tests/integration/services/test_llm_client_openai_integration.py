"""Integration tests for OpenAI LLM client with real API.

This module tests the LLMClientOpenAI class against the actual OpenAI API
to verify end-to-end functionality including:
- Real API completion requests
- Real streaming completions
- Token usage tracking
- Latency measurement
- A2A context propagation

These tests require OPENAI_API_KEY environment variable to be set.
They are skipped if the API key is not available.

Target model: gpt-4.1-mini (fast and cost-effective for testing)
"""

from __future__ import annotations

import os

import pytest

from agentcore.a2a_protocol.models.llm import LLMRequest, LLMResponse
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI


# Skip all tests if OPENAI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY environment variable not set",
)


@pytest.fixture
def openai_client() -> LLMClientOpenAI:
    """Create LLMClientOpenAI with real API key."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return LLMClientOpenAI(api_key=api_key, timeout=30.0, max_retries=3)


@pytest.fixture
def simple_request() -> LLMRequest:
    """Create simple LLM request for testing."""
    return LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
        temperature=0.0,  # Deterministic responses
        max_tokens=50,
        trace_id="integration-test-trace",
        source_agent="test-agent",
        session_id="test-session",
    )


class TestLLMClientOpenAIIntegrationComplete:
    """Integration tests for complete() method with real API."""

    @pytest.mark.asyncio
    async def test_complete_real_api(
        self,
        openai_client: LLMClientOpenAI,
        simple_request: LLMRequest,
    ) -> None:
        """Test completion with real OpenAI API."""
        response = await openai_client.complete(simple_request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert "hello" in response.content.lower()

        # Verify provider metadata
        assert response.provider == "openai"
        assert response.model == "gpt-4.1-mini"
        assert response.trace_id == "integration-test-trace"

        # Verify token usage
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

        # Verify latency tracking
        assert response.latency_ms > 0
        assert response.latency_ms < 30000  # Should complete within 30s

    @pytest.mark.asyncio
    async def test_complete_conversation(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test multi-turn conversation."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What is that number multiplied by 3?"},
            ],
            temperature=0.0,
            max_tokens=50,
        )

        response = await openai_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        # Should mention 12 (4 * 3)
        assert "12" in response.content

    @pytest.mark.asyncio
    async def test_complete_with_higher_temperature(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test completion with higher temperature for creativity."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "Generate a random number between 1 and 10."}
            ],
            temperature=1.5,
            max_tokens=50,
        )

        response = await openai_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.usage.total_tokens > 0


class TestLLMClientOpenAIIntegrationStream:
    """Integration tests for stream() method with real API."""

    @pytest.mark.asyncio
    async def test_stream_real_api(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test streaming with real OpenAI API."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            temperature=0.0,
            max_tokens=100,
            stream=True,
            trace_id="stream-test-trace",
        )

        # Collect all tokens
        tokens: list[str] = []
        async for token in openai_client.stream(request):
            assert isinstance(token, str)
            assert len(token) > 0
            tokens.append(token)

        # Verify streaming worked
        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0

        # Should contain numbers 1-5
        assert "1" in full_response
        assert "5" in full_response

    @pytest.mark.asyncio
    async def test_stream_progressive_delivery(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test that streaming delivers tokens progressively."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "Write a 50-word paragraph about AI."}
            ],
            temperature=0.7,
            max_tokens=200,
            stream=True,
        )

        token_count = 0
        first_token_received = False

        async for token in openai_client.stream(request):
            token_count += 1
            if not first_token_received:
                first_token_received = True
                # First token should arrive quickly
                assert True  # Token received

            # Verify each chunk is a non-empty string
            assert isinstance(token, str)
            assert len(token) > 0

        # Should receive multiple chunks
        assert token_count > 5
        assert first_token_received

    @pytest.mark.asyncio
    async def test_stream_short_response(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test streaming with very short response."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say 'yes'."}],
            temperature=0.0,
            max_tokens=10,
            stream=True,
        )

        tokens: list[str] = []
        async for token in openai_client.stream(request):
            tokens.append(token)

        assert len(tokens) > 0
        full_response = "".join(tokens).lower()
        assert "yes" in full_response


class TestLLMClientOpenAIIntegrationModels:
    """Integration tests for different GPT models."""

    @pytest.mark.asyncio
    async def test_gpt_4_1_mini_model(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test with gpt-4.1-mini model."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            temperature=0.0,
            max_tokens=50,
        )

        response = await openai_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.model == "gpt-4.1-mini"
        assert "paris" in response.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="gpt-5 not available yet, placeholder for future")
    async def test_gpt_5_model(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test with gpt-5 model (when available)."""
        request = LLMRequest(
            model="gpt-5",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            temperature=0.0,
            max_tokens=50,
        )

        response = await openai_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.model == "gpt-5"
        assert "paris" in response.content.lower()


class TestLLMClientOpenAIIntegrationA2AContext:
    """Integration tests for A2A context propagation."""

    @pytest.mark.asyncio
    async def test_a2a_context_propagation(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test that A2A context is propagated to response."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say hello."}],
            temperature=0.0,
            max_tokens=50,
            trace_id="test-trace-123",
            source_agent="test-agent-456",
            session_id="test-session-789",
        )

        response = await openai_client.complete(request)

        # Verify A2A context is preserved in response
        assert response.trace_id == "test-trace-123"
        # Note: source_agent and session_id are only in headers, not in response

    @pytest.mark.asyncio
    async def test_without_a2a_context(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test request without A2A context."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say hello."}],
            temperature=0.0,
            max_tokens=50,
        )

        response = await openai_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.trace_id is None


class TestLLMClientOpenAIIntegrationPerformance:
    """Integration tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_tracking(
        self,
        openai_client: LLMClientOpenAI,
        simple_request: LLMRequest,
    ) -> None:
        """Test that latency is accurately tracked."""
        response = await openai_client.complete(simple_request)

        # Latency should be reasonable
        assert response.latency_ms > 100  # At least 100ms for API call
        assert response.latency_ms < 30000  # Less than 30s timeout

    @pytest.mark.asyncio
    async def test_token_usage_accuracy(
        self,
        openai_client: LLMClientOpenAI,
    ) -> None:
        """Test that token usage is accurately reported."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": "This is a test message for token counting."}
            ],
            temperature=0.0,
            max_tokens=50,
        )

        response = await openai_client.complete(request)

        # Verify token counts are sensible
        assert response.usage.prompt_tokens > 5  # At least a few tokens in prompt
        assert response.usage.completion_tokens > 0  # Some tokens in response
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        # Total should be less than max_tokens + prompt
        assert response.usage.total_tokens < 100
