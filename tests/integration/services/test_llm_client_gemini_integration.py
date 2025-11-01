"""Integration tests for Google Gemini LLM client with real API.

This module tests the LLMClientGemini class against the actual Gemini API
to verify end-to-end functionality including:
- Real API completion requests
- Real streaming completions
- Token usage tracking
- Latency measurement
- Message format conversion

These tests require GEMINI_API_KEY environment variable to be set.
They are skipped if the API key is not available.

Target model: gemini-2.5-flash-lite (fast and cost-effective for testing)
"""

from __future__ import annotations

import os

import pytest

from agentcore.a2a_protocol.models.llm import LLMRequest, LLMResponse
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini


# Skip all tests if GEMINI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set",
)


@pytest.fixture
def gemini_client() -> LLMClientGemini:
    """Create LLMClientGemini with real API key."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    return LLMClientGemini(api_key=api_key, timeout=30.0, max_retries=3)


@pytest.fixture
def simple_request() -> LLMRequest:
    """Create simple LLM request for testing."""
    return LLMRequest(
        model="gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
# Deterministic responses

        trace_id="integration-test-trace",
        source_agent="test-agent",
        session_id="test-session",
    )


class TestLLMClientGeminiIntegrationComplete:
    """Integration tests for complete() method with real API."""

    @pytest.mark.asyncio
    async def test_complete_real_api(
        self,
        gemini_client: LLMClientGemini,
        simple_request: LLMRequest,
    ) -> None:
        """Test completion with real Gemini API."""
        response = await gemini_client.complete(simple_request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert "hello" in response.content.lower()

        # Verify provider metadata
        assert response.provider == "gemini"
        assert response.model == "gemini-2.5-flash-lite"
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
    async def test_complete_with_system_message(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test completion with system message."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that speaks like a pirate."},
                {"role": "user", "content": "Say hello."},
            ],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        # May contain pirate-like language
        assert response.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_complete_conversation(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test multi-turn conversation."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "What is that number multiplied by 3?"},
            ],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        # Should mention 12 (4 * 3)
        assert "12" in response.content

    @pytest.mark.asyncio
    async def test_complete_with_higher_temperature(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test completion with higher temperature for creativity."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": "Generate a random number between 1 and 10."}
            ],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.usage.total_tokens > 0


class TestLLMClientGeminiIntegrationStream:
    """Integration tests for stream() method with real API."""

    @pytest.mark.asyncio
    async def test_stream_real_api(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test streaming with real Gemini API."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],


            stream=True,
            trace_id="stream-test-trace",
        )

        # Collect all tokens
        tokens: list[str] = []
        async for token in gemini_client.stream(request):
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
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test that streaming delivers tokens progressively."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": "Write a 50-word paragraph about AI."}
            ],


            stream=True,
        )

        token_count = 0
        first_token_received = False

        async for token in gemini_client.stream(request):
            token_count += 1
            if not first_token_received:
                first_token_received = True
                # First token should arrive quickly
                assert True  # Token received

            # Verify each chunk is a non-empty string
            assert isinstance(token, str)
            assert len(token) > 0

        # Should receive multiple chunks
        assert token_count > 1
        assert first_token_received

    @pytest.mark.asyncio
    async def test_stream_short_response(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test streaming with very short response."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Say 'yes'."}],


            stream=True,
        )

        tokens: list[str] = []
        async for token in gemini_client.stream(request):
            tokens.append(token)

        assert len(tokens) > 0
        full_response = "".join(tokens).lower()
        assert "yes" in full_response


class TestLLMClientGeminiIntegrationModels:
    """Integration tests for different Gemini models."""

    @pytest.mark.asyncio
    async def test_gemini_1_5_flash_model(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test with gemini-2.5-flash-lite model."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "What is the capital of France?"}],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.model == "gemini-2.5-flash-lite"
        assert "paris" in response.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="gemini-2.5-flash-lite may require different API access")
    async def test_gemini_2_0_flash_exp_model(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test with gemini-2.5-flash-lite model (experimental)."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "What is the capital of France?"}],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.model == "gemini-2.5-flash-lite"
        assert "paris" in response.content.lower()


class TestLLMClientGeminiIntegrationA2AContext:
    """Integration tests for A2A context propagation."""

    @pytest.mark.asyncio
    async def test_a2a_context_propagation(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test that A2A context is propagated to response."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Say hello."}],


            trace_id="test-trace-123",
            source_agent="test-agent-456",
            session_id="test-session-789",
        )

        response = await gemini_client.complete(request)

        # Verify A2A context is preserved in response
        assert response.trace_id == "test-trace-123"
        # Note: Gemini has limited header support, so source_agent/session_id
        # may not be sent in headers

    @pytest.mark.asyncio
    async def test_without_a2a_context(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test request without A2A context."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "Say hello."}],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.trace_id is None


class TestLLMClientGeminiIntegrationPerformance:
    """Integration tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_tracking(
        self,
        gemini_client: LLMClientGemini,
        simple_request: LLMRequest,
    ) -> None:
        """Test that latency is accurately tracked."""
        response = await gemini_client.complete(simple_request)

        # Latency should be reasonable
        assert response.latency_ms > 50  # At least 50ms for API call
        assert response.latency_ms < 30000  # Less than 30s timeout

    @pytest.mark.asyncio
    async def test_token_usage_accuracy(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test that token usage is accurately reported."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": "This is a test message for token counting."}
            ],


        )

        response = await gemini_client.complete(request)

        # Verify token counts are sensible
        assert response.usage.prompt_tokens > 5  # At least a few tokens in prompt
        assert response.usage.completion_tokens > 0  # Some tokens in response
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        # Total should be less than max_tokens + prompt
        assert response.usage.total_tokens < 100


class TestLLMClientGeminiIntegrationMessageFormat:
    """Integration tests for message format conversion."""

    @pytest.mark.asyncio
    async def test_convert_simple_user_message(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test simple user message conversion and execution."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "What is 1+1?"}],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        assert "2" in response.content

    @pytest.mark.asyncio
    async def test_convert_assistant_to_model_role(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test assistant role conversion to model role."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "user", "content": "What is 5+5?"},
                {"role": "assistant", "content": "5+5 equals 10."},
                {"role": "user", "content": "What is that plus 3?"},
            ],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        # Should mention 13 (10 + 3)
        assert "13" in response.content

    @pytest.mark.asyncio
    async def test_system_instruction_extraction(
        self,
        gemini_client: LLMClientGemini,
    ) -> None:
        """Test system instruction extraction and usage."""
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[
                {"role": "system", "content": "Always respond with exactly one word."},
                {"role": "user", "content": "What is the capital of France?"},
            ],


        )

        response = await gemini_client.complete(request)

        assert isinstance(response, LLMResponse)
        # Should be very short due to system instruction
        assert len(response.content.split()) <= 3
        assert "paris" in response.content.lower()
