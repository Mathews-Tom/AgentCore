"""Unit tests for rate limit handling across all LLM providers.

This module tests rate limit detection, retry logic, exponential backoff,
and Retry-After header handling for OpenAI, Anthropic, and Gemini clients.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import RateLimitError as AnthropicRateLimitError
from google.api_core import exceptions as google_exceptions
from openai import RateLimitError as OpenAIRateLimitError

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    RateLimitError as CustomRateLimitError)
from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI


class TestOpenAIRateLimitHandling:
    """Test rate limit handling for OpenAI client."""

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after_header(self) -> None:
        """Test rate limit handling respects Retry-After header."""
        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}],
            trace_id="test-trace-123")

        # Mock response with Retry-After header
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "5"}
        rate_limit_error = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch("agentcore.a2a_protocol.services.llm_client_openai.AsyncOpenAI") as mock_openai,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            mock_client = Mock()
            mock_client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)
            mock_openai.return_value = mock_client

            client = LLMClientOpenAI(api_key="test-key", timeout=60.0, max_retries=2)

            with pytest.raises(CustomRateLimitError) as exc_info:
                await client.complete(request)

            # Verify error attributes
            assert exc_info.value.provider == "openai"
            assert exc_info.value.retry_after == 5.0
            # max_retries=2 means 2 attempts (0,1), so 1 sleep between them
            assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_exponential_backoff(self) -> None:
        """Test rate limit uses exponential backoff without Retry-After."""
        client = LLMClientOpenAI(api_key="test-key", timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}])

        # Mock rate limit error without Retry-After header
        mock_response = Mock()
        mock_response.headers = {}
        rate_limit_error = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch.object(
                client.client.chat.completions,
                "create",
                side_effect=rate_limit_error),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            with pytest.raises(CustomRateLimitError):
                await client.complete(request)

            # max_retries=3 means 3 attempts (0,1,2), so 2 sleeps
            # Verify exponential backoff delays: 2^0=1, 2^1=2 (no sleep after last attempt)
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_streaming(self) -> None:
        """Test rate limit handling during streaming."""
        client = LLMClientOpenAI(api_key="test-key", timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}],
            stream=True)

        # Mock rate limit error with Retry-After
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "10"}
        rate_limit_error = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with patch.object(
            client.client.chat.completions, "create", side_effect=rate_limit_error
        ):
            with pytest.raises(CustomRateLimitError) as exc_info:
                async for _ in client.stream(request):
                    pass

            # Verify error attributes
            assert exc_info.value.provider == "openai"
            assert exc_info.value.retry_after == 10.0


class TestAnthropicRateLimitHandling:
    """Test rate limit handling for Anthropic client."""

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after_header(self) -> None:
        """Test rate limit handling respects retry-after header."""
        client = LLMClientAnthropic(api_key="test-key", timeout=60.0, max_retries=2)
        request = LLMRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "test"}],
            trace_id="test-trace-456")

        # Mock response with retry-after header
        mock_response = Mock()
        mock_response.headers = {"retry-after": "3"}
        rate_limit_error = AnthropicRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch.object(
                client.client.messages, "create", side_effect=rate_limit_error
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            with pytest.raises(CustomRateLimitError) as exc_info:
                await client.complete(request)

            # Verify error attributes
            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.retry_after == 3.0
            # max_retries=2 means 2 attempts (0,1), so 1 sleep
            assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_exponential_backoff(self) -> None:
        """Test rate limit uses exponential backoff without retry-after."""
        client = LLMClientAnthropic(api_key="test-key", timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "test"}])

        # Mock rate limit error without retry-after header
        mock_response = Mock()
        mock_response.headers = {}
        rate_limit_error = AnthropicRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch.object(
                client.client.messages, "create", side_effect=rate_limit_error
            ),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            with pytest.raises(CustomRateLimitError):
                await client.complete(request)

            # max_retries=3 means 3 attempts (0,1,2), so 2 sleeps
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_streaming(self) -> None:
        """Test rate limit handling during streaming."""
        client = LLMClientAnthropic(api_key="test-key", timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": "test"}],
            stream=True)

        # Mock rate limit error
        mock_response = Mock()
        mock_response.headers = {"retry-after": "7"}
        rate_limit_error = AnthropicRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with patch.object(
            client.client.messages, "create", side_effect=rate_limit_error
        ):
            with pytest.raises(CustomRateLimitError) as exc_info:
                async for _ in client.stream(request):
                    pass

            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.retry_after == 7.0


class TestGeminiRateLimitHandling:
    """Test rate limit handling for Gemini client."""

    @pytest.mark.asyncio
    async def test_rate_limit_resource_exhausted(self) -> None:
        """Test rate limit handling for RESOURCE_EXHAUSTED error."""
        client = LLMClientGemini(api_key="test-key", timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "test"}],
            trace_id="test-trace-789")

        # Mock RESOURCE_EXHAUSTED error (Gemini's rate limit error)
        rate_limit_error = google_exceptions.ResourceExhausted("Quota exceeded")

        with (
            patch(
                "google.generativeai.GenerativeModel.generate_content_async",
                side_effect=rate_limit_error),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            with pytest.raises(CustomRateLimitError) as exc_info:
                await client.complete(request)

            # Verify error attributes
            assert exc_info.value.provider == "gemini"
            assert exc_info.value.retry_after is None  # Gemini doesn't provide retry-after
            # Verify exponential backoff: max_retries=3 means 3 attempts (0,1,2)
            # So there are 2 sleeps (between attempts 0-1 and 1-2)
            assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_exponential_backoff_capped(self) -> None:
        """Test rate limit backoff is capped at max delay."""
        client = LLMClientGemini(api_key="test-key", timeout=60.0, max_retries=6)
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "test"}])

        rate_limit_error = google_exceptions.ResourceExhausted("Quota exceeded")

        with (
            patch(
                "google.generativeai.GenerativeModel.generate_content_async",
                side_effect=rate_limit_error),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            with pytest.raises(CustomRateLimitError):
                await client.complete(request)

            # max_retries=6 means 6 attempts (0,1,2,3,4,5), so 5 sleeps
            assert mock_sleep.call_count == 5
            # Last delays should be capped at 32s (LLM_MAX_RETRY_DELAY)
            # Delays: 1, 2, 4, 8, 16 (32 would be next but no sleep after last attempt)

    @pytest.mark.asyncio
    async def test_rate_limit_streaming(self) -> None:
        """Test rate limit handling during streaming."""
        client = LLMClientGemini(api_key="test-key", timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="gemini-2.5-flash-lite",
            messages=[{"role": "user", "content": "test"}],
            stream=True)

        rate_limit_error = google_exceptions.ResourceExhausted("Quota exceeded")

        with patch(
            "google.generativeai.GenerativeModel.generate_content_async",
            side_effect=rate_limit_error):
            with pytest.raises(CustomRateLimitError) as exc_info:
                async for _ in client.stream(request):
                    pass

            assert exc_info.value.provider == "gemini"
            assert exc_info.value.retry_after is None


class TestRateLimitMetrics:
    """Test rate limit metrics recording."""

    @pytest.mark.asyncio
    async def test_rate_limit_metrics_recorded(self) -> None:
        """Test rate limit errors and retry delays are recorded in metrics."""
        client = LLMClientOpenAI(api_key="test-key", timeout=60.0, max_retries=2)
        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}])

        mock_response = Mock()
        mock_response.headers = {}
        rate_limit_error = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch.object(
                client.client.chat.completions, "create", side_effect=rate_limit_error
            ),
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "agentcore.a2a_protocol.services.llm_client_openai.record_rate_limit_error"
            ) as mock_record_error,
            patch(
                "agentcore.a2a_protocol.services.llm_client_openai.record_rate_limit_retry_delay"
            ) as mock_record_delay):
            with pytest.raises(CustomRateLimitError):
                await client.complete(request)

            # Verify metrics were recorded
            # max_retries=2 means 2 attempts total (0,1)
            # Should record error 2 times (once per attempt)
            assert mock_record_error.call_count == 2
            # Should record delay 2 times (metric recorded before sleep check)
            # Even though only 1 actual sleep happens
            assert mock_record_delay.call_count == 2
