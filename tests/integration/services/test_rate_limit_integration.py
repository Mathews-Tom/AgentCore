"""Integration tests for rate limit handling across all LLM providers.

This module tests rate limit handling in real-world scenarios with mock providers
that simulate rate limit responses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    RateLimitError as CustomRateLimitError)
from agentcore.a2a_protocol.services.llm_service import LLMService, ProviderRegistry


@pytest.fixture(autouse=True)
def clear_provider_cache() -> None:
    """Clear ProviderRegistry instance cache before each test.

    ProviderRegistry uses a class-level _instances dict to cache provider
    instances. This causes test isolation issues where providers created in
    one test (with specific max_retries) are reused in subsequent tests.

    This fixture ensures each test gets fresh provider instances.
    """
    # Clear the class-level cache before each test
    ProviderRegistry._instances.clear()
    yield
    # Clear again after test to prevent contamination
    ProviderRegistry._instances.clear()


class TestRateLimitIntegration:
    """Integration tests for rate limit handling through LLMService."""

    @pytest.mark.asyncio
    async def test_openai_rate_limit_with_retry_and_recovery(self) -> None:
        """Test OpenAI rate limit with successful retry after backoff."""
        from openai import RateLimitError as OpenAIRateLimitError

        service = LLMService(timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            trace_id="integration-test-001")

        # Mock response for successful retry
        mock_success_response = Mock()
        mock_success_response.choices = [Mock()]
        mock_success_response.choices[0].message.content = "Success after retry"
        mock_success_response.usage = Mock()
        mock_success_response.usage.prompt_tokens = 10
        mock_success_response.usage.completion_tokens = 5
        mock_success_response.usage.total_tokens = 15

        # Mock rate limit error followed by success
        mock_response = Mock()
        mock_response.headers = {"Retry-After": "1"}
        rate_limit_error = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        call_count = 0

        async def mock_create(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with rate limit
                raise rate_limit_error
            # Second call succeeds
            return mock_success_response

        with (
            patch(
                "agentcore.a2a_protocol.services.llm_client_openai.AsyncOpenAI"
            ) as mock_openai_class,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            mock_client = Mock()
            mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
            mock_openai_class.return_value = mock_client

            # Should succeed after retry
            response = await service.complete(request)

            assert response.content == "Success after retry"
            assert response.usage.total_tokens == 15
            assert call_count == 2  # First failed, second succeeded
            mock_sleep.assert_called_once()  # Should have waited before retry

    @pytest.mark.asyncio
    async def test_anthropic_rate_limit_exhausted_retries(self) -> None:
        """Test Anthropic rate limit with all retries exhausted."""
        from anthropic import RateLimitError as AnthropicRateLimitError

        service = LLMService(timeout=60.0, max_retries=2)
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "test"}],
            trace_id="integration-test-002")

        # Mock persistent rate limit error
        mock_response = Mock()
        mock_response.headers = {"retry-after": "2"}
        rate_limit_error = AnthropicRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch(
                "agentcore.a2a_protocol.services.llm_client_anthropic.AsyncAnthropic"
            ) as mock_anthropic_class,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            mock_client = Mock()
            mock_client.messages.create = AsyncMock(side_effect=rate_limit_error)
            mock_anthropic_class.return_value = mock_client

            # Should fail after exhausting retries
            with pytest.raises(CustomRateLimitError) as exc_info:
                await service.complete(request)

            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.retry_after == 2.0
            # max_retries=2 means 2 attempts (0,1), so 1 sleep
            assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_gemini_rate_limit_with_exponential_backoff(self) -> None:
        """Test Gemini rate limit with exponential backoff progression."""
        from google.api_core import exceptions as google_exceptions

        service = LLMService(timeout=60.0, max_retries=4)
        request = LLMRequest(
            model="gemini-2.0-flash-exp",
            messages=[{"role": "user", "content": "test"}],
            trace_id="integration-test-003")

        rate_limit_error = google_exceptions.ResourceExhausted("Quota exceeded")

        with (
            patch(
                "google.generativeai.GenerativeModel.generate_content_async",
                side_effect=rate_limit_error),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            # Should fail after exhausting retries
            with pytest.raises(CustomRateLimitError) as exc_info:
                await service.complete(request)

            assert exc_info.value.provider == "gemini"
            assert exc_info.value.retry_after is None

            # max_retries=4 means 4 attempts (0,1,2,3), so 3 sleeps
            # Verify exponential backoff: 1, 2, 4 seconds
            assert mock_sleep.call_count == 3
            # Check that delays are increasing exponentially (approximately)
            calls = [call.args[0] for call in mock_sleep.call_args_list]
            assert calls[0] == 1  # 2^0
            assert calls[1] == 2  # 2^1
            assert calls[2] == 4  # 2^2

    @pytest.mark.asyncio
    async def test_rate_limit_streaming_fails_immediately(self) -> None:
        """Test rate limit in streaming mode fails without retry."""
        from openai import RateLimitError as OpenAIRateLimitError

        service = LLMService(timeout=60.0, max_retries=3)
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            stream=True,
            trace_id="integration-test-004")

        mock_response = Mock()
        mock_response.headers = {"Retry-After": "5"}
        rate_limit_error = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch(
                "agentcore.a2a_protocol.services.llm_client_openai.AsyncOpenAI"
            ) as mock_openai_class,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep):
            mock_client = Mock()
            mock_client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)
            mock_openai_class.return_value = mock_client

            # Streaming should fail immediately without retry
            with pytest.raises(CustomRateLimitError) as exc_info:
                async for _ in service.stream(request):
                    pass

            assert exc_info.value.provider == "openai"
            assert exc_info.value.retry_after == 5.0
            # No retries in streaming mode
            mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_metrics_integration(self) -> None:
        """Test rate limit metrics are recorded correctly in integration scenario."""
        from openai import RateLimitError as OpenAIRateLimitError

        service = LLMService(timeout=60.0, max_retries=2)
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            trace_id="integration-test-005")

        mock_response = Mock()
        mock_response.headers = {}
        rate_limit_error = OpenAIRateLimitError(
            "Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}})

        with (
            patch(
                "agentcore.a2a_protocol.services.llm_client_openai.AsyncOpenAI"
            ) as mock_openai_class,
            patch("asyncio.sleep", new_callable=AsyncMock),
            patch(
                "agentcore.a2a_protocol.services.llm_client_openai.record_rate_limit_error"
            ) as mock_record_error,
            patch(
                "agentcore.a2a_protocol.services.llm_client_openai.record_rate_limit_retry_delay"
            ) as mock_record_delay,
            patch(
                "agentcore.a2a_protocol.services.llm_service.record_llm_request"
            ) as mock_record_request,
            patch(
                "agentcore.a2a_protocol.services.llm_service.record_llm_error"
            ) as mock_record_llm_error):
            mock_client = Mock()
            mock_client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)
            mock_openai_class.return_value = mock_client

            with pytest.raises(CustomRateLimitError):
                await service.complete(request)

            # Verify rate limit specific metrics
            # max_retries=2 means 2 attempts (0,1)
            assert mock_record_error.call_count == 2  # 2 error recordings
            assert mock_record_delay.call_count == 2  # 2 delay recordings (metric recorded before sleep check)

            # Verify general error metrics
            mock_record_request.assert_called_with("openai", "gpt-4.1-mini", "error")
            mock_record_llm_error.assert_called_with(
                "openai", "gpt-4.1-mini", "RateLimitError"
            )
