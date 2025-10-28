"""Unit tests for OpenAI LLM client implementation.

This module tests the LLMClientOpenAI class with mocked OpenAI SDK to verify:
- Complete and streaming completion methods
- Response normalization
- Error handling (API errors, timeouts, authentication)
- Retry logic with exponential backoff
- A2A context propagation via extra_headers
- Token usage extraction

Target coverage: 90%+
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError)

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError as CustomRateLimitError)
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI


@pytest.fixture
def mock_openai_client() -> Mock:
    """Create mock AsyncOpenAI client."""
    return Mock()


@pytest.fixture
def llm_client(mock_openai_client: Mock) -> LLMClientOpenAI:
    """Create LLMClientOpenAI with mocked OpenAI client."""
    with patch("agentcore.a2a_protocol.services.llm_client_openai.AsyncOpenAI") as mock_class:
        mock_class.return_value = mock_openai_client
        client = LLMClientOpenAI(api_key="test-key", timeout=60.0, max_retries=3)
        client.client = mock_openai_client
        return client


@pytest.fixture
def sample_request() -> LLMRequest:
    """Create sample LLM request."""
    return LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello"}], trace_id="trace-123",
        source_agent="agent-1",
        session_id="session-456")


@pytest.fixture
def mock_openai_response() -> Mock:
    """Create mock OpenAI response."""
    response = Mock()
    response.choices = [
        Mock(
            message=Mock(content="Hello! How can I help you?"))
    ]
    response.usage = Mock(
        prompt_tokens=10,
        completion_tokens=8,
        total_tokens=18)
    return response


class TestLLMClientOpenAIInit:
    """Test LLMClientOpenAI initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        with patch("agentcore.a2a_protocol.services.llm_client_openai.AsyncOpenAI") as mock_class:
            client = LLMClientOpenAI(api_key="test-key")
            assert client.timeout == 60.0
            assert client.max_retries == 3
            mock_class.assert_called_once_with(api_key="test-key", timeout=60.0)

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        with patch("agentcore.a2a_protocol.services.llm_client_openai.AsyncOpenAI") as mock_class:
            client = LLMClientOpenAI(api_key="test-key", timeout=30.0, max_retries=5)
            assert client.timeout == 30.0
            assert client.max_retries == 5
            mock_class.assert_called_once_with(api_key="test-key", timeout=30.0)


class TestLLMClientOpenAIComplete:
    """Test complete() method."""

    @pytest.mark.asyncio
    async def test_complete_success(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest,
        mock_openai_response: Mock) -> None:
        """Test successful completion request."""
        # Setup mock
        llm_client.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        # Execute
        response = await llm_client.complete(sample_request)

        # Verify
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.provider == "openai"
        assert response.model == "gpt-4.1-mini"
        assert response.trace_id == "trace-123"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8
        assert response.usage.total_tokens == 18
        assert response.latency_ms >= 0

        # Verify API call (no temperature/max_tokens per CLAUDE.md)
        llm_client.client.chat.completions.create.assert_called_once()
        call_kwargs = llm_client.client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4.1-mini"
        assert call_kwargs["extra_headers"]["X-Trace-ID"] == "trace-123"
        assert call_kwargs["extra_headers"]["X-Source-Agent"] == "agent-1"
        assert call_kwargs["extra_headers"]["X-Session-ID"] == "session-456"

    @pytest.mark.asyncio
    async def test_complete_without_a2a_context(
        self,
        llm_client: LLMClientOpenAI,
        mock_openai_response: Mock) -> None:
        """Test completion without A2A context."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello"}])
        llm_client.client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

        response = await llm_client.complete(request)

        assert response.trace_id is None
        call_kwargs = llm_client.client.chat.completions.create.call_args[1]
        assert "extra_headers" not in call_kwargs or call_kwargs["extra_headers"] is None

    @pytest.mark.asyncio
    async def test_complete_timeout_error(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test timeout error handling (no retry)."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=Mock())
        )

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await llm_client.complete(sample_request)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.timeout_seconds == 60.0
        # Should not retry timeout errors
        assert llm_client.client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_authentication_error(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test authentication error handling (no retry)."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError(
                message="Invalid API key",
                response=Mock(),
                body=None)
        )

        with pytest.raises(ProviderError) as exc_info:
            await llm_client.complete(sample_request)

        assert exc_info.value.provider == "openai"
        # Should not retry authentication errors
        assert llm_client.client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_bad_request_error(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test bad request error handling (no retry)."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=BadRequestError(
                message="Invalid parameters",
                response=Mock(),
                body=None)
        )

        with pytest.raises(ProviderError) as exc_info:
            await llm_client.complete(sample_request)

        assert exc_info.value.provider == "openai"
        # Should not retry bad request errors
        assert llm_client.client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_retry_on_rate_limit(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest,
        mock_openai_response: Mock) -> None:
        """Test retry logic on rate limit error."""
        # Fail twice with rate limit, then succeed
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=[
                RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(),
                    body=None),
                RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(),
                    body=None),
                mock_openai_response,
            ]
        )

        # Mock sleep to avoid actual delays in tests
        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await llm_client.complete(sample_request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        # Should retry 3 times total
        assert llm_client.client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_max_retries_exceeded(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test max retries exceeded."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                message="Rate limit exceeded",
                response=Mock(),
                body=None)
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(CustomRateLimitError) as exc_info:
                await llm_client.complete(sample_request)

        assert exc_info.value.provider == "openai"
        # Should try 3 times (max_retries=3)
        assert llm_client.client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_retry_on_connection_error(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest,
        mock_openai_response: Mock) -> None:
        """Test retry logic on connection error."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=[
                APIConnectionError(request=Mock()),
                mock_openai_response,
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await llm_client.complete(sample_request)

        assert isinstance(response, LLMResponse)
        assert llm_client.client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_complete_exponential_backoff(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test exponential backoff timing."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                message="Rate limit exceeded",
                response=Mock(),
                body=None)
        )

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(CustomRateLimitError):
                await llm_client.complete(sample_request)

            # Verify exponential backoff: 1s, 2s (only 2 sleeps for 3 attempts)
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(1)  # 2^0
            mock_sleep.assert_any_call(2)  # 2^1


class TestLLMClientOpenAIStream:
    """Test stream() method."""

    @pytest.mark.asyncio
    async def test_stream_success(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test successful streaming completion."""
        # Mock streaming response
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=" there"))]),
            Mock(choices=[Mock(delta=Mock(content="!"))]),
        ]

        async def mock_stream_iter() -> AsyncMock:
            """Create async iterator for chunks."""
            for chunk in chunks:
                yield chunk

        mock_stream = mock_stream_iter()
        llm_client.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        # Collect tokens
        tokens: list[str] = []
        async for token in llm_client.stream(sample_request):
            tokens.append(token)

        assert tokens == ["Hello", " there", "!"]
        assert "".join(tokens) == "Hello there!"

        # Verify API call
        llm_client.client.chat.completions.create.assert_called_once()
        call_kwargs = llm_client.client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["extra_headers"]["X-Trace-ID"] == "trace-123"

    @pytest.mark.asyncio
    async def test_stream_with_empty_chunks(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test streaming with empty content chunks (should be filtered)."""
        chunks = [
            Mock(choices=[Mock(delta=Mock(content="Hello"))]),
            Mock(choices=[Mock(delta=Mock(content=None))]),  # Empty chunk
            Mock(choices=[Mock(delta=Mock(content="!"))]),
        ]

        async def mock_stream_iter() -> AsyncMock:
            """Create async iterator for chunks."""
            for chunk in chunks:
                yield chunk

        mock_stream = mock_stream_iter()
        llm_client.client.chat.completions.create = AsyncMock(return_value=mock_stream)

        tokens: list[str] = []
        async for token in llm_client.stream(sample_request):
            tokens.append(token)

        # Empty chunk should be filtered out
        assert tokens == ["Hello", "!"]

    @pytest.mark.asyncio
    async def test_stream_timeout_error(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test streaming timeout error."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=Mock())
        )

        with pytest.raises(ProviderTimeoutError) as exc_info:
            async for _ in llm_client.stream(sample_request):
                pass

        assert exc_info.value.provider == "openai"
        assert exc_info.value.timeout_seconds == 60.0

    @pytest.mark.asyncio
    async def test_stream_authentication_error(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test streaming authentication error."""
        llm_client.client.chat.completions.create = AsyncMock(
            side_effect=AuthenticationError(
                message="Invalid API key",
                response=Mock(),
                body=None)
        )

        with pytest.raises(ProviderError) as exc_info:
            async for _ in llm_client.stream(sample_request):
                pass

        assert exc_info.value.provider == "openai"


class TestLLMClientOpenAINormalizeResponse:
    """Test _normalize_response() method."""

    def test_normalize_response_success(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest,
        mock_openai_response: Mock) -> None:
        """Test successful response normalization."""
        normalized = llm_client._normalize_response(
            (mock_openai_response, 1500),
            sample_request)

        assert isinstance(normalized, LLMResponse)
        assert normalized.content == "Hello! How can I help you?"
        assert normalized.usage.prompt_tokens == 10
        assert normalized.usage.completion_tokens == 8
        assert normalized.usage.total_tokens == 18
        assert normalized.latency_ms == 1500
        assert normalized.provider == "openai"
        assert normalized.model == "gpt-4.1-mini"
        assert normalized.trace_id == "trace-123"

    def test_normalize_response_none_content(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test normalization with None content (should default to empty string)."""
        response = Mock()
        response.choices = [Mock(message=Mock(content=None, annotations=None))]
        response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10)

        normalized = llm_client._normalize_response((response, 100), sample_request)

        assert normalized.content == ""
        assert normalized.usage.completion_tokens == 0

    def test_normalize_response_invalid_format(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test normalization with invalid response format."""
        with pytest.raises(ValueError, match="Invalid response format: expected tuple"):
            llm_client._normalize_response("invalid", sample_request)

    def test_normalize_response_missing_choices(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test normalization with missing choices field."""
        response = Mock(spec=[])  # No attributes
        with pytest.raises(ValueError, match="missing 'choices' field"):
            llm_client._normalize_response((response, 100), sample_request)

    def test_normalize_response_empty_choices(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test normalization with empty choices list."""
        response = Mock()
        response.choices = []
        response.usage = Mock()

        with pytest.raises(ValueError, match="empty 'choices' list"):
            llm_client._normalize_response((response, 100), sample_request)

    def test_normalize_response_missing_usage(
        self,
        llm_client: LLMClientOpenAI,
        sample_request: LLMRequest) -> None:
        """Test normalization with missing usage field."""
        response = Mock()
        response.choices = [Mock(message=Mock(content="test"))]
        del response.usage  # Remove usage attribute

        with pytest.raises(ValueError, match="missing 'usage' field"):
            llm_client._normalize_response((response, 100), sample_request)
