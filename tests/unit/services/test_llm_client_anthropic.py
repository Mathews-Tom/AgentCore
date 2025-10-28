"""Unit tests for Anthropic LLM client implementation.

This module tests the LLMClientAnthropic class with mocked Anthropic SDK to verify:
- Complete and streaming completion methods
- Message format conversion (OpenAI → Anthropic)
- System message extraction
- Response normalization
- Error handling (API errors, timeouts, authentication)
- Retry logic with exponential backoff
- A2A context propagation via extra_headers
- Token usage extraction

Target coverage: 90%+
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    RateLimitError)

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError as CustomRateLimitError)
from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic


@pytest.fixture
def mock_anthropic_client() -> Mock:
    """Create mock AsyncAnthropic client."""
    return Mock()


@pytest.fixture
def llm_client(mock_anthropic_client: Mock) -> LLMClientAnthropic:
    """Create LLMClientAnthropic with mocked Anthropic client."""
    with patch("agentcore.a2a_protocol.services.llm_client_anthropic.AsyncAnthropic") as mock_class:
        mock_class.return_value = mock_anthropic_client
        client = LLMClientAnthropic(api_key="test-key", timeout=60.0, max_retries=3)
        client.client = mock_anthropic_client
        return client


@pytest.fixture
def sample_request() -> LLMRequest:
    """Create sample LLM request."""
    return LLMRequest(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Hello"}], trace_id="trace-123",
        source_agent="agent-1",
        session_id="session-456")


@pytest.fixture
def sample_request_with_system() -> LLMRequest:
    """Create sample LLM request with system message."""
    return LLMRequest(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ])


@pytest.fixture
def mock_anthropic_response() -> Mock:
    """Create mock Anthropic response."""
    response = Mock()
    response.content = [
        Mock(text="Hello! How can I help you?"),
    ]
    response.usage = Mock(
        input_tokens=10,
        output_tokens=8)
    return response


class TestLLMClientAnthropicInit:
    """Test LLMClientAnthropic initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        with patch("agentcore.a2a_protocol.services.llm_client_anthropic.AsyncAnthropic") as mock_class:
            client = LLMClientAnthropic(api_key="test-key")
            assert client.timeout == 60.0
            assert client.max_retries == 3
            mock_class.assert_called_once_with(api_key="test-key", timeout=60.0)

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        with patch("agentcore.a2a_protocol.services.llm_client_anthropic.AsyncAnthropic") as mock_class:
            client = LLMClientAnthropic(api_key="test-key", timeout=30.0, max_retries=5)
            assert client.timeout == 30.0
            assert client.max_retries == 5
            mock_class.assert_called_once_with(api_key="test-key", timeout=30.0)


class TestConvertMessages:
    """Test _convert_messages() method."""

    def test_convert_messages_with_system(self, llm_client: LLMClientAnthropic) -> None:
        """Test message conversion with system message."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        system, converted = llm_client._convert_messages(messages)

        assert system == "You are helpful"
        assert converted == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

    def test_convert_messages_without_system(self, llm_client: LLMClientAnthropic) -> None:
        """Test message conversion without system message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        system, converted = llm_client._convert_messages(messages)

        assert system is None
        assert converted == messages

    def test_convert_messages_multiple_system(self, llm_client: LLMClientAnthropic) -> None:
        """Test message conversion with multiple system messages (takes last)."""
        messages = [
            {"role": "system", "content": "First system"},
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Second system"},
        ]

        system, converted = llm_client._convert_messages(messages)

        assert system == "Second system"
        assert converted == [{"role": "user", "content": "Hello"}]

    def test_convert_messages_empty_list(self, llm_client: LLMClientAnthropic) -> None:
        """Test message conversion with empty list."""
        messages: list[dict[str, str]] = []

        system, converted = llm_client._convert_messages(messages)

        assert system is None
        assert converted == []


class TestLLMClientAnthropicComplete:
    """Test complete() method."""

    @pytest.mark.asyncio
    async def test_complete_success(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest,
        mock_anthropic_response: Mock) -> None:
        """Test successful completion request."""
        # Setup mock
        llm_client.client.messages.create = AsyncMock(return_value=mock_anthropic_response)

        # Execute
        response = await llm_client.complete(sample_request)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.provider == "anthropic"
        assert response.model == "claude-3-5-haiku-20241022"
        assert response.trace_id == "trace-123"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8
        assert response.usage.total_tokens == 18
        assert response.latency_ms >= 0

        # Verify API call (no temperature per CLAUDE.md, max_tokens=4096 is Anthropic requirement)
        llm_client.client.messages.create.assert_called_once()
        call_kwargs = llm_client.client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-haiku-20241022"
        assert call_kwargs["max_tokens"] == 4096  # Anthropic API requirement
        assert "system" not in call_kwargs  # No system message in this request (param omitted)
        assert call_kwargs["extra_headers"]["X-Trace-ID"] == "trace-123"
        assert call_kwargs["extra_headers"]["X-Source-Agent"] == "agent-1"
        assert call_kwargs["extra_headers"]["X-Session-ID"] == "session-456"

    @pytest.mark.asyncio
    async def test_complete_with_system_message(
        self,
        llm_client: LLMClientAnthropic,
        sample_request_with_system: LLMRequest,
        mock_anthropic_response: Mock) -> None:
        """Test completion with system message."""
        llm_client.client.messages.create = AsyncMock(return_value=mock_anthropic_response)

        response = await llm_client.complete(sample_request_with_system)

        assert isinstance(response, LLMResponse)

        # Verify system message was extracted and formatted as list
        call_kwargs = llm_client.client.messages.create.call_args[1]
        assert call_kwargs["system"] == [{"type": "text", "text": "You are a helpful assistant"}]
        # Verify messages don't contain system role
        messages = call_kwargs["messages"]
        assert all(msg["role"] != "system" for msg in messages)

    @pytest.mark.asyncio
    async def test_complete_without_a2a_context(
        self,
        llm_client: LLMClientAnthropic,
        mock_anthropic_response: Mock) -> None:
        """Test completion without A2A context."""
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello"}])
        llm_client.client.messages.create = AsyncMock(return_value=mock_anthropic_response)

        response = await llm_client.complete(request)

        assert response.trace_id is None
        call_kwargs = llm_client.client.messages.create.call_args[1]
        assert "extra_headers" not in call_kwargs  # No A2A context (param omitted)

    @pytest.mark.asyncio
    async def test_complete_without_max_tokens(
        self,
        llm_client: LLMClientAnthropic,
        mock_anthropic_response: Mock) -> None:
        """Test completion defaults max_tokens to 4096 when not provided."""
        request = LLMRequest(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Hello"}])
        llm_client.client.messages.create = AsyncMock(return_value=mock_anthropic_response)

        await llm_client.complete(request)

        call_kwargs = llm_client.client.messages.create.call_args[1]
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_complete_timeout_error(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test timeout error handling (no retry)."""
        llm_client.client.messages.create = AsyncMock(
            side_effect=APITimeoutError(request=Mock())
        )

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await llm_client.complete(sample_request)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.timeout_seconds == 60.0
        # Should not retry timeout errors
        assert llm_client.client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_authentication_error(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test authentication error handling (no retry)."""
        llm_client.client.messages.create = AsyncMock(
            side_effect=AuthenticationError(
                message="Invalid API key",
                response=Mock(),
                body=None)
        )

        with pytest.raises(ProviderError) as exc_info:
            await llm_client.complete(sample_request)

        assert exc_info.value.provider == "anthropic"
        # Should not retry authentication errors
        assert llm_client.client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_bad_request_error(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test bad request error handling (no retry)."""
        llm_client.client.messages.create = AsyncMock(
            side_effect=BadRequestError(
                message="Invalid parameters",
                response=Mock(),
                body=None)
        )

        with pytest.raises(ProviderError) as exc_info:
            await llm_client.complete(sample_request)

        assert exc_info.value.provider == "anthropic"
        # Should not retry bad request errors
        assert llm_client.client.messages.create.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_retry_on_rate_limit(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest,
        mock_anthropic_response: Mock) -> None:
        """Test retry logic on rate limit error."""
        # Fail twice with rate limit, then succeed
        llm_client.client.messages.create = AsyncMock(
            side_effect=[
                RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(),
                    body=None),
                RateLimitError(
                    message="Rate limit exceeded",
                    response=Mock(),
                    body=None),
                mock_anthropic_response,
            ]
        )

        # Mock sleep to avoid actual delays in tests
        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await llm_client.complete(sample_request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        # Should retry 3 times total
        assert llm_client.client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_max_retries_exceeded(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test max retries exceeded."""
        llm_client.client.messages.create = AsyncMock(
            side_effect=RateLimitError(
                message="Rate limit exceeded",
                response=Mock(),
                body=None)
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(CustomRateLimitError) as exc_info:
                await llm_client.complete(sample_request)

        assert exc_info.value.provider == "anthropic"
        # Should try 3 times (max_retries=3)
        assert llm_client.client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_retry_on_connection_error(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest,
        mock_anthropic_response: Mock) -> None:
        """Test retry logic on connection error."""
        llm_client.client.messages.create = AsyncMock(
            side_effect=[
                APIConnectionError(request=Mock()),
                mock_anthropic_response,
            ]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            response = await llm_client.complete(sample_request)

        assert isinstance(response, LLMResponse)
        assert llm_client.client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_complete_exponential_backoff(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test exponential backoff timing."""
        llm_client.client.messages.create = AsyncMock(
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


class TestLLMClientAnthropicStream:
    """Test stream() method."""

    @pytest.mark.asyncio
    async def test_stream_success(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test successful streaming completion."""
        # Mock Anthropic streaming events
        events = [
            Mock(type="content_block_delta", delta=Mock(text="Hello")),
            Mock(type="content_block_delta", delta=Mock(text=" there")),
            Mock(type="content_block_delta", delta=Mock(text="!")),
        ]

        async def mock_stream_iter() -> AsyncMock:
            """Create async iterator for events."""
            for event in events:
                yield event

        mock_stream = mock_stream_iter()
        llm_client.client.messages.create = AsyncMock(return_value=mock_stream)

        # Collect tokens
        tokens: list[str] = []
        async for token in llm_client.stream(sample_request):
            tokens.append(token)

        assert tokens == ["Hello", " there", "!"]
        assert "".join(tokens) == "Hello there!"

        # Verify API call
        llm_client.client.messages.create.assert_called_once()
        call_kwargs = llm_client.client.messages.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert call_kwargs["extra_headers"]["X-Trace-ID"] == "trace-123"

    @pytest.mark.asyncio
    async def test_stream_with_system_message(
        self,
        llm_client: LLMClientAnthropic,
        sample_request_with_system: LLMRequest) -> None:
        """Test streaming with system message."""
        events = [
            Mock(type="content_block_delta", delta=Mock(text="Hello")),
        ]

        async def mock_stream_iter() -> AsyncMock:
            """Create async iterator for events."""
            for event in events:
                yield event

        mock_stream = mock_stream_iter()
        llm_client.client.messages.create = AsyncMock(return_value=mock_stream)

        tokens: list[str] = []
        async for token in llm_client.stream(sample_request_with_system):
            tokens.append(token)

        # Verify system message was extracted and formatted as list
        call_kwargs = llm_client.client.messages.create.call_args[1]
        assert call_kwargs["system"] == [{"type": "text", "text": "You are a helpful assistant"}]

    @pytest.mark.asyncio
    async def test_stream_with_non_delta_events(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test streaming filters non-content_block_delta events."""
        events = [
            Mock(type="message_start"),  # Should be filtered
            Mock(type="content_block_delta", delta=Mock(text="Hello")),
            Mock(type="content_block_stop"),  # Should be filtered
            Mock(type="content_block_delta", delta=Mock(text="!")),
        ]

        async def mock_stream_iter() -> AsyncMock:
            """Create async iterator for events."""
            for event in events:
                yield event

        mock_stream = mock_stream_iter()
        llm_client.client.messages.create = AsyncMock(return_value=mock_stream)

        tokens: list[str] = []
        async for token in llm_client.stream(sample_request):
            tokens.append(token)

        # Only content_block_delta events should be yielded
        assert tokens == ["Hello", "!"]

    @pytest.mark.asyncio
    async def test_stream_timeout_error(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test streaming timeout error."""
        llm_client.client.messages.create = AsyncMock(
            side_effect=APITimeoutError(request=Mock())
        )

        with pytest.raises(ProviderTimeoutError) as exc_info:
            async for _ in llm_client.stream(sample_request):
                pass

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.timeout_seconds == 60.0

    @pytest.mark.asyncio
    async def test_stream_authentication_error(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test streaming authentication error."""
        llm_client.client.messages.create = AsyncMock(
            side_effect=AuthenticationError(
                message="Invalid API key",
                response=Mock(),
                body=None)
        )

        with pytest.raises(ProviderError) as exc_info:
            async for _ in llm_client.stream(sample_request):
                pass

        assert exc_info.value.provider == "anthropic"


class TestLLMClientAnthropicNormalizeResponse:
    """Test _normalize_response() method."""

    def test_normalize_response_success(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest,
        mock_anthropic_response: Mock) -> None:
        """Test successful response normalization."""
        normalized = llm_client._normalize_response(
            (mock_anthropic_response, 1500),
            sample_request)

        assert isinstance(normalized, LLMResponse)
        assert normalized.content == "Hello! How can I help you?"
        assert normalized.usage.prompt_tokens == 10
        assert normalized.usage.completion_tokens == 8
        assert normalized.usage.total_tokens == 18
        assert normalized.latency_ms == 1500
        assert normalized.provider == "anthropic"
        assert normalized.model == "claude-3-5-haiku-20241022"
        assert normalized.trace_id == "trace-123"

    def test_normalize_response_empty_content(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test normalization with empty content."""
        response = Mock()
        response.content = [Mock(text="")]
        response.usage = Mock(
            input_tokens=10,
            output_tokens=0)

        normalized = llm_client._normalize_response((response, 100), sample_request)

        assert normalized.content == ""
        assert normalized.usage.completion_tokens == 0

    def test_normalize_response_invalid_format(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test normalization with invalid response format."""
        with pytest.raises(ValueError, match="Invalid response format: expected tuple"):
            llm_client._normalize_response("invalid", sample_request)

    def test_normalize_response_missing_content(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test normalization with missing content field."""
        response = Mock(spec=[])  # No attributes
        with pytest.raises(ValueError, match="missing 'content' field"):
            llm_client._normalize_response((response, 100), sample_request)

    def test_normalize_response_empty_content_list(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test normalization with empty content list."""
        response = Mock()
        response.content = []
        response.usage = Mock()

        with pytest.raises(ValueError, match="empty 'content' list"):
            llm_client._normalize_response((response, 100), sample_request)

    def test_normalize_response_missing_usage(
        self,
        llm_client: LLMClientAnthropic,
        sample_request: LLMRequest) -> None:
        """Test normalization with missing usage field."""
        response = Mock()
        response.content = [Mock(text="test")]
        del response.usage  # Remove usage attribute

        with pytest.raises(ValueError, match="missing 'usage' field"):
            llm_client._normalize_response((response, 100), sample_request)
