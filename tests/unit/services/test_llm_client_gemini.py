"""Unit tests for Google Gemini LLM client implementation.

This module tests the LLMClientGemini class with mocked Gemini SDK to verify:
- Complete and streaming completion methods
- Message format conversion (OpenAI → Gemini)
- Response normalization
- Error handling (API errors, timeouts, authentication)
- Retry logic with exponential backoff
- Token usage extraction

Target coverage: 90%+
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from google.api_core import exceptions as google_exceptions

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderError,
    ProviderTimeoutError,
)
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini


@pytest.fixture
def llm_client() -> LLMClientGemini:
    """Create LLMClientGemini with test API key."""
    with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.configure"):
        return LLMClientGemini(api_key="test-key", timeout=60.0, max_retries=3)


@pytest.fixture
def sample_request() -> LLMRequest:
    """Create sample LLM request."""
    return LLMRequest(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
        trace_id="trace-123",
        source_agent="agent-1",
        session_id="session-456",
    )


@pytest.fixture
def mock_gemini_response() -> Mock:
    """Create mock Gemini response."""
    response = Mock()
    # Gemini response structure: candidates[0].content.parts[0].text
    candidate = Mock()
    content = Mock()
    part = Mock()
    part.text = "Hello! How can I help you?"
    content.parts = [part]
    candidate.content = content
    response.candidates = [candidate]

    # Gemini usage structure: usage_metadata
    usage_metadata = Mock()
    usage_metadata.prompt_token_count = 10
    usage_metadata.candidates_token_count = 8
    usage_metadata.total_token_count = 18
    response.usage_metadata = usage_metadata

    return response


class TestLLMClientGeminiInit:
    """Test LLMClientGemini initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.configure") as mock_configure:
            client = LLMClientGemini(api_key="test-key")
            assert client.api_key == "test-key"
            assert client.timeout == 60.0
            assert client.max_retries == 3
            mock_configure.assert_called_once_with(api_key="test-key")

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.configure") as mock_configure:
            client = LLMClientGemini(api_key="test-key", timeout=30.0, max_retries=5)
            assert client.timeout == 30.0
            assert client.max_retries == 5
            mock_configure.assert_called_once_with(api_key="test-key")


class TestLLMClientGeminiMessageConversion:
    """Test message format conversion."""

    def test_convert_simple_user_message(self, llm_client: LLMClientGemini) -> None:
        """Test converting simple user message."""
        messages = [{"role": "user", "content": "Hello"}]
        system, gemini_msgs = llm_client._convert_messages(messages)

        assert system is None
        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "user"
        assert gemini_msgs[0]["parts"] == [{"text": "Hello"}]

    def test_convert_with_system_message(self, llm_client: LLMClientGemini) -> None:
        """Test extracting system message."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system, gemini_msgs = llm_client._convert_messages(messages)

        assert system == "You are helpful"
        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "user"

    def test_convert_assistant_to_model(self, llm_client: LLMClientGemini) -> None:
        """Test converting assistant role to model."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        system, gemini_msgs = llm_client._convert_messages(messages)

        assert system is None
        assert len(gemini_msgs) == 2
        assert gemini_msgs[0]["role"] == "user"
        assert gemini_msgs[1]["role"] == "model"
        assert gemini_msgs[1]["parts"] == [{"text": "Hi there!"}]

    def test_convert_multi_turn_conversation(self, llm_client: LLMClientGemini) -> None:
        """Test converting multi-turn conversation."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]
        system, gemini_msgs = llm_client._convert_messages(messages)

        assert system == "You are helpful"
        assert len(gemini_msgs) == 3
        assert gemini_msgs[0]["role"] == "user"
        assert gemini_msgs[1]["role"] == "model"
        assert gemini_msgs[2]["role"] == "user"


class TestLLMClientGeminiComplete:
    """Test complete() method."""

    @pytest.mark.asyncio
    async def test_complete_success(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
        mock_gemini_response: Mock,
    ) -> None:
        """Test successful completion request."""
        # Setup mock
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_gemini_response)

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                response = await llm_client.complete(sample_request)

        # Verify
        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.provider == "gemini"
        assert response.model == "gemini-1.5-flash"
        assert response.trace_id == "trace-123"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8
        assert response.usage.total_tokens == 18
        assert response.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_complete_without_a2a_context(
        self,
        llm_client: LLMClientGemini,
        mock_gemini_response: Mock,
    ) -> None:
        """Test completion without A2A context."""
        request = LLMRequest(
            model="gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )

        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_gemini_response)

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                response = await llm_client.complete(request)

        assert response.trace_id is None

    @pytest.mark.asyncio
    async def test_complete_timeout_error(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test timeout error handling (no retry)."""
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                with pytest.raises(ProviderTimeoutError) as exc_info:
                    await llm_client.complete(sample_request)

        assert exc_info.value.provider == "gemini"
        assert exc_info.value.timeout_seconds == 60.0
        # Should not retry timeout errors
        assert mock_model.generate_content_async.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_authentication_error(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test authentication error handling (no retry)."""
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=google_exceptions.Unauthenticated("Invalid API key")
        )

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                with pytest.raises(ProviderError) as exc_info:
                    await llm_client.complete(sample_request)

        assert exc_info.value.provider == "gemini"
        # Should not retry authentication errors
        assert mock_model.generate_content_async.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_invalid_argument_error(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test invalid argument error handling (no retry)."""
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=google_exceptions.InvalidArgument("Invalid parameters")
        )

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                with pytest.raises(ProviderError) as exc_info:
                    await llm_client.complete(sample_request)

        assert exc_info.value.provider == "gemini"
        # Should not retry invalid argument errors
        assert mock_model.generate_content_async.call_count == 1

    @pytest.mark.asyncio
    async def test_complete_retry_on_resource_exhausted(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
        mock_gemini_response: Mock,
    ) -> None:
        """Test retry logic on resource exhausted error (rate limit)."""
        mock_model = Mock()
        # Fail twice with rate limit, then succeed
        mock_model.generate_content_async = AsyncMock(
            side_effect=[
                google_exceptions.ResourceExhausted("Rate limit exceeded"),
                google_exceptions.ResourceExhausted("Rate limit exceeded"),
                mock_gemini_response,
            ]
        )

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                # Mock sleep to avoid actual delays in tests
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    response = await llm_client.complete(sample_request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        # Should retry 3 times total
        assert mock_model.generate_content_async.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_max_retries_exceeded(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test max retries exceeded."""
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=google_exceptions.ResourceExhausted("Rate limit exceeded")
        )

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    with pytest.raises(ProviderError) as exc_info:
                        await llm_client.complete(sample_request)

        assert exc_info.value.provider == "gemini"
        # Should try 3 times (max_retries=3)
        assert mock_model.generate_content_async.call_count == 3

    @pytest.mark.asyncio
    async def test_complete_retry_on_service_unavailable(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
        mock_gemini_response: Mock,
    ) -> None:
        """Test retry logic on service unavailable error."""
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=[
                google_exceptions.ServiceUnavailable("Service unavailable"),
                mock_gemini_response,
            ]
        )

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    response = await llm_client.complete(sample_request)

        assert isinstance(response, LLMResponse)
        assert mock_model.generate_content_async.call_count == 2


class TestLLMClientGeminiStream:
    """Test stream() method."""

    @pytest.mark.asyncio
    async def test_stream_success(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test successful streaming request."""
        # Create mock streaming chunks
        chunk1 = Mock()
        chunk1_candidate = Mock()
        chunk1_content = Mock()
        chunk1_part = Mock()
        chunk1_part.text = "Hello"
        chunk1_content.parts = [chunk1_part]
        chunk1_candidate.content = chunk1_content
        chunk1.candidates = [chunk1_candidate]

        chunk2 = Mock()
        chunk2_candidate = Mock()
        chunk2_content = Mock()
        chunk2_part = Mock()
        chunk2_part.text = " there!"
        chunk2_content.parts = [chunk2_part]
        chunk2_candidate.content = chunk2_content
        chunk2.candidates = [chunk2_candidate]

        async def mock_stream() -> None:
            """Mock async stream generator."""
            yield chunk1
            yield chunk2

        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_stream())

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                tokens = []
                async for token in llm_client.stream(sample_request):
                    tokens.append(token)

        assert tokens == ["Hello", " there!"]

    @pytest.mark.asyncio
    async def test_stream_timeout_error(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test streaming timeout error."""
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                with pytest.raises(ProviderTimeoutError) as exc_info:
                    async for _ in llm_client.stream(sample_request):
                        pass

        assert exc_info.value.provider == "gemini"

    @pytest.mark.asyncio
    async def test_stream_authentication_error(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test streaming authentication error."""
        mock_model = Mock()
        mock_model.generate_content_async = AsyncMock(
            side_effect=google_exceptions.Unauthenticated("Invalid API key")
        )

        with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerativeModel", return_value=mock_model):
            with patch("agentcore.a2a_protocol.services.llm_client_gemini.genai.GenerationConfig"):
                with pytest.raises(ProviderError) as exc_info:
                    async for _ in llm_client.stream(sample_request):
                        pass

        assert exc_info.value.provider == "gemini"


class TestLLMClientGeminiNormalizeResponse:
    """Test _normalize_response() method."""

    def test_normalize_success(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
        mock_gemini_response: Mock,
    ) -> None:
        """Test successful response normalization."""
        response = llm_client._normalize_response((mock_gemini_response, 100), sample_request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.provider == "gemini"
        assert response.model == "gemini-1.5-flash"
        assert response.trace_id == "trace-123"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 8
        assert response.usage.total_tokens == 18
        assert response.latency_ms == 100

    def test_normalize_invalid_format(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test normalization with invalid response format."""
        with pytest.raises(ValueError, match="Invalid response format"):
            llm_client._normalize_response("not a tuple", sample_request)

    def test_normalize_missing_candidates(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test normalization with missing candidates field."""
        invalid_response = Mock(spec=[])
        with pytest.raises(ValueError, match="missing 'candidates' field"):
            llm_client._normalize_response((invalid_response, 100), sample_request)

    def test_normalize_empty_candidates(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test normalization with empty candidates list."""
        invalid_response = Mock()
        invalid_response.candidates = []
        with pytest.raises(ValueError, match="empty 'candidates' list"):
            llm_client._normalize_response((invalid_response, 100), sample_request)

    def test_normalize_without_usage_metadata(
        self,
        llm_client: LLMClientGemini,
        sample_request: LLMRequest,
    ) -> None:
        """Test normalization without usage metadata (should default to 0)."""
        response = Mock(spec=["candidates"])
        candidate = Mock()
        content = Mock()
        part = Mock()
        part.text = "Hello!"
        content.parts = [part]
        candidate.content = content
        response.candidates = [candidate]
        # No usage_metadata attribute

        normalized = llm_client._normalize_response((response, 100), sample_request)

        assert normalized.usage.prompt_tokens == 0
        assert normalized.usage.completion_tokens == 0
        assert normalized.usage.total_tokens == 0
