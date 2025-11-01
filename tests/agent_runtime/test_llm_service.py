"""Tests for Portkey LLM service."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from agentcore.agent_runtime.services.llm_service import (
    LLMConfig,
    LLMResponse,
    PortkeyLLMService,
    cleanup_llm_service,
    get_llm_service,
    initialize_llm_service,
)


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create test LLM configuration."""
    return LLMConfig(
        portkey_api_key="test-api-key",
        portkey_base_url="https://api.portkey.test",
        default_model="gpt-5-test",
        fallback_models=["gpt-5-mini-test"],
        default_temperature=0.7,
        default_max_tokens=500,
        timeout_seconds=30,
        max_retries=3,
        cache_enabled=True,
    )


@pytest.fixture
async def llm_service(llm_config: LLMConfig) -> PortkeyLLMService:
    """Create LLM service instance."""
    service = PortkeyLLMService(llm_config)
    yield service
    await service.close()


class TestLLMConfig:
    """Test suite for LLM configuration."""

    def test_config_defaults(self) -> None:
        """Test configuration defaults."""
        config = LLMConfig(portkey_api_key="test-key")

        assert config.portkey_api_key == "test-key"
        assert config.portkey_base_url == "https://api.portkey.ai"
        assert config.default_model == "gpt-5"
        assert config.fallback_models == ["gpt-5-mini"]
        assert config.timeout_seconds == 30
        assert config.max_retries == 3
        assert config.cache_enabled is True

    def test_config_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = LLMConfig(
            portkey_api_key="custom-key",
            portkey_base_url="https://custom.api",
            default_model="custom-model",
            fallback_models=["fallback1", "fallback2"],
            default_temperature=0.9,
            default_max_tokens=1000,
            timeout_seconds=60,
            max_retries=5,
            cache_enabled=False,
        )

        assert config.portkey_api_key == "custom-key"
        assert config.portkey_base_url == "https://custom.api"
        assert config.default_model == "custom-model"
        assert config.fallback_models == ["fallback1", "fallback2"]
        assert config.timeout_seconds == 60
        assert config.max_retries == 5
        assert config.cache_enabled is False


@pytest.mark.asyncio
class TestPortkeyLLMService:
    """Test suite for Portkey LLM service."""

    async def test_service_initialization(self, llm_config: LLMConfig) -> None:
        """Test service initialization."""
        service = PortkeyLLMService(llm_config)

        assert service.config == llm_config
        assert service.client is not None
        assert service.client.base_url == llm_config.portkey_base_url

        await service.close()

    @patch("httpx.AsyncClient.post")
    async def test_complete_success(
        self,
        mock_post: AsyncMock,
        llm_service: PortkeyLLMService,
    ) -> None:
        """Test successful completion request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Test response content"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-5-test",
            "usage": {"total_tokens": 150},
        }
        mock_response.headers = {"x-portkey-cache-status": "MISS"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Make request
        response = await llm_service.complete(
            prompt="Test prompt",
            system_prompt="Test system prompt",
        )

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "Test response content"
        assert response.model == "gpt-5-test"
        assert response.tokens_used == 150
        assert response.finish_reason == "stop"
        assert response.cached is False

        # Verify request was made with correct parameters
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "/v1/chat/completions"
        payload = call_args[1]["json"]
        assert payload["model"] == "gpt-5-test"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

    @patch("httpx.AsyncClient.post")
    async def test_complete_with_cache_hit(
        self,
        mock_post: AsyncMock,
        llm_service: PortkeyLLMService,
    ) -> None:
        """Test completion with cache hit."""
        # Mock cached response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Cached response"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gpt-5-test",
            "usage": {"total_tokens": 100},
        }
        mock_response.headers = {"x-portkey-cache-status": "HIT"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = await llm_service.complete(prompt="Test")

        assert response.cached is True
        assert response.content == "Cached response"

    @patch("httpx.AsyncClient.post")
    async def test_complete_with_custom_parameters(
        self,
        mock_post: AsyncMock,
        llm_service: PortkeyLLMService,
    ) -> None:
        """Test completion with custom parameters."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "model": "custom-model",
            "usage": {"total_tokens": 200},
        }
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        response = await llm_service.complete(
            prompt="Test",
            model="custom-model",
            temperature=0.8,
            max_tokens=150,
        )

        # Verify custom parameters were used
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "custom-model"

    @patch("httpx.AsyncClient.post")
    async def test_complete_http_error(
        self,
        mock_post: AsyncMock,
        llm_service: PortkeyLLMService,
    ) -> None:
        """Test completion with HTTP error."""
        mock_post.side_effect = httpx.HTTPError("API error")

        with pytest.raises(httpx.HTTPError):
            await llm_service.complete(prompt="Test")

    @patch("httpx.AsyncClient.post")
    async def test_complete_portkey_headers(
        self,
        mock_post: AsyncMock,
        llm_service: PortkeyLLMService,
    ) -> None:
        """Test that Portkey-specific headers are set."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}, "finish_reason": "stop"}],
            "model": "gpt-5-test",
            "usage": {"total_tokens": 100},
        }
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        await llm_service.complete(prompt="Test")

        # Verify Portkey headers were set
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert "x-portkey-fallback" in headers
        assert "x-portkey-retry" in headers
        assert "x-portkey-cache" in headers

    @patch("httpx.AsyncClient.stream")
    async def test_stream_complete_success(
        self,
        mock_stream: AsyncMock,
        llm_service: PortkeyLLMService,
    ) -> None:
        """Test streaming completion."""
        # Mock streaming response
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()

        async def mock_aiter_lines():
            yield "data: " + '{"choices":[{"delta":{"content":"Hello"}}]}'
            yield "data: " + '{"choices":[{"delta":{"content":" world"}}]}'
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines
        mock_stream.return_value.__aenter__.return_value = mock_response

        # Collect streamed content
        chunks = []
        async for chunk in llm_service.stream_complete(prompt="Test"):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @patch("httpx.AsyncClient.stream")
    async def test_stream_complete_http_error(
        self,
        mock_stream: AsyncMock,
        llm_service: PortkeyLLMService,
    ) -> None:
        """Test streaming with HTTP error."""
        # Mock response that raises HTTP error on raise_for_status
        mock_response = Mock()
        mock_response.raise_for_status = Mock(side_effect=httpx.HTTPError("API error"))

        # Setup async context manager
        mock_stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(httpx.HTTPError):
            async for _ in llm_service.stream_complete(prompt="Test"):
                pass

    async def test_service_close(self, llm_config: LLMConfig) -> None:
        """Test service cleanup."""
        service = PortkeyLLMService(llm_config)
        assert service.client is not None

        await service.close()
        # Client should be closed (no exception raised)


@pytest.mark.asyncio
class TestLLMServiceSingleton:
    """Test suite for LLM service singleton management."""

    async def test_initialize_and_get_service(self, llm_config: LLMConfig) -> None:
        """Test service initialization and retrieval."""
        # Clean up any existing service
        await cleanup_llm_service()

        # Initialize service
        service = initialize_llm_service(llm_config)
        assert service is not None
        assert service.config == llm_config

        # Get same instance
        same_service = get_llm_service()
        assert same_service is service

        # Clean up
        await cleanup_llm_service()

    async def test_get_service_without_initialization(self) -> None:
        """Test getting service without initialization."""
        # Clean up any existing service
        await cleanup_llm_service()

        # Should raise error
        with pytest.raises(RuntimeError, match="LLM service not initialized"):
            get_llm_service()

    async def test_get_service_with_config(self, llm_config: LLMConfig) -> None:
        """Test getting service with config parameter."""
        # Clean up any existing service
        await cleanup_llm_service()

        # Get service with config
        service = get_llm_service(llm_config)
        assert service is not None
        assert service.config == llm_config

        # Clean up
        await cleanup_llm_service()

    async def test_cleanup_service(self, llm_config: LLMConfig) -> None:
        """Test service cleanup."""
        # Initialize service
        initialize_llm_service(llm_config)

        # Cleanup
        await cleanup_llm_service()

        # Should raise error after cleanup
        with pytest.raises(RuntimeError, match="LLM service not initialized"):
            get_llm_service()
