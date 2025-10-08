"""Tests for Portkey client wrapper."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.integration.portkey.client import PortkeyClient
from agentcore.integration.portkey.config import PortkeyConfig
from agentcore.integration.portkey.exceptions import (
    PortkeyAuthenticationError,
    PortkeyError,
    PortkeyProviderError,
    PortkeyRateLimitError,
    PortkeyTimeoutError,
    PortkeyValidationError,
)
from agentcore.integration.portkey.models import LLMRequest, LLMResponse


@pytest.fixture
def mock_config(monkeypatch: pytest.MonkeyPatch) -> PortkeyConfig:
    """Create a mock configuration for testing."""
    monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")
    return PortkeyConfig(
        api_key="test-api-key",
        base_url="https://api.portkey.ai/v1",
        timeout=60.0,
        max_retries=3,
    )


@pytest.fixture
def sample_request() -> LLMRequest:
    """Create a sample LLM request for testing."""
    return LLMRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=100,
        temperature=0.7,
        context={"agent_id": "test-agent"},
    )


@pytest.fixture
def mock_response() -> dict[str, Any]:
    """Create a mock Portkey API response."""
    return {
        "id": "chatcmpl-123",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25,
        },
    }


class TestPortkeyClient:
    """Test suite for PortkeyClient."""

    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    def test_initialization_with_config(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
    ) -> None:
        """Test client initialization with provided config."""
        client = PortkeyClient(config=mock_config)

        assert client.config == mock_config
        assert not client._closed
        mock_portkey.assert_called_once()

    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    def test_initialization_from_env(
        self,
        mock_portkey: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test client initialization from environment variables."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")

        client = PortkeyClient()

        assert client.config.api_key == "test-api-key"
        assert not client._closed

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_success(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
        mock_response: dict[str, Any],
    ) -> None:
        """Test successful completion request."""
        # Setup mock
        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(return_value=MagicMock(model_dump=lambda: mock_response))
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Execute request
        response = await client.complete(sample_request)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.usage == mock_response["usage"]
        assert response.latency_ms is not None
        assert response.latency_ms >= 0

        # Verify API was called correctly
        mock_completion.create.assert_called_once()
        call_kwargs = mock_completion.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["messages"] == sample_request.messages
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_with_trace_id(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        mock_response: dict[str, Any],
    ) -> None:
        """Test completion request with trace ID."""
        # Setup mock
        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(return_value=MagicMock(model_dump=lambda: mock_response))
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Create request with trace ID
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            context={"trace_id": "trace-123"},
        )

        # Execute request
        await client.complete(request)

        # Verify trace ID was included in metadata
        call_kwargs = mock_completion.create.call_args[1]
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["trace_id"] == "trace-123"

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_authentication_error(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
    ) -> None:
        """Test handling of authentication errors."""
        # Setup mock to raise auth error
        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(
            side_effect=Exception("Authentication failed: invalid api key")
        )
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Verify error is mapped correctly
        with pytest.raises(PortkeyAuthenticationError):
            await client.complete(sample_request)

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_rate_limit_error(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
    ) -> None:
        """Test handling of rate limit errors."""
        # Setup mock to raise rate limit error
        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(side_effect=Exception("Rate limit exceeded"))
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Verify error is mapped correctly
        with pytest.raises(PortkeyRateLimitError):
            await client.complete(sample_request)

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_timeout_error(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
    ) -> None:
        """Test handling of timeout errors."""
        # Setup mock to raise timeout error
        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(side_effect=Exception("Request timed out"))
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Verify error is mapped correctly
        with pytest.raises(PortkeyTimeoutError):
            await client.complete(sample_request)

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_provider_error(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
    ) -> None:
        """Test handling of provider errors."""
        # Setup mock to raise provider error
        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(side_effect=Exception("Provider error: model not found"))
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Verify error is mapped correctly
        with pytest.raises(PortkeyProviderError):
            await client.complete(sample_request)

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_validation_error(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
    ) -> None:
        """Test handling of validation errors."""
        # Setup mock to raise validation error
        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(side_effect=Exception("Validation error: invalid parameter"))
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Verify error is mapped correctly
        with pytest.raises(PortkeyValidationError):
            await client.complete(sample_request)

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_complete_after_close(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
    ) -> None:
        """Test that requests fail after client is closed."""
        client = PortkeyClient(config=mock_config)
        await client.close()

        # Verify error is raised
        with pytest.raises(PortkeyError, match="Client has been closed"):
            await client.complete(sample_request)

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_stream_complete(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
        sample_request: LLMRequest,
    ) -> None:
        """Test streaming completion request."""
        # Setup mock stream
        mock_chunks = [
            {"id": "chunk-1", "choices": [{"delta": {"content": "Hello"}}]},
            {"id": "chunk-2", "choices": [{"delta": {"content": " world"}}]},
        ]

        async def mock_stream() -> Any:
            for chunk in mock_chunks:
                yield MagicMock(model_dump=lambda c=chunk: c)

        mock_completion = AsyncMock()
        mock_completion.create = AsyncMock(return_value=mock_stream())
        mock_portkey.return_value.chat.completions = mock_completion

        client = PortkeyClient(config=mock_config)

        # Execute streaming request
        chunks = []
        async for chunk in client.stream_complete(sample_request):
            chunks.append(chunk)

        # Verify chunks
        assert len(chunks) == 2
        assert chunks[0]["id"] == "chunk-1"
        assert chunks[1]["id"] == "chunk-2"

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_context_manager(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
    ) -> None:
        """Test using client as async context manager."""
        async with PortkeyClient(config=mock_config) as client:
            assert not client._closed

        # Client should be closed after exiting context
        assert client._closed

    @pytest.mark.asyncio
    @patch("agentcore.integration.portkey.client.AsyncPortkey")
    async def test_close(
        self,
        mock_portkey: MagicMock,
        mock_config: PortkeyConfig,
    ) -> None:
        """Test closing the client."""
        client = PortkeyClient(config=mock_config)
        assert not client._closed

        await client.close()
        assert client._closed

        # Closing again should be idempotent
        await client.close()
        assert client._closed
