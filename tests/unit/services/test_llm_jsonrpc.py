"""Unit tests for LLM JSON-RPC methods.

Tests all LLM JSON-RPC handlers with mocked llm_service to avoid real API calls.
Covers success cases, error handling, A2A context extraction, and edge cases.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest
from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ModelNotAllowedError,
    ProviderError,
    ProviderTimeoutError)
from agentcore.a2a_protocol.services.llm_jsonrpc import (
    handle_llm_complete,
    handle_llm_metrics,
    handle_llm_models,
    handle_llm_stream)


class TestHandleLLMComplete:
    """Test suite for llm.complete JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_complete_success(self) -> None:
        """Test successful LLM completion via JSON-RPC."""
        # Setup request
        request = JsonRpcRequest(
            method="llm.complete",
            params={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
                "max_tokens": 100,
            },
            id=1,
            a2a_context=A2AContext(
                source_agent="test-agent",
                trace_id="trace-123",
                timestamp="2025-10-26T10:00:00Z"))

        # Mock LLMService.complete
        mock_response = LLMResponse(
            content="Hello! How can I help you?",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=7, total_tokens=17),
            latency_ms=234,
            provider="openai",
            model="gpt-4.1-mini",
            trace_id="trace-123")

        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.llm_service.complete",
            new_callable=AsyncMock,
            return_value=mock_response):
            result = await handle_llm_complete(request)

        # Verify result
        assert result["content"] == "Hello! How can I help you?"
        assert result["usage"]["total_tokens"] == 17
        assert result["latency_ms"] == 234
        assert result["provider"] == "openai"
        assert result["model"] == "gpt-4.1-mini"
        assert result["trace_id"] == "trace-123"

    @pytest.mark.asyncio
    async def test_complete_a2a_context_extraction(self) -> None:
        """Test A2A context extraction from JsonRpcRequest."""
        request = JsonRpcRequest(
            method="llm.complete",
            params={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Test"}],
            },
            id=2,
            a2a_context=A2AContext(
                source_agent="agent-001",
                trace_id="trace-abc",
                session_id="session-xyz",
                timestamp="2025-10-26T10:00:00Z"))

        mock_response = LLMResponse(
            content="Response",
            usage=LLMUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            latency_ms=100,
            provider="openai",
            model="gpt-4.1-mini",
            trace_id="trace-abc")

        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.llm_service.complete",
            new_callable=AsyncMock,
            return_value=mock_response) as mock_complete:
            await handle_llm_complete(request)

            # Verify A2A context was passed to LLMRequest
            call_args = mock_complete.call_args
            llm_request = call_args[0][0]
            assert isinstance(llm_request, LLMRequest)
            assert llm_request.trace_id == "trace-abc"
            assert llm_request.source_agent == "agent-001"
            assert llm_request.session_id == "session-xyz"

    @pytest.mark.asyncio
    async def test_complete_model_not_allowed(self) -> None:
        """Test model governance violation handling."""
        request = JsonRpcRequest(
            method="llm.complete",
            params={
                "model": "gpt-3.5-turbo",  # Not in ALLOWED_MODELS
                "messages": [{"role": "user", "content": "Test"}],
            },
            id=3)

        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.llm_service.complete",
            new_callable=AsyncMock,
            side_effect=ModelNotAllowedError("gpt-3.5-turbo", ["gpt-4.1-mini"])):
            with pytest.raises(ValueError, match="Model 'gpt-3.5-turbo' not allowed"):
                await handle_llm_complete(request)

    @pytest.mark.asyncio
    async def test_complete_provider_timeout(self) -> None:
        """Test provider timeout error handling."""
        request = JsonRpcRequest(
            method="llm.complete",
            params={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Test"}],
            },
            id=4)

        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.llm_service.complete",
            new_callable=AsyncMock,
            side_effect=ProviderTimeoutError("openai", 60.0)):
            with pytest.raises(RuntimeError, match="Request timed out after 60.0s"):
                await handle_llm_complete(request)

    @pytest.mark.asyncio
    async def test_complete_provider_error(self) -> None:
        """Test provider error handling."""
        request = JsonRpcRequest(
            method="llm.complete",
            params={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Test"}],
            },
            id=5)

        original_error = Exception("Rate limit exceeded")
        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.llm_service.complete",
            new_callable=AsyncMock,
            side_effect=ProviderError("openai", original_error)):
            with pytest.raises(RuntimeError, match="Provider error: openai"):
                await handle_llm_complete(request)

    @pytest.mark.asyncio
    async def test_complete_missing_params(self) -> None:
        """Test error handling for missing required parameters."""
        request = JsonRpcRequest(
            method="llm.complete",
            params={},  # Missing model and messages
            id=6)

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_llm_complete(request)


class TestHandleLLMStream:
    """Test suite for llm.stream JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_stream_returns_error_message(self) -> None:
        """Test that streaming via JSON-RPC returns helpful error message."""
        request = JsonRpcRequest(
            method="llm.stream",
            params={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Test"}],
            },
            id=7)

        result = await handle_llm_stream(request)

        # Verify error message
        assert "error" in result
        assert "Streaming not supported via JSON-RPC" in result["error"]
        assert "alternatives" in result
        assert len(result["alternatives"]) > 0
        assert "WebSocket" in result["alternatives"][0] or "SSE" in result["alternatives"][1]


class TestHandleLLMModels:
    """Test suite for llm.models JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_models_returns_allowed_list(self) -> None:
        """Test that llm.models returns ALLOWED_MODELS configuration."""
        request = JsonRpcRequest(
            method="llm.models",
            params={},
            id=8)

        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.llm_service.registry.list_available_models",
            return_value=["gpt-4.1-mini", "claude-3-5-haiku-20241022"]):
            with patch(
                "agentcore.a2a_protocol.config.settings.LLM_DEFAULT_MODEL",
                "gpt-4.1-mini"):
                result = await handle_llm_models(request)

        # Verify result
        assert "allowed_models" in result
        assert "default_model" in result
        assert "count" in result
        assert result["allowed_models"] == ["gpt-4.1-mini", "claude-3-5-haiku-20241022"]
        assert result["default_model"] == "gpt-4.1-mini"
        assert result["count"] == 2


class TestHandleLLMMetrics:
    """Test suite for llm.metrics JSON-RPC handler."""

    @pytest.mark.asyncio
    async def test_metrics_collects_prometheus_data(self) -> None:
        """Test that llm.metrics collects data from Prometheus registry."""
        request = JsonRpcRequest(
            method="llm.metrics",
            params={},
            id=9)

        # Mock Prometheus metric sample
        mock_sample = MagicMock()
        mock_sample.name = "llm_requests_total"
        mock_sample.labels = {"provider": "openai", "model": "gpt-4.1-mini", "status": "success"}
        mock_sample.value = 10

        # Mock Prometheus metric family
        mock_metric_family = MagicMock()
        mock_metric_family.name = "llm_requests_total"
        mock_metric_family.samples = [mock_sample]

        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.REGISTRY.collect",
            return_value=[mock_metric_family]):
            result = await handle_llm_metrics(request)

        # Verify result structure
        assert "total_requests" in result
        assert "total_tokens" in result
        assert "by_provider" in result
        assert "governance_violations" in result

        # Verify request count
        assert result["total_requests"] == 10

    @pytest.mark.asyncio
    async def test_metrics_aggregates_by_provider(self) -> None:
        """Test that llm.metrics correctly aggregates by provider."""
        request = JsonRpcRequest(
            method="llm.metrics",
            params={},
            id=10)

        # Create mock samples for requests
        mock_sample1 = MagicMock()
        mock_sample1.name = "llm_requests_total"
        mock_sample1.labels = {"provider": "openai", "model": "gpt-4.1-mini", "status": "success"}
        mock_sample1.value = 5

        mock_sample2 = MagicMock()
        mock_sample2.name = "llm_requests_total"
        mock_sample2.labels = {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "status": "success"}
        mock_sample2.value = 3

        # Create mock samples for tokens
        mock_sample3 = MagicMock()
        mock_sample3.name = "llm_tokens_total"
        mock_sample3.labels = {"provider": "openai", "model": "gpt-4.1-mini", "token_type": "prompt"}
        mock_sample3.value = 100

        mock_sample4 = MagicMock()
        mock_sample4.name = "llm_tokens_total"
        mock_sample4.labels = {"provider": "anthropic", "model": "claude-3-5-haiku-20241022", "token_type": "prompt"}
        mock_sample4.value = 50

        # Mock families
        mock_family1 = MagicMock()
        mock_family1.name = "llm_requests_total"
        mock_family1.samples = [mock_sample1, mock_sample2]

        mock_family2 = MagicMock()
        mock_family2.name = "llm_tokens_total"
        mock_family2.samples = [mock_sample3, mock_sample4]

        with patch(
            "agentcore.a2a_protocol.services.llm_jsonrpc.REGISTRY.collect",
            return_value=[mock_family1, mock_family2]):
            result = await handle_llm_metrics(request)

        # Verify aggregation
        assert result["total_requests"] == 8  # 5 + 3
        assert result["total_tokens"] == 150  # 100 + 50
        assert "openai" in result["by_provider"]
        assert "anthropic" in result["by_provider"]
        assert result["by_provider"]["openai"]["requests"] == 5
        assert result["by_provider"]["anthropic"]["requests"] == 3
