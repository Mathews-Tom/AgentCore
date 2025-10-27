"""Unit tests for LLM client data models and enums.

Tests cover:
- Enum values and types
- Model validation (temperature, max_tokens)
- Custom exception instantiation
- Pydantic serialization/deserialization
- Edge cases and boundary conditions
"""

import pytest
from pydantic import ValidationError

from agentcore.a2a_protocol.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ModelNotAllowedError,
    ModelTier,
    Provider,
    ProviderError,
    ProviderTimeoutError,
)


class TestProviderEnum:
    """Test Provider enumeration."""

    def test_provider_values(self) -> None:
        """Test that Provider enum has correct values."""
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.GEMINI.value == "gemini"

    def test_provider_count(self) -> None:
        """Test that Provider enum has exactly 3 members."""
        assert len(Provider) == 3

    def test_provider_string_comparison(self) -> None:
        """Test that Provider enum values compare with strings."""
        assert Provider.OPENAI == "openai"
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.GEMINI == "gemini"


class TestModelTierEnum:
    """Test ModelTier enumeration."""

    def test_model_tier_values(self) -> None:
        """Test that ModelTier enum has correct values."""
        assert ModelTier.FAST.value == "fast"
        assert ModelTier.BALANCED.value == "balanced"
        assert ModelTier.PREMIUM.value == "premium"

    def test_model_tier_count(self) -> None:
        """Test that ModelTier enum has exactly 3 members."""
        assert len(ModelTier) == 3

    def test_model_tier_string_comparison(self) -> None:
        """Test that ModelTier enum values compare with strings."""
        assert ModelTier.FAST == "fast"
        assert ModelTier.BALANCED == "balanced"
        assert ModelTier.PREMIUM == "premium"


class TestModelNotAllowedError:
    """Test ModelNotAllowedError exception."""

    def test_exception_creation(self) -> None:
        """Test that ModelNotAllowedError can be instantiated."""
        allowed = ["gpt-4.1-mini", "claude-3-5-haiku"]
        error = ModelNotAllowedError("gpt-3.5-turbo", allowed)

        assert error.model == "gpt-3.5-turbo"
        assert error.allowed == allowed
        assert "gpt-3.5-turbo" in str(error)
        assert "not allowed" in str(error)

    def test_exception_raise_and_catch(self) -> None:
        """Test that ModelNotAllowedError can be raised and caught."""
        with pytest.raises(ModelNotAllowedError) as exc_info:
            raise ModelNotAllowedError("invalid-model", ["valid-model"])

        assert exc_info.value.model == "invalid-model"
        assert exc_info.value.allowed == ["valid-model"]


class TestProviderError:
    """Test ProviderError exception."""

    def test_exception_creation(self) -> None:
        """Test that ProviderError can be instantiated."""
        original = ValueError("API key invalid")
        error = ProviderError("openai", original)

        assert error.provider == "openai"
        assert error.original_error is original
        assert "openai" in str(error)
        assert "API key invalid" in str(error)

    def test_exception_raise_and_catch(self) -> None:
        """Test that ProviderError can be raised and caught."""
        original = RuntimeError("Connection failed")
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("anthropic", original)

        assert exc_info.value.provider == "anthropic"
        assert exc_info.value.original_error is original


class TestProviderTimeoutError:
    """Test ProviderTimeoutError exception."""

    def test_exception_creation(self) -> None:
        """Test that ProviderTimeoutError can be instantiated."""
        error = ProviderTimeoutError("gemini", 60.0)

        assert error.provider == "gemini"
        assert error.timeout_seconds == 60.0
        assert "gemini" in str(error)
        assert "60" in str(error)
        assert "timed out" in str(error)

    def test_exception_raise_and_catch(self) -> None:
        """Test that ProviderTimeoutError can be raised and caught."""
        with pytest.raises(ProviderTimeoutError) as exc_info:
            raise ProviderTimeoutError("openai", 30.0)

        assert exc_info.value.provider == "openai"
        assert exc_info.value.timeout_seconds == 30.0


class TestLLMRequest:
    """Test LLMRequest model."""

    def test_valid_request_minimal(self) -> None:
        """Test creating LLMRequest with minimal required fields."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert request.model == "gpt-4.1-mini"
        assert len(request.messages) == 1
        assert request.stream is False  # Default
        assert request.trace_id is None
        assert request.source_agent is None
        assert request.session_id is None

    def test_valid_request_with_all_fields(self) -> None:
        """Test creating LLMRequest with all fields."""
        request = LLMRequest(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            stream=True,
            reasoning_effort="medium",
            trace_id="trace-123",
            source_agent="agent-001",
            session_id="session-456",
        )

        assert request.model == "claude-3-5-sonnet-20241022"
        assert len(request.messages) == 2
        assert request.stream is True
        assert request.reasoning_effort == "medium"
        assert request.trace_id == "trace-123"
        assert request.source_agent == "agent-001"
        assert request.session_id == "session-456"

    def test_serialization_to_dict(self) -> None:
        """Test that LLMRequest can be serialized to dict."""
        request = LLMRequest(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": "test"}],
            trace_id="trace-123",
        )
        data = request.model_dump()

        assert data["model"] == "gpt-4.1-mini"
        assert data["messages"] == [{"role": "user", "content": "test"}]
        assert data["trace_id"] == "trace-123"

    def test_deserialization_from_dict(self) -> None:
        """Test that LLMRequest can be deserialized from dict."""
        data = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "reasoning_effort": "high",
        }
        request = LLMRequest(**data)

        assert request.model == "gpt-4.1-mini"
        assert request.reasoning_effort == "high"


class TestLLMUsage:
    """Test LLMUsage model."""

    def test_valid_usage(self) -> None:
        """Test creating LLMUsage with valid values."""
        usage = LLMUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_serialization_to_dict(self) -> None:
        """Test that LLMUsage can be serialized to dict."""
        usage = LLMUsage(
            prompt_tokens=5, completion_tokens=15, total_tokens=20
        )
        data = usage.model_dump()

        assert data["prompt_tokens"] == 5
        assert data["completion_tokens"] == 15
        assert data["total_tokens"] == 20

    def test_deserialization_from_dict(self) -> None:
        """Test that LLMUsage can be deserialized from dict."""
        data = {
            "prompt_tokens": 8,
            "completion_tokens": 12,
            "total_tokens": 20,
        }
        usage = LLMUsage(**data)

        assert usage.prompt_tokens == 8
        assert usage.completion_tokens == 12
        assert usage.total_tokens == 20


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_valid_response_minimal(self) -> None:
        """Test creating LLMResponse with minimal required fields."""
        usage = LLMUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        response = LLMResponse(
            content="Hello, world!",
            usage=usage,
            latency_ms=250,
            provider="openai",
            model="gpt-4.1-mini",
        )

        assert response.content == "Hello, world!"
        assert response.usage.total_tokens == 30
        assert response.latency_ms == 250
        assert response.provider == "openai"
        assert response.model == "gpt-4.1-mini"
        assert response.trace_id is None

    def test_valid_response_with_trace_id(self) -> None:
        """Test creating LLMResponse with trace_id."""
        usage = LLMUsage(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        response = LLMResponse(
            content="Test response",
            usage=usage,
            latency_ms=300,
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            trace_id="trace-456",
        )

        assert response.trace_id == "trace-456"

    def test_serialization_to_dict(self) -> None:
        """Test that LLMResponse can be serialized to dict."""
        usage = LLMUsage(
            prompt_tokens=5, completion_tokens=10, total_tokens=15
        )
        response = LLMResponse(
            content="Response",
            usage=usage,
            latency_ms=200,
            provider="gemini",
            model="gemini-1.5-flash",
            trace_id="trace-789",
        )
        data = response.model_dump()

        assert data["content"] == "Response"
        assert data["usage"]["total_tokens"] == 15
        assert data["latency_ms"] == 200
        assert data["provider"] == "gemini"
        assert data["model"] == "gemini-1.5-flash"
        assert data["trace_id"] == "trace-789"

    def test_deserialization_from_dict(self) -> None:
        """Test that LLMResponse can be deserialized from dict."""
        data = {
            "content": "Test",
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 13,
                "total_tokens": 20,
            },
            "latency_ms": 180,
            "provider": "openai",
            "model": "gpt-5",
        }
        response = LLMResponse(**data)

        assert response.content == "Test"
        assert response.usage.total_tokens == 20
        assert response.latency_ms == 180
        assert response.provider == "openai"
        assert response.model == "gpt-5"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_content_in_response(self) -> None:
        """Test that LLMResponse accepts empty content string."""
        usage = LLMUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        response = LLMResponse(
            content="",
            usage=usage,
            latency_ms=100,
            provider="openai",
            model="gpt-4.1-mini",
        )
        assert response.content == ""

    def test_very_long_model_name(self) -> None:
        """Test that LLMRequest accepts long model names."""
        request = LLMRequest(
            model="very-long-model-name-with-version-123456789",
            messages=[{"role": "user", "content": "test"}],
        )
        assert len(request.model) > 20

    def test_multiple_messages_in_request(self) -> None:
        """Test that LLMRequest accepts multiple messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First assistant response"},
            {"role": "user", "content": "Second user message"},
        ]
        request = LLMRequest(model="gpt-4.1-mini", messages=messages)
        assert len(request.messages) == 4
