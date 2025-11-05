"""Unit tests for LLM Gateway exceptions."""

import pytest

from agentcore.llm_gateway.exceptions import (
    LLMGatewayAuthenticationError,
    LLMGatewayConfigurationError,
    LLMGatewayError,
    LLMGatewayProviderError,
    LLMGatewayRateLimitError,
    LLMGatewayTimeoutError,
    LLMGatewayValidationError,
)


class TestLLMGatewayError:
    """Test base LLMGatewayError exception."""

    def test_init_with_message(self) -> None:
        """Test exception initialization with message."""
        error = LLMGatewayError("Test error message")
        assert str(error) == "Test error message"

    def test_init_with_empty_message(self) -> None:
        """Test exception initialization with empty message."""
        error = LLMGatewayError("")
        assert str(error) == ""

    def test_inheritance(self) -> None:
        """Test that LLMGatewayError inherits from Exception."""
        error = LLMGatewayError("test")
        assert isinstance(error, Exception)


class TestLLMGatewayConfigurationError:
    """Test LLMGatewayConfigurationError exception."""

    def test_inheritance(self) -> None:
        """Test proper inheritance."""
        error = LLMGatewayConfigurationError("config error")
        assert isinstance(error, LLMGatewayError)
        assert isinstance(error, Exception)

    def test_message(self) -> None:
        """Test error message."""
        error = LLMGatewayConfigurationError("Invalid config")
        assert str(error) == "Configuration error: Invalid config"


class TestLLMGatewayProviderError:
    """Test LLMGatewayProviderError exception."""

    def test_inheritance(self) -> None:
        """Test proper inheritance."""
        error = LLMGatewayProviderError("provider error")
        assert isinstance(error, LLMGatewayError)

    def test_message_with_provider_name(self) -> None:
        """Test error message with provider details."""
        error = LLMGatewayProviderError("OpenAI provider unavailable")
        assert "OpenAI" in str(error)


class TestLLMGatewayRateLimitError:
    """Test LLMGatewayRateLimitError exception."""

    def test_inheritance(self) -> None:
        """Test proper inheritance."""
        error = LLMGatewayRateLimitError("rate limit exceeded")
        assert isinstance(error, LLMGatewayError)

    def test_rate_limit_message(self) -> None:
        """Test rate limit error message."""
        error = LLMGatewayRateLimitError("Rate limit: 60 requests/min exceeded")
        assert "Rate limit" in str(error)
        assert "60 requests/min" in str(error)


class TestLLMGatewayTimeoutError:
    """Test LLMGatewayTimeoutError exception."""

    def test_inheritance(self) -> None:
        """Test proper inheritance."""
        error = LLMGatewayTimeoutError("request timeout", timeout=30.0)
        assert isinstance(error, LLMGatewayError)

    def test_timeout_message(self) -> None:
        """Test timeout error message."""
        error = LLMGatewayTimeoutError("OpenAI completion", timeout=30.0)
        assert "timed out" in str(error)
        assert "30.0s" in str(error)
        assert error.timeout == 30.0


class TestLLMGatewayAuthenticationError:
    """Test LLMGatewayAuthenticationError exception."""

    def test_inheritance(self) -> None:
        """Test proper inheritance."""
        error = LLMGatewayAuthenticationError("auth failed")
        assert isinstance(error, LLMGatewayError)

    def test_auth_error_message(self) -> None:
        """Test authentication error message."""
        error = LLMGatewayAuthenticationError("Invalid API key")
        assert "API key" in str(error)


class TestLLMGatewayValidationError:
    """Test LLMGatewayValidationError exception."""

    def test_inheritance(self) -> None:
        """Test proper inheritance."""
        error = LLMGatewayValidationError("validation failed")
        assert isinstance(error, LLMGatewayError)

    def test_validation_error_message(self) -> None:
        """Test validation error message."""
        error = LLMGatewayValidationError("Invalid request parameters")
        assert "parameters" in str(error)


class TestExceptionRaising:
    """Test that exceptions can be raised and caught properly."""

    def test_raise_and_catch_base_error(self) -> None:
        """Test raising and catching base LLMGatewayError."""
        with pytest.raises(LLMGatewayError) as exc_info:
            raise LLMGatewayError("test error")

        assert str(exc_info.value) == "test error"

    def test_catch_specific_error_as_base(self) -> None:
        """Test catching specific error as base error."""
        with pytest.raises(LLMGatewayError):
            raise LLMGatewayRateLimitError("rate limit")

    def test_raise_multiple_error_types(self) -> None:
        """Test raising different error types."""
        errors = [
            LLMGatewayConfigurationError("config"),
            LLMGatewayProviderError("provider"),
            LLMGatewayRateLimitError("rate"),
            LLMGatewayTimeoutError("timeout", timeout=30.0),
            LLMGatewayAuthenticationError("auth"),
            LLMGatewayValidationError("validation"),
        ]

        for error in errors:
            with pytest.raises(LLMGatewayError):
                raise error
