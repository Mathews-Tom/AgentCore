"""Tests for Portkey exception classes."""

from __future__ import annotations

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
    """Test suite for base LLMGatewayError."""

    def test_basic_error(self) -> None:
        """Test creating basic error."""
        error = LLMGatewayError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.status_code is None

    def test_error_with_status_code(self) -> None:
        """Test error with HTTP status code."""
        error = LLMGatewayError("Bad request", status_code=400)

        assert error.message == "Bad request"
        assert error.status_code == 400


class TestLLMGatewayConfigurationError:
    """Test suite for LLMGatewayConfigurationError."""

    def test_configuration_error(self) -> None:
        """Test configuration error message formatting."""
        error = LLMGatewayConfigurationError("Missing API key")

        assert "Configuration error: Missing API key" in str(error)
        assert error.status_code is None


class TestLLMGatewayAuthenticationError:
    """Test suite for LLMGatewayAuthenticationError."""

    def test_authentication_error(self) -> None:
        """Test authentication error."""
        error = LLMGatewayAuthenticationError("Invalid API key")

        assert "Authentication error: Invalid API key" in str(error)
        assert error.status_code == 401


class TestLLMGatewayProviderError:
    """Test suite for LLMGatewayProviderError."""

    def test_provider_error_without_provider(self) -> None:
        """Test provider error without provider name."""
        error = LLMGatewayProviderError("Model not found")

        assert "Provider error: Model not found" in str(error)
        assert error.provider is None

    def test_provider_error_with_provider(self) -> None:
        """Test provider error with provider name."""
        error = LLMGatewayProviderError("Service unavailable", provider="openai")

        assert "Provider error (openai): Service unavailable" in str(error)
        assert error.provider == "openai"


class TestLLMGatewayRateLimitError:
    """Test suite for LLMGatewayRateLimitError."""

    def test_rate_limit_error(self) -> None:
        """Test rate limit error."""
        error = LLMGatewayRateLimitError("Too many requests")

        assert "Rate limit exceeded: Too many requests" in str(error)
        assert error.status_code == 429
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self) -> None:
        """Test rate limit error with retry_after."""
        error = LLMGatewayRateLimitError("Too many requests", retry_after=60)

        assert error.retry_after == 60
        assert error.status_code == 429


class TestLLMGatewayTimeoutError:
    """Test suite for LLMGatewayTimeoutError."""

    def test_timeout_error(self) -> None:
        """Test timeout error."""
        error = LLMGatewayTimeoutError("Request timed out", timeout=30.0)

        assert "Request timed out after 30.0s" in str(error)
        assert error.timeout == 30.0
        assert error.status_code == 408


class TestLLMGatewayValidationError:
    """Test suite for LLMGatewayValidationError."""

    def test_validation_error_without_field(self) -> None:
        """Test validation error without field name."""
        error = LLMGatewayValidationError("Invalid input")

        assert "Validation error: Invalid input" in str(error)
        assert error.field is None
        assert error.status_code == 400

    def test_validation_error_with_field(self) -> None:
        """Test validation error with field name."""
        error = LLMGatewayValidationError("Must be positive", field="max_tokens")

        assert "Validation error (field: max_tokens): Must be positive" in str(error)
        assert error.field == "max_tokens"
        assert error.status_code == 400


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        """Test that all exceptions inherit from LLMGatewayError."""
        exceptions = [
            LLMGatewayConfigurationError("test"),
            LLMGatewayAuthenticationError("test"),
            LLMGatewayProviderError("test"),
            LLMGatewayRateLimitError("test"),
            LLMGatewayTimeoutError("test", timeout=30.0),
            LLMGatewayValidationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, LLMGatewayError)
            assert isinstance(exc, Exception)

    def test_exceptions_can_be_caught_as_base(self) -> None:
        """Test that specific exceptions can be caught as base exception."""
        try:
            raise LLMGatewayAuthenticationError("Invalid API key")
        except LLMGatewayError as e:
            assert isinstance(e, LLMGatewayAuthenticationError)
            assert "Authentication error" in str(e)
