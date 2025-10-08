"""Tests for Portkey exception classes."""

from __future__ import annotations

from agentcore.integration.portkey.exceptions import (
    PortkeyAuthenticationError,
    PortkeyConfigurationError,
    PortkeyError,
    PortkeyProviderError,
    PortkeyRateLimitError,
    PortkeyTimeoutError,
    PortkeyValidationError,
)


class TestPortkeyError:
    """Test suite for base PortkeyError."""

    def test_basic_error(self) -> None:
        """Test creating basic error."""
        error = PortkeyError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.status_code is None

    def test_error_with_status_code(self) -> None:
        """Test error with HTTP status code."""
        error = PortkeyError("Bad request", status_code=400)

        assert error.message == "Bad request"
        assert error.status_code == 400


class TestPortkeyConfigurationError:
    """Test suite for PortkeyConfigurationError."""

    def test_configuration_error(self) -> None:
        """Test configuration error message formatting."""
        error = PortkeyConfigurationError("Missing API key")

        assert "Configuration error: Missing API key" in str(error)
        assert error.status_code is None


class TestPortkeyAuthenticationError:
    """Test suite for PortkeyAuthenticationError."""

    def test_authentication_error(self) -> None:
        """Test authentication error."""
        error = PortkeyAuthenticationError("Invalid API key")

        assert "Authentication error: Invalid API key" in str(error)
        assert error.status_code == 401


class TestPortkeyProviderError:
    """Test suite for PortkeyProviderError."""

    def test_provider_error_without_provider(self) -> None:
        """Test provider error without provider name."""
        error = PortkeyProviderError("Model not found")

        assert "Provider error: Model not found" in str(error)
        assert error.provider is None

    def test_provider_error_with_provider(self) -> None:
        """Test provider error with provider name."""
        error = PortkeyProviderError("Service unavailable", provider="openai")

        assert "Provider error (openai): Service unavailable" in str(error)
        assert error.provider == "openai"


class TestPortkeyRateLimitError:
    """Test suite for PortkeyRateLimitError."""

    def test_rate_limit_error(self) -> None:
        """Test rate limit error."""
        error = PortkeyRateLimitError("Too many requests")

        assert "Rate limit exceeded: Too many requests" in str(error)
        assert error.status_code == 429
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self) -> None:
        """Test rate limit error with retry_after."""
        error = PortkeyRateLimitError("Too many requests", retry_after=60)

        assert error.retry_after == 60
        assert error.status_code == 429


class TestPortkeyTimeoutError:
    """Test suite for PortkeyTimeoutError."""

    def test_timeout_error(self) -> None:
        """Test timeout error."""
        error = PortkeyTimeoutError("Request timed out", timeout=30.0)

        assert "Request timed out after 30.0s" in str(error)
        assert error.timeout == 30.0
        assert error.status_code == 408


class TestPortkeyValidationError:
    """Test suite for PortkeyValidationError."""

    def test_validation_error_without_field(self) -> None:
        """Test validation error without field name."""
        error = PortkeyValidationError("Invalid input")

        assert "Validation error: Invalid input" in str(error)
        assert error.field is None
        assert error.status_code == 400

    def test_validation_error_with_field(self) -> None:
        """Test validation error with field name."""
        error = PortkeyValidationError("Must be positive", field="max_tokens")

        assert "Validation error (field: max_tokens): Must be positive" in str(error)
        assert error.field == "max_tokens"
        assert error.status_code == 400


class TestExceptionHierarchy:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_base(self) -> None:
        """Test that all exceptions inherit from PortkeyError."""
        exceptions = [
            PortkeyConfigurationError("test"),
            PortkeyAuthenticationError("test"),
            PortkeyProviderError("test"),
            PortkeyRateLimitError("test"),
            PortkeyTimeoutError("test", timeout=30.0),
            PortkeyValidationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, PortkeyError)
            assert isinstance(exc, Exception)

    def test_exceptions_can_be_caught_as_base(self) -> None:
        """Test that specific exceptions can be caught as base exception."""
        try:
            raise PortkeyAuthenticationError("Invalid API key")
        except PortkeyError as e:
            assert isinstance(e, PortkeyAuthenticationError)
            assert "Authentication error" in str(e)
