"""Portkey integration exceptions.

Custom exception hierarchy for handling Portkey API errors and integration issues.
"""

from __future__ import annotations


class PortkeyError(Exception):
    """Base exception for all Portkey integration errors.

    All Portkey-specific exceptions inherit from this base class.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message describing what went wrong
            status_code: Optional HTTP status code if applicable
        """
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class PortkeyConfigurationError(PortkeyError):
    """Exception raised for configuration errors.

    This exception is raised when:
    - Required configuration is missing
    - Configuration values are invalid
    - Configuration file cannot be loaded
    """

    def __init__(self, message: str) -> None:
        """Initialize the configuration error.

        Args:
            message: Description of the configuration issue
        """
        super().__init__(f"Configuration error: {message}")


class PortkeyAuthenticationError(PortkeyError):
    """Exception raised for authentication failures.

    This exception is raised when:
    - API key is missing or invalid
    - Virtual key authentication fails
    - Authorization headers are malformed
    """

    def __init__(self, message: str) -> None:
        """Initialize the authentication error.

        Args:
            message: Description of the authentication failure
        """
        super().__init__(f"Authentication error: {message}", status_code=401)


class PortkeyProviderError(PortkeyError):
    """Exception raised for LLM provider errors.

    This exception is raised when:
    - Provider is unavailable or down
    - Provider returns an error response
    - Model is not available for the provider
    - Provider-specific quota is exceeded
    """

    def __init__(self, message: str, provider: str | None = None) -> None:
        """Initialize the provider error.

        Args:
            message: Description of the provider issue
            provider: Name of the LLM provider that failed
        """
        self.provider = provider
        provider_info = f" ({provider})" if provider else ""
        super().__init__(f"Provider error{provider_info}: {message}")


class PortkeyRateLimitError(PortkeyError):
    """Exception raised when rate limits are exceeded.

    This exception is raised when:
    - Portkey API rate limit is exceeded
    - Provider rate limit is exceeded
    - Request quota is exhausted
    """

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: Description of the rate limit issue
            retry_after: Optional seconds to wait before retrying
        """
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded: {message}", status_code=429)


class PortkeyTimeoutError(PortkeyError):
    """Exception raised when requests timeout.

    This exception is raised when:
    - Request to Portkey API times out
    - Provider request exceeds timeout threshold
    - No response received within configured timeout
    """

    def __init__(self, message: str, timeout: float) -> None:
        """Initialize the timeout error.

        Args:
            message: Description of what timed out
            timeout: The timeout value in seconds that was exceeded
        """
        self.timeout = timeout
        super().__init__(f"Request timed out after {timeout}s: {message}", status_code=408)


class PortkeyValidationError(PortkeyError):
    """Exception raised for request validation errors.

    This exception is raised when:
    - Request parameters are invalid
    - Model requirements cannot be satisfied
    - Request format is malformed
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        """Initialize the validation error.

        Args:
            message: Description of the validation failure
            field: Optional name of the field that failed validation
        """
        self.field = field
        field_info = f" (field: {field})" if field else ""
        super().__init__(f"Validation error{field_info}: {message}", status_code=400)
