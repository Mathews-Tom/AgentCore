"""API integration exceptions.

Custom exceptions for API client operations with specific error handling.
"""

from __future__ import annotations


class APIError(Exception):
    """Base exception for all API integration errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        """Initialize API error.

        Args:
            message: Error message
            status_code: Optional HTTP status code
            response_body: Optional response body
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class APIConnectionError(APIError):
    """Connection to API failed."""

    pass


class APITimeoutError(APIError):
    """API request timeout."""

    pass


class APIAuthenticationError(APIError):
    """Authentication failed for API request."""

    pass


class APIAuthorizationError(APIError):
    """Authorization failed - insufficient permissions."""

    pass


class APIRateLimitError(APIError):
    """API rate limit exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: int | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds until rate limit resets
            status_code: Optional HTTP status code
            response_body: Optional response body
        """
        super().__init__(message, status_code, response_body)
        self.retry_after = retry_after


class APIValidationError(APIError):
    """Request validation failed."""

    pass


class APINotFoundError(APIError):
    """API resource not found."""

    pass


class APIServerError(APIError):
    """API server error (5xx)."""

    pass


class APITransformationError(APIError):
    """Response transformation failed."""

    pass


class APIConfigurationError(APIError):
    """API configuration error."""

    pass
