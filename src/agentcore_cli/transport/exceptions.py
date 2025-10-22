"""Transport layer exceptions.

These exceptions are raised by the transport layer when network-level
errors occur. They have no knowledge of JSON-RPC or business logic.
"""

from __future__ import annotations


class TransportError(Exception):
    """Base exception for transport layer errors.

    Raised when network-level communication fails. This is the base class
    for all transport-specific errors.

    Args:
        message: Human-readable error description
        status_code: HTTP status code if applicable
        cause: Original exception that caused this error

    Attributes:
        message: Error message
        status_code: HTTP status code (or None)
        cause: Original exception (or None)
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize TransportError.

        Args:
            message: Human-readable error description
            status_code: HTTP status code if applicable
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation of error.

        Returns:
            Formatted error message with status code if present
        """
        if self.status_code:
            return f"{self.message} (status: {self.status_code})"
        return self.message


class NetworkError(TransportError):
    """Network-level error occurred.

    Raised when network communication fails due to connection issues,
    DNS resolution failures, or other network-level problems.

    Examples:
        - Connection refused
        - DNS lookup failed
        - Network unreachable
    """

    pass


class TimeoutError(TransportError):
    """Request timed out.

    Raised when an HTTP request exceeds the configured timeout duration.
    This is distinct from network errors as the connection was established
    but the server did not respond in time.

    Examples:
        - Read timeout (server too slow to respond)
        - Connection timeout (slow to establish)
    """

    pass


class HttpError(TransportError):
    """HTTP error response received.

    Raised when the server returns an HTTP error status code (4xx or 5xx).
    The status code is available in the status_code attribute.

    Examples:
        - 400 Bad Request
        - 404 Not Found
        - 500 Internal Server Error
        - 502 Bad Gateway
    """

    pass
