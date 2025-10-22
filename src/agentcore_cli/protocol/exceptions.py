"""Protocol layer exceptions.

These exceptions are raised by the protocol layer when JSON-RPC 2.0
specification violations or protocol-level errors occur.
"""

from __future__ import annotations

from typing import Any


class ProtocolError(Exception):
    """Base exception for protocol layer errors.

    Raised when JSON-RPC 2.0 protocol violations occur. This is the base
    class for all protocol-specific errors.

    Args:
        message: Human-readable error description
        code: JSON-RPC error code (optional)
        data: Additional error data (optional)

    Attributes:
        message: Error message
        code: JSON-RPC error code (or None)
        data: Additional error information (or None)
    """

    def __init__(
        self,
        message: str,
        code: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ProtocolError.

        Args:
            message: Human-readable error description
            code: JSON-RPC error code (optional)
            data: Additional error data (optional)
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data

    def __str__(self) -> str:
        """Return string representation of error.

        Returns:
            Formatted error message with code if present
        """
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class JsonRpcProtocolError(ProtocolError):
    """JSON-RPC protocol error returned by server.

    Raised when the server returns an error response. The error details
    are parsed from the JSON-RPC error object.

    Examples:
        - Invalid params
        - Method not found
        - Internal server error
    """

    pass


class InvalidRequestError(ProtocolError):
    """Invalid JSON-RPC request format.

    Raised when a request does not conform to JSON-RPC 2.0 specification.

    Error code: -32600

    Examples:
        - Missing required fields
        - Invalid field types
        - Malformed JSON
    """

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Initialize InvalidRequestError.

        Args:
            message: Error description
            data: Additional error data
        """
        super().__init__(message, code=-32600, data=data)


class MethodNotFoundError(ProtocolError):
    """Method does not exist on server.

    Raised when the requested method is not available.

    Error code: -32601

    Example:
        >>> raise MethodNotFoundError("agent.unknown")
    """

    def __init__(self, method: str, data: dict[str, Any] | None = None) -> None:
        """Initialize MethodNotFoundError.

        Args:
            method: Method name that was not found
            data: Additional error data
        """
        super().__init__(f"Method not found: {method}", code=-32601, data=data)
        self.method = method


class InvalidParamsError(ProtocolError):
    """Invalid method parameters.

    Raised when method parameters do not match expected schema.

    Error code: -32602

    Examples:
        - Missing required parameter
        - Invalid parameter type
        - Extra unexpected parameters
    """

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Initialize InvalidParamsError.

        Args:
            message: Error description
            data: Additional error data
        """
        super().__init__(message, code=-32602, data=data)


class InternalError(ProtocolError):
    """Internal JSON-RPC error.

    Raised when the server encounters an internal error during processing.

    Error code: -32603

    Example:
        >>> raise InternalError("Database connection failed")
    """

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Initialize InternalError.

        Args:
            message: Error description
            data: Additional error data
        """
        super().__init__(message, code=-32603, data=data)


class ParseError(ProtocolError):
    """Invalid JSON received.

    Raised when the server fails to parse JSON request.

    Error code: -32700

    Example:
        >>> raise ParseError("Unexpected token in JSON")
    """

    def __init__(self, message: str, data: dict[str, Any] | None = None) -> None:
        """Initialize ParseError.

        Args:
            message: Error description
            data: Additional error data
        """
        super().__init__(message, code=-32700, data=data)
