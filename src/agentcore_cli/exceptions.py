"""Custom exceptions for AgentCore CLI."""

from __future__ import annotations


class AgentCoreError(Exception):
    """Base exception for AgentCore CLI errors."""

    pass


class ConnectionError(AgentCoreError):
    """Error connecting to AgentCore API."""

    pass


class AuthenticationError(AgentCoreError):
    """Authentication failed."""

    pass


class ValidationError(AgentCoreError):
    """Invalid input or configuration."""

    pass


class JsonRpcError(AgentCoreError):
    """JSON-RPC protocol error."""

    def __init__(self, error_data: dict[str, object]) -> None:
        """Initialize with JSON-RPC error data."""
        code_value = error_data.get("code", -32603)
        if isinstance(code_value, int):
            self.code: int = code_value
        else:
            self.code = -32603

        message_value = error_data.get("message", "Unknown error")
        if isinstance(message_value, str):
            self.message: str = message_value
        else:
            self.message = "Unknown error"

        self.data: object = error_data.get("data")
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return user-friendly error message."""
        base_msg = f"JSON-RPC Error {self.code}: {self.message}"
        if self.data:
            base_msg += f"\nDetails: {self.data}"
        return base_msg


class TimeoutError(AgentCoreError):
    """Operation timed out."""

    pass


class ApiError(AgentCoreError):
    """API returned an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with HTTP status code and message."""
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")
