"""Service layer exceptions.

Domain-specific exceptions for business logic errors. These exceptions
are independent of JSON-RPC protocol errors and represent business rule
violations or domain-level failures.
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """Base exception for service layer errors.

    All service-level exceptions inherit from this base class.

    Attributes:
        message: Human-readable error message
        details: Additional error context

    Example:
        >>> raise ServiceError("Operation failed", details={"reason": "timeout"})
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize service error.

        Args:
            message: Error message
            details: Optional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(ServiceError):
    """Raised when parameter validation fails.

    Indicates that input parameters do not meet business requirements.

    Example:
        >>> raise ValidationError("At least one capability required")
    """

    pass


class AgentNotFoundError(ServiceError):
    """Raised when agent is not found.

    Example:
        >>> raise AgentNotFoundError("Agent 'agent-123' not found")
    """

    pass


class TaskNotFoundError(ServiceError):
    """Raised when task is not found.

    Example:
        >>> raise TaskNotFoundError("Task 'task-456' not found")
    """

    pass


class SessionNotFoundError(ServiceError):
    """Raised when session is not found.

    Example:
        >>> raise SessionNotFoundError("Session 'session-789' not found")
    """

    pass


class WorkflowNotFoundError(ServiceError):
    """Raised when workflow is not found.

    Example:
        >>> raise WorkflowNotFoundError("Workflow 'workflow-001' not found")
    """

    pass


class OperationError(ServiceError):
    """Raised when operation fails.

    Generic error for failed operations that don't fit other categories.

    Example:
        >>> raise OperationError("Failed to execute operation", details={"error": "timeout"})
    """

    pass
