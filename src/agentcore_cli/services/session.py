"""Session service for managing session state.

This service provides high-level operations for session management without
any knowledge of JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Any

from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.exceptions import (
    ValidationError,
    SessionNotFoundError,
    OperationError,
)


class SessionService:
    """Service for session operations.

    Provides business operations for session state management:
    - Session creation
    - Session listing and filtering
    - Session information retrieval
    - Session deletion
    - Session restoration

    This service abstracts JSON-RPC protocol details and focuses on
    business logic and domain validation.

    Args:
        client: JSON-RPC client for API communication

    Attributes:
        client: JSON-RPC client instance

    Example:
        >>> transport = HttpTransport("http://localhost:8001")
        >>> client = JsonRpcClient(transport)
        >>> service = SessionService(client)
        >>> session_id = service.create("analysis-session")
        >>> print(session_id)
        'session-001'
    """

    def __init__(self, client: JsonRpcClient) -> None:
        """Initialize session service.

        Args:
            client: JSON-RPC client for API communication
        """
        self.client = client

    def create(
        self,
        name: str,
        description: str | None = None,
        context: dict[str, Any] | None = None,
        owner_agent: str = "agentcore.cli",
    ) -> str:
        """Create a new session.

        Args:
            name: Session name
            description: Optional session description
            context: Optional session context (initial state)
            owner_agent: Agent that owns this session (default: "agentcore.cli")

        Returns:
            Session ID (string)

        Raises:
            ValidationError: If validation fails
            OperationError: If session creation fails

        Example:
            >>> session_id = service.create(
            ...     "analysis-session",
            ...     description="Code analysis session",
            ...     context={"user": "alice", "project": "foo"}
            ... )
            >>> print(session_id)
            'session-001'
        """
        # Business validation
        if not name or not name.strip():
            raise ValidationError("Session name cannot be empty")

        # Prepare parameters - API requires name and owner_agent
        params: dict[str, Any] = {
            "name": name.strip(),
            "owner_agent": owner_agent,
        }

        if description:
            params["description"] = description

        if context:
            params["initial_context"] = context

        # Call JSON-RPC method
        try:
            result = self.client.call("session.create", params)
        except Exception as e:
            raise OperationError(f"Session creation failed: {str(e)}")

        # Validate result
        session_id = result.get("session_id")
        if not session_id:
            raise OperationError("API did not return session_id")

        return str(session_id)

    def list_sessions(
        self,
        state: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List sessions with optional filtering.

        Args:
            state: Optional state filter ("active", "paused", "suspended", "completed", "failed", "expired")
            limit: Maximum number of sessions to return (default: 100)
            offset: Number of sessions to skip (default: 0)

        Returns:
            List of session dictionaries

        Raises:
            ValidationError: If parameters are invalid
            OperationError: If listing fails

        Example:
            >>> sessions = service.list_sessions(state="active", limit=10)
            >>> for session in sessions:
            ...     print(session["name"])
            'analysis-session'
            'test-session'
        """
        # Validation
        if limit <= 0:
            raise ValidationError("Limit must be positive")

        if offset < 0:
            raise ValidationError("Offset cannot be negative")

        valid_states = ["active", "paused", "suspended", "completed", "failed", "expired"]
        if state and state not in valid_states:
            raise ValidationError(
                f"Invalid state: {state}. Must be one of {valid_states}"
            )

        # Prepare parameters - API uses session.query method
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if state:
            params["state"] = state

        # Call JSON-RPC method - session.query instead of session.list
        try:
            result = self.client.call("session.query", params)
        except Exception as e:
            raise OperationError(f"Session listing failed: {str(e)}")

        # Extract sessions - API returns {sessions: [], total: int, has_more: bool}
        sessions = result.get("sessions", [])
        if not isinstance(sessions, list):
            raise OperationError("API returned invalid sessions list")

        return sessions

    def get(self, session_id: str) -> dict[str, Any]:
        """Get session information by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session information dictionary (SessionSnapshot)

        Raises:
            ValidationError: If session_id is empty
            SessionNotFoundError: If session does not exist
            OperationError: If retrieval fails

        Example:
            >>> info = service.get("session-001")
            >>> print(info["name"])
            'analysis-session'
        """
        # Validation
        if not session_id or not session_id.strip():
            raise ValidationError("Session ID cannot be empty")

        # Call JSON-RPC method
        try:
            result = self.client.call("session.get", {"session_id": session_id.strip()})
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            raise OperationError(f"Session retrieval failed: {str(e)}")

        # API returns session directly as SessionSnapshot (not wrapped)
        if not result or not isinstance(result, dict):
            raise OperationError("API did not return session information")

        return dict(result)

    def delete(self, session_id: str, hard_delete: bool = False) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier
            hard_delete: Permanent deletion vs soft delete (default: False)

        Returns:
            True if successful

        Raises:
            ValidationError: If session_id is empty
            SessionNotFoundError: If session does not exist
            OperationError: If deletion fails

        Example:
            >>> success = service.delete("session-001", hard_delete=True)
            >>> print(success)
            True
        """
        # Validation
        if not session_id or not session_id.strip():
            raise ValidationError("Session ID cannot be empty")

        # Prepare parameters - API uses hard_delete, not force
        params: dict[str, Any] = {
            "session_id": session_id.strip(),
            "hard_delete": hard_delete,
        }

        # Call JSON-RPC method
        try:
            result = self.client.call("session.delete", params)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            raise OperationError(f"Session deletion failed: {str(e)}")

        # Validate result
        success = result.get("success", False)
        return bool(success)

    def resume(self, session_id: str) -> dict[str, Any]:
        """Resume a paused or suspended session.

        Args:
            session_id: Session identifier

        Returns:
            Resume result with session_id and success status

        Raises:
            ValidationError: If session_id is empty
            SessionNotFoundError: If session does not exist
            OperationError: If resume fails

        Example:
            >>> result = service.resume("session-001")
            >>> print(result["success"])
            True
        """
        # Validation
        if not session_id or not session_id.strip():
            raise ValidationError("Session ID cannot be empty")

        # Call JSON-RPC method - API uses session.resume
        try:
            result = self.client.call("session.resume", {"session_id": session_id.strip()})
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            raise OperationError(f"Session resume failed: {str(e)}")

        # Validate result - API returns {success, session_id, message}
        if not result or not isinstance(result, dict):
            raise OperationError("API did not return resume result")

        return dict(result)
