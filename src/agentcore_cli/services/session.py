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
        context: dict[str, Any] | None = None,
    ) -> str:
        """Create a new session.

        Args:
            name: Session name
            context: Optional session context (initial state)

        Returns:
            Session ID (string)

        Raises:
            ValidationError: If validation fails
            OperationError: If session creation fails

        Example:
            >>> session_id = service.create(
            ...     "analysis-session",
            ...     context={"user": "alice", "project": "foo"}
            ... )
            >>> print(session_id)
            'session-001'
        """
        # Business validation
        if not name or not name.strip():
            raise ValidationError("Session name cannot be empty")

        # Prepare parameters
        params: dict[str, Any] = {
            "name": name.strip(),
        }

        if context:
            params["context"] = context

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
            state: Optional state filter ("active", "inactive", "archived")
            limit: Maximum number of sessions to return (default: 100)
            offset: Number of sessions to skip (default: 0)

        Returns:
            List of session dictionaries

        Raises:
            ValidationError: If parameters are invalid
            OperationError: If listing fails

        Example:
            >>> sessions = service.list(state="active", limit=10)
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

        valid_states = ["active", "inactive", "archived"]
        if state and state not in valid_states:
            raise ValidationError(
                f"Invalid state: {state}. Must be one of {valid_states}"
            )

        # Prepare parameters
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if state:
            params["state"] = state

        # Call JSON-RPC method
        try:
            result = self.client.call("session.list", params)
        except Exception as e:
            raise OperationError(f"Session listing failed: {str(e)}")

        # Extract sessions
        sessions = result.get("sessions", [])
        if not isinstance(sessions, list):
            raise OperationError("API returned invalid sessions list")

        return sessions

    def get(self, session_id: str) -> dict[str, Any]:
        """Get session information by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session information dictionary

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

        # Validate result
        session = result.get("session")
        if not session:
            raise OperationError("API did not return session information")

        return dict(session)

    def delete(self, session_id: str, force: bool = False) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier
            force: Force deletion even if session is active (default: False)

        Returns:
            True if successful

        Raises:
            ValidationError: If session_id is empty
            SessionNotFoundError: If session does not exist
            OperationError: If deletion fails

        Example:
            >>> success = service.delete("session-001", force=True)
            >>> print(success)
            True
        """
        # Validation
        if not session_id or not session_id.strip():
            raise ValidationError("Session ID cannot be empty")

        # Prepare parameters
        params: dict[str, Any] = {
            "session_id": session_id.strip(),
            "force": force,
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

    def restore(self, session_id: str) -> dict[str, Any]:
        """Restore a session.

        Args:
            session_id: Session identifier

        Returns:
            Restored session context

        Raises:
            ValidationError: If session_id is empty
            SessionNotFoundError: If session does not exist
            OperationError: If restoration fails

        Example:
            >>> context = service.restore("session-001")
            >>> print(context["user"])
            'alice'
        """
        # Validation
        if not session_id or not session_id.strip():
            raise ValidationError("Session ID cannot be empty")

        # Call JSON-RPC method
        try:
            result = self.client.call("session.restore", {"session_id": session_id.strip()})
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise SessionNotFoundError(f"Session '{session_id}' not found")
            raise OperationError(f"Session restoration failed: {str(e)}")

        # Validate result
        context = result.get("context")
        if context is None:
            raise OperationError("API did not return session context")

        return dict(context) if isinstance(context, dict) else {}
