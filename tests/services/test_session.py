"""Unit tests for SessionService.

Tests cover:
- Business validation
- Parameter transformation
- JSON-RPC method calls
- Error handling
- Result validation
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from agentcore_cli.services.session import SessionService
from agentcore_cli.services.exceptions import (
    ValidationError,
    SessionNotFoundError,
    OperationError)


class TestSessionServiceCreate:
    """Test SessionService.create() method."""

    def test_create_success(self) -> None:
        """Test successful session creation."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"session_id": "session-001"}
        service = SessionService(mock_client)

        # Act
        session_id = service.create("test-session")

        # Assert
        assert session_id == "session-001"
        mock_client.call.assert_called_once_with(
            "session.create",
            {"name": "test-session", "owner_agent": "agentcore.cli"})

    def test_create_with_context(self) -> None:
        """Test creation with context."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"session_id": "session-002"}
        service = SessionService(mock_client)

        # Act
        session_id = service.create(
            "test-session",
            context={"user": "alice", "project": "foo"})

        # Assert
        assert session_id == "session-002"
        mock_client.call.assert_called_once_with(
            "session.create",
            {
                "name": "test-session",
                "owner_agent": "agentcore.cli",
                "initial_context": {"user": "alice", "project": "foo"},
            })

    def test_create_with_description(self) -> None:
        """Test creation with description."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"session_id": "session-003"}
        service = SessionService(mock_client)

        # Act
        session_id = service.create(
            "test-session",
            description="My test session")

        # Assert
        assert session_id == "session-003"
        mock_client.call.assert_called_once_with(
            "session.create",
            {
                "name": "test-session",
                "owner_agent": "agentcore.cli",
                "description": "My test session",
            })

    def test_create_empty_name_raises_validation_error(self) -> None:
        """Test that empty name raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Session name cannot be empty"):
            service.create("")


class TestSessionServiceListSessions:
    """Test SessionService.list_sessions() method."""

    def test_list_success(self) -> None:
        """Test successful session listing."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "sessions": [
                {"session_id": "session-001", "name": "session-1"},
                {"session_id": "session-002", "name": "session-2"},
            ]
        }
        service = SessionService(mock_client)

        # Act
        sessions = service.list_sessions()

        # Assert
        assert len(sessions) == 2
        mock_client.call.assert_called_once_with(
            "session.query",
            {"limit": 100, "offset": 0})

    def test_list_with_state_filter(self) -> None:
        """Test listing with state filter."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"sessions": []}
        service = SessionService(mock_client)

        # Act
        service.list_sessions(state="active", limit=10)

        # Assert
        mock_client.call.assert_called_once_with(
            "session.query",
            {"limit": 10, "offset": 0, "state": "active"})

    def test_list_invalid_state_raises_validation_error(self) -> None:
        """Test that invalid state raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid state"):
            service.list_sessions(state="invalid")

    def test_list_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Session listing failed"):
            service.list_sessions()

    def test_list_invalid_response_raises_operation_error(self) -> None:
        """Test that invalid response raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"sessions": "not-a-list"}
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API returned invalid sessions list"):
            service.list_sessions()


class TestSessionServiceGet:
    """Test SessionService.get() method."""

    def test_get_success(self) -> None:
        """Test successful session retrieval."""
        # Arrange
        mock_client = Mock()
        # Service expects raw dict response (not wrapped in "session")
        mock_client.call.return_value = {
            "session_id": "session-001",
            "name": "test-session",
        }
        service = SessionService(mock_client)

        # Act
        session = service.get("session-001")

        # Assert
        assert session["session_id"] == "session-001"
        assert session["name"] == "test-session"

    def test_get_not_found_raises_session_not_found_error(self) -> None:
        """Test that 'not found' error raises SessionNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Session not found")
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(SessionNotFoundError, match="Session 'session-001' not found"):
            service.get("session-001")

    def test_get_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Session retrieval failed"):
            service.get("session-001")

    def test_get_missing_session_raises_operation_error(self) -> None:
        """Test that missing session raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {}
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API did not return session information"):
            service.get("session-001")


class TestSessionServiceDelete:
    """Test SessionService.delete() method."""

    def test_delete_success(self) -> None:
        """Test successful session deletion."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = SessionService(mock_client)

        # Act
        success = service.delete("session-001")

        # Assert
        assert success is True
        mock_client.call.assert_called_once_with(
            "session.delete",
            {"session_id": "session-001", "hard_delete": False})

    def test_delete_with_hard_delete(self) -> None:
        """Test deletion with hard_delete flag."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = SessionService(mock_client)

        # Act
        success = service.delete("session-001", hard_delete=True)

        # Assert
        assert success is True
        mock_client.call.assert_called_once_with(
            "session.delete",
            {"session_id": "session-001", "hard_delete": True})

    def test_delete_not_found_raises_session_not_found_error(self) -> None:
        """Test that 'not found' error raises SessionNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Session not found")
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(SessionNotFoundError, match="Session 'session-001' not found"):
            service.delete("session-001")

    def test_delete_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Session deletion failed"):
            service.delete("session-001")


class TestSessionServiceResume:
    """Test SessionService.resume() method."""

    def test_resume_success(self) -> None:
        """Test successful session resume."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "success": True,
            "session_id": "session-001",
            "message": "Session resumed",
        }
        service = SessionService(mock_client)

        # Act
        result = service.resume("session-001")

        # Assert
        assert result["success"] is True
        assert result["session_id"] == "session-001"
        mock_client.call.assert_called_once_with(
            "session.resume",
            {"session_id": "session-001"})

    def test_resume_not_found_raises_session_not_found_error(self) -> None:
        """Test that 'not found' error raises SessionNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Session not found")
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(SessionNotFoundError, match="Session 'session-001' not found"):
            service.resume("session-001")

    def test_resume_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Session resume failed"):
            service.resume("session-001")

    def test_resume_empty_session_id_raises_validation_error(self) -> None:
        """Test that empty session_id raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Session ID cannot be empty"):
            service.resume("")

    def test_resume_missing_result_raises_operation_error(self) -> None:
        """Test that missing result raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = None
        service = SessionService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API did not return resume result"):
            service.resume("session-001")
