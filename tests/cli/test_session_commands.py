"""Integration tests for session commands.

These tests verify that the session commands properly use the service layer
and send JSON-RPC 2.0 compliant requests to the API.
"""

from __future__ import annotations

from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app
from agentcore_cli.services.exceptions import (
    ValidationError,
    SessionNotFoundError,
    OperationError)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_session_service() -> Mock:
    """Create a mock session service."""
    return Mock()


class TestSessionCreateCommand:
    """Test suite for session create command."""

    def test_create_success(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test successful session creation."""
        # Mock service response
        mock_session_service.create.return_value = "session-001"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "create",
                    "--name",
                    "analysis-session",
                ])

        # Verify exit code
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify output
        assert "Session created successfully" in result.output
        assert "session-001" in result.output
        assert "analysis-session" in result.output

        # Verify service was called correctly
        mock_session_service.create.assert_called_once_with(
            name="analysis-session",
            description=None,
            context=None)

    def test_create_with_context(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session creation with context."""
        # Mock service response
        mock_session_service.create.return_value = "session-002"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "create",
                    "--name",
                    "test-session",
                    "--context",
                    '{"user": "alice", "project": "foo"}',
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Session created successfully" in result.output
        assert "session-002" in result.output

        # Verify service was called correctly
        mock_session_service.create.assert_called_once_with(
            name="test-session",
            description=None,
            context={"user": "alice", "project": "foo"})

    def test_create_json_output(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session creation with JSON output."""
        # Mock service response
        mock_session_service.create.return_value = "session-003"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "create",
                    "--name",
                    "my-session",
                    "--json",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output is valid JSON
        import json
        output = json.loads(result.output)
        assert output["session_id"] == "session-003"
        assert output["name"] == "my-session"

    def test_create_validation_error(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session creation with validation error."""
        # Mock service to raise validation error
        mock_session_service.create.side_effect = ValidationError(
            "Session name cannot be empty"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "create",
                    "--name",
                    "",
                ])

        # Verify exit code (2 = validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output
        assert "Session name cannot be empty" in result.output

    def test_create_operation_error(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session creation with operation error."""
        # Mock service to raise operation error
        mock_session_service.create.side_effect = OperationError(
            "Session creation failed: API error"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "create",
                    "--name",
                    "test-session",
                ])

        # Verify exit code (1 = operation error)
        assert result.exit_code == 1

        # Verify error message
        assert "Operation failed" in result.output
        assert "API error" in result.output

    def test_create_invalid_json_context(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session creation with invalid JSON context."""
        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "create",
                    "--name",
                    "test-session",
                    "--context",
                    "invalid-json",
                ])

        # Verify exit code (2 = validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Invalid JSON in context" in result.output


class TestSessionListCommand:
    """Test suite for session list command."""

    def test_list_success(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test successful session listing."""
        # Mock service response
        mock_session_service.list_sessions.return_value = [
            {
                "session_id": "session-001",
                "name": "analysis-session",
                "state": "active",
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "session_id": "session-002",
                "name": "test-session",
                "state": "inactive",
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(app, ["session", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "session-001" in result.output
        assert "analysis-session" in result.output
        assert "session-002" in result.output
        assert "test-session" in result.output

        # Verify service was called correctly
        mock_session_service.list_sessions.assert_called_once_with(
            state=None,
            limit=100,
            offset=0)

    def test_list_with_state_filter(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session listing with state filter."""
        # Mock service response
        mock_session_service.list_sessions.return_value = [
            {
                "session_id": "session-001",
                "name": "active-session",
                "state": "active",
                "created_at": "2024-01-01T00:00:00Z",
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "list",
                    "--state",
                    "active",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with filter
        mock_session_service.list_sessions.assert_called_once_with(
            state="active",
            limit=100,
            offset=0)

    def test_list_with_limit(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session listing with limit."""
        # Mock service response
        mock_session_service.list_sessions.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "list",
                    "--limit",
                    "10",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with limit
        mock_session_service.list_sessions.assert_called_once_with(
            state=None,
            limit=10,
            offset=0)

    def test_list_with_pagination(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session listing with pagination."""
        # Mock service response
        mock_session_service.list_sessions.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "list",
                    "--limit",
                    "10",
                    "--offset",
                    "20",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with pagination
        mock_session_service.list_sessions.assert_called_once_with(
            state=None,
            limit=10,
            offset=20)

    def test_list_json_output(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session listing with JSON output."""
        # Mock service response
        sessions = [
            {
                "session_id": "session-001",
                "name": "test-session",
                "state": "active",
            }
        ]
        mock_session_service.list_sessions.return_value = sessions

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "list",
                    "--json",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output is valid JSON
        import json
        output = json.loads(result.output)
        assert len(output) == 1
        assert output[0]["session_id"] == "session-001"

    def test_list_empty(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session listing with no sessions."""
        # Mock service response
        mock_session_service.list_sessions.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(app, ["session", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "No sessions found" in result.output

    def test_list_validation_error(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session listing with validation error."""
        # Mock service to raise validation error
        mock_session_service.list_sessions.side_effect = ValidationError(
            "Limit must be positive"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "list",
                ])

        # Verify exit code (2 = validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output


class TestSessionInfoCommand:
    """Test suite for session info command."""

    def test_info_success(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test successful session info retrieval."""
        # Mock service response
        mock_session_service.get.return_value = {
            "session_id": "session-001",
            "name": "analysis-session",
            "state": "active",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "context": {"user": "alice"},
        }

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "info",
                    "session-001",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Session Information" in result.output
        assert "session-001" in result.output
        assert "analysis-session" in result.output
        assert "active" in result.output

        # Verify service was called correctly
        mock_session_service.get.assert_called_once_with("session-001")

    def test_info_json_output(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session info with JSON output."""
        # Mock service response
        session_data = {
            "session_id": "session-001",
            "name": "test-session",
            "state": "active",
        }
        mock_session_service.get.return_value = session_data

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "info",
                    "session-001",
                    "--json",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output is valid JSON
        import json
        output = json.loads(result.output)
        assert output["session_id"] == "session-001"
        assert output["name"] == "test-session"

    def test_info_not_found(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session info for non-existent session."""
        # Mock service to raise not found error
        mock_session_service.get.side_effect = SessionNotFoundError(
            "Session 'session-999' not found"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "info",
                    "session-999",
                ])

        # Verify exit code (1 = error)
        assert result.exit_code == 1

        # Verify error message
        assert "Session not found" in result.output
        assert "session-999" in result.output

    def test_info_validation_error(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session info with validation error."""
        # Mock service to raise validation error
        mock_session_service.get.side_effect = ValidationError(
            "Session ID cannot be empty"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "info",
                    "",
                ])

        # Verify exit code (2 = validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output


class TestSessionDeleteCommand:
    """Test suite for session delete command."""

    def test_delete_success(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test successful session deletion."""
        # Mock service response
        mock_session_service.delete.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "delete",
                    "session-001",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Session deleted successfully" in result.output
        assert "session-001" in result.output

        # Verify service was called correctly (uses hard_delete, not force)
        mock_session_service.delete.assert_called_once_with(
            "session-001",
            hard_delete=False)

    def test_delete_with_hard(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session deletion with hard flag."""
        # Mock service response
        mock_session_service.delete.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "delete",
                    "session-001",
                    "--hard",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with hard_delete=True
        mock_session_service.delete.assert_called_once_with(
            "session-001",
            hard_delete=True)

    def test_delete_json_output(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session deletion with JSON output."""
        # Mock service response
        mock_session_service.delete.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "delete",
                    "session-001",
                    "--json",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output is valid JSON
        import json
        output = json.loads(result.output)
        assert output["success"] is True
        assert output["session_id"] == "session-001"

    def test_delete_not_found(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session deletion for non-existent session."""
        # Mock service to raise not found error
        mock_session_service.delete.side_effect = SessionNotFoundError(
            "Session 'session-999' not found"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "delete",
                    "session-999",
                ])

        # Verify exit code (1 = error)
        assert result.exit_code == 1

        # Verify error message
        assert "Session not found" in result.output
        assert "session-999" in result.output

    def test_delete_failed(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session deletion failure."""
        # Mock service to return False
        mock_session_service.delete.return_value = False

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "delete",
                    "session-001",
                ])

        # Verify exit code (1 = error)
        assert result.exit_code == 1

        # Verify error message
        assert "Failed to delete session" in result.output


class TestSessionResumeCommand:
    """Test suite for session resume command."""

    def test_resume_success(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test successful session resume."""
        # Mock service response
        mock_session_service.resume.return_value = {
            "success": True,
            "session_id": "session-001",
            "message": "Session resumed successfully",
        }

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "resume",
                    "session-001",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Session resumed successfully" in result.output
        assert "session-001" in result.output

        # Verify service was called correctly
        mock_session_service.resume.assert_called_once_with("session-001")

    def test_resume_json_output(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session resume with JSON output."""
        # Mock service response
        result_data = {
            "success": True,
            "session_id": "session-001",
            "message": "Session resumed",
        }
        mock_session_service.resume.return_value = result_data

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "resume",
                    "session-001",
                    "--json",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output is valid JSON
        import json
        output = json.loads(result.output)
        assert output["success"] is True
        assert output["session_id"] == "session-001"

    def test_resume_not_found(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session resume for non-existent session."""
        # Mock service to raise not found error
        mock_session_service.resume.side_effect = SessionNotFoundError(
            "Session 'session-999' not found"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "resume",
                    "session-999",
                ])

        # Verify exit code (1 = error)
        assert result.exit_code == 1

        # Verify error message
        assert "Session not found" in result.output
        assert "session-999" in result.output

    def test_resume_validation_error(
        self, runner: CliRunner, mock_session_service: Mock
    ) -> None:
        """Test session resume with validation error."""
        # Mock service to raise validation error
        mock_session_service.resume.side_effect = ValidationError(
            "Session ID cannot be empty"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.session.get_session_service",
            return_value=mock_session_service):
            result = runner.invoke(
                app,
                [
                    "session",
                    "resume",
                    "",
                ])

        # Verify exit code (2 = validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output
