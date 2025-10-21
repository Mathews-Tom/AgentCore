"""Integration tests for session commands with realistic workflows."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("agentcore_cli.commands.session.Config") as mock_config_class:
        config = MagicMock()
        config.api.url = "http://localhost:8001"
        config.api.timeout = 30
        config.api.retries = 3
        config.api.verify_ssl = True
        config.auth.type = "none"
        config.auth.token = None
        mock_config_class.load.return_value = config
        yield config


@pytest.fixture
def mock_client():
    """Mock AgentCore client."""
    with patch("agentcore_cli.commands.session.AgentCoreClient") as mock_client_class:
        client = MagicMock()
        mock_client_class.return_value = client
        yield client


class TestSessionWorkflows:
    """Integration tests for complete session workflows."""

    def test_save_list_info_workflow(self, mock_config, mock_client):
        """Test complete workflow: save → list → info."""
        # Step 1: Save session
        mock_client.call.return_value = {
            "session_id": "session-abc123",
            "name": "feature-dev",
            "status": "active",
        }

        save_result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
            "--description", "Building new feature",
            "--tag", "backend",
        ])

        assert save_result.exit_code == 0
        assert "Session saved: session-abc123" in save_result.stdout

        # Step 2: List sessions
        mock_client.call.return_value = {
            "sessions": [
                {
                    "session_id": "session-abc123",
                    "name": "feature-dev",
                    "status": "active",
                    "tasks_count": 5,
                    "created_at": "2025-10-21T10:00:00Z",
                }
            ]
        }

        list_result = runner.invoke(app, ["session", "list"])

        assert list_result.exit_code == 0
        assert "session-abc123" in list_result.stdout
        assert "feature-dev" in list_result.stdout

        # Step 3: Get session info
        mock_client.call.return_value = {
            "session_id": "session-abc123",
            "name": "feature-dev",
            "status": "active",
            "description": "Building new feature",
            "tags": ["backend"],
            "tasks_count": 5,
            "agents_count": 2,
            "created_at": "2025-10-21T10:00:00Z",
        }

        info_result = runner.invoke(app, ["session", "info", "session-abc123"])

        assert info_result.exit_code == 0
        assert "Session ID: session-abc123" in info_result.stdout
        assert "feature-dev" in info_result.stdout

    def test_save_export_workflow(self, mock_config, mock_client, tmp_path):
        """Test workflow: save → export."""
        # Step 1: Save session
        mock_client.call.return_value = {
            "session_id": "session-export-test",
            "name": "export-session",
            "status": "active",
        }

        save_result = runner.invoke(app, [
            "session", "save",
            "--name", "export-session",
        ])

        assert save_result.exit_code == 0

        # Step 2: Export session
        mock_client.call.return_value = {
            "session_id": "session-export-test",
            "name": "export-session",
            "status": "active",
            "tasks": [],
            "agents": [],
        }

        output_file = tmp_path / "session_export.json"
        export_result = runner.invoke(app, [
            "session", "export", "session-export-test",
            "--output", str(output_file),
            "--pretty",
        ])

        assert export_result.exit_code == 0
        assert output_file.exists()

        # Verify exported content
        import json
        with open(output_file) as f:
            data = json.load(f)
            assert data["session_id"] == "session-export-test"
            assert data["name"] == "export-session"

    def test_save_resume_workflow(self, mock_config, mock_client):
        """Test workflow: save → (work) → save again → resume."""
        # Step 1: Save initial session
        mock_client.call.return_value = {
            "session_id": "session-resume-test",
            "name": "resumable-session",
            "status": "active",
        }

        save_result1 = runner.invoke(app, [
            "session", "save",
            "--name", "resumable-session",
        ])

        assert save_result1.exit_code == 0

        # Step 2: Save again (snapshot)
        mock_client.call.return_value = {
            "session_id": "session-resume-test",
            "name": "resumable-session",
            "status": "active",
        }

        save_result2 = runner.invoke(app, [
            "session", "save",
            "--name", "resumable-session",
        ])

        assert save_result2.exit_code == 0

        # Step 3: Resume session
        mock_client.call.return_value = {
            "session_id": "session-resume-test",
            "name": "resumable-session",
            "tasks_count": 5,
            "agents_count": 2,
        }

        resume_result = runner.invoke(app, [
            "session", "resume", "session-resume-test"
        ])

        assert resume_result.exit_code == 0
        assert "Session resumed" in resume_result.stdout
        assert "Tasks restored: 5" in resume_result.stdout

    def test_save_delete_workflow_with_confirmation(self, mock_config, mock_client):
        """Test workflow: save → delete (with confirmation)."""
        # Step 1: Save session
        mock_client.call.return_value = {
            "session_id": "session-delete-test",
            "name": "deletable-session",
            "status": "active",
        }

        save_result = runner.invoke(app, [
            "session", "save",
            "--name", "deletable-session",
        ])

        assert save_result.exit_code == 0

        # Step 2: Delete with confirmation (user confirms)
        mock_client.call.side_effect = [
            # First call: get session info for confirmation
            {
                "session_id": "session-delete-test",
                "name": "deletable-session",
                "status": "active",
            },
            # Second call: actual delete
            {"success": True}
        ]

        delete_result = runner.invoke(app, [
            "session", "delete", "session-delete-test"
        ], input="y\n")

        assert delete_result.exit_code == 0
        assert "Session deleted" in delete_result.stdout

    def test_save_delete_workflow_declined(self, mock_config, mock_client):
        """Test workflow: save → delete (declined)."""
        # Save session
        mock_client.call.return_value = {
            "session_id": "session-decline-test",
            "name": "keep-session",
            "status": "active",
        }

        save_result = runner.invoke(app, [
            "session", "save",
            "--name", "keep-session",
        ])

        assert save_result.exit_code == 0

        # Try to delete but decline
        mock_client.call.return_value = {
            "session_id": "session-decline-test",
            "name": "keep-session",
            "status": "active",
        }

        delete_result = runner.invoke(app, [
            "session", "delete", "session-decline-test"
        ], input="n\n")

        assert delete_result.exit_code == 0
        assert "Operation cancelled" in delete_result.stdout


class TestSessionFilteringAndSorting:
    """Integration tests for session filtering and sorting."""

    def test_list_with_status_filter(self, mock_config, mock_client):
        """Test listing sessions with status filter."""
        mock_client.call.return_value = {
            "sessions": [
                {
                    "session_id": "session-1",
                    "name": "active-session",
                    "status": "active",
                    "tasks_count": 3,
                    "created_at": "2025-10-21T10:00:00Z",
                },
            ]
        }

        result = runner.invoke(app, [
            "session", "list",
            "--status", "active",
        ])

        assert result.exit_code == 0
        assert "active-session" in result.stdout

        # Verify API was called with filter
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args[0][1]
        assert call_args["status"] == "active"

    def test_list_with_limit(self, mock_config, mock_client):
        """Test listing sessions with limit."""
        mock_client.call.return_value = {
            "sessions": [
                {"session_id": f"session-{i}", "name": f"session-{i}", "status": "active", "tasks_count": i}
                for i in range(3)
            ]
        }

        result = runner.invoke(app, [
            "session", "list",
            "--limit", "3",
        ])

        assert result.exit_code == 0
        assert "session-0" in result.stdout
        assert "session-1" in result.stdout
        assert "session-2" in result.stdout

    def test_list_empty_results(self, mock_config, mock_client):
        """Test listing when no sessions exist."""
        mock_client.call.return_value = {"sessions": []}

        result = runner.invoke(app, ["session", "list"])

        assert result.exit_code == 0
        assert "No sessions found" in result.stdout


class TestSessionOutputFormats:
    """Integration tests for different output formats."""

    def test_save_json_output(self, mock_config, mock_client):
        """Test session save with JSON output."""
        mock_client.call.return_value = {
            "session_id": "session-json-test",
            "name": "json-session",
            "status": "active",
        }

        result = runner.invoke(app, [
            "session", "save",
            "--name", "json-session",
            "--json",
        ])

        assert result.exit_code == 0
        import json
        output = json.loads(result.stdout)
        assert output["session_id"] == "session-json-test"

    def test_list_json_output(self, mock_config, mock_client):
        """Test session list with JSON output."""
        mock_client.call.return_value = {
            "sessions": [
                {"session_id": "session-1", "name": "test", "status": "active", "tasks_count": 0},
            ]
        }

        result = runner.invoke(app, ["session", "list", "--json"])

        assert result.exit_code == 0
        import json
        output = json.loads(result.stdout)
        assert len(output) == 1

    def test_info_json_output(self, mock_config, mock_client):
        """Test session info with JSON output."""
        mock_client.call.return_value = {
            "session_id": "session-123",
            "name": "test-session",
            "status": "active",
            "tasks_count": 5,
        }

        result = runner.invoke(app, [
            "session", "info", "session-123", "--json"
        ])

        assert result.exit_code == 0
        import json
        output = json.loads(result.stdout)
        assert output["session_id"] == "session-123"


class TestSessionMetadata:
    """Integration tests for session metadata handling."""

    def test_save_with_complex_metadata(self, mock_config, mock_client):
        """Test saving session with complex metadata."""
        mock_client.call.return_value = {
            "session_id": "session-metadata-test",
            "name": "meta-session",
        }

        metadata = {
            "branch": "feature/auth",
            "sprint": "S1",
            "priority": "high",
            "assignee": "developer@example.com",
            "nested": {
                "key1": "value1",
                "key2": 42,
            }
        }

        result = runner.invoke(app, [
            "session", "save",
            "--name", "meta-session",
            "--metadata", f"{metadata}".replace("'", '"'),
        ])

        assert result.exit_code == 0

        # Verify metadata was passed to API
        call_args = mock_client.call.call_args[0][1]
        assert "metadata" in call_args

    def test_save_with_multiple_tags(self, mock_config, mock_client):
        """Test saving session with multiple tags."""
        mock_client.call.return_value = {
            "session_id": "session-tags-test",
            "name": "tagged-session",
        }

        result = runner.invoke(app, [
            "session", "save",
            "--name", "tagged-session",
            "--tag", "backend",
            "--tag", "auth",
            "--tag", "api",
        ])

        assert result.exit_code == 0

        # Verify tags were passed to API
        call_args = mock_client.call.call_args[0][1]
        assert call_args["tags"] == ["backend", "auth", "api"]


class TestSessionEdgeCases:
    """Integration tests for edge cases and error scenarios."""

    def test_resume_nonexistent_session(self, mock_config, mock_client):
        """Test resuming a session that doesn't exist."""
        from agentcore_cli.exceptions import JsonRpcError

        mock_client.call.side_effect = JsonRpcError({
            "code": -32602,
            "message": "Session not found"
        })

        result = runner.invoke(app, ["session", "resume", "nonexistent-session"])

        assert result.exit_code == 1
        assert "Session not found" in result.stdout

    def test_export_with_connection_error(self, mock_config, mock_client, tmp_path):
        """Test export with connection error."""
        from agentcore_cli.exceptions import ConnectionError as CliConnectionError

        mock_client.call.side_effect = CliConnectionError("Cannot connect")

        output_file = tmp_path / "export.json"
        result = runner.invoke(app, [
            "session", "export", "session-123",
            "--output", str(output_file),
        ])

        assert result.exit_code == 3
        assert "Cannot connect" in result.stdout
        assert not output_file.exists()

    def test_delete_force_without_confirmation(self, mock_config, mock_client):
        """Test deleting session with --force (no confirmation needed)."""
        mock_client.call.return_value = {"success": True}

        result = runner.invoke(app, [
            "session", "delete", "session-123", "--force"
        ])

        assert result.exit_code == 0
        assert "Session deleted" in result.stdout

        # Should only call delete API, not info API
        assert mock_client.call.call_count == 1


class TestSessionCrossFunctionality:
    """Integration tests for session functionality with other commands."""

    def test_session_with_task_tracking(self, mock_config, mock_client):
        """Test session info shows task information."""
        mock_client.call.return_value = {
            "session_id": "session-tasks",
            "name": "task-session",
            "status": "active",
            "tasks_count": 10,
            "tasks_completed": 7,
            "tasks_running": 2,
            "tasks_pending": 1,
        }

        result = runner.invoke(app, ["session", "info", "session-tasks"])

        assert result.exit_code == 0
        assert "Tasks: 10" in result.stdout

    def test_session_with_agent_tracking(self, mock_config, mock_client):
        """Test session info shows agent information."""
        mock_client.call.return_value = {
            "session_id": "session-agents",
            "name": "agent-session",
            "status": "active",
            "agents_count": 3,
            "tasks_count": 0,
        }

        result = runner.invoke(app, ["session", "info", "session-agents"])

        assert result.exit_code == 0
        assert "Agents: 3" in result.stdout
