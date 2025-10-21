"""Unit tests for session commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentcore_cli.exceptions import (
    AuthenticationError,
    ConnectionError as CliConnectionError,
    JsonRpcError,
)
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


class TestSessionSave:
    """Tests for session save command."""

    def test_save_success(self, mock_config, mock_client):
        """Test successful session save."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
            "status": "active",
        }

        result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
        ])

        assert result.exit_code == 0
        assert "Session saved: session-12345" in result.stdout
        assert "Name: feature-dev" in result.stdout
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args
        assert call_args[0][0] == "session.save"
        assert call_args[0][1]["name"] == "feature-dev"
        assert call_args[0][1]["description"] == ""
        assert call_args[0][1]["tags"] == []

    def test_save_with_all_options(self, mock_config, mock_client):
        """Test session save with all options."""
        mock_client.call.return_value = {"session_id": "session-12345"}

        result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
            "--description", "Building authentication",
            "--tag", "auth",
            "--tag", "backend",
            "--metadata", '{"branch": "main", "sprint": "S1"}',
        ])

        assert result.exit_code == 0
        call_args = mock_client.call.call_args[0][1]
        assert call_args["name"] == "feature-dev"
        assert call_args["description"] == "Building authentication"
        assert call_args["tags"] == ["auth", "backend"]
        assert call_args["metadata"] == {"branch": "main", "sprint": "S1"}

    def test_save_json_output(self, mock_config, mock_client):
        """Test session save with JSON output."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
        }

        result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
            "--json",
        ])

        assert result.exit_code == 0
        assert '"session_id": "session-12345"' in result.stdout

    def test_save_invalid_metadata_json(self, mock_config, mock_client):
        """Test session save with invalid metadata JSON."""
        result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
            "--metadata", "invalid-json",
        ])

        assert result.exit_code == 2
        assert "Invalid JSON in metadata" in result.stdout

    def test_save_metadata_not_dict(self, mock_config, mock_client):
        """Test session save with metadata not being a dict."""
        result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
            "--metadata", '["not", "a", "dict"]',
        ])

        assert result.exit_code == 2
        assert "Metadata must be a JSON object" in result.stdout

    def test_save_connection_error(self, mock_config, mock_client):
        """Test session save with connection error."""
        mock_client.call.side_effect = CliConnectionError("Cannot connect")

        result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
        ])

        assert result.exit_code == 3
        assert "Cannot connect" in result.stdout

    def test_save_authentication_error(self, mock_config, mock_client):
        """Test session save with authentication error."""
        mock_client.call.side_effect = AuthenticationError("Auth failed")

        result = runner.invoke(app, [
            "session", "save",
            "--name", "feature-dev",
        ])

        assert result.exit_code == 4
        assert "Auth failed" in result.stdout


class TestSessionResume:
    """Tests for session resume command."""

    def test_resume_success(self, mock_config, mock_client):
        """Test successful session resume."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
            "tasks_count": 5,
            "agents_count": 2,
        }

        result = runner.invoke(app, ["session", "resume", "session-12345"])

        assert result.exit_code == 0
        assert "Session resumed: session-12345" in result.stdout
        assert "Tasks restored: 5" in result.stdout
        assert "Agents restored: 2" in result.stdout
        mock_client.call.assert_called_once_with("session.resume", {
            "session_id": "session-12345"
        })

    def test_resume_json_output(self, mock_config, mock_client):
        """Test session resume with JSON output."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "tasks_count": 5,
            "agents_count": 2,
        }

        result = runner.invoke(app, [
            "session", "resume", "session-12345", "--json"
        ])

        assert result.exit_code == 0
        assert '"session_id": "session-12345"' in result.stdout

    def test_resume_not_found(self, mock_config, mock_client):
        """Test resume for non-existent session."""
        mock_client.call.side_effect = JsonRpcError({
            "code": -32602,
            "message": "Session not found"
        })

        result = runner.invoke(app, ["session", "resume", "session-99999"])

        assert result.exit_code == 1
        assert "Session not found" in result.stdout


class TestSessionList:
    """Tests for session list command."""

    def test_list_success(self, mock_config, mock_client):
        """Test successful session listing."""
        mock_client.call.return_value = {
            "sessions": [
                {
                    "session_id": "session-1",
                    "name": "feature-dev",
                    "status": "active",
                    "tasks_count": 5,
                    "created_at": "2025-10-21T10:00:00Z",
                },
                {
                    "session_id": "session-2",
                    "name": "bugfix",
                    "status": "completed",
                    "tasks_count": 3,
                    "created_at": "2025-10-21T09:00:00Z",
                },
            ]
        }

        result = runner.invoke(app, ["session", "list"])

        assert result.exit_code == 0
        assert "session-1" in result.stdout
        assert "session-2" in result.stdout
        assert "Total: 2 session(s)" in result.stdout
        mock_client.call.assert_called_once_with("session.list", {"limit": 100})

    def test_list_with_status_filter(self, mock_config, mock_client):
        """Test session listing with status filter."""
        mock_client.call.return_value = {"sessions": []}

        result = runner.invoke(app, [
            "session", "list",
            "--status", "active",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("session.list", {
            "limit": 100,
            "status": "active",
        })

    def test_list_with_limit(self, mock_config, mock_client):
        """Test session listing with custom limit."""
        mock_client.call.return_value = {"sessions": []}

        result = runner.invoke(app, [
            "session", "list",
            "--limit", "10",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("session.list", {"limit": 10})

    def test_list_empty(self, mock_config, mock_client):
        """Test listing when no sessions exist."""
        mock_client.call.return_value = {"sessions": []}

        result = runner.invoke(app, ["session", "list"])

        assert result.exit_code == 0
        assert "No sessions found" in result.stdout

    def test_list_json_output(self, mock_config, mock_client):
        """Test session listing with JSON output."""
        mock_client.call.return_value = {
            "sessions": [{"session_id": "session-1", "name": "test"}]
        }

        result = runner.invoke(app, ["session", "list", "--json"])

        assert result.exit_code == 0
        assert '"session_id": "session-1"' in result.stdout


class TestSessionInfo:
    """Tests for session info command."""

    def test_info_success(self, mock_config, mock_client):
        """Test successful session info retrieval."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
            "status": "active",
            "description": "Building authentication",
            "tags": ["auth", "backend"],
            "tasks_count": 5,
            "agents_count": 2,
            "created_at": "2025-10-21T10:00:00Z",
            "metadata": {"branch": "main"},
        }

        result = runner.invoke(app, ["session", "info", "session-12345"])

        assert result.exit_code == 0
        assert "Session ID: session-12345" in result.stdout
        assert "Name: feature-dev" in result.stdout
        assert "Status:" in result.stdout
        assert "Tasks: 5" in result.stdout
        assert "Agents: 2" in result.stdout
        mock_client.call.assert_called_once_with("session.info", {
            "session_id": "session-12345"
        })

    def test_info_json_output(self, mock_config, mock_client):
        """Test session info with JSON output."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
        }

        result = runner.invoke(app, [
            "session", "info", "session-12345", "--json"
        ])

        assert result.exit_code == 0
        assert '"session_id": "session-12345"' in result.stdout

    def test_info_not_found(self, mock_config, mock_client):
        """Test info for non-existent session."""
        mock_client.call.side_effect = JsonRpcError({
            "code": -32602,
            "message": "Session not found"
        })

        result = runner.invoke(app, ["session", "info", "session-99999"])

        assert result.exit_code == 1
        assert "Session not found" in result.stdout


class TestSessionDelete:
    """Tests for session delete command."""

    def test_delete_with_force(self, mock_config, mock_client):
        """Test session deletion with force flag."""
        mock_client.call.return_value = {"success": True}

        result = runner.invoke(app, [
            "session", "delete", "session-12345", "--force"
        ])

        assert result.exit_code == 0
        assert "Session deleted: session-12345" in result.stdout
        mock_client.call.assert_called_once_with("session.delete", {
            "session_id": "session-12345"
        })

    def test_delete_with_confirmation(self, mock_config, mock_client):
        """Test session deletion with confirmation prompt."""
        # Mock session info call, then delete call
        mock_client.call.side_effect = [
            {"session_id": "session-12345", "name": "feature-dev", "status": "active"},
            {"success": True}
        ]

        result = runner.invoke(app, [
            "session", "delete", "session-12345"
        ], input="y\n")

        assert result.exit_code == 0
        assert "Session deleted: session-12345" in result.stdout
        assert mock_client.call.call_count == 2

    def test_delete_declined(self, mock_config, mock_client):
        """Test session deletion declined by user."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
            "status": "active"
        }

        result = runner.invoke(app, [
            "session", "delete", "session-12345"
        ], input="n\n")

        assert result.exit_code == 0
        assert "Operation cancelled" in result.stdout

    def test_delete_json_output(self, mock_config, mock_client):
        """Test session deletion with JSON output."""
        mock_client.call.return_value = {"success": True}

        result = runner.invoke(app, [
            "session", "delete", "session-12345", "--force", "--json"
        ])

        assert result.exit_code == 0
        assert '"success": true' in result.stdout


class TestSessionExport:
    """Tests for session export command."""

    def test_export_success(self, mock_config, mock_client, tmp_path):
        """Test successful session export."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
            "tasks": [],
        }

        output_file = tmp_path / "session.json"
        result = runner.invoke(app, [
            "session", "export", "session-12345",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert "Session exported to:" in result.stdout
        assert output_file.exists()

        # Verify file contents
        import json
        with open(output_file) as f:
            data = json.load(f)
            assert data["session_id"] == "session-12345"

    def test_export_pretty_json(self, mock_config, mock_client, tmp_path):
        """Test session export with pretty-printing (default)."""
        mock_client.call.return_value = {
            "session_id": "session-12345",
            "name": "feature-dev",
        }

        output_file = tmp_path / "session.json"
        result = runner.invoke(app, [
            "session", "export", "session-12345",
            "--output", str(output_file),
            "--pretty",
        ])

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify file has pretty formatting (indentation)
        import json
        with open(output_file) as f:
            data = json.load(f)
            # Verify it's valid JSON and has expected content
            assert data["session_id"] == "session-12345"

        # Check file has indentation
        with open(output_file) as f:
            content = f.read()
            assert "  " in content  # Has indentation

    def test_export_not_found(self, mock_config, mock_client, tmp_path):
        """Test export for non-existent session."""
        mock_client.call.side_effect = JsonRpcError({
            "code": -32602,
            "message": "Session not found"
        })

        output_file = tmp_path / "session.json"
        result = runner.invoke(app, [
            "session", "export", "session-99999",
            "--output", str(output_file),
        ])

        assert result.exit_code == 1
        assert "Session not found" in result.stdout
        assert not output_file.exists()

    def test_export_connection_error(self, mock_config, mock_client, tmp_path):
        """Test export with connection error."""
        mock_client.call.side_effect = CliConnectionError("Cannot connect")

        output_file = tmp_path / "session.json"
        result = runner.invoke(app, [
            "session", "export", "session-12345",
            "--output", str(output_file),
        ])

        assert result.exit_code == 3
        assert "Cannot connect" in result.stdout
