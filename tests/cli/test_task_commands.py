"""Unit tests for task commands."""

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
    with patch("agentcore_cli.commands.task.Config") as mock_config_class:
        config = MagicMock()
        config.api.url = "http://localhost:8001"
        config.api.timeout = 30
        config.api.retries = 3
        config.api.verify_ssl = True
        config.auth.type = "none"
        config.auth.token = None
        config.defaults.task.priority = "medium"
        config.defaults.task.timeout = 3600
        config.defaults.task.requirements = {}
        mock_config_class.load.return_value = config
        yield config


@pytest.fixture
def mock_client():
    """Mock AgentCore client."""
    with patch("agentcore_cli.commands.task.AgentCoreClient") as mock_client_class:
        client = MagicMock()
        mock_client_class.return_value = client
        yield client


class TestTaskCreate:
    """Tests for task create command."""

    def test_create_success(self, mock_config, mock_client):
        """Test successful task creation."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "type": "code-review",
            "status": "pending",
        }

        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
        ])

        assert result.exit_code == 0
        assert "Task created: task-12345" in result.stdout
        assert "Type: code-review" in result.stdout
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args
        assert call_args[0][0] == "task.create"
        assert call_args[0][1]["type"] == "code-review"
        assert call_args[0][1]["priority"] == "medium"

    def test_create_with_all_options(self, mock_config, mock_client):
        """Test task creation with all options."""
        mock_client.call.return_value = {"task_id": "task-12345"}

        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
            "--input", "src/**/*.py",
            "--requirements", '{"language": "python"}',
            "--priority", "high",
            "--timeout", "7200",
        ])

        assert result.exit_code == 0
        call_args = mock_client.call.call_args[0][1]
        assert call_args["type"] == "code-review"
        assert call_args["input"] == "src/**/*.py"
        assert call_args["requirements"] == {"language": "python"}
        assert call_args["priority"] == "high"
        assert call_args["timeout"] == 7200

    def test_create_json_output(self, mock_config, mock_client):
        """Test task creation with JSON output."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "type": "code-review",
        }

        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
            "--json",
        ])

        assert result.exit_code == 0
        assert '"task_id": "task-12345"' in result.stdout

    def test_create_invalid_priority(self, mock_config, mock_client):
        """Test task creation with invalid priority."""
        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
            "--priority", "invalid",
        ])

        assert result.exit_code == 2
        assert "Invalid priority" in result.stdout

    def test_create_invalid_requirements_json(self, mock_config, mock_client):
        """Test task creation with invalid requirements JSON."""
        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
            "--requirements", "invalid-json",
        ])

        assert result.exit_code == 2
        assert "Invalid JSON in requirements" in result.stdout

    def test_create_requirements_not_dict(self, mock_config, mock_client):
        """Test task creation with requirements not being a dict."""
        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
            "--requirements", '["not", "a", "dict"]',
        ])

        assert result.exit_code == 2
        assert "Requirements must be a JSON object" in result.stdout

    def test_create_connection_error(self, mock_config, mock_client):
        """Test task creation with connection error."""
        mock_client.call.side_effect = CliConnectionError("Cannot connect")

        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
        ])

        assert result.exit_code == 3
        assert "Cannot connect" in result.stdout

    def test_create_authentication_error(self, mock_config, mock_client):
        """Test task creation with authentication error."""
        mock_client.call.side_effect = AuthenticationError("Auth failed")

        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
        ])

        assert result.exit_code == 4
        assert "Auth failed" in result.stdout


class TestTaskStatus:
    """Tests for task status command."""

    def test_status_success(self, mock_config, mock_client):
        """Test successful task status retrieval."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "type": "code-review",
            "status": "running",
            "priority": "high",
            "progress": 50,
            "created_at": "2025-10-21T10:00:00Z",
        }

        result = runner.invoke(app, ["task", "status", "task-12345"])

        assert result.exit_code == 0
        assert "Task ID: task-12345" in result.stdout
        assert "Type: code-review" in result.stdout
        mock_client.call.assert_called_once_with("task.status", {
            "task_id": "task-12345"
        })

    def test_status_json_output(self, mock_config, mock_client):
        """Test task status with JSON output."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "status": "running",
        }

        result = runner.invoke(app, [
            "task", "status", "task-12345", "--json"
        ])

        assert result.exit_code == 0
        assert '"task_id": "task-12345"' in result.stdout

    def test_status_not_found(self, mock_config, mock_client):
        """Test status for non-existent task."""
        mock_client.call.side_effect = JsonRpcError({
            "code": -32602,
            "message": "Task not found"
        })

        result = runner.invoke(app, ["task", "status", "task-99999"])

        assert result.exit_code == 1
        assert "Task not found" in result.stdout

    def test_status_watch_mode_completed(self, mock_config, mock_client):
        """Test watch mode when task completes."""
        # Return completed status immediately
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "status": "completed",
            "type": "code-review",
        }

        result = runner.invoke(app, [
            "task", "status", "task-12345", "--watch"
        ])

        assert result.exit_code == 0
        assert "Task completed successfully" in result.stdout


class TestTaskList:
    """Tests for task list command."""

    def test_list_success(self, mock_config, mock_client):
        """Test successful task listing."""
        mock_client.call.return_value = {
            "tasks": [
                {
                    "task_id": "task-1",
                    "type": "code-review",
                    "status": "running",
                    "priority": "high",
                    "created_at": "2025-10-21T10:00:00Z",
                },
                {
                    "task_id": "task-2",
                    "type": "testing",
                    "status": "completed",
                    "priority": "medium",
                    "created_at": "2025-10-21T09:00:00Z",
                },
            ]
        }

        result = runner.invoke(app, ["task", "list"])

        assert result.exit_code == 0
        assert "task-1" in result.stdout
        assert "task-2" in result.stdout
        assert "Total: 2 task(s)" in result.stdout
        mock_client.call.assert_called_once_with("task.list", {"limit": 100})

    def test_list_with_status_filter(self, mock_config, mock_client):
        """Test task listing with status filter."""
        mock_client.call.return_value = {"tasks": []}

        result = runner.invoke(app, [
            "task", "list",
            "--status", "running",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("task.list", {
            "limit": 100,
            "status": "running",
        })

    def test_list_with_limit(self, mock_config, mock_client):
        """Test task listing with custom limit."""
        mock_client.call.return_value = {"tasks": []}

        result = runner.invoke(app, [
            "task", "list",
            "--limit", "10",
        ])

        assert result.exit_code == 0
        mock_client.call.assert_called_once_with("task.list", {"limit": 10})

    def test_list_empty(self, mock_config, mock_client):
        """Test listing when no tasks exist."""
        mock_client.call.return_value = {"tasks": []}

        result = runner.invoke(app, ["task", "list"])

        assert result.exit_code == 0
        assert "No tasks found" in result.stdout

    def test_list_json_output(self, mock_config, mock_client):
        """Test task listing with JSON output."""
        mock_client.call.return_value = {
            "tasks": [{"task_id": "task-1", "type": "test"}]
        }

        result = runner.invoke(app, ["task", "list", "--json"])

        assert result.exit_code == 0
        assert '"task_id": "task-1"' in result.stdout


class TestTaskCancel:
    """Tests for task cancel command."""

    def test_cancel_with_force(self, mock_config, mock_client):
        """Test task cancellation with force flag."""
        mock_client.call.return_value = {"success": True}

        result = runner.invoke(app, [
            "task", "cancel", "task-12345", "--force"
        ])

        assert result.exit_code == 0
        assert "Task cancelled: task-12345" in result.stdout
        mock_client.call.assert_called_once_with("task.cancel", {
            "task_id": "task-12345"
        })

    def test_cancel_with_confirmation(self, mock_config, mock_client):
        """Test task cancellation with confirmation prompt."""
        # Mock task status call, then cancel call
        mock_client.call.side_effect = [
            {"task_id": "task-12345", "type": "code-review", "status": "running"},
            {"success": True}
        ]

        result = runner.invoke(app, [
            "task", "cancel", "task-12345"
        ], input="y\n")

        assert result.exit_code == 0
        assert "Task cancelled: task-12345" in result.stdout
        assert mock_client.call.call_count == 2

    def test_cancel_declined(self, mock_config, mock_client):
        """Test task cancellation declined by user."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "type": "code-review",
            "status": "running"
        }

        result = runner.invoke(app, [
            "task", "cancel", "task-12345"
        ], input="n\n")

        assert result.exit_code == 0
        assert "Operation cancelled" in result.stdout

    def test_cancel_json_output(self, mock_config, mock_client):
        """Test task cancellation with JSON output."""
        mock_client.call.return_value = {"success": True}

        result = runner.invoke(app, [
            "task", "cancel", "task-12345", "--force", "--json"
        ])

        assert result.exit_code == 0
        assert '"success": true' in result.stdout


class TestTaskResult:
    """Tests for task result command."""

    def test_result_success(self, mock_config, mock_client):
        """Test successful task result retrieval."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "status": "completed",
            "output": {"result": "success"},
            "artifacts": [
                {
                    "name": "report.pdf",
                    "type": "pdf",
                    "size": 1024,
                }
            ],
        }

        result = runner.invoke(app, ["task", "result", "task-12345"])

        assert result.exit_code == 0
        assert "Task ID: task-12345" in result.stdout
        assert "Output:" in result.stdout
        assert "Artifacts:" in result.stdout
        mock_client.call.assert_called_once_with("task.result", {
            "task_id": "task-12345"
        })

    def test_result_json_output(self, mock_config, mock_client):
        """Test task result with JSON output."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "status": "completed",
            "output": {"result": "success"},
        }

        result = runner.invoke(app, [
            "task", "result", "task-12345", "--json"
        ])

        assert result.exit_code == 0
        assert '"task_id": "task-12345"' in result.stdout

    def test_result_save_to_file(self, mock_config, mock_client, tmp_path):
        """Test saving result to file."""
        mock_client.call.return_value = {
            "task_id": "task-12345",
            "status": "completed",
            "output": {"result": "success"},
        }

        output_file = tmp_path / "result.json"
        result = runner.invoke(app, [
            "task", "result", "task-12345",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert "Result saved to:" in result.stdout
        assert output_file.exists()

        # Verify file contents
        import json
        with open(output_file) as f:
            data = json.load(f)
            assert data["task_id"] == "task-12345"


class TestTaskRetry:
    """Tests for task retry command."""

    def test_retry_success(self, mock_config, mock_client):
        """Test successful task retry."""
        mock_client.call.return_value = {
            "task_id": "task-67890",
            "original_task_id": "task-12345",
        }

        result = runner.invoke(app, ["task", "retry", "task-12345"])

        assert result.exit_code == 0
        assert "Task retried: task-67890" in result.stdout
        assert "Original task: task-12345" in result.stdout
        mock_client.call.assert_called_once_with("task.retry", {
            "task_id": "task-12345"
        })

    def test_retry_json_output(self, mock_config, mock_client):
        """Test task retry with JSON output."""
        mock_client.call.return_value = {
            "task_id": "task-67890",
        }

        result = runner.invoke(app, [
            "task", "retry", "task-12345", "--json"
        ])

        assert result.exit_code == 0
        assert '"task_id": "task-67890"' in result.stdout

    def test_retry_not_found(self, mock_config, mock_client):
        """Test retry for non-existent task."""
        mock_client.call.side_effect = JsonRpcError({
            "code": -32602,
            "message": "Task not found"
        })

        result = runner.invoke(app, ["task", "retry", "task-99999"])

        assert result.exit_code == 1
        assert "Task not found" in result.stdout

    def test_retry_connection_error(self, mock_config, mock_client):
        """Test retry with connection error."""
        mock_client.call.side_effect = CliConnectionError("Cannot connect")

        result = runner.invoke(app, ["task", "retry", "task-12345"])

        assert result.exit_code == 3
        assert "Cannot connect" in result.stdout
