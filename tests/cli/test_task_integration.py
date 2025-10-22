"""Integration tests for task commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("agentcore_cli.container.get_config") as mock_get_config:
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
        mock_get_config.return_value = config
        yield config


@pytest.fixture
def mock_client():
    """Mock AgentCore client."""
    with patch("agentcore_cli.container.get_jsonrpc_client") as mock_get_client:
        client = MagicMock()
        mock_get_client.return_value = client
        yield client


class TestTaskCreateIntegration:
    """Integration tests for task create command."""

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_create_full_workflow(self, mock_config, mock_client):
        """Test complete task creation workflow."""
        mock_client.call.return_value = {
            "task_id": "task-abc123",
            "type": "code-review",
            "status": "pending",
            "priority": "high",
            "created_at": "2025-10-21T10:00:00Z",
        }

        result = runner.invoke(app, [
            "task", "create",
            "--type", "code-review",
            "--input", "src/**/*.py",
            "--requirements", '{"language": "python", "framework": "fastapi"}',
            "--priority", "high",
            "--timeout", "7200",
        ])

        assert result.exit_code == 0
        assert "Task created: task-abc123" in result.stdout
        assert "Type: code-review" in result.stdout
        assert "Priority: high" in result.stdout

        # Verify API call
        call_args = mock_client.call.call_args[0][1]
        assert call_args["type"] == "code-review"
        assert call_args["input"] == "src/**/*.py"
        assert call_args["requirements"]["language"] == "python"
        assert call_args["priority"] == "high"
        assert call_args["timeout"] == 7200

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_create_with_complex_requirements(self, mock_config, mock_client):
        """Test task creation with complex requirements."""
        mock_client.call.return_value = {"task_id": "task-xyz789"}

        requirements = {
            "language": "python",
            "framework": "fastapi",
            "min_coverage": 90,
            "tools": ["pytest", "mypy", "ruff"],
            "exclude_patterns": ["tests/", "migrations/"],
        }

        result = runner.invoke(app, [
            "task", "create",
            "--type", "test-analysis",
            "--requirements", f"{requirements}".replace("'", '"'),
        ])

        assert result.exit_code == 0


class TestTaskStatusIntegration:
    """Integration tests for task status command."""

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_status_running_task(self, mock_config, mock_client):
        """Test status of running task."""
        mock_client.call.return_value = {
            "task_id": "task-123",
            "type": "code-review",
            "status": "running",
            "priority": "high",
            "progress": 65,
            "agent_id": "agent-456",
            "created_at": "2025-10-21T10:00:00Z",
            "started_at": "2025-10-21T10:01:00Z",
        }

        result = runner.invoke(app, ["task", "status", "task-123"])

        assert result.exit_code == 0
        assert "Task ID: task-123" in result.stdout
        assert "Type: code-review" in result.stdout
        assert "running" in result.stdout.lower()


class TestTaskListIntegration:
    """Integration tests for task list command."""

    def test_list_all_tasks(self, mock_config, mock_client):
        """Test listing all tasks."""
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
                {
                    "task_id": "task-3",
                    "type": "deployment",
                    "status": "failed",
                    "priority": "critical",
                    "created_at": "2025-10-21T08:00:00Z",
                },
            ]
        }

        result = runner.invoke(app, ["task", "list"])

        assert result.exit_code == 0
        assert "task-1" in result.stdout
        assert "task-2" in result.stdout
        assert "task-3" in result.stdout
        assert "Tasks (3)" in result.stdout

    def test_list_running_tasks(self, mock_config, mock_client):
        """Test listing only running tasks."""
        mock_client.call.return_value = {
            "tasks": [
                {
                    "task_id": "task-1",
                    "type": "code-review",
                    "status": "running",
                    "priority": "high",
                    "created_at": "2025-10-21T10:00:00Z",
                },
            ]
        }

        result = runner.invoke(app, [
            "task", "list",
            "--status", "running",
        ])

        assert result.exit_code == 0
        assert "task-1" in result.stdout
        assert "Tasks (1)" in result.stdout

    def test_list_with_json_output(self, mock_config, mock_client):
        """Test listing tasks with JSON output."""
        mock_client.call.return_value = {
            "tasks": [
                {"task_id": "task-1", "type": "test", "status": "completed"},
            ]
        }

        result = runner.invoke(app, ["task", "list", "--json"])

        assert result.exit_code == 0
        assert '"task_id": "task-1"' in result.stdout
        assert '"type": "test"' in result.stdout


class TestTaskWorkflow:
    """Integration tests for task workflows."""

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_create_status_result_workflow(self, mock_config, mock_client):
        """Test complete workflow: create -> status -> result."""
        # Step 1: Create task
        mock_client.call.return_value = {
            "task_id": "task-workflow-123",
            "status": "pending",
        }

        create_result = runner.invoke(app, [
            "task", "create",
            "--type", "integration-test",
        ])

        assert create_result.exit_code == 0
        assert "task-workflow-123" in create_result.stdout

        # Step 2: Check status
        mock_client.call.return_value = {
            "task_id": "task-workflow-123",
            "status": "completed",
            "type": "integration-test",
        }

        status_result = runner.invoke(app, [
            "task", "status", "task-workflow-123"
        ])

        assert status_result.exit_code == 0
        assert "completed" in status_result.stdout.lower()

        # Step 3: Get result
        mock_client.call.return_value = {
            "task_id": "task-workflow-123",
            "status": "completed",
            "output": {"test_results": "all passed"},
            "artifacts": [{"name": "report.xml", "type": "xml", "size": 2048}],
        }

        result_result = runner.invoke(app, [
            "task", "result", "task-workflow-123"
        ])

        assert result_result.exit_code == 0
        assert "Output:" in result_result.stdout
        assert "Artifacts:" in result_result.stdout

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_create_cancel_workflow(self, mock_config, mock_client):
        """Test create then cancel workflow."""
        # Create task
        mock_client.call.return_value = {"task_id": "task-cancel-test"}

        create_result = runner.invoke(app, [
            "task", "create",
            "--type", "long-running",
        ])

        assert create_result.exit_code == 0

        # Cancel task
        mock_client.call.return_value = {"success": True}

        cancel_result = runner.invoke(app, [
            "task", "cancel", "task-cancel-test", "--force"
        ])

        assert cancel_result.exit_code == 0
        assert "Task cancelled" in cancel_result.stdout

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_create_fail_retry_workflow(self, mock_config, mock_client):
        """Test create -> fail -> retry workflow."""
        # Create task
        mock_client.call.return_value = {"task_id": "task-retry-test"}

        create_result = runner.invoke(app, [
            "task", "create",
            "--type", "flaky-test",
        ])

        assert create_result.exit_code == 0

        # Retry task
        mock_client.call.return_value = {"task_id": "task-retry-test-2"}

        retry_result = runner.invoke(app, [
            "task", "retry", "task-retry-test"
        ])

        assert retry_result.exit_code == 0
        assert "Task retried" in retry_result.stdout
        assert "task-retry-test-2" in retry_result.stdout


class TestTaskEdgeCases:
    """Edge case tests for task commands."""

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_create_with_empty_input(self, mock_config, mock_client):
        """Test creating task with empty input."""
        mock_client.call.return_value = {"task_id": "task-empty-input"}

        result = runner.invoke(app, [
            "task", "create",
            "--type", "no-input-task",
            "--input", "",
        ])

        assert result.exit_code == 0

    def test_list_with_small_limit(self, mock_config, mock_client):
        """Test listing tasks with small limit."""
        mock_client.call.return_value = {
            "tasks": [
                {"task_id": f"task-{i}", "type": "test", "status": "completed"}
                for i in range(3)
            ]
        }

        result = runner.invoke(app, [
            "task", "list",
            "--limit", "3",
        ])

        assert result.exit_code == 0
        assert "Tasks (3)" in result.stdout

    @pytest.mark.skip(reason="Task command features not yet fully implemented")
    def test_result_save_to_relative_path(self, mock_config, mock_client, tmp_path):
        """Test saving result to relative path."""
        import os
        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            mock_client.call.return_value = {
                "task_id": "task-123",
                "status": "completed",
                "output": {"success": True},
            }

            result = runner.invoke(app, [
                "task", "result", "task-123",
                "--output", "result.json",
            ])

            assert result.exit_code == 0
            assert (tmp_path / "result.json").exists()
        finally:
            os.chdir(original_cwd)
