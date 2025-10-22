"""Unit tests for workflow commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from typer.testing import CliRunner

from agentcore_cli.exceptions import AgentCoreError, AuthenticationError, ConnectionError
from agentcore_cli.main import app

runner = CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch("agentcore_cli.commands.workflow.Config.load") as mock:
        config = Mock()
        config.api.url = "http://localhost:8001"
        config.api.timeout = 30
        config.api.retries = 3
        config.api.verify_ssl = True
        config.auth.type = "none"
        config.auth.token = None
        mock.return_value = config
        yield mock


@pytest.fixture
def mock_client():
    """Mock AgentCore client."""
    with patch("agentcore_cli.commands.workflow.AgentCoreClient") as mock:
        client_instance = Mock()
        mock.return_value = client_instance
        yield client_instance


@pytest.fixture
def sample_workflow_yaml(tmp_path):
    """Create a sample workflow YAML file."""
    workflow_data = {
        "name": "test-workflow",
        "description": "Test workflow for unit testing",
        "version": "1.0",
        "tasks": [
            {
                "name": "task-1",
                "type": "test-type",
                "requirements": {"test": "requirement"},
            },
            {
                "name": "task-2",
                "type": "test-type",
                "depends_on": ["task-1"],
            },
        ],
        "max_retries": 3,
        "timeout": 3600,
    }

    workflow_file = tmp_path / "workflow.yaml"
    with open(workflow_file, "w") as f:
        yaml.dump(workflow_data, f)

    return workflow_file


class TestWorkflowCreate:
    """Tests for workflow create command."""

    def test_create_workflow_success(self, mock_config, mock_client, sample_workflow_yaml):
        """Test successful workflow creation."""
        # Setup mock response
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "name": "test-workflow",
            "tasks_count": 2,
        }

        # Run command
        result = runner.invoke(app, ["workflow", "create", "--file", str(sample_workflow_yaml)])

        # Assert success
        assert result.exit_code == 0
        assert "Workflow created: workflow-12345" in result.stdout
        assert "test-workflow" in result.stdout

        # Verify API call
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args
        assert call_args[0][0] == "workflow.create"
        params = call_args[0][1]
        assert params["name"] == "test-workflow"
        assert len(params["tasks"]) == 2

    def test_create_workflow_with_json_output(self, mock_config, mock_client, sample_workflow_yaml):
        """Test workflow creation with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "name": "test-workflow",
        }

        result = runner.invoke(app, ["workflow", "create", "--file", str(sample_workflow_yaml), "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["workflow_id"] == "workflow-12345"

    def test_create_workflow_validate_only(self, mock_config, mock_client, sample_workflow_yaml):
        """Test workflow validation without creation."""
        result = runner.invoke(app, ["workflow", "create", "--file", str(sample_workflow_yaml), "--validate-only"])

        assert result.exit_code == 0
        assert "Workflow definition is valid" in result.stdout
        # API should not be called
        mock_client.call.assert_not_called()

    def test_create_workflow_file_not_found(self, mock_config, mock_client):
        """Test workflow creation with missing file."""
        result = runner.invoke(app, ["workflow", "create", "--file", "nonexistent.yaml"])

        assert result.exit_code == 2
        assert "Workflow file not found" in result.stdout

    def test_create_workflow_invalid_yaml(self, mock_config, mock_client, tmp_path):
        """Test workflow creation with invalid YAML syntax."""
        invalid_yaml = tmp_path / "invalid.yaml"
        with open(invalid_yaml, "w") as f:
            f.write("invalid: yaml: syntax: [")

        result = runner.invoke(app, ["workflow", "create", "--file", str(invalid_yaml)])

        assert result.exit_code == 2
        assert "Invalid YAML syntax" in result.stdout

    def test_create_workflow_validation_error(self, mock_config, mock_client, tmp_path):
        """Test workflow creation with validation error."""
        invalid_workflow = tmp_path / "invalid_workflow.yaml"
        with open(invalid_workflow, "w") as f:
            yaml.dump({"name": "test"}, f)  # Missing required 'tasks' field

        result = runner.invoke(app, ["workflow", "create", "--file", str(invalid_workflow)])

        assert result.exit_code == 2
        assert "Workflow validation failed" in result.stdout

    def test_create_workflow_api_error(self, mock_config, mock_client, sample_workflow_yaml):
        """Test workflow creation with API error."""
        mock_client.call.side_effect = AgentCoreError("API error")

        result = runner.invoke(app, ["workflow", "create", "--file", str(sample_workflow_yaml)])

        assert result.exit_code == 1
        assert "API error" in result.stdout

    def test_create_workflow_auth_error(self, mock_config, mock_client, sample_workflow_yaml):
        """Test workflow creation with authentication error."""
        mock_client.call.side_effect = AuthenticationError("Auth failed")

        result = runner.invoke(app, ["workflow", "create", "--file", str(sample_workflow_yaml)])

        assert result.exit_code == 4

    def test_create_workflow_connection_error(self, mock_config, mock_client, sample_workflow_yaml):
        """Test workflow creation with connection error."""
        mock_client.call.side_effect = ConnectionError("Cannot connect")

        result = runner.invoke(app, ["workflow", "create", "--file", str(sample_workflow_yaml)])

        assert result.exit_code == 3


class TestWorkflowExecute:
    """Tests for workflow execute command."""

    def test_execute_workflow_success(self, mock_config, mock_client):
        """Test successful workflow execution."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "running",
        }

        result = runner.invoke(app, ["workflow", "execute", "workflow-12345"])

        assert result.exit_code == 0
        assert "Workflow execution started: workflow-12345" in result.stdout

        # Verify API call
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args
        assert call_args[0][0] == "workflow.execute"
        assert call_args[0][1]["workflow_id"] == "workflow-12345"

    def test_execute_workflow_with_json_output(self, mock_config, mock_client):
        """Test workflow execution with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "running",
        }

        result = runner.invoke(app, ["workflow", "execute", "workflow-12345", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["workflow_id"] == "workflow-12345"

    def test_execute_workflow_watch_mode(self, mock_config, mock_client):
        """Test workflow execution with watch mode."""
        # Mock status responses - workflow completes after 2 checks
        mock_client.call.side_effect = [
            {"workflow_id": "workflow-12345", "status": "running"},  # execute call
            {"workflow_id": "workflow-12345", "status": "running", "tasks_total": 2, "tasks_completed": 1},  # status 1
            {"workflow_id": "workflow-12345", "status": "completed", "tasks_total": 2, "tasks_completed": 2},  # status 2
        ]

        with patch("agentcore_cli.commands.workflow.time.sleep"):  # Speed up test
            result = runner.invoke(app, ["workflow", "execute", "workflow-12345", "--watch"])

        assert result.exit_code == 0
        assert "Workflow execution started" in result.stdout
        assert "Workflow completed successfully" in result.stdout

    def test_execute_workflow_api_error(self, mock_config, mock_client):
        """Test workflow execution with API error."""
        mock_client.call.side_effect = AgentCoreError("Workflow not found")

        result = runner.invoke(app, ["workflow", "execute", "workflow-12345"])

        assert result.exit_code == 1


class TestWorkflowStatus:
    """Tests for workflow status command."""

    def test_status_workflow_success(self, mock_config, mock_client):
        """Test successful workflow status retrieval."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "name": "test-workflow",
            "status": "running",
            "tasks_total": 3,
            "tasks_completed": 1,
            "tasks_running": 1,
            "tasks_pending": 1,
            "tasks_failed": 0,
        }

        result = runner.invoke(app, ["workflow", "status", "workflow-12345"])

        assert result.exit_code == 0
        assert "workflow-12345" in result.stdout
        assert "test-workflow" in result.stdout
        assert "running" in result.stdout

    def test_status_workflow_with_json_output(self, mock_config, mock_client):
        """Test workflow status with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "running",
        }

        result = runner.invoke(app, ["workflow", "status", "workflow-12345", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["workflow_id"] == "workflow-12345"


class TestWorkflowList:
    """Tests for workflow list command."""

    def test_list_workflows_success(self, mock_config, mock_client):
        """Test successful workflow listing."""
        mock_client.call.return_value = {
            "workflows": [
                {
                    "workflow_id": "workflow-1",
                    "name": "workflow-1",
                    "status": "running",
                    "tasks_total": 3,
                    "tasks_completed": 1,
                },
                {
                    "workflow_id": "workflow-2",
                    "name": "workflow-2",
                    "status": "completed",
                    "tasks_total": 2,
                    "tasks_completed": 2,
                },
            ]
        }

        result = runner.invoke(app, ["workflow", "list"])

        assert result.exit_code == 0
        assert "workflow-1" in result.stdout
        assert "workflow-2" in result.stdout
        assert "Total: 2 workflow(s)" in result.stdout

    def test_list_workflows_empty(self, mock_config, mock_client):
        """Test listing workflows when none exist."""
        mock_client.call.return_value = {"workflows": []}

        result = runner.invoke(app, ["workflow", "list"])

        assert result.exit_code == 0
        assert "No workflows found" in result.stdout

    def test_list_workflows_with_filter(self, mock_config, mock_client):
        """Test listing workflows with status filter."""
        mock_client.call.return_value = {
            "workflows": [
                {
                    "workflow_id": "workflow-1",
                    "name": "workflow-1",
                    "status": "running",
                    "tasks_total": 3,
                    "tasks_completed": 1,
                },
            ]
        }

        result = runner.invoke(app, ["workflow", "list", "--status", "running"])

        assert result.exit_code == 0
        # Verify filter was passed to API
        call_args = mock_client.call.call_args
        assert call_args[0][1]["status"] == "running"

    def test_list_workflows_with_json_output(self, mock_config, mock_client):
        """Test listing workflows with JSON output."""
        mock_client.call.return_value = {
            "workflows": [
                {"workflow_id": "workflow-1", "name": "test"},
            ]
        }

        result = runner.invoke(app, ["workflow", "list", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert len(output) == 1
        assert output[0]["workflow_id"] == "workflow-1"


class TestWorkflowVisualize:
    """Tests for workflow visualize command."""

    def test_visualize_workflow_success(self, mock_config, mock_client):
        """Test successful workflow visualization."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "name": "test-workflow",
            "status": "running",
            "tasks": [
                {"name": "task-1", "type": "test", "status": "completed", "depends_on": []},
                {"name": "task-2", "type": "test", "status": "running", "depends_on": ["task-1"]},
            ],
        }

        result = runner.invoke(app, ["workflow", "visualize", "workflow-12345"])

        assert result.exit_code == 0
        assert "test-workflow" in result.stdout
        assert "task-1" in result.stdout
        assert "task-2" in result.stdout

    def test_visualize_workflow_save_to_file(self, mock_config, mock_client, tmp_path):
        """Test saving workflow visualization to file."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "name": "test-workflow",
            "status": "running",
            "tasks": [
                {"name": "task-1", "type": "test", "status": "completed", "depends_on": []},
            ],
        }

        output_file = tmp_path / "graph.txt"
        result = runner.invoke(app, ["workflow", "visualize", "workflow-12345", "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Workflow graph saved to" in result.stdout

    def test_visualize_workflow_with_json_output(self, mock_config, mock_client):
        """Test workflow visualization with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "running",
            "tasks": [],
        }

        result = runner.invoke(app, ["workflow", "visualize", "workflow-12345", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["workflow_id"] == "workflow-12345"


class TestWorkflowPause:
    """Tests for workflow pause command."""

    def test_pause_workflow_success(self, mock_config, mock_client):
        """Test successful workflow pause."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "paused",
        }

        result = runner.invoke(app, ["workflow", "pause", "workflow-12345"])

        assert result.exit_code == 0
        assert "Workflow paused: workflow-12345" in result.stdout
        assert "Resume with:" in result.stdout

        # Verify API call
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args
        assert call_args[0][0] == "workflow.pause"
        assert call_args[0][1]["workflow_id"] == "workflow-12345"

    def test_pause_workflow_with_json_output(self, mock_config, mock_client):
        """Test workflow pause with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "paused",
        }

        result = runner.invoke(app, ["workflow", "pause", "workflow-12345", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["status"] == "paused"

    def test_pause_workflow_api_error(self, mock_config, mock_client):
        """Test workflow pause with API error."""
        mock_client.call.side_effect = AgentCoreError("Cannot pause completed workflow")

        result = runner.invoke(app, ["workflow", "pause", "workflow-12345"])

        assert result.exit_code == 1


class TestWorkflowResume:
    """Tests for workflow resume command."""

    def test_resume_workflow_success(self, mock_config, mock_client):
        """Test successful workflow resume."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "running",
        }

        result = runner.invoke(app, ["workflow", "resume", "workflow-12345"])

        assert result.exit_code == 0
        assert "Workflow resumed: workflow-12345" in result.stdout
        assert "Check status:" in result.stdout

        # Verify API call
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args
        assert call_args[0][0] == "workflow.resume"
        assert call_args[0][1]["workflow_id"] == "workflow-12345"

    def test_resume_workflow_with_json_output(self, mock_config, mock_client):
        """Test workflow resume with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-12345",
            "status": "running",
        }

        result = runner.invoke(app, ["workflow", "resume", "workflow-12345", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["status"] == "running"

    def test_resume_workflow_api_error(self, mock_config, mock_client):
        """Test workflow resume with API error."""
        mock_client.call.side_effect = AgentCoreError("Workflow not in paused state")

        result = runner.invoke(app, ["workflow", "resume", "workflow-12345"])

        assert result.exit_code == 1


class TestWorkflowHelpers:
    """Tests for workflow helper functions."""

    def test_format_workflow_status(self):
        """Test workflow status formatting."""
        from agentcore_cli.commands.workflow import _format_workflow_status

        assert "green" in _format_workflow_status("completed")
        assert "yellow" in _format_workflow_status("running")
        assert "blue" in _format_workflow_status("pending")
        assert "red" in _format_workflow_status("failed")
        assert "dim" in _format_workflow_status("paused")

    def test_format_task_status(self):
        """Test task status formatting."""
        from agentcore_cli.commands.workflow import _format_task_status

        assert "green" in _format_task_status("completed")
        assert "yellow" in _format_task_status("running")
        assert "blue" in _format_task_status("pending")
        assert "red" in _format_task_status("failed")
        assert "dim" in _format_task_status("skipped")

    def test_create_progress_bar(self):
        """Test progress bar creation."""
        from agentcore_cli.commands.workflow import _create_progress_bar

        bar_0 = _create_progress_bar(0)
        assert "0.0%" in bar_0
        assert "░" in bar_0

        bar_50 = _create_progress_bar(50)
        assert "50.0%" in bar_50
        assert "█" in bar_50
        assert "░" in bar_50

        bar_100 = _create_progress_bar(100)
        assert "100.0%" in bar_100
        assert "█" in bar_100

    def test_create_workflow_graph(self):
        """Test workflow graph creation."""
        from agentcore_cli.commands.workflow import _create_workflow_graph

        workflow_data = {
            "name": "test-workflow",
            "status": "running",
            "tasks": [
                {"name": "task-1", "type": "test", "status": "completed", "depends_on": []},
                {"name": "task-2", "type": "test", "status": "running", "depends_on": ["task-1"]},
            ],
        }

        tree = _create_workflow_graph(workflow_data)
        assert tree is not None
        # Tree should have nodes for tasks
        assert len(tree.children) > 0
