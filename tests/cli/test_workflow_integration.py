"""Integration tests for workflow commands with realistic scenarios."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from typer.testing import CliRunner

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
def simple_workflow_file(tmp_path: Path) -> Path:
    """Create a simple workflow YAML file."""
    workflow_data = {
        "name": "simple-workflow",
        "description": "A simple test workflow",
        "version": "1.0",
        "tasks": [
            {
                "name": "task-1",
                "type": "analysis",
                "requirements": {"language": "python"},
            },
            {
                "name": "task-2",
                "type": "testing",
                "depends_on": ["task-1"],
            },
        ],
        "max_retries": 2,
        "timeout": 1800,
    }

    workflow_file = tmp_path / "simple_workflow.yaml"
    with open(workflow_file, "w") as f:
        yaml.dump(workflow_data, f)

    return workflow_file


@pytest.fixture
def complex_workflow_file(tmp_path: Path) -> Path:
    """Create a complex workflow YAML file."""
    workflow_data = {
        "name": "complex-workflow",
        "description": "A complex multi-stage workflow",
        "version": "2.0",
        "tasks": [
            {
                "name": "setup",
                "type": "initialization",
                "requirements": {"env": "production"},
            },
            {
                "name": "build",
                "type": "compilation",
                "depends_on": ["setup"],
                "requirements": {"memory": "2GB"},
            },
            {
                "name": "test-unit",
                "type": "testing",
                "depends_on": ["build"],
            },
            {
                "name": "test-integration",
                "type": "testing",
                "depends_on": ["build"],
            },
            {
                "name": "deploy",
                "type": "deployment",
                "depends_on": ["test-unit", "test-integration"],
                "requirements": {"approval": "required"},
            },
        ],
        "max_retries": 3,
        "timeout": 7200,
        "on_failure": "rollback",
    }

    workflow_file = tmp_path / "complex_workflow.yaml"
    with open(workflow_file, "w") as f:
        yaml.dump(workflow_data, f)

    return workflow_file


class TestWorkflowCreationWorkflows:
    """Integration tests for workflow creation workflows."""

    def test_create_execute_status_workflow(self, mock_config, mock_client, simple_workflow_file):
        """Test complete workflow: create → execute → check status."""
        # Step 1: Create workflow
        mock_client.call.return_value = {
            "workflow_id": "workflow-create-123",
            "name": "simple-workflow",
            "tasks_count": 2,
        }

        create_result = runner.invoke(app, [
            "workflow", "create",
            "--file", str(simple_workflow_file)
        ])

        assert create_result.exit_code == 0
        assert "Workflow created: workflow-create-123" in create_result.stdout

        # Step 2: Execute workflow
        mock_client.call.return_value = {
            "workflow_id": "workflow-create-123",
            "status": "running",
        }

        execute_result = runner.invoke(app, [
            "workflow", "execute", "workflow-create-123"
        ])

        assert execute_result.exit_code == 0
        assert "Workflow execution started" in execute_result.stdout

        # Step 3: Check status
        mock_client.call.return_value = {
            "workflow_id": "workflow-create-123",
            "name": "simple-workflow",
            "status": "running",
            "tasks_total": 2,
            "tasks_completed": 1,
            "tasks_running": 1,
            "tasks_pending": 0,
            "tasks_failed": 0,
        }

        status_result = runner.invoke(app, [
            "workflow", "status", "workflow-create-123"
        ])

        assert status_result.exit_code == 0
        assert "running" in status_result.stdout

    def test_create_validate_only_workflow(self, mock_config, mock_client, simple_workflow_file):
        """Test workflow: create with --validate-only."""
        result = runner.invoke(app, [
            "workflow", "create",
            "--file", str(simple_workflow_file),
            "--validate-only"
        ])

        assert result.exit_code == 0
        assert "Workflow definition is valid" in result.stdout

        # API should not be called for validation-only
        mock_client.call.assert_not_called()

    def test_create_with_invalid_file(self, mock_config, mock_client):
        """Test creating workflow with non-existent file."""
        result = runner.invoke(app, [
            "workflow", "create",
            "--file", "nonexistent.yaml"
        ])

        assert result.exit_code == 2
        assert "Workflow file not found" in result.stdout


class TestWorkflowExecutionWorkflows:
    """Integration tests for workflow execution workflows."""

    def test_execute_with_watch_mode(self, mock_config, mock_client):
        """Test executing workflow with watch mode until completion."""
        # Mock progressive status updates
        mock_client.call.side_effect = [
            # Execute call
            {"workflow_id": "workflow-watch", "status": "running"},
            # Status call 1
            {
                "workflow_id": "workflow-watch",
                "status": "running",
                "tasks_total": 3,
                "tasks_completed": 0,
                "tasks_running": 1,
            },
            # Status call 2
            {
                "workflow_id": "workflow-watch",
                "status": "running",
                "tasks_total": 3,
                "tasks_completed": 1,
                "tasks_running": 1,
            },
            # Status call 3 - completed
            {
                "workflow_id": "workflow-watch",
                "status": "completed",
                "tasks_total": 3,
                "tasks_completed": 3,
                "tasks_running": 0,
            },
        ]

        with patch("agentcore_cli.commands.workflow.time.sleep"):  # Speed up test
            result = runner.invoke(app, [
                "workflow", "execute", "workflow-watch", "--watch"
            ])

        assert result.exit_code == 0
        assert "Workflow execution started" in result.stdout
        assert "Workflow completed successfully" in result.stdout

    def test_execute_pause_resume_workflow(self, mock_config, mock_client):
        """Test workflow: execute → pause → resume."""
        # Step 1: Execute
        mock_client.call.return_value = {
            "workflow_id": "workflow-pause-test",
            "status": "running",
        }

        execute_result = runner.invoke(app, [
            "workflow", "execute", "workflow-pause-test"
        ])

        assert execute_result.exit_code == 0

        # Step 2: Pause
        mock_client.call.return_value = {
            "workflow_id": "workflow-pause-test",
            "status": "paused",
        }

        pause_result = runner.invoke(app, [
            "workflow", "pause", "workflow-pause-test"
        ])

        assert pause_result.exit_code == 0
        assert "Workflow paused" in pause_result.stdout

        # Step 3: Resume
        mock_client.call.return_value = {
            "workflow_id": "workflow-pause-test",
            "status": "running",
        }

        resume_result = runner.invoke(app, [
            "workflow", "resume", "workflow-pause-test"
        ])

        assert resume_result.exit_code == 0
        assert "Workflow resumed" in resume_result.stdout


class TestWorkflowVisualizationWorkflows:
    """Integration tests for workflow visualization workflows."""

    def test_status_visualize_workflow(self, mock_config, mock_client):
        """Test workflow: status → visualize."""
        # Step 1: Check status
        mock_client.call.return_value = {
            "workflow_id": "workflow-viz",
            "name": "visualization-test",
            "status": "running",
            "tasks_total": 3,
            "tasks_completed": 1,
        }

        status_result = runner.invoke(app, [
            "workflow", "status", "workflow-viz"
        ])

        assert status_result.exit_code == 0

        # Step 2: Visualize workflow
        mock_client.call.return_value = {
            "workflow_id": "workflow-viz",
            "name": "visualization-test",
            "status": "running",
            "tasks": [
                {"name": "task-1", "type": "test", "status": "completed", "depends_on": []},
                {"name": "task-2", "type": "test", "status": "running", "depends_on": ["task-1"]},
                {"name": "task-3", "type": "test", "status": "pending", "depends_on": ["task-2"]},
            ],
        }

        visualize_result = runner.invoke(app, [
            "workflow", "visualize", "workflow-viz"
        ])

        assert visualize_result.exit_code == 0
        assert "task-1" in visualize_result.stdout
        assert "task-2" in visualize_result.stdout

    def test_visualize_save_to_file(self, mock_config, mock_client, tmp_path: Path):
        """Test visualizing and saving to file."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-save-viz",
            "name": "save-test",
            "status": "running",
            "tasks": [
                {"name": "task-1", "type": "test", "status": "completed", "depends_on": []},
            ],
        }

        output_file = tmp_path / "workflow_graph.txt"
        result = runner.invoke(app, [
            "workflow", "visualize", "workflow-save-viz",
            "--output", str(output_file)
        ])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Workflow graph saved to" in result.stdout


class TestWorkflowListingAndFiltering:
    """Integration tests for workflow listing and filtering."""

    def test_list_all_workflows(self, mock_config, mock_client):
        """Test listing all workflows."""
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

    def test_list_with_status_filter(self, mock_config, mock_client):
        """Test listing workflows with status filter."""
        mock_client.call.return_value = {
            "workflows": [
                {
                    "workflow_id": "workflow-running",
                    "name": "running-workflow",
                    "status": "running",
                    "tasks_total": 3,
                    "tasks_completed": 1,
                },
            ]
        }

        result = runner.invoke(app, [
            "workflow", "list",
            "--status", "running"
        ])

        assert result.exit_code == 0
        assert "running-workflow" in result.stdout

        # Verify filter was passed to API
        call_args = mock_client.call.call_args
        assert call_args[0][1]["status"] == "running"

    def test_list_empty_results(self, mock_config, mock_client):
        """Test listing when no workflows exist."""
        mock_client.call.return_value = {"workflows": []}

        result = runner.invoke(app, ["workflow", "list"])

        assert result.exit_code == 0
        assert "No workflows found" in result.stdout


class TestWorkflowOutputFormats:
    """Integration tests for different output formats."""

    def test_create_json_output(self, mock_config, mock_client, simple_workflow_file):
        """Test workflow creation with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-json",
            "name": "simple-workflow",
        }

        result = runner.invoke(app, [
            "workflow", "create",
            "--file", str(simple_workflow_file),
            "--json"
        ])

        assert result.exit_code == 0
        import json
        output = json.loads(result.stdout)
        assert output["workflow_id"] == "workflow-json"

    def test_status_json_output(self, mock_config, mock_client):
        """Test workflow status with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-123",
            "status": "running",
            "tasks_total": 3,
        }

        result = runner.invoke(app, [
            "workflow", "status", "workflow-123", "--json"
        ])

        assert result.exit_code == 0
        import json
        output = json.loads(result.stdout)
        assert output["workflow_id"] == "workflow-123"

    def test_list_json_output(self, mock_config, mock_client):
        """Test workflow list with JSON output."""
        mock_client.call.return_value = {
            "workflows": [
                {"workflow_id": "workflow-1", "name": "test", "status": "running"},
            ]
        }

        result = runner.invoke(app, ["workflow", "list", "--json"])

        assert result.exit_code == 0
        import json
        output = json.loads(result.stdout)
        assert len(output) == 1

    def test_visualize_json_output(self, mock_config, mock_client):
        """Test workflow visualization with JSON output."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-viz-json",
            "status": "running",
            "tasks": [],
        }

        result = runner.invoke(app, [
            "workflow", "visualize", "workflow-viz-json", "--json"
        ])

        assert result.exit_code == 0
        import json
        output = json.loads(result.stdout)
        assert "workflow_id" in output


class TestWorkflowComplexScenarios:
    """Integration tests for complex workflow scenarios."""

    def test_complex_dependency_workflow(self, mock_config, mock_client, complex_workflow_file):
        """Test workflow with complex dependencies."""
        mock_client.call.return_value = {
            "workflow_id": "workflow-complex",
            "name": "complex-workflow",
            "tasks_count": 5,
        }

        create_result = runner.invoke(app, [
            "workflow", "create",
            "--file", str(complex_workflow_file)
        ])

        assert create_result.exit_code == 0
        assert "Workflow created" in create_result.stdout

        # Verify task dependencies
        call_args = mock_client.call.call_args[0][1]
        tasks = call_args["tasks"]
        assert len(tasks) == 5

        # Verify deploy task depends on both test tasks
        deploy_task = next(t for t in tasks if t["name"] == "deploy")
        assert "test-unit" in deploy_task["depends_on"]
        assert "test-integration" in deploy_task["depends_on"]

    def test_workflow_failure_handling(self, mock_config, mock_client):
        """Test workflow with task failures."""
        # Execute workflow
        mock_client.call.return_value = {
            "workflow_id": "workflow-fail",
            "status": "running",
        }

        execute_result = runner.invoke(app, [
            "workflow", "execute", "workflow-fail"
        ])

        assert execute_result.exit_code == 0

        # Check status showing failure
        mock_client.call.return_value = {
            "workflow_id": "workflow-fail",
            "name": "failing-workflow",
            "status": "failed",
            "tasks_total": 3,
            "tasks_completed": 1,
            "tasks_running": 0,
            "tasks_pending": 1,
            "tasks_failed": 1,
        }

        status_result = runner.invoke(app, [
            "workflow", "status", "workflow-fail"
        ])

        assert status_result.exit_code == 0
        assert "failed" in status_result.stdout.lower()


class TestWorkflowEdgeCases:
    """Integration tests for edge cases and error scenarios."""

    def test_invalid_yaml_syntax(self, mock_config, mock_client, tmp_path: Path):
        """Test creating workflow with invalid YAML syntax."""
        invalid_yaml = tmp_path / "invalid.yaml"
        with open(invalid_yaml, "w") as f:
            f.write("invalid: yaml: syntax: [")

        result = runner.invoke(app, [
            "workflow", "create",
            "--file", str(invalid_yaml)
        ])

        assert result.exit_code == 2
        assert "Invalid YAML syntax" in result.stdout

    def test_missing_required_fields(self, mock_config, mock_client, tmp_path: Path):
        """Test creating workflow with missing required fields."""
        invalid_workflow = tmp_path / "invalid_workflow.yaml"
        with open(invalid_workflow, "w") as f:
            yaml.dump({"name": "incomplete"}, f)  # Missing 'tasks' field

        result = runner.invoke(app, [
            "workflow", "create",
            "--file", str(invalid_workflow)
        ])

        assert result.exit_code == 2
        assert "Workflow validation failed" in result.stdout

    def test_execute_nonexistent_workflow(self, mock_config, mock_client):
        """Test executing a workflow that doesn't exist."""
        from agentcore_cli.exceptions import AgentCoreError

        mock_client.call.side_effect = AgentCoreError("Workflow not found")

        result = runner.invoke(app, [
            "workflow", "execute", "nonexistent-workflow"
        ])

        assert result.exit_code == 1
        assert "Workflow not found" in result.stdout

    def test_pause_completed_workflow(self, mock_config, mock_client):
        """Test pausing a workflow that is already completed."""
        from agentcore_cli.exceptions import AgentCoreError

        mock_client.call.side_effect = AgentCoreError("Cannot pause completed workflow")

        result = runner.invoke(app, [
            "workflow", "pause", "completed-workflow"
        ])

        assert result.exit_code == 1
        assert "Cannot pause completed workflow" in result.stdout


class TestWorkflowCrossFunctionality:
    """Integration tests for workflow with other CLI features."""

    def test_workflow_with_session_integration(self, mock_config, mock_client, simple_workflow_file):
        """Test creating and executing workflow within a session context."""
        # This would test if workflows properly integrate with sessions
        # For now, just test that workflow commands work independently
        mock_client.call.return_value = {
            "workflow_id": "workflow-session-test",
            "name": "simple-workflow",
            "tasks_count": 2,
        }

        result = runner.invoke(app, [
            "workflow", "create",
            "--file", str(simple_workflow_file)
        ])

        assert result.exit_code == 0
        assert "Workflow created" in result.stdout
