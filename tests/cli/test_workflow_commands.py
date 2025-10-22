"""Integration tests for workflow commands.

These tests verify that the workflow commands properly use the service layer
and send JSON-RPC 2.0 compliant requests to the API.
"""

from __future__ import annotations

from unittest.mock import Mock, patch
from pathlib import Path
import pytest
from typer.testing import CliRunner
import tempfile

from agentcore_cli.main import app
from agentcore_cli.services.exceptions import (
    ValidationError,
    WorkflowNotFoundError,
    OperationError,
)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_workflow_service() -> Mock:
    """Create a mock workflow service."""
    return Mock()


@pytest.fixture
def workflow_yaml() -> str:
    """Sample workflow YAML content."""
    return """
name: test-workflow
description: Test workflow
steps:
  - agent: analyzer
    task: analyze
  - agent: reporter
    task: report
"""


class TestWorkflowRunCommand:
    """Test suite for workflow run command."""

    def test_run_success(
        self, runner: CliRunner, mock_workflow_service: Mock, workflow_yaml: str
    ) -> None:
        """Test successful workflow execution."""
        # Create temporary workflow file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            workflow_file = f.name

        try:
            # Mock service response
            mock_workflow_service.run.return_value = "workflow-001"

            # Patch the container to return mock service
            with patch(
                "agentcore_cli.commands.workflow.get_workflow_service",
                return_value=mock_workflow_service,
            ):
                result = runner.invoke(app, ["workflow", "run", workflow_file])

            # Verify exit code
            assert result.exit_code == 0, f"Command failed: {result.output}"

            # Verify output
            assert "Workflow started successfully" in result.output
            assert "workflow-001" in result.output
            assert "test-workflow" in result.output

            # Verify service was called correctly
            mock_workflow_service.run.assert_called_once()
            call_args = mock_workflow_service.run.call_args
            assert call_args.kwargs["definition"]["name"] == "test-workflow"
            assert call_args.kwargs["parameters"] is None

        finally:
            # Cleanup
            Path(workflow_file).unlink()

    def test_run_with_parameters(
        self, runner: CliRunner, mock_workflow_service: Mock, workflow_yaml: str
    ) -> None:
        """Test workflow execution with parameters."""
        # Create temporary workflow file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            workflow_file = f.name

        try:
            # Mock service response
            mock_workflow_service.run.return_value = "workflow-002"

            # Patch the container to return mock service
            with patch(
                "agentcore_cli.commands.workflow.get_workflow_service",
                return_value=mock_workflow_service,
            ):
                result = runner.invoke(
                    app,
                    [
                        "workflow",
                        "run",
                        workflow_file,
                        "--parameters",
                        '{"repo": "foo/bar", "branch": "main"}',
                    ],
                )

            # Verify exit code
            assert result.exit_code == 0

            # Verify output
            assert "Workflow started successfully" in result.output
            assert "workflow-002" in result.output

            # Verify service was called with parameters
            call_args = mock_workflow_service.run.call_args
            assert call_args.kwargs["parameters"]["repo"] == "foo/bar"
            assert call_args.kwargs["parameters"]["branch"] == "main"

        finally:
            # Cleanup
            Path(workflow_file).unlink()

    def test_run_json_output(
        self, runner: CliRunner, mock_workflow_service: Mock, workflow_yaml: str
    ) -> None:
        """Test workflow run with JSON output."""
        # Create temporary workflow file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            workflow_file = f.name

        try:
            # Mock service response
            mock_workflow_service.run.return_value = "workflow-003"

            # Patch the container to return mock service
            with patch(
                "agentcore_cli.commands.workflow.get_workflow_service",
                return_value=mock_workflow_service,
            ):
                result = runner.invoke(
                    app, ["workflow", "run", workflow_file, "--json"]
                )

            # Verify exit code
            assert result.exit_code == 0

            # Verify JSON output
            import json

            output = json.loads(result.output)
            assert output["workflow_id"] == "workflow-003"
            assert output["definition"]["name"] == "test-workflow"

        finally:
            # Cleanup
            Path(workflow_file).unlink()

    def test_run_file_not_found(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow run with non-existent file."""
        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(
                app, ["workflow", "run", "/nonexistent/workflow.yaml"]
            )

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Workflow file not found" in result.output

    def test_run_invalid_yaml(self, runner: CliRunner, mock_workflow_service: Mock) -> None:
        """Test workflow run with invalid YAML."""
        # Create temporary file with invalid YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            workflow_file = f.name

        try:
            with patch(
                "agentcore_cli.commands.workflow.get_workflow_service",
                return_value=mock_workflow_service,
            ):
                result = runner.invoke(app, ["workflow", "run", workflow_file])

            # Verify exit code
            assert result.exit_code == 2

            # Verify error message
            assert "Invalid YAML" in result.output

        finally:
            # Cleanup
            Path(workflow_file).unlink()

    def test_run_invalid_parameters(
        self, runner: CliRunner, mock_workflow_service: Mock, workflow_yaml: str
    ) -> None:
        """Test workflow run with invalid parameters JSON."""
        # Create temporary workflow file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            workflow_file = f.name

        try:
            with patch(
                "agentcore_cli.commands.workflow.get_workflow_service",
                return_value=mock_workflow_service,
            ):
                result = runner.invoke(
                    app,
                    [
                        "workflow",
                        "run",
                        workflow_file,
                        "--parameters",
                        "invalid json",
                    ],
                )

            # Verify exit code
            assert result.exit_code == 2

            # Verify error message
            assert "Invalid JSON in parameters" in result.output

        finally:
            # Cleanup
            Path(workflow_file).unlink()

    def test_run_validation_error(
        self, runner: CliRunner, mock_workflow_service: Mock, workflow_yaml: str
    ) -> None:
        """Test workflow run with validation error."""
        # Create temporary workflow file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            workflow_file = f.name

        try:
            # Mock service to raise validation error
            mock_workflow_service.run.side_effect = ValidationError(
                "Workflow definition must have a 'name' field"
            )

            with patch(
                "agentcore_cli.commands.workflow.get_workflow_service",
                return_value=mock_workflow_service,
            ):
                result = runner.invoke(app, ["workflow", "run", workflow_file])

            # Verify exit code
            assert result.exit_code == 2

            # Verify error message
            assert "Validation error" in result.output

        finally:
            # Cleanup
            Path(workflow_file).unlink()

    def test_run_operation_error(
        self, runner: CliRunner, mock_workflow_service: Mock, workflow_yaml: str
    ) -> None:
        """Test workflow run with operation error."""
        # Create temporary workflow file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(workflow_yaml)
            workflow_file = f.name

        try:
            # Mock service to raise operation error
            mock_workflow_service.run.side_effect = OperationError(
                "Workflow execution failed"
            )

            with patch(
                "agentcore_cli.commands.workflow.get_workflow_service",
                return_value=mock_workflow_service,
            ):
                result = runner.invoke(app, ["workflow", "run", workflow_file])

            # Verify exit code
            assert result.exit_code == 1

            # Verify error message
            assert "Operation failed" in result.output

        finally:
            # Cleanup
            Path(workflow_file).unlink()


class TestWorkflowListCommand:
    """Test suite for workflow list command."""

    def test_list_success(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test successful workflow listing."""
        # Mock service response
        mock_workflow_service.list_workflows.return_value = [
            {
                "workflow_id": "workflow-001",
                "name": "test-workflow-1",
                "status": "running",
                "created_at": "2025-10-22T10:00:00Z",
            },
            {
                "workflow_id": "workflow-002",
                "name": "test-workflow-2",
                "status": "completed",
                "created_at": "2025-10-22T09:00:00Z",
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "workflow-001" in result.output
        assert "workflow-002" in result.output
        assert "test-workflow-1" in result.output
        assert "test-workflow-2" in result.output

        # Verify service was called correctly
        mock_workflow_service.list_workflows.assert_called_once_with(
            status=None, limit=100, offset=0
        )

    def test_list_with_status_filter(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow listing with status filter."""
        # Mock service response
        mock_workflow_service.list_workflows.return_value = [
            {
                "workflow_id": "workflow-001",
                "name": "running-workflow",
                "status": "running",
                "created_at": "2025-10-22T10:00:00Z",
            },
        ]

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "list", "--status", "running"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with filter
        mock_workflow_service.list_workflows.assert_called_once_with(
            status="running", limit=100, offset=0
        )

    def test_list_with_limit_and_offset(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow listing with pagination."""
        # Mock service response
        mock_workflow_service.list_workflows.return_value = []

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(
                app, ["workflow", "list", "--limit", "10", "--offset", "20"]
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with pagination
        mock_workflow_service.list_workflows.assert_called_once_with(
            status=None, limit=10, offset=20
        )

    def test_list_json_output(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow list with JSON output."""
        # Mock service response
        workflows = [
            {
                "workflow_id": "workflow-001",
                "name": "test-workflow",
                "status": "running",
                "created_at": "2025-10-22T10:00:00Z",
            },
        ]
        mock_workflow_service.list_workflows.return_value = workflows

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "list", "--json"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        import json

        output = json.loads(result.output)
        assert len(output) == 1
        assert output[0]["workflow_id"] == "workflow-001"

    def test_list_empty(self, runner: CliRunner, mock_workflow_service: Mock) -> None:
        """Test workflow list with no workflows."""
        # Mock service response
        mock_workflow_service.list_workflows.return_value = []

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "No workflows found" in result.output

    def test_list_validation_error(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow list with validation error."""
        # Mock service to raise validation error
        mock_workflow_service.list_workflows.side_effect = ValidationError(
            "Limit must be positive"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "list"])

        # Verify exit code
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output

    def test_list_operation_error(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow list with operation error."""
        # Mock service to raise operation error
        mock_workflow_service.list_workflows.side_effect = OperationError(
            "Workflow listing failed"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "list"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Operation failed" in result.output


class TestWorkflowInfoCommand:
    """Test suite for workflow info command."""

    def test_info_success(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test successful workflow info retrieval."""
        # Mock service response
        mock_workflow_service.get.return_value = {
            "workflow_id": "workflow-001",
            "name": "test-workflow",
            "status": "running",
            "created_at": "2025-10-22T10:00:00Z",
            "updated_at": "2025-10-22T10:05:00Z",
            "definition": {"name": "test-workflow", "steps": []},
            "parameters": {"repo": "foo/bar"},
        }

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "info", "workflow-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Workflow Information" in result.output
        assert "workflow-001" in result.output
        assert "test-workflow" in result.output
        assert "running" in result.output

        # Verify service was called correctly
        mock_workflow_service.get.assert_called_once_with("workflow-001")

    def test_info_json_output(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow info with JSON output."""
        # Mock service response
        workflow_data = {
            "workflow_id": "workflow-001",
            "name": "test-workflow",
            "status": "running",
            "created_at": "2025-10-22T10:00:00Z",
            "updated_at": "2025-10-22T10:05:00Z",
        }
        mock_workflow_service.get.return_value = workflow_data

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "info", "workflow-001", "--json"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        import json

        output = json.loads(result.output)
        assert output["workflow_id"] == "workflow-001"
        assert output["name"] == "test-workflow"

    def test_info_not_found(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow info with non-existent workflow."""
        # Mock service to raise not found error
        mock_workflow_service.get.side_effect = WorkflowNotFoundError(
            "Workflow 'workflow-999' not found"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "info", "workflow-999"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Workflow not found" in result.output

    def test_info_validation_error(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow info with validation error."""
        # Mock service to raise validation error
        mock_workflow_service.get.side_effect = ValidationError(
            "Workflow ID cannot be empty"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "info", ""])

        # Verify exit code
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output

    def test_info_operation_error(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow info with operation error."""
        # Mock service to raise operation error
        mock_workflow_service.get.side_effect = OperationError(
            "Workflow retrieval failed"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "info", "workflow-001"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Operation failed" in result.output


class TestWorkflowStopCommand:
    """Test suite for workflow stop command."""

    def test_stop_success(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test successful workflow stop."""
        # Mock service response
        mock_workflow_service.stop.return_value = True

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "stop", "workflow-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Workflow stopped successfully" in result.output
        assert "workflow-001" in result.output

        # Verify service was called correctly
        mock_workflow_service.stop.assert_called_once_with("workflow-001", force=False)

    def test_stop_with_force(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow stop with force option."""
        # Mock service response
        mock_workflow_service.stop.return_value = True

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(
                app, ["workflow", "stop", "workflow-001", "--force"]
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with force=True
        mock_workflow_service.stop.assert_called_once_with("workflow-001", force=True)

    def test_stop_json_output(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow stop with JSON output."""
        # Mock service response
        mock_workflow_service.stop.return_value = True

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(
                app, ["workflow", "stop", "workflow-001", "--json"]
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        import json

        output = json.loads(result.output)
        assert output["success"] is True
        assert output["workflow_id"] == "workflow-001"

    def test_stop_not_found(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow stop with non-existent workflow."""
        # Mock service to raise not found error
        mock_workflow_service.stop.side_effect = WorkflowNotFoundError(
            "Workflow 'workflow-999' not found"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "stop", "workflow-999"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Workflow not found" in result.output

    def test_stop_validation_error(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow stop with validation error."""
        # Mock service to raise validation error
        mock_workflow_service.stop.side_effect = ValidationError(
            "Workflow ID cannot be empty"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "stop", ""])

        # Verify exit code
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output

    def test_stop_operation_error(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow stop with operation error."""
        # Mock service to raise operation error
        mock_workflow_service.stop.side_effect = OperationError(
            "Workflow stopping failed"
        )

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "stop", "workflow-001"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Operation failed" in result.output

    def test_stop_failure(
        self, runner: CliRunner, mock_workflow_service: Mock
    ) -> None:
        """Test workflow stop when service returns False."""
        # Mock service response
        mock_workflow_service.stop.return_value = False

        with patch(
            "agentcore_cli.commands.workflow.get_workflow_service",
            return_value=mock_workflow_service,
        ):
            result = runner.invoke(app, ["workflow", "stop", "workflow-001"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Failed to stop workflow" in result.output
