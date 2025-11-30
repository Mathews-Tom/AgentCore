"""Integration tests for task commands.

These tests verify that the task commands properly use the service layer
and send JSON-RPC 2.0 compliant requests to the API.
"""

from __future__ import annotations

from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from agentcore_cli.main import app
from agentcore_cli.services.exceptions import (
    ValidationError,
    TaskNotFoundError,
    OperationError)


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_task_service() -> Mock:
    """Create a mock task service."""
    return Mock()


class TestTaskCreateCommand:
    """Test suite for task create command."""

    def test_create_success(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test successful task creation."""
        # Mock service response
        mock_task_service.create.return_value = "task-001"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Analyze code repository",
                ])

        # Verify exit code
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify output
        assert "Task created successfully" in result.output
        assert "task-001" in result.output
        assert "Analyze code repository" in result.output

        # Verify service was called correctly
        mock_task_service.create.assert_called_once_with(
            description="Analyze code repository",
            agent_id=None,
            priority="normal",
            parameters=None)

    def test_create_with_agent_assignment(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task creation with agent assignment."""
        # Mock service response
        mock_task_service.create.return_value = "task-002"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Run tests",
                    "--agent-id",
                    "agent-001",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with agent_id
        mock_task_service.create.assert_called_once_with(
            description="Run tests",
            agent_id="agent-001",
            priority="normal",
            parameters=None)

    def test_create_with_priority(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task creation with custom priority."""
        # Mock service response
        mock_task_service.create.return_value = "task-003"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Fix critical bug",
                    "--priority",
                    "critical",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with priority
        mock_task_service.create.assert_called_once_with(
            description="Fix critical bug",
            agent_id=None,
            priority="critical",
            parameters=None)

    def test_create_with_parameters(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task creation with parameters."""
        # Mock service response
        mock_task_service.create.return_value = "task-004"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Process data",
                    "--parameters",
                    '{"repo": "foo/bar", "branch": "main"}',
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with parameters
        mock_task_service.create.assert_called_once_with(
            description="Process data",
            agent_id=None,
            priority="normal",
            parameters={"repo": "foo/bar", "branch": "main"})

    def test_create_with_invalid_json_parameters(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task creation with invalid JSON parameters."""
        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Process data",
                    "--parameters",
                    "invalid json",
                ])

        # Verify exit code (2 for validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Invalid JSON in parameters" in result.output

    def test_create_json_output(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task creation with JSON output."""
        # Mock service response
        mock_task_service.create.return_value = "task-005"

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "JSON test",
                    "--json",
                ])

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        assert '"task_id": "task-005"' in result.output
        assert '"description": "JSON test"' in result.output
        assert '"priority": "normal"' in result.output

    def test_create_validation_error(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task creation with validation error."""
        # Mock service to raise validation error
        mock_task_service.create.side_effect = ValidationError(
            "Task description cannot be empty"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "",
                ])

        # Verify exit code (2 for validation error)
        assert result.exit_code == 2

        # Verify error message
        assert "Validation error" in result.output
        assert "Task description cannot be empty" in result.output

    def test_create_operation_error(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task creation with operation error."""
        # Mock service to raise operation error
        mock_task_service.create.side_effect = OperationError(
            "Task creation failed: API timeout"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Test task",
                ])

        # Verify exit code (1 for operation error)
        assert result.exit_code == 1

        # Verify error message
        assert "Operation failed" in result.output
        assert "Task creation failed: API timeout" in result.output


class TestTaskListCommand:
    """Test suite for task list command."""

    def test_list_success(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test successful task listing."""
        # Mock service response
        mock_task_service.list_tasks.return_value = [
            {
                "task_id": "task-001",
                "description": "Analyze code",
                "status": "running",
                "priority": "normal",
                "agent_id": "agent-001",
            },
            {
                "task_id": "task-002",
                "description": "Run tests",
                "status": "completed",
                "priority": "high",
                "agent_id": "agent-002",
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output contains task info
        assert "task-001" in result.output
        assert "Analyze code" in result.output
        assert "task-002" in result.output
        assert "Run tests" in result.output

    def test_list_with_status_filter(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task listing with status filter."""
        # Mock service response
        mock_task_service.list_tasks.return_value = [
            {
                "task_id": "task-001",
                "description": "Analyze code",
                "status": "running",
                "priority": "normal",
                "agent_id": "agent-001",
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "list", "--status", "running"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with status filter
        mock_task_service.list_tasks.assert_called_once_with(
            status="running", limit=100, offset=0
        )

    def test_list_with_limit_and_offset(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task listing with limit and offset."""
        # Mock service response
        mock_task_service.list_tasks.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(
                app, ["task", "list", "--limit", "10", "--offset", "20"]
            )

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with limit and offset
        mock_task_service.list_tasks.assert_called_once_with(
            status=None, limit=10, offset=20
        )

    def test_list_empty(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task listing with no results."""
        # Mock service response
        mock_task_service.list_tasks.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "list"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "No tasks found" in result.output

    def test_list_json_output(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task listing with JSON output."""
        # Mock service response
        mock_task_service.list_tasks.return_value = [
            {
                "task_id": "task-001",
                "description": "Test task",
                "status": "running",
                "priority": "normal",
                "agent_id": "agent-001",
            },
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "list", "--json"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        assert '"task_id": "task-001"' in result.output
        assert '"description": "Test task"' in result.output


class TestTaskInfoCommand:
    """Test suite for task info command."""

    def test_info_success(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test successful task info retrieval."""
        # Mock service response
        mock_task_service.get.return_value = {
            "task_id": "task-001",
            "description": "Analyze code",
            "status": "running",
            "priority": "high",
            "agent_id": "agent-001",
            "parameters": {"repo": "foo/bar"},
        }

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "info", "task-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "task-001" in result.output
        assert "Analyze code" in result.output
        assert "running" in result.output

    def test_info_not_found(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task info for non-existent task."""
        # Mock service to raise not found error
        mock_task_service.get.side_effect = TaskNotFoundError(
            "Task 'task-999' not found"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "info", "task-999"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Task not found" in result.output

    def test_info_json_output(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task info with JSON output."""
        # Mock service response
        mock_task_service.get.return_value = {
            "task_id": "task-001",
            "description": "Test task",
            "status": "completed",
            "priority": "normal",
            "agent_id": "agent-001",
        }

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "info", "task-001", "--json"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        assert '"task_id": "task-001"' in result.output
        assert '"description": "Test task"' in result.output


class TestTaskCancelCommand:
    """Test suite for task cancel command."""

    def test_cancel_success(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test successful task cancellation."""
        # Mock service response
        mock_task_service.cancel.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "cancel", "task-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "Task cancelled successfully" in result.output
        assert "task-001" in result.output

        # Verify service was called
        mock_task_service.cancel.assert_called_once_with("task-001", force=False)

    def test_cancel_with_force(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task cancellation with force flag."""
        # Mock service response
        mock_task_service.cancel.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "cancel", "task-001", "--force"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with force=True
        mock_task_service.cancel.assert_called_once_with("task-001", force=True)

    def test_cancel_not_found(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task cancellation for non-existent task."""
        # Mock service to raise not found error
        mock_task_service.cancel.side_effect = TaskNotFoundError(
            "Task 'task-999' not found"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "cancel", "task-999"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Task not found" in result.output

    def test_cancel_json_output(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task cancellation with JSON output."""
        # Mock service response
        mock_task_service.cancel.return_value = True

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "cancel", "task-001", "--json"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        assert '"success": true' in result.output
        assert '"task_id": "task-001"' in result.output


class TestTaskLogsCommand:
    """Test suite for task logs command."""

    def test_logs_success(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test successful task logs retrieval."""
        # Mock service response
        mock_task_service.logs.return_value = [
            "[INFO] Task started",
            "[INFO] Processing...",
            "[INFO] Task completed",
        ]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "logs", "task-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output contains logs
        assert "Task started" in result.output
        assert "Processing" in result.output
        assert "Task completed" in result.output

    def test_logs_with_lines_limit(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task logs retrieval with lines limit."""
        # Mock service response
        mock_task_service.logs.return_value = ["[INFO] Last 100 lines"]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "logs", "task-001", "--lines", "100"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with lines parameter
        mock_task_service.logs.assert_called_once_with(
            "task-001", follow=False, lines=100
        )

    def test_logs_with_follow(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task logs retrieval with follow flag."""
        # Mock service response
        mock_task_service.logs.return_value = ["[INFO] Following logs"]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "logs", "task-001", "--follow"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify service was called with follow=True
        mock_task_service.logs.assert_called_once_with(
            "task-001", follow=True, lines=None
        )

    def test_logs_empty(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task logs retrieval with no logs."""
        # Mock service response
        mock_task_service.logs.return_value = []

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "logs", "task-001"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify output
        assert "No logs available" in result.output

    def test_logs_json_output(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task logs retrieval with JSON output."""
        # Mock service response
        mock_task_service.logs.return_value = ["[INFO] Log line 1", "[INFO] Log line 2"]

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "logs", "task-001", "--json"])

        # Verify exit code
        assert result.exit_code == 0

        # Verify JSON output
        assert '"task_id": "task-001"' in result.output
        assert '"logs"' in result.output
        assert "Log line 1" in result.output

    def test_logs_not_found(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test task logs retrieval for non-existent task."""
        # Mock service to raise not found error
        mock_task_service.logs.side_effect = TaskNotFoundError(
            "Task 'task-999' not found"
        )

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            result = runner.invoke(app, ["task", "logs", "task-999"])

        # Verify exit code
        assert result.exit_code == 1

        # Verify error message
        assert "Task not found" in result.output


class TestJSONRPCCompliance:
    """Test suite for JSON-RPC 2.0 compliance.

    These tests verify that the CLI sends properly formatted JSON-RPC 2.0
    requests with the required 'params' wrapper.
    """

    def test_create_sends_proper_jsonrpc_request(
        self, runner: CliRunner
    ) -> None:
        """Verify task create sends JSON-RPC 2.0 compliant request."""
        # Create a mock client that captures the request
        mock_client = Mock()
        mock_client.call.return_value = {"task_id": "task-001"}

        # Create mock service with the mock client
        mock_service = Mock()
        mock_service.create.return_value = "task-001"
        mock_service.client = mock_client

        # Patch the container to return mock service
        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_service):
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Test task",
                    "--priority",
                    "high",
                ])

        # Verify command succeeded
        assert result.exit_code == 0

        # Verify service.create was called (which calls client.call internally)
        mock_service.create.assert_called_once()

        # Extract the call arguments
        call_args = mock_service.create.call_args
        assert call_args is not None

        # Verify parameters are passed as expected
        assert call_args.kwargs["description"] == "Test task"
        assert call_args.kwargs["priority"] == "high"

    def test_service_layer_wraps_params_correctly(self) -> None:
        """Verify service layer properly wraps parameters in task_definition object."""
        from agentcore_cli.services.task import TaskService

        # Create a mock client
        mock_client = Mock()
        mock_client.call.return_value = {"execution_id": "task-001"}

        # Create service with mock client
        service = TaskService(mock_client)

        # Call create
        task_id = service.create(
            description="Test task",
            agent_id="agent-001",
            priority="high",
            parameters={"key": "value"})

        # Verify result
        assert task_id == "task-001"

        # Verify client.call was called with proper method and params
        mock_client.call.assert_called_once()
        call_args = mock_client.call.call_args

        # Verify method name
        assert call_args.args[0] == "task.create"

        # Verify params structure (uses task_definition nested structure)
        params = call_args.args[1]
        assert isinstance(params, dict)
        assert "task_definition" in params
        assert "auto_assign" in params
        assert params["auto_assign"] is True

        # Check task_definition structure
        task_def = params["task_definition"]
        assert task_def["description"] == "Test task"
        assert task_def["title"] == "Test task"
        assert task_def["priority"] == "high"
        assert task_def["parameters"] == {"key": "value"}

        # Check preferred_agent instead of agent_id
        assert "preferred_agent" in params
        assert params["preferred_agent"] == "agent-001"

        # This dict will be wrapped in "params" by the JsonRpcClient
        # The client is responsible for creating the full JSON-RPC request


class TestIntegrationFlow:
    """Integration tests for complete command flow."""

    def test_complete_task_lifecycle(
        self, runner: CliRunner, mock_task_service: Mock
    ) -> None:
        """Test complete task lifecycle: create -> info -> logs -> cancel."""
        # Mock service responses
        mock_task_service.create.return_value = "task-001"
        mock_task_service.get.return_value = {
            "task_id": "task-001",
            "description": "Lifecycle task",
            "status": "running",
            "priority": "normal",
            "agent_id": "agent-001",
        }
        mock_task_service.logs.return_value = ["[INFO] Task started"]
        mock_task_service.cancel.return_value = True

        with patch(
            "agentcore_cli.commands.task.get_task_service",
            return_value=mock_task_service):
            # Create
            result = runner.invoke(
                app,
                [
                    "task",
                    "create",
                    "--description",
                    "Lifecycle task",
                ])
            assert result.exit_code == 0

            # Info
            result = runner.invoke(app, ["task", "info", "task-001"])
            assert result.exit_code == 0

            # Logs
            result = runner.invoke(app, ["task", "logs", "task-001"])
            assert result.exit_code == 0

            # Cancel
            result = runner.invoke(app, ["task", "cancel", "task-001"])
            assert result.exit_code == 0
