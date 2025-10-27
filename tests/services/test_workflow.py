"""Unit tests for WorkflowService.

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

from agentcore_cli.services.workflow import WorkflowService
from agentcore_cli.services.exceptions import (
    ValidationError,
    WorkflowNotFoundError,
    OperationError)


class TestWorkflowServiceRun:
    """Test WorkflowService.run() method."""

    def test_run_success(self) -> None:
        """Test successful workflow execution."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"workflow_id": "workflow-001"}
        service = WorkflowService(mock_client)

        definition = {
            "name": "test-workflow",
            "steps": [{"agent": "analyzer", "task": "analyze"}],
        }

        # Act
        workflow_id = service.run(definition)

        # Assert
        assert workflow_id == "workflow-001"
        mock_client.call.assert_called_once_with(
            "workflow.run",
            {"definition": definition})

    def test_run_with_parameters(self) -> None:
        """Test execution with parameters."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"workflow_id": "workflow-002"}
        service = WorkflowService(mock_client)

        definition = {
            "name": "test-workflow",
            "steps": [{"agent": "analyzer", "task": "analyze"}],
        }

        # Act
        workflow_id = service.run(definition, parameters={"repo": "foo/bar"})

        # Assert
        assert workflow_id == "workflow-002"
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["parameters"] == {"repo": "foo/bar"}

    def test_run_empty_definition_raises_validation_error(self) -> None:
        """Test that empty definition raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Workflow definition cannot be empty"):
            service.run({})

    def test_run_non_dict_definition_raises_validation_error(self) -> None:
        """Test that non-dict definition raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Workflow definition must be a dictionary"):
            service.run("not-a-dict")  # type: ignore

    def test_run_missing_name_raises_validation_error(self) -> None:
        """Test that missing name raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Workflow definition must have a 'name' field"):
            service.run({"steps": []})

    def test_run_missing_steps_raises_validation_error(self) -> None:
        """Test that missing steps raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Workflow definition must have a 'steps' field"):
            service.run({"name": "test"})

    def test_run_non_list_steps_raises_validation_error(self) -> None:
        """Test that non-list steps raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Workflow 'steps' must be a list"):
            service.run({"name": "test", "steps": "not-a-list"})

    def test_run_empty_steps_raises_validation_error(self) -> None:
        """Test that empty steps raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Workflow must have at least one step"):
            service.run({"name": "test", "steps": []})


class TestWorkflowServiceListWorkflows:
    """Test WorkflowService.list() method."""

    def test_list_success(self) -> None:
        """Test successful workflow listing."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "workflows": [
                {"workflow_id": "workflow-001", "name": "workflow-1"},
                {"workflow_id": "workflow-002", "name": "workflow-2"},
            ]
        }
        service = WorkflowService(mock_client)

        # Act
        workflows = service.list_workflows()

        # Assert
        assert len(workflows) == 2
        mock_client.call.assert_called_once_with(
            "workflow.list",
            {"limit": 100, "offset": 0})

    def test_list_with_status_filter(self) -> None:
        """Test listing with status filter."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"workflows": []}
        service = WorkflowService(mock_client)

        # Act
        service.list_workflows(status="running", limit=10)

        # Assert
        mock_client.call.assert_called_once_with(
            "workflow.list",
            {"limit": 10, "offset": 0, "status": "running"})

    def test_list_invalid_status_raises_validation_error(self) -> None:
        """Test that invalid status raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid status"):
            service.list_workflows(status="invalid")

    def test_list_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Workflow listing failed"):
            service.list_workflows()

    def test_list_invalid_response_raises_operation_error(self) -> None:
        """Test that invalid response raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"workflows": "not-a-list"}
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API returned invalid workflows list"):
            service.list_workflows()


class TestWorkflowServiceGet:
    """Test WorkflowService.get() method."""

    def test_get_success(self) -> None:
        """Test successful workflow retrieval."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "workflow": {"workflow_id": "workflow-001", "name": "test-workflow"}
        }
        service = WorkflowService(mock_client)

        # Act
        workflow = service.get("workflow-001")

        # Assert
        assert workflow["workflow_id"] == "workflow-001"

    def test_get_not_found_raises_workflow_not_found_error(self) -> None:
        """Test that 'not found' error raises WorkflowNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Workflow not found")
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(WorkflowNotFoundError, match="Workflow 'workflow-001' not found"):
            service.get("workflow-001")

    def test_get_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Workflow retrieval failed"):
            service.get("workflow-001")

    def test_get_missing_workflow_raises_operation_error(self) -> None:
        """Test that missing workflow raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {}
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API did not return workflow information"):
            service.get("workflow-001")


class TestWorkflowServiceStop:
    """Test WorkflowService.stop() method."""

    def test_stop_success(self) -> None:
        """Test successful workflow stopping."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = WorkflowService(mock_client)

        # Act
        success = service.stop("workflow-001")

        # Assert
        assert success is True
        mock_client.call.assert_called_once_with(
            "workflow.stop",
            {"workflow_id": "workflow-001", "force": False})

    def test_stop_with_force(self) -> None:
        """Test stopping with force flag."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = WorkflowService(mock_client)

        # Act
        service.stop("workflow-001", force=True)

        # Assert
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["force"] is True

    def test_stop_not_found_raises_workflow_not_found_error(self) -> None:
        """Test that 'not found' error raises WorkflowNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Workflow not found")
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(WorkflowNotFoundError, match="Workflow 'workflow-001' not found"):
            service.stop("workflow-001")

    def test_stop_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = WorkflowService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Workflow stopping failed"):
            service.stop("workflow-001")
