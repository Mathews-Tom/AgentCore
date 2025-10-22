"""Unit tests for TaskService.

Tests cover:
- Business validation
- Parameter transformation
- JSON-RPC method calls
- Error handling
- Result validation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from agentcore_cli.services.task import TaskService
from agentcore_cli.services.exceptions import (
    ValidationError,
    TaskNotFoundError,
    OperationError,
)


class TestTaskServiceCreate:
    """Test TaskService.create() method."""

    def test_create_success(self) -> None:
        """Test successful task creation."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"task_id": "task-001"}
        service = TaskService(mock_client)

        # Act
        task_id = service.create("Analyze code")

        # Assert
        assert task_id == "task-001"
        mock_client.call.assert_called_once_with(
            "task.create",
            {"description": "Analyze code", "priority": "normal"},
        )

    def test_create_with_all_parameters(self) -> None:
        """Test creation with all optional parameters."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"task_id": "task-002"}
        service = TaskService(mock_client)

        # Act
        task_id = service.create(
            "Analyze code",
            agent_id="agent-001",
            priority="high",
            parameters={"repo": "foo/bar"},
        )

        # Assert
        assert task_id == "task-002"
        mock_client.call.assert_called_once_with(
            "task.create",
            {
                "description": "Analyze code",
                "priority": "high",
                "agent_id": "agent-001",
                "parameters": {"repo": "foo/bar"},
            },
        )

    def test_create_empty_description_raises_validation_error(self) -> None:
        """Test that empty description raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Task description cannot be empty"):
            service.create("")

    def test_create_invalid_priority_raises_validation_error(self) -> None:
        """Test that invalid priority raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid priority"):
            service.create("Analyze code", priority="invalid")

    def test_create_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Task creation failed"):
            service.create("Analyze code")

    def test_create_missing_task_id_raises_operation_error(self) -> None:
        """Test that missing task_id raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {}
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API did not return task_id"):
            service.create("Analyze code")


class TestTaskServiceListTasks:
    """Test TaskService.list() method."""

    def test_list_success(self) -> None:
        """Test successful task listing."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "tasks": [
                {"task_id": "task-001", "description": "Test 1"},
                {"task_id": "task-002", "description": "Test 2"},
            ]
        }
        service = TaskService(mock_client)

        # Act
        tasks = service.list_tasks()

        # Assert
        assert len(tasks) == 2
        mock_client.call.assert_called_once_with(
            "task.list",
            {"limit": 100, "offset": 0},
        )

    def test_list_with_status_filter(self) -> None:
        """Test listing with status filter."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"tasks": []}
        service = TaskService(mock_client)

        # Act
        service.list_tasks(status="running", limit=10, offset=5)

        # Assert
        mock_client.call.assert_called_once_with(
            "task.list",
            {"limit": 10, "offset": 5, "status": "running"},
        )

    def test_list_invalid_limit_raises_validation_error(self) -> None:
        """Test that invalid limit raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Limit must be positive"):
            service.list_tasks(limit=-1)

    def test_list_invalid_status_raises_validation_error(self) -> None:
        """Test that invalid status raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Invalid status"):
            service.list_tasks(status="invalid")

    def test_list_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Task listing failed"):
            service.list_tasks()

    def test_list_invalid_response_raises_operation_error(self) -> None:
        """Test that invalid response raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"tasks": "not-a-list"}
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API returned invalid tasks list"):
            service.list_tasks()


class TestTaskServiceGet:
    """Test TaskService.get() method."""

    def test_get_success(self) -> None:
        """Test successful task retrieval."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {
            "task": {"task_id": "task-001", "description": "Test task"}
        }
        service = TaskService(mock_client)

        # Act
        task = service.get("task-001")

        # Assert
        assert task["task_id"] == "task-001"
        mock_client.call.assert_called_once_with(
            "task.get",
            {"task_id": "task-001"},
        )

    def test_get_empty_id_raises_validation_error(self) -> None:
        """Test that empty task_id raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Task ID cannot be empty"):
            service.get("")

    def test_get_not_found_raises_task_not_found_error(self) -> None:
        """Test that 'not found' error raises TaskNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Task not found")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(TaskNotFoundError, match="Task 'task-001' not found"):
            service.get("task-001")

    def test_get_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Task retrieval failed"):
            service.get("task-001")

    def test_get_missing_task_raises_operation_error(self) -> None:
        """Test that missing task raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {}
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API did not return task information"):
            service.get("task-001")


class TestTaskServiceCancel:
    """Test TaskService.cancel() method."""

    def test_cancel_success(self) -> None:
        """Test successful task cancellation."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = TaskService(mock_client)

        # Act
        success = service.cancel("task-001")

        # Assert
        assert success is True
        mock_client.call.assert_called_once_with(
            "task.cancel",
            {"task_id": "task-001", "force": False},
        )

    def test_cancel_with_force(self) -> None:
        """Test cancellation with force flag."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"success": True}
        service = TaskService(mock_client)

        # Act
        service.cancel("task-001", force=True)

        # Assert
        call_args = mock_client.call.call_args[0]
        assert call_args[1]["force"] is True

    def test_cancel_not_found_raises_task_not_found_error(self) -> None:
        """Test that 'not found' error raises TaskNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Task not found")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(TaskNotFoundError, match="Task 'task-001' not found"):
            service.cancel("task-001")

    def test_cancel_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Task cancellation failed"):
            service.cancel("task-001")


class TestTaskServiceLogs:
    """Test TaskService.logs() method."""

    def test_logs_success(self) -> None:
        """Test successful log retrieval."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"logs": ["[INFO] Log 1", "[INFO] Log 2"]}
        service = TaskService(mock_client)

        # Act
        logs = service.logs("task-001")

        # Assert
        assert len(logs) == 2
        assert logs[0] == "[INFO] Log 1"
        mock_client.call.assert_called_once_with(
            "task.logs",
            {"task_id": "task-001", "follow": False},
        )

    def test_logs_with_parameters(self) -> None:
        """Test log retrieval with optional parameters."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"logs": []}
        service = TaskService(mock_client)

        # Act
        service.logs("task-001", follow=True, lines=100)

        # Assert
        mock_client.call.assert_called_once_with(
            "task.logs",
            {"task_id": "task-001", "follow": True, "lines": 100},
        )

    def test_logs_invalid_lines_raises_validation_error(self) -> None:
        """Test that invalid lines raises ValidationError."""
        # Arrange
        mock_client = Mock()
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(ValidationError, match="Lines must be positive"):
            service.logs("task-001", lines=-1)

    def test_logs_not_found_raises_task_not_found_error(self) -> None:
        """Test that 'not found' error raises TaskNotFoundError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("Task not found")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(TaskNotFoundError, match="Task 'task-001' not found"):
            service.logs("task-001")

    def test_logs_api_error_raises_operation_error(self) -> None:
        """Test that API errors raise OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.side_effect = Exception("API error")
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="Task log retrieval failed"):
            service.logs("task-001")

    def test_logs_invalid_response_raises_operation_error(self) -> None:
        """Test that invalid response raises OperationError."""
        # Arrange
        mock_client = Mock()
        mock_client.call.return_value = {"logs": "not-a-list"}
        service = TaskService(mock_client)

        # Act & Assert
        with pytest.raises(OperationError, match="API returned invalid logs list"):
            service.logs("task-001")
