"""Task service for managing task creation and execution.

This service provides high-level operations for task management without
any knowledge of JSON-RPC protocol details.
"""

from __future__ import annotations

from typing import Any

from agentcore_cli.protocol.jsonrpc import JsonRpcClient
from agentcore_cli.services.exceptions import (
    ValidationError,
    TaskNotFoundError,
    OperationError,
)


class TaskService:
    """Service for task operations.

    Provides business operations for task lifecycle management:
    - Task creation
    - Task listing and filtering
    - Task information retrieval
    - Task cancellation
    - Task logs retrieval

    This service abstracts JSON-RPC protocol details and focuses on
    business logic and domain validation.

    Args:
        client: JSON-RPC client for API communication

    Attributes:
        client: JSON-RPC client instance

    Example:
        >>> transport = HttpTransport("http://localhost:8001")
        >>> client = JsonRpcClient(transport)
        >>> service = TaskService(client)
        >>> task_id = service.create("Analyze code repository", {"repo": "foo/bar"})
        >>> print(task_id)
        'task-001'
    """

    def __init__(self, client: JsonRpcClient) -> None:
        """Initialize task service.

        Args:
            client: JSON-RPC client for API communication
        """
        self.client = client

    def create(
        self,
        description: str,
        agent_id: str | None = None,
        priority: str = "normal",
        parameters: dict[str, Any] | None = None,
        task_type: str | None = None,
    ) -> str:
        """Create a new task.

        Args:
            description: Task description (used as title if no separate title)
            agent_id: Optional agent ID to assign task to
            priority: Task priority ("low", "normal", "high", "critical")
            parameters: Optional task parameters
            task_type: Optional task type (default: "cli.task")

        Returns:
            Task ID (string)

        Raises:
            ValidationError: If validation fails
            OperationError: If task creation fails

        Example:
            >>> task_id = service.create(
            ...     "Analyze code",
            ...     agent_id="agent-001",
            ...     priority="high",
            ...     parameters={"repo": "foo/bar"}
            ... )
            >>> print(task_id)
            'task-001'
        """
        # Business validation
        if not description or not description.strip():
            raise ValidationError("Task description cannot be empty")

        valid_priorities = ["low", "normal", "high", "critical"]
        if priority not in valid_priorities:
            raise ValidationError(
                f"Invalid priority: {priority}. Must be one of {valid_priorities}"
            )

        # Build task_definition object as expected by the A2A API
        task_definition: dict[str, Any] = {
            "task_type": task_type if task_type else "cli.task",
            "title": description.strip(),
            "description": description.strip(),
            "priority": priority,
        }

        if parameters:
            task_definition["parameters"] = parameters

        # Build request parameters
        params: dict[str, Any] = {
            "task_definition": task_definition,
            "auto_assign": True,
        }

        if agent_id:
            params["preferred_agent"] = agent_id

        # Call JSON-RPC method
        try:
            result = self.client.call("task.create", params)
        except Exception as e:
            raise OperationError(f"Task creation failed: {str(e)}")

        # Validate result - API returns execution_id and task_id
        # Use execution_id for all operations (get, cancel, etc.)
        execution_id = result.get("execution_id") or result.get("task_id")
        if not execution_id:
            raise OperationError("API did not return execution_id")

        return str(execution_id)

    def list_tasks(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List tasks with optional filtering.

        Args:
            status: Optional status filter ("pending", "running", "completed", "failed", "cancelled")
            limit: Maximum number of tasks to return (default: 100)
            offset: Number of tasks to skip (default: 0)

        Returns:
            List of task dictionaries

        Raises:
            ValidationError: If parameters are invalid
            OperationError: If listing fails

        Example:
            >>> tasks = service.list(status="running", limit=10)
            >>> for task in tasks:
            ...     print(task["description"])
            'Analyze code'
            'Run tests'
        """
        # Validation
        if limit <= 0:
            raise ValidationError("Limit must be positive")

        if offset < 0:
            raise ValidationError("Offset cannot be negative")

        valid_statuses = ["pending", "assigned", "running", "completed", "failed", "cancelled", "timeout"]
        if status and status not in valid_statuses:
            raise ValidationError(
                f"Invalid status: {status}. Must be one of {valid_statuses}"
            )

        # Prepare parameters - API uses task.query method
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if status:
            params["status"] = status

        # Call JSON-RPC method - using task.query instead of task.list
        try:
            result = self.client.call("task.query", params)
        except Exception as e:
            raise OperationError(f"Task listing failed: {str(e)}")

        # Extract tasks - task.query returns {tasks: [], total: int, has_more: bool}
        tasks = result.get("tasks", [])
        if not isinstance(tasks, list):
            raise OperationError("API returned invalid tasks list")

        return tasks

    def get(self, task_id: str) -> dict[str, Any]:
        """Get task information by ID.

        Args:
            task_id: Task identifier (execution_id in A2A protocol)

        Returns:
            Task information dictionary

        Raises:
            ValidationError: If task_id is empty
            TaskNotFoundError: If task does not exist
            OperationError: If retrieval fails

        Example:
            >>> info = service.get("task-001")
            >>> print(info["description"])
            'Analyze code'
        """
        # Validation
        if not task_id or not task_id.strip():
            raise ValidationError("Task ID cannot be empty")

        # Call JSON-RPC method - API uses execution_id
        try:
            result = self.client.call("task.get", {"execution_id": task_id.strip()})
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise TaskNotFoundError(f"Task '{task_id}' not found")
            raise OperationError(f"Task retrieval failed: {str(e)}")

        # Validate result
        task = result.get("task")
        if not task:
            raise OperationError("API did not return task information")

        return dict(task)

    def cancel(self, task_id: str, force: bool = False) -> bool:
        """Cancel a task.

        Args:
            task_id: Task identifier (execution_id in A2A protocol)
            force: Force cancellation even if task is running (ignored by API)

        Returns:
            True if successful

        Raises:
            ValidationError: If task_id is empty
            TaskNotFoundError: If task does not exist
            OperationError: If cancellation fails

        Example:
            >>> success = service.cancel("task-001", force=True)
            >>> print(success)
            True
        """
        # Validation
        if not task_id or not task_id.strip():
            raise ValidationError("Task ID cannot be empty")

        # Prepare parameters - API uses execution_id
        params: dict[str, Any] = {
            "execution_id": task_id.strip(),
        }

        # Call JSON-RPC method
        try:
            result = self.client.call("task.cancel", params)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise TaskNotFoundError(f"Task '{task_id}' not found")
            raise OperationError(f"Task cancellation failed: {str(e)}")

        # Validate result
        success = result.get("success", False)
        return bool(success)

    def logs(
        self,
        task_id: str,
        follow: bool = False,
        lines: int | None = None,
    ) -> list[str]:
        """Get task logs.

        Args:
            task_id: Task identifier
            follow: Whether to follow logs in real-time (default: False)
            lines: Number of lines to retrieve (None for all)

        Returns:
            List of log lines

        Raises:
            ValidationError: If parameters are invalid
            TaskNotFoundError: If task does not exist
            OperationError: If log retrieval fails

        Example:
            >>> logs = service.logs("task-001", lines=100)
            >>> for line in logs:
            ...     print(line)
            '[INFO] Task started'
            '[INFO] Processing...'
        """
        # Validation
        if not task_id or not task_id.strip():
            raise ValidationError("Task ID cannot be empty")

        if lines is not None and lines <= 0:
            raise ValidationError("Lines must be positive")

        # Prepare parameters
        params: dict[str, Any] = {
            "task_id": task_id.strip(),
            "follow": follow,
        }

        if lines is not None:
            params["lines"] = lines

        # Call JSON-RPC method
        try:
            result = self.client.call("task.logs", params)
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise TaskNotFoundError(f"Task '{task_id}' not found")
            raise OperationError(f"Task log retrieval failed: {str(e)}")

        # Extract logs
        logs = result.get("logs", [])
        if not isinstance(logs, list):
            raise OperationError("API returned invalid logs list")

        return logs
