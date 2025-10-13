"""Tests for task handler service."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentcore.agent_runtime.models.agent_state import AgentExecutionState
from agentcore.agent_runtime.services.a2a_client import A2AClient, A2AClientError
from agentcore.agent_runtime.services.agent_lifecycle import AgentLifecycleManager, AgentNotFoundException
from agentcore.agent_runtime.services.task_handler import (
    TaskExecutionError,
    TaskHandler,
    TaskHandlerError,
)


@pytest.fixture
def mock_a2a_client() -> A2AClient:
    """Create mock A2A client."""
    client = MagicMock(spec=A2AClient)
    client.accept_task = AsyncMock()
    client.start_task = AsyncMock()
    client.complete_task = AsyncMock()
    client.fail_task = AsyncMock()
    return client


@pytest.fixture
def mock_lifecycle_manager() -> AgentLifecycleManager:
    """Create mock lifecycle manager."""
    manager = MagicMock(spec=AgentLifecycleManager)
    manager.get_agent_status = AsyncMock(return_value=AgentExecutionState(
        agent_id="test-agent-001",
        status="running",
        container_id="test-container-123",
    ))
    return manager


@pytest.fixture
def task_handler(mock_a2a_client: A2AClient, mock_lifecycle_manager: AgentLifecycleManager) -> TaskHandler:
    """Create task handler with mocked dependencies."""
    return TaskHandler(
        a2a_client=mock_a2a_client,
        lifecycle_manager=mock_lifecycle_manager,
    )


@pytest.fixture
def task_data() -> dict[str, Any]:
    """Create test task data."""
    return {
        "goal": "Test task goal",
        "input": {"param1": "value1"},
        "max_steps": 10,
    }


@pytest.mark.asyncio
class TestTaskHandler:
    """Test task handler functionality."""

    async def test_assign_task_success(
        self,
        task_handler: TaskHandler,
        mock_a2a_client: A2AClient,
        task_data: dict[str, Any],
    ) -> None:
        """Test successful task assignment."""
        result = await task_handler.assign_task(
            task_id="task-001",
            agent_id="test-agent-001",
            task_data=task_data,
        )

        assert result is True
        mock_a2a_client.accept_task.assert_called_once_with(
            task_id="task-001",
            agent_id="test-agent-001",
        )

        # Task executor should be created
        assert "task-001" in task_handler._task_executors

    async def test_assign_task_agent_not_running(
        self,
        task_handler: TaskHandler,
        mock_lifecycle_manager: AgentLifecycleManager,
        task_data: dict[str, Any],
    ) -> None:
        """Test task assignment when agent not running."""
        mock_lifecycle_manager.get_agent_status = AsyncMock(return_value=AgentExecutionState(
            agent_id="test-agent-001",
            status="paused",
            container_id="test-container-123",
        ))

        with pytest.raises(TaskHandlerError, match="not in running state"):
            await task_handler.assign_task(
                task_id="task-001",
                agent_id="test-agent-001",
                task_data=task_data,
            )

    async def test_assign_task_agent_not_found(
        self,
        task_handler: TaskHandler,
        mock_lifecycle_manager: AgentLifecycleManager,
        task_data: dict[str, Any],
    ) -> None:
        """Test task assignment when agent doesn't exist."""
        mock_lifecycle_manager.get_agent_status = AsyncMock(
            side_effect=AgentNotFoundException("Agent not found")
        )

        with pytest.raises(TaskHandlerError):
            await task_handler.assign_task(
                task_id="task-001",
                agent_id="nonexistent",
                task_data=task_data,
            )

    async def test_assign_task_a2a_error(
        self,
        task_handler: TaskHandler,
        mock_a2a_client: A2AClient,
        task_data: dict[str, Any],
    ) -> None:
        """Test task assignment with A2A client error."""
        mock_a2a_client.accept_task = AsyncMock(
            side_effect=A2AClientError("A2A protocol error")
        )

        with pytest.raises(TaskHandlerError):
            await task_handler.assign_task(
                task_id="task-001",
                agent_id="test-agent-001",
                task_data=task_data,
            )

    async def test_execute_task_success(
        self,
        task_handler: TaskHandler,
        mock_a2a_client: A2AClient,
        task_data: dict[str, Any],
    ) -> None:
        """Test successful task execution."""
        await task_handler.assign_task(
            task_id="task-001",
            agent_id="test-agent-001",
            task_data=task_data,
        )

        # Wait for task execution to complete
        await asyncio.sleep(1.5)

        # Verify task lifecycle calls
        mock_a2a_client.start_task.assert_called_once_with(
            task_id="task-001",
            agent_id="test-agent-001",
        )
        mock_a2a_client.complete_task.assert_called_once()

        # Task executor should be cleaned up
        assert "task-001" not in task_handler._task_executors

    async def test_execute_task_failure(
        self,
        task_handler: TaskHandler,
        mock_a2a_client: A2AClient,
        task_data: dict[str, Any],
    ) -> None:
        """Test task execution failure."""
        # Make start_task fail
        mock_a2a_client.start_task = AsyncMock(
            side_effect=A2AClientError("Start failed")
        )

        await task_handler.assign_task(
            task_id="task-001",
            agent_id="test-agent-001",
            task_data=task_data,
        )

        # Wait for task execution to fail
        await asyncio.sleep(1.5)

        # Verify failure was reported
        mock_a2a_client.fail_task.assert_called_once()

        # Task executor should be cleaned up
        assert "task-001" not in task_handler._task_executors

    async def test_execute_task_failure_report_error(
        self,
        task_handler: TaskHandler,
        mock_a2a_client: A2AClient,
        task_data: dict[str, Any],
    ) -> None:
        """Test task execution failure when failure reporting also fails."""
        # Make both start_task and fail_task fail
        mock_a2a_client.start_task = AsyncMock(
            side_effect=A2AClientError("Start failed")
        )
        mock_a2a_client.fail_task = AsyncMock(
            side_effect=A2AClientError("Fail report failed")
        )

        await task_handler.assign_task(
            task_id="task-001",
            agent_id="test-agent-001",
            task_data=task_data,
        )

        # Wait for task execution to fail
        await asyncio.sleep(1.5)

        # Both calls should have been attempted
        mock_a2a_client.start_task.assert_called_once()
        mock_a2a_client.fail_task.assert_called_once()

        # Task executor should still be cleaned up
        assert "task-001" not in task_handler._task_executors

    async def test_cancel_task_success(
        self,
        task_handler: TaskHandler,
        task_data: dict[str, Any],
    ) -> None:
        """Test successful task cancellation."""
        await task_handler.assign_task(
            task_id="task-001",
            agent_id="test-agent-001",
            task_data=task_data,
        )

        # Cancel task
        result = await task_handler.cancel_task("task-001")

        assert result is True

    async def test_cancel_task_not_found(self, task_handler: TaskHandler) -> None:
        """Test cancelling non-existent task."""
        with pytest.raises(TaskHandlerError, match="not found or not running"):
            await task_handler.cancel_task("nonexistent")

    async def test_get_active_tasks(
        self,
        task_handler: TaskHandler,
        task_data: dict[str, Any],
    ) -> None:
        """Test getting active tasks."""
        # Initially empty
        active = await task_handler.get_active_tasks()
        assert len(active) == 0

        # Assign multiple tasks
        await task_handler.assign_task("task-001", "test-agent-001", task_data)
        await task_handler.assign_task("task-002", "test-agent-001", task_data)
        await task_handler.assign_task("task-003", "test-agent-001", task_data)

        # Should have 3 active tasks
        active = await task_handler.get_active_tasks()
        assert len(active) == 3
        assert "task-001" in active
        assert "task-002" in active
        assert "task-003" in active

    async def test_shutdown_with_running_tasks(
        self,
        task_handler: TaskHandler,
        task_data: dict[str, Any],
    ) -> None:
        """Test shutdown with running tasks."""
        # Assign multiple tasks
        await task_handler.assign_task("task-001", "test-agent-001", task_data)
        await task_handler.assign_task("task-002", "test-agent-001", task_data)

        # Verify tasks are running
        active = await task_handler.get_active_tasks()
        assert len(active) == 2

        # Shutdown
        await task_handler.shutdown()

        # All tasks should be cleared
        active = await task_handler.get_active_tasks()
        assert len(active) == 0
        assert len(task_handler._task_executors) == 0

    async def test_shutdown_empty(self, task_handler: TaskHandler) -> None:
        """Test shutdown with no running tasks."""
        # Should complete without error
        await task_handler.shutdown()

        assert len(task_handler._task_executors) == 0

    async def test_shutdown_timeout_handling(
        self,
        task_handler: TaskHandler,
        task_data: dict[str, Any],
    ) -> None:
        """Test shutdown handles task timeout."""
        await task_handler.assign_task("task-001", "test-agent-001", task_data)

        # Shutdown should handle timeout gracefully
        await task_handler.shutdown()

        # All tasks should be cleared despite timeout
        assert len(task_handler._task_executors) == 0

    async def test_multiple_concurrent_tasks(
        self,
        task_handler: TaskHandler,
        mock_a2a_client: A2AClient,
        task_data: dict[str, Any],
    ) -> None:
        """Test handling multiple concurrent tasks."""
        # Assign multiple tasks concurrently
        await asyncio.gather(
            task_handler.assign_task("task-001", "test-agent-001", task_data),
            task_handler.assign_task("task-002", "test-agent-001", task_data),
            task_handler.assign_task("task-003", "test-agent-001", task_data),
        )

        # All tasks should be tracked
        active = await task_handler.get_active_tasks()
        assert len(active) == 3

        # Wait for all tasks to complete
        await asyncio.sleep(2.0)

        # All should complete successfully
        assert mock_a2a_client.complete_task.call_count == 3

        # All tasks should be cleaned up
        active = await task_handler.get_active_tasks()
        assert len(active) == 0
