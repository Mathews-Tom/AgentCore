"""
A2A task handler for agent runtime.

This module handles task assignment, execution, and status reporting
through the A2A protocol integration.
"""

import asyncio
from typing import Any

import structlog

from .a2a_client import A2AClient, A2AClientError
from .agent_lifecycle import AgentLifecycleManager

logger = structlog.get_logger()


class TaskHandlerError(Exception):
    """Base exception for task handler errors."""


class TaskExecutionError(TaskHandlerError):
    """Raised when task execution fails."""


class TaskHandler:
    """Handles A2A task assignment and execution for agents."""

    def __init__(
        self,
        a2a_client: A2AClient,
        lifecycle_manager: AgentLifecycleManager,
    ) -> None:
        """
        Initialize task handler.

        Args:
            a2a_client: A2A protocol client
            lifecycle_manager: Agent lifecycle manager
        """
        self._a2a_client = a2a_client
        self._lifecycle_manager = lifecycle_manager
        self._task_executors: dict[str, asyncio.Task[None]] = {}

    async def assign_task(
        self,
        task_id: str,
        agent_id: str,
        task_data: dict[str, Any],
    ) -> bool:
        """
        Assign task to agent.

        Args:
            task_id: Task ID
            agent_id: Agent ID
            task_data: Task input data

        Returns:
            True if task accepted

        Raises:
            TaskHandlerError: If assignment fails
        """
        try:
            # Verify agent exists and is running
            agent_state = await self._lifecycle_manager.get_agent_status(agent_id)

            if agent_state.status != "running":
                raise TaskHandlerError(
                    f"Agent {agent_id} not in running state: {agent_state.status}"
                )

            # Accept task in A2A protocol
            await self._a2a_client.accept_task(task_id=task_id, agent_id=agent_id)

            # Start task execution in background
            executor_task = asyncio.create_task(
                self._execute_task(task_id, agent_id, task_data)
            )
            self._task_executors[task_id] = executor_task

            logger.info(
                "task_assigned",
                task_id=task_id,
                agent_id=agent_id,
            )

            return True

        except Exception as e:
            logger.error(
                "task_assignment_failed",
                task_id=task_id,
                agent_id=agent_id,
                error=str(e),
            )
            raise TaskHandlerError(f"Failed to assign task: {e}") from e

    async def _execute_task(
        self,
        task_id: str,
        agent_id: str,
        task_data: dict[str, Any],
    ) -> None:
        """
        Execute task and report status to A2A protocol.

        Args:
            task_id: Task ID
            agent_id: Agent ID
            task_data: Task input data
        """
        try:
            # Mark task as started
            await self._a2a_client.start_task(task_id=task_id, agent_id=agent_id)

            logger.info(
                "task_execution_started",
                task_id=task_id,
                agent_id=agent_id,
            )

            # Execute task (placeholder - actual execution depends on agent type)
            # In real implementation, this would:
            # 1. Pass task data to agent container
            # 2. Monitor execution
            # 3. Collect results
            await asyncio.sleep(1)  # Placeholder for actual execution

            # Simulate task result
            result = {
                "status": "success",
                "output": "Task executed successfully",
                "agent_id": agent_id,
                "task_id": task_id,
            }

            # Mark task as completed
            await self._a2a_client.complete_task(
                task_id=task_id,
                agent_id=agent_id,
                result=result,
            )

            logger.info(
                "task_execution_completed",
                task_id=task_id,
                agent_id=agent_id,
            )

        except Exception as e:
            # Mark task as failed
            try:
                await self._a2a_client.fail_task(
                    task_id=task_id,
                    agent_id=agent_id,
                    error=str(e),
                )
            except A2AClientError as fail_error:
                logger.error(
                    "task_failure_report_failed",
                    task_id=task_id,
                    agent_id=agent_id,
                    original_error=str(e),
                    report_error=str(fail_error),
                )

            logger.error(
                "task_execution_failed",
                task_id=task_id,
                agent_id=agent_id,
                error=str(e),
            )

        finally:
            # Cleanup executor tracking
            if task_id in self._task_executors:
                del self._task_executors[task_id]

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel running task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled successfully

        Raises:
            TaskHandlerError: If cancellation fails
        """
        if task_id not in self._task_executors:
            raise TaskHandlerError(f"Task {task_id} not found or not running")

        try:
            # Cancel executor task
            self._task_executors[task_id].cancel()
            await asyncio.sleep(0.1)  # Allow cancellation to propagate

            logger.info("task_cancelled", task_id=task_id)

            return True

        except Exception as e:
            logger.error(
                "task_cancellation_failed",
                task_id=task_id,
                error=str(e),
            )
            raise TaskHandlerError(f"Failed to cancel task: {e}") from e

    async def get_active_tasks(self) -> list[str]:
        """
        Get list of currently active task IDs.

        Returns:
            List of task IDs
        """
        return list(self._task_executors.keys())

    async def shutdown(self) -> None:
        """Shutdown task handler and cancel all running tasks."""
        logger.info("task_handler_shutting_down", active_tasks=len(self._task_executors))

        # Cancel all running tasks
        for task_id, executor_task in list(self._task_executors.items()):
            try:
                executor_task.cancel()
                await asyncio.wait_for(executor_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger.warning(
                    "task_cleanup_failed",
                    task_id=task_id,
                    error=str(e),
                )

        self._task_executors.clear()

        logger.info("task_handler_shutdown_complete")
