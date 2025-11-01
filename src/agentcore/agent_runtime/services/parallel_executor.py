"""Parallel tool execution with dependency management.

Implements GAP (Graph-based Async Processing) for executing multiple tools
concurrently while respecting dependencies.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import structlog

from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionRequest,
    ToolExecutionStatus,
    ToolResult,
)
from agentcore.agent_runtime.services.tool_executor import ToolExecutor

logger = structlog.get_logger()


@dataclass
class ParallelTask:
    """Task for parallel execution."""

    task_id: str
    request: ToolExecutionRequest
    dependencies: list[str] = field(default_factory=list)
    result: ToolResult | None = None
    status: str = "pending"  # pending, running, completed, failed


class ParallelExecutor:
    """Execute multiple tools in parallel with dependency management."""

    def __init__(self, executor: ToolExecutor):
        """Initialize parallel executor.

        Args:
            executor: Tool executor instance
        """
        self.executor = executor

    async def execute_parallel(
        self,
        tasks: list[ParallelTask],
        max_concurrent: int = 10,
    ) -> dict[str, ToolResult]:
        """Execute multiple tasks in parallel respecting dependencies.

        Args:
            tasks: List of tasks to execute
            max_concurrent: Maximum concurrent executions

        Returns:
            Dictionary mapping task_id to execution results
        """
        # Build dependency graph
        task_map = {task.task_id: task for task in tasks}
        results: dict[str, ToolResult] = {}
        pending_tasks = set(task.task_id for task in tasks)
        running_tasks: set[str] = set()

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_task(task: ParallelTask) -> None:
            """Execute a single task."""
            async with semaphore:
                try:
                    task.status = "running"
                    running_tasks.add(task.task_id)

                    logger.info(
                        "parallel_task_started",
                        task_id=task.task_id,
                        tool_id=task.request.tool_id,
                    )

                    # Execute the tool
                    result = await self.executor.execute(task.request)
                    task.result = result
                    results[task.task_id] = result

                    if result.status == ToolExecutionStatus.SUCCESS:
                        task.status = "completed"
                        logger.info(
                            "parallel_task_completed",
                            task_id=task.task_id,
                            tool_id=task.request.tool_id,
                        )
                    else:
                        task.status = "failed"
                        logger.error(
                            "parallel_task_failed",
                            task_id=task.task_id,
                            tool_id=task.request.tool_id,
                            error=result.error,
                        )

                except Exception as e:
                    task.status = "failed"
                    logger.error(
                        "parallel_task_exception",
                        task_id=task.task_id,
                        error=str(e),
                    )
                    # Create error result
                    from datetime import datetime, timezone
                    from uuid import uuid4

                    task.result = ToolResult(
                        request_id=str(uuid4()),
                        tool_id=task.request.tool_id,
                        status=ToolExecutionStatus.FAILED,
                        error=str(e),
                        error_type=type(e).__name__,
                        execution_time_ms=0,
                        timestamp=datetime.now(timezone.utc),
                    )
                    results[task.task_id] = task.result

                finally:
                    running_tasks.discard(task.task_id)
                    pending_tasks.discard(task.task_id)

        def can_execute(task: ParallelTask) -> bool:
            """Check if task dependencies are satisfied."""
            if task.status != "pending":
                return False
            for dep_id in task.dependencies:
                dep_task = task_map.get(dep_id)
                if not dep_task or dep_task.status != "completed":
                    return False
            return True

        # Execute tasks
        active_tasks: set[asyncio.Task] = set()

        while pending_tasks or active_tasks:
            # Find tasks ready to execute
            ready_tasks = [
                task for task in tasks if can_execute(task) and task.task_id in pending_tasks
            ]

            # Check for deadlock (no ready tasks, no active tasks, but tasks still pending)
            if not ready_tasks and not active_tasks and pending_tasks:
                logger.error(
                    "parallel_execution_deadlock",
                    pending_tasks=list(pending_tasks),
                )
                raise RuntimeError(
                    f"Execution deadlock detected. "
                    f"Pending tasks with unsatisfied dependencies: {pending_tasks}"
                )

            # Launch ready tasks
            for task in ready_tasks:
                coro = execute_task(task)
                active_task = asyncio.create_task(coro)
                active_tasks.add(active_task)

            # Wait for at least one task to complete
            if active_tasks:
                done, active_tasks = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

        logger.info(
            "parallel_execution_completed",
            total_tasks=len(tasks),
            successful=sum(
                1 for t in tasks if t.status == "completed"
            ),
            failed=sum(1 for t in tasks if t.status == "failed"),
        )

        return results

    async def execute_batch(
        self,
        requests: list[ToolExecutionRequest],
        max_concurrent: int = 10,
    ) -> list[ToolResult]:
        """Execute multiple tool requests in parallel (no dependencies).

        Args:
            requests: List of tool execution requests
            max_concurrent: Maximum concurrent executions

        Returns:
            List of execution results in same order as requests
        """
        # Create tasks without dependencies
        tasks = [
            ParallelTask(
                task_id=f"task_{i}",
                request=request,
            )
            for i, request in enumerate(requests)
        ]

        # Execute in parallel
        results_map = await self.execute_parallel(tasks, max_concurrent)

        # Return results in original order
        return [results_map[f"task_{i}"] for i in range(len(requests))]


async def execute_with_timeout(
    executor: ToolExecutor,
    request: ToolExecutionRequest,
    timeout_seconds: float,
) -> ToolResult:
    """Execute tool with timeout.

    Args:
        executor: Tool executor
        request: Execution request
        timeout_seconds: Timeout in seconds

    Returns:
        Execution result

    Raises:
        asyncio.TimeoutError: If execution exceeds timeout
    """
    try:
        result = await asyncio.wait_for(
            executor.execute(request),
            timeout=timeout_seconds,
        )
        return result

    except asyncio.TimeoutError:
        logger.error(
            "tool_execution_timeout",
            tool_id=request.tool_id,
            timeout=timeout_seconds,
        )
        # Create timeout result
        from datetime import datetime, timezone
        from uuid import uuid4

        return ToolResult(
            request_id=str(uuid4()),
            tool_id=request.tool_id,
            status=ToolExecutionStatus.TIMEOUT,
            error=f"Execution exceeded timeout of {timeout_seconds}s",
            error_type="TimeoutError",
            execution_time_ms=int(timeout_seconds * 1000),
            timestamp=datetime.now(timezone.utc),
        )


async def execute_with_fallback(
    executor: ToolExecutor,
    primary_request: ToolExecutionRequest,
    fallback_request: ToolExecutionRequest,
) -> ToolResult:
    """Execute tool with fallback on failure.

    Args:
        executor: Tool executor
        primary_request: Primary execution request
        fallback_request: Fallback execution request

    Returns:
        Execution result from primary or fallback
    """
    # Try primary execution
    result = await executor.execute(primary_request)

    if result.status == ToolExecutionStatus.SUCCESS:
        return result

    # Primary failed, try fallback
    logger.warning(
        "executing_fallback_tool",
        primary_tool=primary_request.tool_id,
        fallback_tool=fallback_request.tool_id,
        primary_error=result.error,
    )

    fallback_result = await executor.execute(fallback_request)

    # Add metadata about fallback
    if fallback_result.metadata is None:
        fallback_result.metadata = {}
    fallback_result.metadata["fallback_used"] = True
    fallback_result.metadata["primary_tool"] = primary_request.tool_id
    fallback_result.metadata["primary_error"] = result.error

    return fallback_result
