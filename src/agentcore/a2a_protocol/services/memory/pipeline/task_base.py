"""TaskBase abstract class for ECL Pipeline tasks.

This module provides the foundation for all ECL pipeline tasks with:
- Abstract execute() method
- Task metadata (name, description, dependencies)
- Error handling hooks with retry logic
- Logging integration
- Async execution support

References:
    - FR-9.5: Task Registry and Composition
    - MEM-010: ECL Pipeline Base Classes
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a pipeline task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RetryStrategy(str, Enum):
    """Retry strategy for failed tasks."""

    NONE = "none"  # No retry
    FIXED = "fixed"  # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff


@dataclass
class TaskResult:
    """Result of a task execution.

    Attributes:
        task_name: Name of the executed task
        status: Current status of the task
        output: Output data from successful execution
        error: Exception from failed execution
        started_at: Timestamp when task started
        completed_at: Timestamp when task completed
        execution_time_ms: Total execution time in milliseconds
        retry_count: Number of retry attempts
    """

    task_name: str
    status: TaskStatus
    output: dict[str, Any] | None = None
    error: Exception | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_time_ms: float | None = None
    retry_count: int = 0

    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def is_failure(self) -> bool:
        """Check if task failed."""
        return self.status == TaskStatus.FAILED


class TaskBase(ABC):
    """Abstract base class for ECL pipeline tasks.

    Tasks are the building blocks of the ECL pipeline. Each task represents a discrete
    unit of work that can be composed and chained together.

    The TaskBase provides:
    - Abstract execute() method that subclasses must implement
    - Retry logic with configurable strategies (none, fixed, exponential)
    - Error handling and logging
    - Execution timing and metrics
    - Dependency tracking

    Example:
        ```python
        class MyTask(TaskBase):
            def __init__(self):
                super().__init__(
                    name="my_task",
                    description="Does something useful",
                    retry_strategy=RetryStrategy.EXPONENTIAL,
                    max_retries=3
                )

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                # Implement task logic
                result = await do_work(input_data)
                return {"result": result}
        ```

    Attributes:
        name: Unique name identifying the task
        description: Human-readable description of what the task does
        dependencies: List of task names that must complete before this task runs
        retry_strategy: Strategy for retrying failed executions
        max_retries: Maximum number of retry attempts
        retry_delay_ms: Base delay between retries in milliseconds
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        dependencies: list[str] | None = None,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        max_retries: int = 3,
        retry_delay_ms: int = 1000,
    ):
        """Initialize task.

        Args:
            name: Unique task identifier
            description: Task description
            dependencies: List of task names this task depends on
            retry_strategy: Retry strategy for failed executions
            max_retries: Maximum retry attempts (0 = no retry)
            retry_delay_ms: Base delay between retries in milliseconds
        """
        self.name = name
        self.description = description
        self.dependencies = dependencies or []
        self.retry_strategy = retry_strategy
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms

    @abstractmethod
    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the task with given input data.

        This is the main entry point for task execution. Subclasses must implement
        the actual task logic here.

        Args:
            input_data: Input data dictionary containing task parameters.
                       May include outputs from dependent tasks.

        Returns:
            Output data dictionary with task results

        Raises:
            Exception: Task-specific exceptions on failure
        """
        pass

    async def run_with_retry(self, input_data: dict[str, Any]) -> TaskResult:
        """Execute task with retry logic.

        This method wraps execute() with retry logic based on the configured
        retry strategy. It handles:
        - Exponential backoff
        - Retry count tracking
        - Execution timing
        - Error capture

        Args:
            input_data: Input data for task execution

        Returns:
            TaskResult containing execution status and output
        """
        result = TaskResult(
            task_name=self.name,
            status=TaskStatus.PENDING,
            started_at=datetime.now(),
        )

        for attempt in range(self.max_retries + 1):
            try:
                result.status = TaskStatus.RUNNING
                result.retry_count = attempt

                # Execute the task
                output = await self.execute(input_data)

                # Success
                result.status = TaskStatus.COMPLETED
                result.output = output
                result.completed_at = datetime.now()
                result.execution_time_ms = (
                    (result.completed_at - result.started_at).total_seconds() * 1000
                    if result.started_at
                    else None
                )

                if attempt > 0:
                    logger.info(
                        f"Task {self.name} succeeded on retry {attempt}/{self.max_retries}"
                    )

                return result

            except Exception as e:
                logger.warning(
                    f"Task {self.name} failed on attempt {attempt + 1}/{self.max_retries + 1}: {e!s}"
                )

                # Last attempt - mark as failed
                if attempt >= self.max_retries:
                    result.status = TaskStatus.FAILED
                    result.error = e
                    result.completed_at = datetime.now()
                    result.execution_time_ms = (
                        (result.completed_at - result.started_at).total_seconds()
                        * 1000
                        if result.started_at
                        else None
                    )
                    logger.error(
                        f"Task {self.name} failed after {self.max_retries + 1} attempts"
                    )
                    return result

                # Calculate retry delay
                if self.retry_strategy == RetryStrategy.NONE:
                    break  # No retry
                elif self.retry_strategy == RetryStrategy.FIXED:
                    delay_ms = self.retry_delay_ms
                elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
                    delay_ms = self.retry_delay_ms * (2**attempt)
                else:
                    delay_ms = self.retry_delay_ms

                # Wait before retry
                await asyncio.sleep(delay_ms / 1000)

        # Should not reach here, but handle gracefully
        result.status = TaskStatus.FAILED
        result.error = RuntimeError(f"Task {self.name} exceeded retry limit")
        return result

    def __repr__(self) -> str:
        """String representation of task."""
        return f"TaskBase(name={self.name}, dependencies={self.dependencies})"


class TaskRegistry:
    """Registry for discovering and managing available ECL tasks.

    The registry provides a centralized place to register and retrieve tasks.
    It supports dynamic task discovery and validation.

    Example:
        ```python
        registry = TaskRegistry()

        # Register a task
        @registry.register
        class MyTask(TaskBase):
            async def execute(self, input_data):
                return {"result": "done"}

        # Get a task
        task = registry.get_task("MyTask")
        ```
    """

    def __init__(self):
        """Initialize task registry."""
        self._tasks: dict[str, type[TaskBase]] = {}
        self._instances: dict[str, TaskBase] = {}

    def register(
        self, task_class: type[TaskBase] | None = None, *, name: str | None = None
    ) -> type[TaskBase] | Callable[[type[TaskBase]], type[TaskBase]]:
        """Register a task class in the registry.

        Can be used as a decorator or called directly.

        Args:
            task_class: The task class to register
            name: Optional custom name (defaults to class name)

        Returns:
            The registered task class (for decorator usage)

        Example:
            ```python
            @registry.register
            class MyTask(TaskBase):
                pass

            # Or with custom name
            @registry.register(name="custom_name")
            class MyTask(TaskBase):
                pass
            ```
        """

        def decorator(cls: type[TaskBase]) -> type[TaskBase]:
            task_name = name or cls.__name__
            if task_name in self._tasks:
                logger.warning(f"Task {task_name} already registered, overwriting")
            self._tasks[task_name] = cls
            logger.info(f"Registered task: {task_name}")
            return cls

        if task_class is None:
            # Called with arguments: @register(name="...")
            return decorator
        else:
            # Called without arguments: @register
            return decorator(task_class)

    def get_task(self, name: str, **kwargs: Any) -> TaskBase:
        """Get a task instance by name.

        Creates a new instance if one doesn't exist in the cache.

        Args:
            name: Name of the task to retrieve
            **kwargs: Arguments to pass to task constructor

        Returns:
            Task instance

        Raises:
            KeyError: If task not found in registry
        """
        if name not in self._tasks:
            raise KeyError(f"Task {name} not found in registry")

        # Return cached instance if no kwargs provided
        if not kwargs and name in self._instances:
            return self._instances[name]

        # Create new instance
        task_class = self._tasks[name]
        instance = task_class(**kwargs)

        # Cache if no custom kwargs
        if not kwargs:
            self._instances[name] = instance

        return instance

    def list_tasks(self) -> list[str]:
        """List all registered task names.

        Returns:
            List of task names in alphabetical order
        """
        return sorted(self._tasks.keys())

    def clear(self) -> None:
        """Clear all registered tasks.

        Useful for testing or dynamic reloading.
        """
        self._tasks.clear()
        self._instances.clear()
        logger.info("Cleared task registry")


# Global task registry instance
task_registry = TaskRegistry()
