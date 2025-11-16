"""ECL Pipeline - Extract, Cognify, Load architecture for modular memory processing.

This module implements the ECL (Extract, Cognify, Load) pipeline pattern inspired by Cognee
for task-based, modular processing of memory data. The pipeline supports:
- Task composition and chaining
- Parallel task execution where dependencies allow
- Task-level error handling and retry logic
- Dependency resolution

References:
    - FR-9: ECL Pipeline Architecture (spec.md)
    - Cognee: https://github.com/topoteretes/cognee
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
    """Result of a task execution."""

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


@dataclass
class PipelineResult:
    """Result of pipeline execution."""

    pipeline_id: str
    status: TaskStatus
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_execution_time_ms: float | None = None

    def is_success(self) -> bool:
        """Check if all tasks completed successfully."""
        return all(result.is_success() for result in self.task_results.values())

    def get_failed_tasks(self) -> list[str]:
        """Get names of failed tasks."""
        return [
            name for name, result in self.task_results.items() if result.is_failure()
        ]

    def get_successful_tasks(self) -> list[str]:
        """Get names of successful tasks."""
        return [
            name for name, result in self.task_results.items() if result.is_success()
        ]


class ECLTask(ABC):
    """Abstract base class for ECL pipeline tasks.

    Tasks are the building blocks of the ECL pipeline. Each task represents a discrete
    unit of work that can be composed and chained together.

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
        """Initialize ECL task.

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
            input_data: Input data dictionary containing task parameters

        Returns:
            Output data dictionary with task results

        Raises:
            Exception: Task-specific exceptions on failure
        """
        pass

    async def run_with_retry(self, input_data: dict[str, Any]) -> TaskResult:
        """Execute task with retry logic.

        This method wraps execute() with retry logic based on the configured
        retry strategy. It handles exponential backoff, tracks retry count,
        and manages execution timing.

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
        return f"ECLTask(name={self.name}, dependencies={self.dependencies})"


class TaskRegistry:
    """Registry for discovering and managing available ECL tasks.

    The registry provides a centralized place to register and retrieve tasks.
    It supports dynamic task discovery and validation.

    Example:
        ```python
        registry = TaskRegistry()

        # Register a task
        @registry.register
        class MyTask(ECLTask):
            async def execute(self, input_data):
                return {"result": "done"}

        # Get a task
        task = registry.get_task("MyTask")
        ```
    """

    def __init__(self):
        """Initialize task registry."""
        self._tasks: dict[str, type[ECLTask]] = {}
        self._instances: dict[str, ECLTask] = {}

    def register(
        self, task_class: type[ECLTask] | None = None, *, name: str | None = None
    ) -> type[ECLTask] | Callable[[type[ECLTask]], type[ECLTask]]:
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
            class MyTask(ECLTask):
                pass

            # Or with custom name
            @registry.register(name="custom_name")
            class MyTask(ECLTask):
                pass
            ```
        """

        def decorator(cls: type[ECLTask]) -> type[ECLTask]:
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

    def get_task(self, name: str, **kwargs: Any) -> ECLTask:
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


class Pipeline:
    """ECL Pipeline for composing and executing tasks.

    The pipeline manages task execution, dependency resolution, and parallel execution
    where possible. It supports:
    - Sequential execution (default)
    - Parallel execution where dependencies allow
    - Automatic dependency resolution
    - Error handling and rollback

    Example:
        ```python
        pipeline = Pipeline(pipeline_id="data_processing")
        pipeline.add_task(extract_task)
        pipeline.add_task(transform_task)
        pipeline.add_task(load_task)

        result = await pipeline.execute({"source": "data.csv"})
        ```
    """

    def __init__(
        self,
        pipeline_id: str,
        parallel_execution: bool = False,
        max_parallel: int = 4,
    ):
        """Initialize pipeline.

        Args:
            pipeline_id: Unique identifier for this pipeline
            parallel_execution: Enable parallel task execution
            max_parallel: Maximum number of tasks to run in parallel
        """
        self.pipeline_id = pipeline_id
        self.parallel_execution = parallel_execution
        self.max_parallel = max_parallel
        self._tasks: list[ECLTask] = []
        self._task_map: dict[str, ECLTask] = {}

    def add_task(self, task: ECLTask) -> None:
        """Add a task to the pipeline.

        Args:
            task: The task to add

        Raises:
            ValueError: If task with same name already exists
        """
        if task.name in self._task_map:
            raise ValueError(f"Task {task.name} already exists in pipeline")

        self._tasks.append(task)
        self._task_map[task.name] = task
        logger.debug(f"Added task {task.name} to pipeline {self.pipeline_id}")

    def _build_dependency_graph(self) -> dict[str, set[str]]:
        """Build dependency graph for task execution order.

        Returns:
            Dictionary mapping task name to set of dependency names
        """
        graph: dict[str, set[str]] = {}
        for task in self._tasks:
            graph[task.name] = set(task.dependencies)
        return graph

    def _topological_sort(self, graph: dict[str, set[str]]) -> list[list[str]]:
        """Perform topological sort to determine execution order.

        Groups tasks into levels where tasks in the same level can run in parallel.

        Args:
            graph: Dependency graph

        Returns:
            List of task name groups (each group can run in parallel)

        Raises:
            ValueError: If circular dependency detected
        """
        # Count incoming edges (dependencies)
        in_degree: dict[str, int] = {name: len(deps) for name, deps in graph.items()}

        # Find all tasks with no dependencies (level 0)
        levels: list[list[str]] = []
        remaining = set(graph.keys())

        while remaining:
            # Find tasks with all dependencies satisfied
            ready = [name for name in remaining if in_degree[name] == 0]

            if not ready:
                # Circular dependency detected
                raise ValueError(
                    f"Circular dependency detected in pipeline {self.pipeline_id}. "
                    f"Remaining tasks: {remaining}"
                )

            levels.append(ready)

            # Remove ready tasks and update in-degrees
            for name in ready:
                remaining.remove(name)
                # Decrease in-degree for dependent tasks
                for other_name in remaining:
                    if name in graph[other_name]:
                        in_degree[other_name] -= 1

        return levels

    async def execute(self, input_data: dict[str, Any]) -> PipelineResult:
        """Execute the pipeline with given input data.

        Executes tasks in dependency order. If parallel_execution is enabled,
        tasks at the same dependency level run concurrently.

        Args:
            input_data: Initial input data for the pipeline

        Returns:
            PipelineResult with execution status and task results
        """
        result = PipelineResult(
            pipeline_id=self.pipeline_id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(),
        )

        if not self._tasks:
            logger.warning(f"Pipeline {self.pipeline_id} has no tasks")
            result.status = TaskStatus.COMPLETED
            result.completed_at = datetime.now()
            return result

        try:
            # Build dependency graph and execution order
            graph = self._build_dependency_graph()
            levels = self._topological_sort(graph)

            logger.info(
                f"Executing pipeline {self.pipeline_id} with {len(self._tasks)} tasks "
                f"in {len(levels)} levels"
            )

            # Track outputs for passing between tasks
            task_outputs: dict[str, Any] = {"input": input_data}

            # Execute tasks level by level
            for level_idx, level_tasks in enumerate(levels):
                logger.debug(
                    f"Pipeline {self.pipeline_id} level {level_idx + 1}/{len(levels)}: "
                    f"{len(level_tasks)} tasks"
                )

                # Prepare input for each task (merge outputs from dependencies)
                level_inputs = {}
                for task_name in level_tasks:
                    task = self._task_map[task_name]
                    task_input = {"input": input_data}

                    # Add outputs from dependency tasks
                    for dep_name in task.dependencies:
                        if dep_name in task_outputs:
                            task_input[dep_name] = task_outputs[dep_name]

                    level_inputs[task_name] = task_input

                # Execute tasks in parallel or sequential
                if self.parallel_execution and len(level_tasks) > 1:
                    # Parallel execution with semaphore for concurrency limit
                    semaphore = asyncio.Semaphore(self.max_parallel)

                    async def run_task_limited(task_name: str) -> TaskResult:
                        async with semaphore:
                            task = self._task_map[task_name]
                            return await task.run_with_retry(level_inputs[task_name])

                    # Execute all tasks in this level concurrently
                    level_results = await asyncio.gather(
                        *[run_task_limited(name) for name in level_tasks],
                        return_exceptions=False,
                    )

                    # Store results
                    for task_name, task_result in zip(level_tasks, level_results):
                        result.task_results[task_name] = task_result
                        if task_result.is_success() and task_result.output:
                            task_outputs[task_name] = task_result.output

                else:
                    # Sequential execution
                    for task_name in level_tasks:
                        task = self._task_map[task_name]
                        task_result = await task.run_with_retry(level_inputs[task_name])
                        result.task_results[task_name] = task_result

                        if task_result.is_success() and task_result.output:
                            task_outputs[task_name] = task_result.output

                # Check if any task in this level failed
                failed_tasks = [
                    name
                    for name in level_tasks
                    if result.task_results[name].is_failure()
                ]

                if failed_tasks:
                    logger.error(
                        f"Pipeline {self.pipeline_id} failed at level {level_idx + 1}. "
                        f"Failed tasks: {failed_tasks}"
                    )
                    result.status = TaskStatus.FAILED
                    break

            # Mark pipeline as completed if no failures
            if result.status == TaskStatus.RUNNING:
                result.status = TaskStatus.COMPLETED
                logger.info(
                    f"Pipeline {self.pipeline_id} completed successfully. "
                    f"Executed {len(result.task_results)} tasks."
                )

        except Exception as e:
            logger.exception(f"Pipeline {self.pipeline_id} failed with exception: {e}")
            result.status = TaskStatus.FAILED

        finally:
            result.completed_at = datetime.now()
            result.total_execution_time_ms = (
                (result.completed_at - result.started_at).total_seconds() * 1000
                if result.started_at
                else None
            )

        return result

    def get_task(self, name: str) -> ECLTask | None:
        """Get a task by name.

        Args:
            name: Task name

        Returns:
            Task instance or None if not found
        """
        return self._task_map.get(name)

    def list_tasks(self) -> list[str]:
        """List all task names in the pipeline.

        Returns:
            List of task names in order added
        """
        return [task.name for task in self._tasks]

    def clear(self) -> None:
        """Remove all tasks from the pipeline."""
        self._tasks.clear()
        self._task_map.clear()
        logger.debug(f"Cleared pipeline {self.pipeline_id}")


# Global task registry instance
task_registry = TaskRegistry()
