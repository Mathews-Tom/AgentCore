"""Pipeline class for ECL task composition and execution.

This module provides the Pipeline orchestrator for composing and executing
ECL tasks with:
- Task registry and composition
- Sequential/parallel execution support
- Automatic dependency resolution via topological sort
- Error propagation and handling
- Context passing between tasks

References:
    - FR-9.5: Task Registry and Composition
    - MEM-010: ECL Pipeline Base Classes
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agentcore.a2a_protocol.services.memory.pipeline.task_base import (
    TaskBase,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of pipeline execution.

    Attributes:
        pipeline_id: Unique identifier for the pipeline
        status: Overall status of the pipeline
        task_results: Dictionary mapping task names to their results
        started_at: Timestamp when pipeline started
        completed_at: Timestamp when pipeline completed
        total_execution_time_ms: Total execution time in milliseconds
    """

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


class Pipeline:
    """ECL Pipeline for composing and executing tasks.

    The pipeline manages task execution, dependency resolution, and parallel execution
    where possible. It supports:
    - Sequential execution (default)
    - Parallel execution where dependencies allow
    - Automatic dependency resolution via topological sort
    - Error handling and propagation
    - Context passing between dependent tasks

    Tasks are organized into "levels" based on their dependencies. Tasks at the
    same level can run in parallel, while tasks at different levels run sequentially.

    Example:
        ```python
        pipeline = Pipeline(pipeline_id="data_processing", parallel_execution=True)
        pipeline.add_task(extract_task)
        pipeline.add_task(transform_task)  # depends on extract
        pipeline.add_task(load_task)       # depends on transform

        result = await pipeline.execute({"source": "data.csv"})
        if result.is_success():
            print("Pipeline completed successfully!")
        ```

    Attributes:
        pipeline_id: Unique identifier for this pipeline
        parallel_execution: Enable parallel task execution
        max_parallel: Maximum number of tasks to run in parallel
    """

    def __init__(
        self,
        pipeline_id: str | None = None,
        parallel_execution: bool = False,
        max_parallel: int = 4,
    ):
        """Initialize pipeline.

        Args:
            pipeline_id: Unique identifier for this pipeline (auto-generated if not provided)
            parallel_execution: Enable parallel task execution
            max_parallel: Maximum number of tasks to run in parallel
        """
        from uuid import uuid4
        self.pipeline_id = pipeline_id or f"pipeline-{uuid4()}"
        self.parallel_execution = parallel_execution
        self.max_parallel = max_parallel
        self._tasks: list[TaskBase] = []
        self._task_map: dict[str, TaskBase] = {}

    def add_task(self, task: TaskBase) -> None:
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
        Uses Kahn's algorithm for topological sorting.

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
        tasks at the same dependency level run concurrently (up to max_parallel).

        The pipeline:
        1. Builds dependency graph
        2. Performs topological sort to determine execution order
        3. Executes tasks level by level
        4. Passes outputs between dependent tasks
        5. Stops on first failure

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

    def get_task(self, name: str) -> TaskBase | None:
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
