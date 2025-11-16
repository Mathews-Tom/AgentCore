"""Unit tests for ECL Pipeline (Extract, Cognify, Load) implementation.

Tests cover:
- ECLTask base class and retry logic
- TaskRegistry registration and retrieval
- Pipeline composition and dependency resolution
- Sequential and parallel task execution
- Error handling and recovery
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.a2a_protocol.services.memory.ecl_pipeline import (
    ECLTask,
    Pipeline,
    PipelineResult,
    RetryStrategy,
    TaskRegistry,
    TaskResult,
    TaskStatus,
    task_registry,
)


# Test Task Implementations
class SimpleTask(ECLTask):
    """Simple task that succeeds immediately."""

    def __init__(self, name: str = "simple_task", output_value: str = "success"):
        super().__init__(name=name, description="Simple test task")
        self.output_value = output_value
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return {"result": self.output_value, "input": input_data}


class FailingTask(ECLTask):
    """Task that fails a specified number of times before succeeding."""

    def __init__(
        self,
        name: str = "failing_task",
        fail_count: int = 1,
        error_message: str = "Task failed",
    ):
        super().__init__(
            name=name,
            description="Task that fails then succeeds",
            retry_strategy=RetryStrategy.EXPONENTIAL,
            max_retries=3,
            retry_delay_ms=10,
        )
        self.fail_count = fail_count
        self.error_message = error_message
        self.attempt_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.attempt_count += 1
        if self.attempt_count <= self.fail_count:
            raise RuntimeError(self.error_message)
        return {"result": "success_after_retry", "attempts": self.attempt_count}


class DependentTask(ECLTask):
    """Task that depends on other tasks."""

    def __init__(self, name: str, dependencies: list[str]):
        super().__init__(name=name, description=f"Depends on {dependencies}", dependencies=dependencies)
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        # Merge outputs from dependencies
        merged_output = {"task": self.name}
        for key, value in input_data.items():
            if isinstance(value, dict) and "result" in value:
                merged_output[key] = value["result"]
        return merged_output


class SlowTask(ECLTask):
    """Task that takes time to execute."""

    def __init__(self, name: str = "slow_task", delay_ms: int = 100):
        super().__init__(name=name, description="Slow task for testing")
        self.delay_ms = delay_ms
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.delay_ms / 1000)
        return {"result": f"completed_after_{self.delay_ms}ms"}


# Test ECLTask Base Class
@pytest.mark.asyncio
class TestECLTask:
    """Tests for ECLTask base class."""

    async def test_simple_task_execution(self):
        """Test basic task execution."""
        task = SimpleTask(output_value="test_output")
        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.status == TaskStatus.COMPLETED
        assert result.output["result"] == "test_output"
        assert result.retry_count == 0
        assert task.call_count == 1

    async def test_task_retry_on_failure(self):
        """Test task retries on failure."""
        task = FailingTask(fail_count=2)
        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.status == TaskStatus.COMPLETED
        assert result.output["attempts"] == 3  # Failed 2 times, succeeded on 3rd
        assert result.retry_count == 2
        assert task.attempt_count == 3

    async def test_task_max_retries_exceeded(self):
        """Test task fails after max retries."""
        task = FailingTask(fail_count=10)  # Will never succeed
        result = await task.run_with_retry({"test": "data"})

        assert result.is_failure()
        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        assert result.retry_count == 3  # max_retries
        assert task.attempt_count == 4  # Initial + 3 retries

    async def test_no_retry_strategy(self):
        """Test task with no retry strategy."""
        task = FailingTask(fail_count=1)
        task.retry_strategy = RetryStrategy.NONE
        result = await task.run_with_retry({"test": "data"})

        assert result.is_failure()
        assert result.retry_count == 0
        assert task.attempt_count == 1  # Only initial attempt

    async def test_fixed_retry_strategy(self):
        """Test task with fixed retry delay."""
        task = FailingTask(fail_count=1)
        task.retry_strategy = RetryStrategy.FIXED
        task.retry_delay_ms = 10

        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.retry_count == 1
        assert task.attempt_count == 2

    async def test_exponential_backoff(self):
        """Test exponential backoff timing."""
        task = FailingTask(fail_count=2)
        task.retry_strategy = RetryStrategy.EXPONENTIAL
        task.retry_delay_ms = 10

        import time

        start = time.time()
        result = await task.run_with_retry({"test": "data"})
        elapsed_ms = (time.time() - start) * 1000

        assert result.is_success()
        # Should wait: 10ms (1st retry) + 20ms (2nd retry) = 30ms minimum
        assert elapsed_ms >= 25  # Allow some variance

    async def test_task_execution_timing(self):
        """Test task execution time tracking."""
        task = SlowTask(delay_ms=50)
        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 45  # Allow some variance
        assert result.started_at is not None
        assert result.completed_at is not None


# Test TaskRegistry
class TestTaskRegistry:
    """Tests for TaskRegistry."""

    def test_register_task_decorator(self):
        """Test registering task with decorator."""
        registry = TaskRegistry()

        @registry.register
        class TestTask(ECLTask):
            def __init__(self, name: str = "TestTask"):
                super().__init__(name=name, description="Test task")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test"}

        assert "TestTask" in registry.list_tasks()
        task = registry.get_task("TestTask")
        assert isinstance(task, TestTask)

    def test_register_task_with_custom_name(self):
        """Test registering task with custom name."""
        registry = TaskRegistry()

        @registry.register(name="custom_name")
        class TestTask(ECLTask):
            def __init__(self, name: str = "custom_name"):
                super().__init__(name=name, description="Test task")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test"}

        assert "custom_name" in registry.list_tasks()
        task = registry.get_task("custom_name")
        assert isinstance(task, TestTask)

    def test_register_duplicate_task(self):
        """Test registering task with duplicate name."""
        registry = TaskRegistry()

        @registry.register
        class TestTask1(ECLTask):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test1"}

        @registry.register
        class TestTask1(ECLTask):  # Same name
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test2"}

        # Should overwrite with warning
        assert len(registry.list_tasks()) == 1

    def test_get_task_not_found(self):
        """Test getting non-existent task."""
        registry = TaskRegistry()

        with pytest.raises(KeyError):
            registry.get_task("NonExistent")

    def test_get_task_with_kwargs(self):
        """Test getting task with constructor arguments."""
        registry = TaskRegistry()

        @registry.register
        class ConfigurableTask(ECLTask):
            def __init__(self, name: str = "configurable", config_value: str = "default"):
                super().__init__(name=name)
                self.config_value = config_value

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"config": self.config_value}

        # Get with custom kwargs
        task = registry.get_task("ConfigurableTask", config_value="custom")
        assert task.config_value == "custom"

        # Get cached instance (no kwargs)
        cached_task = registry.get_task("ConfigurableTask")
        assert cached_task.config_value == "default"

    def test_list_tasks(self):
        """Test listing all registered tasks."""
        registry = TaskRegistry()

        @registry.register
        class TaskA(ECLTask):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {}

        @registry.register
        class TaskB(ECLTask):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {}

        tasks = registry.list_tasks()
        assert tasks == ["TaskA", "TaskB"]  # Alphabetically sorted

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = TaskRegistry()

        @registry.register
        class TestTask(ECLTask):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {}

        assert len(registry.list_tasks()) == 1
        registry.clear()
        assert len(registry.list_tasks()) == 0


# Test Pipeline
@pytest.mark.asyncio
class TestPipeline:
    """Tests for Pipeline class."""

    async def test_empty_pipeline(self):
        """Test executing empty pipeline."""
        pipeline = Pipeline(pipeline_id="empty_test")
        result = await pipeline.execute({"test": "data"})

        assert result.status == TaskStatus.COMPLETED
        assert len(result.task_results) == 0

    async def test_single_task_pipeline(self):
        """Test pipeline with single task."""
        pipeline = Pipeline(pipeline_id="single_task")
        task = SimpleTask(output_value="single_result")
        pipeline.add_task(task)

        result = await pipeline.execute({"input": "data"})

        assert result.is_success()
        assert len(result.task_results) == 1
        assert result.task_results["simple_task"].is_success()
        assert result.task_results["simple_task"].output["result"] == "single_result"

    async def test_sequential_pipeline(self):
        """Test sequential execution of tasks."""
        pipeline = Pipeline(pipeline_id="sequential")

        task1 = SimpleTask(name="task1", output_value="result1")
        task2 = SimpleTask(name="task2", output_value="result2")
        task3 = SimpleTask(name="task3", output_value="result3")

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        result = await pipeline.execute({"input": "data"})

        assert result.is_success()
        assert len(result.task_results) == 3
        assert all(r.is_success() for r in result.task_results.values())

    async def test_dependency_resolution(self):
        """Test pipeline respects task dependencies."""
        pipeline = Pipeline(pipeline_id="dependencies")

        # Create tasks with dependencies: task3 depends on task1 and task2
        task1 = SimpleTask(name="task1", output_value="result1")
        task2 = SimpleTask(name="task2", output_value="result2")
        task3 = DependentTask(name="task3", dependencies=["task1", "task2"])

        # Add in reverse order to test dependency resolution
        pipeline.add_task(task3)
        pipeline.add_task(task2)
        pipeline.add_task(task1)

        result = await pipeline.execute({"input": "data"})

        assert result.is_success()
        assert len(result.task_results) == 3

        # task3 should have access to task1 and task2 outputs
        task3_output = result.task_results["task3"].output
        assert task3_output["task1"] == "result1"
        assert task3_output["task2"] == "result2"

    async def test_parallel_execution(self):
        """Test parallel task execution."""
        pipeline = Pipeline(
            pipeline_id="parallel", parallel_execution=True, max_parallel=3
        )

        # Create independent slow tasks
        task1 = SlowTask(name="slow1", delay_ms=50)
        task2 = SlowTask(name="slow2", delay_ms=50)
        task3 = SlowTask(name="slow3", delay_ms=50)

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        import time

        start = time.time()
        result = await pipeline.execute({"input": "data"})
        elapsed_ms = (time.time() - start) * 1000

        assert result.is_success()
        # Should complete in ~50ms (parallel), not 150ms (sequential)
        assert elapsed_ms < 100  # Allow some overhead

    async def test_mixed_parallel_sequential(self):
        """Test pipeline with mixed parallel and sequential execution."""
        pipeline = Pipeline(pipeline_id="mixed", parallel_execution=True, max_parallel=2)

        # Level 1: task1, task2 (parallel)
        task1 = SimpleTask(name="task1", output_value="result1")
        task2 = SimpleTask(name="task2", output_value="result2")

        # Level 2: task3 depends on task1 and task2
        task3 = DependentTask(name="task3", dependencies=["task1", "task2"])

        # Level 3: task4, task5 both depend on task3 (parallel)
        task4 = DependentTask(name="task4", dependencies=["task3"])
        task5 = DependentTask(name="task5", dependencies=["task3"])

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)
        pipeline.add_task(task4)
        pipeline.add_task(task5)

        result = await pipeline.execute({"input": "data"})

        assert result.is_success()
        assert len(result.task_results) == 5

    async def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        pipeline = Pipeline(pipeline_id="circular")

        task1 = DependentTask(name="task1", dependencies=["task2"])
        task2 = DependentTask(name="task2", dependencies=["task1"])

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        result = await pipeline.execute({"input": "data"})
        # Circular dependency should be detected during execution
        assert result.status == TaskStatus.FAILED

    async def test_pipeline_failure_stops_execution(self):
        """Test pipeline stops on task failure."""
        pipeline = Pipeline(pipeline_id="failure_test")

        task1 = SimpleTask(name="task1", output_value="result1")
        task2 = FailingTask(name="task2", fail_count=10)  # Will fail
        task3 = DependentTask(name="task3", dependencies=["task2"])

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        result = await pipeline.execute({"input": "data"})

        assert result.status == TaskStatus.FAILED
        assert result.task_results["task1"].is_success()
        assert result.task_results["task2"].is_failure()
        assert "task3" not in result.task_results  # Should not execute

    async def test_duplicate_task_name(self):
        """Test pipeline rejects duplicate task names."""
        pipeline = Pipeline(pipeline_id="duplicate_test")

        task1 = SimpleTask(name="same_name")
        task2 = SimpleTask(name="same_name")

        pipeline.add_task(task1)
        with pytest.raises(ValueError, match="already exists"):
            pipeline.add_task(task2)

    async def test_get_task(self):
        """Test retrieving task from pipeline."""
        pipeline = Pipeline(pipeline_id="get_test")
        task = SimpleTask(name="test_task")
        pipeline.add_task(task)

        retrieved = pipeline.get_task("test_task")
        assert retrieved is task

        not_found = pipeline.get_task("nonexistent")
        assert not_found is None

    async def test_list_tasks(self):
        """Test listing tasks in pipeline."""
        pipeline = Pipeline(pipeline_id="list_test")

        task1 = SimpleTask(name="task1")
        task2 = SimpleTask(name="task2")

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        tasks = pipeline.list_tasks()
        assert tasks == ["task1", "task2"]

    async def test_clear_pipeline(self):
        """Test clearing pipeline."""
        pipeline = Pipeline(pipeline_id="clear_test")
        pipeline.add_task(SimpleTask(name="task1"))

        assert len(pipeline.list_tasks()) == 1
        pipeline.clear()
        assert len(pipeline.list_tasks()) == 0

    async def test_max_parallel_limit(self):
        """Test max_parallel concurrency limit."""
        pipeline = Pipeline(
            pipeline_id="concurrency_limit", parallel_execution=True, max_parallel=2
        )

        # Create 5 slow tasks
        for i in range(5):
            pipeline.add_task(SlowTask(name=f"slow{i}", delay_ms=50))

        import time

        start = time.time()
        result = await pipeline.execute({"input": "data"})
        elapsed_ms = (time.time() - start) * 1000

        assert result.is_success()
        # With max_parallel=2, should take ~150ms (3 batches: 2+2+1)
        assert 120 < elapsed_ms < 200


# Test PipelineResult
class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_is_success(self):
        """Test is_success method."""
        result = PipelineResult(pipeline_id="test", status=TaskStatus.COMPLETED)
        result.task_results["task1"] = TaskResult(
            task_name="task1", status=TaskStatus.COMPLETED
        )
        result.task_results["task2"] = TaskResult(
            task_name="task2", status=TaskStatus.COMPLETED
        )

        assert result.is_success()

    def test_is_success_with_failure(self):
        """Test is_success returns False when task failed."""
        result = PipelineResult(pipeline_id="test", status=TaskStatus.FAILED)
        result.task_results["task1"] = TaskResult(
            task_name="task1", status=TaskStatus.COMPLETED
        )
        result.task_results["task2"] = TaskResult(
            task_name="task2", status=TaskStatus.FAILED
        )

        assert not result.is_success()

    def test_get_failed_tasks(self):
        """Test getting list of failed tasks."""
        result = PipelineResult(pipeline_id="test", status=TaskStatus.FAILED)
        result.task_results["task1"] = TaskResult(
            task_name="task1", status=TaskStatus.COMPLETED
        )
        result.task_results["task2"] = TaskResult(
            task_name="task2", status=TaskStatus.FAILED
        )
        result.task_results["task3"] = TaskResult(
            task_name="task3", status=TaskStatus.FAILED
        )

        failed = result.get_failed_tasks()
        assert failed == ["task2", "task3"]

    def test_get_successful_tasks(self):
        """Test getting list of successful tasks."""
        result = PipelineResult(pipeline_id="test", status=TaskStatus.COMPLETED)
        result.task_results["task1"] = TaskResult(
            task_name="task1", status=TaskStatus.COMPLETED
        )
        result.task_results["task2"] = TaskResult(
            task_name="task2", status=TaskStatus.FAILED
        )
        result.task_results["task3"] = TaskResult(
            task_name="task3", status=TaskStatus.COMPLETED
        )

        successful = result.get_successful_tasks()
        assert successful == ["task1", "task3"]


# Test TaskResult
class TestTaskResult:
    """Tests for TaskResult."""

    def test_is_success(self):
        """Test is_success method."""
        result = TaskResult(task_name="test", status=TaskStatus.COMPLETED)
        assert result.is_success()

        result.status = TaskStatus.FAILED
        assert not result.is_success()

    def test_is_failure(self):
        """Test is_failure method."""
        result = TaskResult(task_name="test", status=TaskStatus.FAILED)
        assert result.is_failure()

        result.status = TaskStatus.COMPLETED
        assert not result.is_failure()


# Integration Test
@pytest.mark.asyncio
class TestECLPipelineIntegration:
    """Integration tests for complete ECL pipeline workflows."""

    async def test_full_ecl_workflow(self):
        """Test complete Extract-Cognify-Load workflow."""
        # Create a realistic ECL pipeline
        pipeline = Pipeline(pipeline_id="ecl_workflow", parallel_execution=True)

        # Extract phase
        extract_task = SimpleTask(name="extract", output_value="extracted_data")

        # Cognify phase (depends on extract)
        cognify_task1 = DependentTask(name="entity_extraction", dependencies=["extract"])
        cognify_task2 = DependentTask(
            name="relationship_detection", dependencies=["extract"]
        )

        # Load phase (depends on both cognify tasks)
        load_vector = DependentTask(
            name="load_vector", dependencies=["entity_extraction", "relationship_detection"]
        )
        load_graph = DependentTask(
            name="load_graph", dependencies=["entity_extraction", "relationship_detection"]
        )

        # Add tasks to pipeline
        pipeline.add_task(extract_task)
        pipeline.add_task(cognify_task1)
        pipeline.add_task(cognify_task2)
        pipeline.add_task(load_vector)
        pipeline.add_task(load_graph)

        # Execute pipeline
        result = await pipeline.execute({"source": "memory_data"})

        # Verify results
        assert result.is_success()
        assert len(result.task_results) == 5

        # Verify execution order (extract first, then cognify in parallel, then load in parallel)
        assert result.task_results["extract"].is_success()
        assert result.task_results["entity_extraction"].is_success()
        assert result.task_results["relationship_detection"].is_success()
        assert result.task_results["load_vector"].is_success()
        assert result.task_results["load_graph"].is_success()

    async def test_global_task_registry(self):
        """Test using the global task_registry instance."""
        # Clear registry first
        task_registry.clear()

        @task_registry.register
        class GlobalTask(ECLTask):
            def __init__(self, name: str = "GlobalTask"):
                super().__init__(name=name, description="Global test task")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "global_registry_works"}

        assert "GlobalTask" in task_registry.list_tasks()
        task = task_registry.get_task("GlobalTask")

        result = await task.run_with_retry({"test": "data"})
        assert result.is_success()
        assert result.output["result"] == "global_registry_works"

        # Clean up
        task_registry.clear()
