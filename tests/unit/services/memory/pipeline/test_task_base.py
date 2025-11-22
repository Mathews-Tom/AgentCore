"""Unit tests for TaskBase abstract class.

Tests cover:
- TaskBase initialization and configuration
- execute() abstract method
- run_with_retry() retry logic
- Retry strategies (none, fixed, exponential)
- Error handling and logging
- TaskResult data class
- TaskRegistry registration and retrieval
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.a2a_protocol.services.memory.pipeline.task_base import (
    RetryStrategy,
    TaskBase,
    TaskRegistry,
    TaskResult,
    TaskStatus,
    task_registry,
)


# Test Task Implementations
class SimpleTask(TaskBase):
    """Simple task that succeeds immediately."""

    def __init__(self, name: str = "simple_task", output_value: str = "success"):
        super().__init__(name=name, description="Simple test task")
        self.output_value = output_value
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return {"result": self.output_value, "input": input_data}


class FailingTask(TaskBase):
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


class SlowTask(TaskBase):
    """Task that takes time to execute."""

    def __init__(self, name: str = "slow_task", delay_ms: int = 100):
        super().__init__(name=name, description="Slow task for testing")
        self.delay_ms = delay_ms
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.delay_ms / 1000)
        return {"result": f"completed_after_{self.delay_ms}ms"}


# Test TaskBase
@pytest.mark.asyncio
class TestTaskBase:
    """Tests for TaskBase abstract class."""

    async def test_task_initialization(self):
        """Test task initialization with various parameters."""
        task = SimpleTask(name="test_task", output_value="test")

        assert task.name == "test_task"
        assert task.description == "Simple test task"
        assert task.dependencies == []
        assert task.retry_strategy == RetryStrategy.EXPONENTIAL
        assert task.max_retries == 3

    async def test_task_with_dependencies(self):
        """Test task initialization with dependencies."""
        task = SimpleTask(name="dependent_task")
        task.dependencies = ["task1", "task2"]

        assert task.dependencies == ["task1", "task2"]

    async def test_simple_task_execution(self):
        """Test basic task execution without retries."""
        task = SimpleTask(output_value="test_output")
        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.status == TaskStatus.COMPLETED
        assert result.output["result"] == "test_output"
        assert result.retry_count == 0
        assert task.call_count == 1

    async def test_task_retry_on_failure(self):
        """Test task retries on failure and eventually succeeds."""
        task = FailingTask(fail_count=2)
        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.status == TaskStatus.COMPLETED
        assert result.output["attempts"] == 3
        assert result.retry_count == 2
        assert task.attempt_count == 3

    async def test_task_max_retries_exceeded(self):
        """Test task fails after max retries exceeded."""
        task = FailingTask(fail_count=10)
        result = await task.run_with_retry({"test": "data"})

        assert result.is_failure()
        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        assert isinstance(result.error, RuntimeError)
        assert result.retry_count == 3
        assert task.attempt_count == 4

    async def test_no_retry_strategy(self):
        """Test task with no retry strategy fails immediately."""
        task = FailingTask(fail_count=1)
        task.retry_strategy = RetryStrategy.NONE
        result = await task.run_with_retry({"test": "data"})

        assert result.is_failure()
        assert result.retry_count == 0
        assert task.attempt_count == 1

    async def test_fixed_retry_strategy(self):
        """Test task with fixed retry delay."""
        task = FailingTask(fail_count=1)
        task.retry_strategy = RetryStrategy.FIXED
        task.retry_delay_ms = 10

        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.retry_count == 1
        assert task.attempt_count == 2

    async def test_exponential_backoff_timing(self):
        """Test exponential backoff retry timing."""
        task = FailingTask(fail_count=2)
        task.retry_strategy = RetryStrategy.EXPONENTIAL
        task.retry_delay_ms = 10

        import time
        start = time.time()
        result = await task.run_with_retry({"test": "data"})
        elapsed_ms = (time.time() - start) * 1000

        assert result.is_success()
        # Should wait: 10ms (1st retry) + 20ms (2nd retry) = 30ms minimum
        assert elapsed_ms >= 25

    async def test_task_execution_timing(self):
        """Test task execution time tracking."""
        task = SlowTask(delay_ms=50)
        result = await task.run_with_retry({"test": "data"})

        assert result.is_success()
        assert result.execution_time_ms is not None
        assert result.execution_time_ms >= 45
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at > result.started_at

    async def test_task_repr(self):
        """Test task string representation."""
        task = SimpleTask(name="test_task")
        task.dependencies = ["dep1", "dep2"]

        repr_str = repr(task)
        assert "test_task" in repr_str
        assert "dep1" in repr_str


# Test TaskResult
class TestTaskResult:
    """Tests for TaskResult data class."""

    def test_task_result_success(self):
        """Test successful task result."""
        result = TaskResult(task_name="test", status=TaskStatus.COMPLETED)

        assert result.is_success()
        assert not result.is_failure()

    def test_task_result_failure(self):
        """Test failed task result."""
        result = TaskResult(
            task_name="test",
            status=TaskStatus.FAILED,
            error=RuntimeError("Test error"),
        )

        assert result.is_failure()
        assert not result.is_success()
        assert result.error is not None

    def test_task_result_with_output(self):
        """Test task result with output data."""
        result = TaskResult(
            task_name="test",
            status=TaskStatus.COMPLETED,
            output={"key": "value"},
        )

        assert result.output["key"] == "value"

    def test_task_result_retry_count(self):
        """Test task result retry count tracking."""
        result = TaskResult(
            task_name="test",
            status=TaskStatus.COMPLETED,
            retry_count=3,
        )

        assert result.retry_count == 3


# Test TaskRegistry
class TestTaskRegistry:
    """Tests for TaskRegistry."""

    def test_register_task_decorator(self):
        """Test registering task with decorator."""
        registry = TaskRegistry()

        @registry.register
        class TestTask(TaskBase):
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
        class TestTask(TaskBase):
            def __init__(self, name: str = "custom_name"):
                super().__init__(name=name, description="Test task")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test"}

        assert "custom_name" in registry.list_tasks()
        task = registry.get_task("custom_name")
        assert isinstance(task, TestTask)

    def test_register_duplicate_task_overwrites(self):
        """Test registering task with duplicate name overwrites."""
        registry = TaskRegistry()

        @registry.register
        class TestTask1(TaskBase):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test1"}

        @registry.register
        class TestTask1(TaskBase):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "test2"}

        assert len(registry.list_tasks()) == 1

    def test_get_task_not_found(self):
        """Test getting non-existent task raises KeyError."""
        registry = TaskRegistry()

        with pytest.raises(KeyError):
            registry.get_task("NonExistent")

    def test_get_task_with_kwargs(self):
        """Test getting task with constructor arguments."""
        registry = TaskRegistry()

        @registry.register
        class ConfigurableTask(TaskBase):
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

    def test_list_tasks_sorted(self):
        """Test listing tasks returns sorted list."""
        registry = TaskRegistry()

        @registry.register
        class TaskB(TaskBase):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {}

        @registry.register
        class TaskA(TaskBase):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {}

        tasks = registry.list_tasks()
        assert tasks == ["TaskA", "TaskB"]

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = TaskRegistry()

        @registry.register
        class TestTask(TaskBase):
            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {}

        assert len(registry.list_tasks()) == 1
        registry.clear()
        assert len(registry.list_tasks()) == 0

    def test_global_registry_instance(self):
        """Test using the global task_registry instance."""
        task_registry.clear()

        @task_registry.register
        class GlobalTask(TaskBase):
            def __init__(self, name: str = "GlobalTask"):
                super().__init__(name=name)

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"result": "global"}

        assert "GlobalTask" in task_registry.list_tasks()
        task = task_registry.get_task("GlobalTask")
        assert isinstance(task, GlobalTask)

        task_registry.clear()


# Integration tests
@pytest.mark.asyncio
class TestTaskBaseIntegration:
    """Integration tests for TaskBase with realistic scenarios."""

    async def test_task_error_propagation(self):
        """Test that task errors are properly captured."""
        class ErrorTask(TaskBase):
            def __init__(self):
                super().__init__(
                    name="error_task",
                    retry_strategy=RetryStrategy.NONE,
                    max_retries=0,
                )

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                raise ValueError("Expected error")

        task = ErrorTask()
        result = await task.run_with_retry({"test": "data"})

        assert result.is_failure()
        assert result.error is not None
        # Error should be captured (either original or wrapped)
        assert "error" in str(result.error).lower()

    async def test_task_input_data_passthrough(self):
        """Test that input data is passed correctly to execute."""
        class DataPassTask(TaskBase):
            def __init__(self):
                super().__init__(name="data_pass")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"received": input_data.copy()}

        task = DataPassTask()
        input_data = {"key1": "value1", "key2": 42}
        result = await task.run_with_retry(input_data)

        assert result.is_success()
        assert result.output["received"] == input_data

    async def test_multiple_tasks_in_registry(self):
        """Test managing multiple tasks in registry."""
        registry = TaskRegistry()

        @registry.register
        class Task1(TaskBase):
            def __init__(self):
                super().__init__(name="Task1")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"task": "1"}

        @registry.register
        class Task2(TaskBase):
            def __init__(self):
                super().__init__(name="Task2")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"task": "2"}

        @registry.register
        class Task3(TaskBase):
            def __init__(self):
                super().__init__(name="Task3")

            async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
                return {"task": "3"}

        assert len(registry.list_tasks()) == 3
        assert set(registry.list_tasks()) == {"Task1", "Task2", "Task3"}

        task1 = registry.get_task("Task1")
        task2 = registry.get_task("Task2")

        result1 = await task1.run_with_retry({})
        result2 = await task2.run_with_retry({})

        assert result1.output["task"] == "1"
        assert result2.output["task"] == "2"
