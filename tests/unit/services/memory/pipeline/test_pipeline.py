"""Unit tests for Pipeline class.

Tests cover:
- Pipeline composition and task management
- Dependency resolution via topological sort
- Sequential and parallel execution
- Error handling and propagation
- Context passing between tasks
- PipelineResult data class
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from agentcore.a2a_protocol.services.memory.pipeline.pipeline import (
    Pipeline,
    PipelineResult,
)
from agentcore.a2a_protocol.services.memory.pipeline.task_base import (
    TaskBase,
    TaskResult,
    TaskStatus,
)


# Test Task Implementations
class SimpleTask(TaskBase):
    """Simple task that succeeds immediately."""

    def __init__(self, name: str, output_value: str = "success"):
        super().__init__(name=name, description=f"Simple task: {name}")
        self.output_value = output_value
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        return {"result": self.output_value, "task": self.name}


class DependentTask(TaskBase):
    """Task that depends on other tasks."""

    def __init__(self, name: str, dependencies: list[str]):
        super().__init__(
            name=name,
            description=f"Depends on {dependencies}",
            dependencies=dependencies,
        )
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        # Merge outputs from dependencies
        merged = {"task": self.name}
        for key, value in input_data.items():
            if isinstance(value, dict) and "result" in value:
                merged[key] = value["result"]
        return merged


class SlowTask(TaskBase):
    """Task that takes time to execute."""

    def __init__(self, name: str, delay_ms: int = 100):
        super().__init__(name=name, description=f"Slow task ({delay_ms}ms)")
        self.delay_ms = delay_ms
        self.call_count = 0

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        await asyncio.sleep(self.delay_ms / 1000)
        return {"result": f"completed", "delay_ms": self.delay_ms}


class FailingTask(TaskBase):
    """Task that always fails."""

    def __init__(self, name: str = "failing_task"):
        super().__init__(name=name, max_retries=0)

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"Task {self.name} failed")


# Test Pipeline
@pytest.mark.asyncio
class TestPipeline:
    """Tests for Pipeline class."""

    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = Pipeline(
            pipeline_id="test_pipeline",
            parallel_execution=True,
            max_parallel=8,
        )

        assert pipeline.pipeline_id == "test_pipeline"
        assert pipeline.parallel_execution is True
        assert pipeline.max_parallel == 8
        assert len(pipeline.list_tasks()) == 0

    async def test_empty_pipeline_execution(self):
        """Test executing empty pipeline."""
        pipeline = Pipeline(pipeline_id="empty_test")
        result = await pipeline.execute({"test": "data"})

        assert result.status == TaskStatus.COMPLETED
        assert len(result.task_results) == 0
        assert result.is_success()

    async def test_single_task_pipeline(self):
        """Test pipeline with single task."""
        pipeline = Pipeline(pipeline_id="single_task")
        task = SimpleTask(name="task1", output_value="result1")
        pipeline.add_task(task)

        result = await pipeline.execute({"input": "data"})

        assert result.is_success()
        assert len(result.task_results) == 1
        assert result.task_results["task1"].is_success()
        assert result.task_results["task1"].output["result"] == "result1"

    async def test_sequential_pipeline(self):
        """Test sequential execution of independent tasks."""
        pipeline = Pipeline(pipeline_id="sequential", parallel_execution=False)

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

        # task3 depends on task1 and task2
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
        """Test parallel task execution is faster than sequential."""
        pipeline_parallel = Pipeline(
            pipeline_id="parallel",
            parallel_execution=True,
            max_parallel=3,
        )
        pipeline_sequential = Pipeline(
            pipeline_id="sequential",
            parallel_execution=False,
        )

        # Create independent slow tasks
        for pipeline in [pipeline_parallel, pipeline_sequential]:
            pipeline.add_task(SlowTask(name="slow1", delay_ms=50))
            pipeline.add_task(SlowTask(name="slow2", delay_ms=50))
            pipeline.add_task(SlowTask(name="slow3", delay_ms=50))

        # Parallel execution
        start = time.time()
        result_parallel = await pipeline_parallel.execute({"input": "data"})
        parallel_time = (time.time() - start) * 1000

        # Sequential execution
        start = time.time()
        result_sequential = await pipeline_sequential.execute({"input": "data"})
        sequential_time = (time.time() - start) * 1000

        assert result_parallel.is_success()
        assert result_sequential.is_success()

        # Parallel should be significantly faster (< 100ms vs ~150ms)
        assert parallel_time < 100
        assert sequential_time > 140

    async def test_mixed_parallel_sequential_levels(self):
        """Test pipeline with mixed parallel and sequential execution."""
        pipeline = Pipeline(
            pipeline_id="mixed",
            parallel_execution=True,
            max_parallel=2,
        )

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

        # Verify all tasks executed
        for task_name in ["task1", "task2", "task3", "task4", "task5"]:
            assert result.task_results[task_name].is_success()

    async def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        pipeline = Pipeline(pipeline_id="circular")

        task1 = DependentTask(name="task1", dependencies=["task2"])
        task2 = DependentTask(name="task2", dependencies=["task1"])

        pipeline.add_task(task1)
        pipeline.add_task(task2)

        result = await pipeline.execute({"input": "data"})
        assert result.status == TaskStatus.FAILED

    async def test_pipeline_stops_on_failure(self):
        """Test pipeline stops execution when task fails."""
        pipeline = Pipeline(pipeline_id="failure_test")

        task1 = SimpleTask(name="task1", output_value="result1")
        task2 = FailingTask(name="task2")
        task3 = DependentTask(name="task3", dependencies=["task2"])

        pipeline.add_task(task1)
        pipeline.add_task(task2)
        pipeline.add_task(task3)

        result = await pipeline.execute({"input": "data"})

        assert result.status == TaskStatus.FAILED
        assert result.task_results["task1"].is_success()
        assert result.task_results["task2"].is_failure()
        assert "task3" not in result.task_results  # Should not execute

    async def test_duplicate_task_name_raises_error(self):
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
            pipeline_id="concurrency_limit",
            parallel_execution=True,
            max_parallel=2,
        )

        # Create 5 slow tasks
        for i in range(5):
            pipeline.add_task(SlowTask(name=f"slow{i}", delay_ms=50))

        start = time.time()
        result = await pipeline.execute({"input": "data"})
        elapsed_ms = (time.time() - start) * 1000

        assert result.is_success()
        # With max_parallel=2, should take ~150ms (3 batches: 2+2+1)
        assert 120 < elapsed_ms < 200

    async def test_pipeline_execution_timing(self):
        """Test pipeline execution time tracking."""
        pipeline = Pipeline(pipeline_id="timing_test")
        pipeline.add_task(SlowTask(name="slow1", delay_ms=50))

        result = await pipeline.execute({"input": "data"})

        assert result.is_success()
        assert result.total_execution_time_ms is not None
        assert result.total_execution_time_ms >= 45
        assert result.started_at is not None
        assert result.completed_at is not None


# Test PipelineResult
class TestPipelineResult:
    """Tests for PipelineResult data class."""

    def test_is_success_all_completed(self):
        """Test is_success when all tasks completed."""
        result = PipelineResult(pipeline_id="test", status=TaskStatus.COMPLETED)
        result.task_results["task1"] = TaskResult(
            task_name="task1", status=TaskStatus.COMPLETED
        )
        result.task_results["task2"] = TaskResult(
            task_name="task2", status=TaskStatus.COMPLETED
        )

        assert result.is_success()

    def test_is_success_with_failure(self):
        """Test is_success returns False when any task failed."""
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
        assert set(failed) == {"task2", "task3"}

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
        assert set(successful) == {"task1", "task3"}


# Integration tests
@pytest.mark.asyncio
class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""

    async def test_complex_dependency_graph(self):
        """Test complex multi-level dependency graph."""
        pipeline = Pipeline(pipeline_id="complex", parallel_execution=True)

        # Level 0: independent tasks
        task_a = SimpleTask(name="a", output_value="result_a")
        task_b = SimpleTask(name="b", output_value="result_b")

        # Level 1: depends on level 0
        task_c = DependentTask(name="c", dependencies=["a"])
        task_d = DependentTask(name="d", dependencies=["b"])

        # Level 2: depends on level 1
        task_e = DependentTask(name="e", dependencies=["c", "d"])

        # Add in random order
        pipeline.add_task(task_e)
        pipeline.add_task(task_b)
        pipeline.add_task(task_d)
        pipeline.add_task(task_a)
        pipeline.add_task(task_c)

        result = await pipeline.execute({"input": "data"})

        assert result.is_success()
        assert len(result.task_results) == 5

        # Verify all tasks completed successfully
        for task_name in ["a", "b", "c", "d", "e"]:
            assert result.task_results[task_name].is_success()

    async def test_full_ecl_workflow(self):
        """Test complete Extract-Cognify-Load workflow."""
        pipeline = Pipeline(pipeline_id="ecl_workflow", parallel_execution=True)

        # Extract phase
        extract = SimpleTask(name="extract", output_value="extracted_data")

        # Cognify phase (parallel, both depend on extract)
        entity = DependentTask(name="entity_extraction", dependencies=["extract"])
        relationship = DependentTask(
            name="relationship_detection", dependencies=["extract"]
        )

        # Load phase (parallel, both depend on cognify tasks)
        load_vector = DependentTask(
            name="load_vector",
            dependencies=["entity_extraction", "relationship_detection"],
        )
        load_graph = DependentTask(
            name="load_graph",
            dependencies=["entity_extraction", "relationship_detection"],
        )

        pipeline.add_task(extract)
        pipeline.add_task(entity)
        pipeline.add_task(relationship)
        pipeline.add_task(load_vector)
        pipeline.add_task(load_graph)

        result = await pipeline.execute({"source": "memory_data"})

        assert result.is_success()
        assert len(result.task_results) == 5

        # Verify all phases completed
        for task_name in ["extract", "entity_extraction", "relationship_detection",
                          "load_vector", "load_graph"]:
            assert result.task_results[task_name].is_success()
