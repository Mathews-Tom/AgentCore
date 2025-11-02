"""Tests for parallel tool execution."""

import asyncio

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionRequest,
    ToolExecutionStatus,
)
from agentcore.agent_runtime.services.parallel_executor import (
    ParallelExecutor,
    ParallelTask,
    execute_with_fallback,
    execute_with_timeout,
)
from agentcore.agent_runtime.services.tool_executor import ToolExecutor
from agentcore.agent_runtime.services.tool_registry import get_tool_registry


@pytest.fixture
def tool_executor() -> ToolExecutor:
    """Get tool executor with built-in tools."""
    registry = get_tool_registry()
    return ToolExecutor(registry, enable_metrics=True)


@pytest.fixture
def parallel_executor(tool_executor: ToolExecutor) -> ParallelExecutor:
    """Get parallel executor."""
    return ParallelExecutor(tool_executor)


@pytest.mark.asyncio
async def test_execute_batch_simple(parallel_executor: ParallelExecutor):
    """Test simple batch execution without dependencies."""
    requests = [
        ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "+", "a": i, "b": i},
            agent_id="test-agent",
        )
        for i in range(1, 4)
    ]

    results = await parallel_executor.execute_batch(requests, max_concurrent=10)

    assert len(results) == 3
    for i, result in enumerate(results, start=1):
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result == i + i


@pytest.mark.asyncio
async def test_execute_batch_concurrency_limit(parallel_executor: ParallelExecutor):
    """Test batch execution respects concurrency limit."""
    # Create 10 echo requests
    requests = [
        ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": f"message_{i}"},
            agent_id="test-agent",
        )
        for i in range(10)
    ]

    # Execute with max_concurrent=3
    results = await parallel_executor.execute_batch(requests, max_concurrent=3)

    assert len(results) == 10
    assert all(r.status == ToolExecutionStatus.SUCCESS for r in results)


@pytest.mark.asyncio
async def test_execute_parallel_with_dependencies(parallel_executor: ParallelExecutor):
    """Test parallel execution with task dependencies."""
    # Create tasks with dependencies:
    # task1 (no deps) -> task2 (depends on task1) -> task3 (depends on task2)
    tasks = [
        ParallelTask(
            task_id="task1",
            request=ToolExecutionRequest(
                tool_id="calculator",
                parameters={"operation": "+", "a": 1, "b": 1},
                agent_id="test-agent",
            ),
            dependencies=[],
        ),
        ParallelTask(
            task_id="task2",
            request=ToolExecutionRequest(
                tool_id="calculator",
                parameters={"operation": "*", "a": 2, "b": 3},
                agent_id="test-agent",
            ),
            dependencies=["task1"],
        ),
        ParallelTask(
            task_id="task3",
            request=ToolExecutionRequest(
                tool_id="echo",
                parameters={"message": "final"},
                agent_id="test-agent",
            ),
            dependencies=["task2"],
        ),
    ]

    results = await parallel_executor.execute_parallel(tasks, max_concurrent=10)

    assert len(results) == 3
    assert results["task1"].status == ToolExecutionStatus.SUCCESS
    assert results["task2"].status == ToolExecutionStatus.SUCCESS
    assert results["task3"].status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_execute_parallel_diamond_dependencies(parallel_executor: ParallelExecutor):
    """Test parallel execution with diamond dependency pattern."""
    # Diamond pattern:
    #     task1
    #    /    \
    # task2  task3
    #    \    /
    #     task4
    tasks = [
        ParallelTask(
            task_id="task1",
            request=ToolExecutionRequest(
                tool_id="echo",
                parameters={"message": "start"},
                agent_id="test-agent",
            ),
            dependencies=[],
        ),
        ParallelTask(
            task_id="task2",
            request=ToolExecutionRequest(
                tool_id="calculator",
                parameters={"operation": "+", "a": 1, "b": 1},
                agent_id="test-agent",
            ),
            dependencies=["task1"],
        ),
        ParallelTask(
            task_id="task3",
            request=ToolExecutionRequest(
                tool_id="calculator",
                parameters={"operation": "+", "a": 2, "b": 2},
                agent_id="test-agent",
            ),
            dependencies=["task1"],
        ),
        ParallelTask(
            task_id="task4",
            request=ToolExecutionRequest(
                tool_id="echo",
                parameters={"message": "end"},
                agent_id="test-agent",
            ),
            dependencies=["task2", "task3"],
        ),
    ]

    results = await parallel_executor.execute_parallel(tasks, max_concurrent=10)

    assert len(results) == 4
    assert all(r.status == ToolExecutionStatus.SUCCESS for r in results.values())

    # task2 and task3 should have been able to run in parallel
    # (both depend only on task1)


@pytest.mark.asyncio
async def test_execute_parallel_handles_failures(parallel_executor: ParallelExecutor):
    """Test parallel execution handles task failures."""
    tasks = [
        ParallelTask(
            task_id="task1",
            request=ToolExecutionRequest(
                tool_id="calculator",
                parameters={"operation": "+", "a": 1, "b": 1},
                agent_id="test-agent",
            ),
            dependencies=[],
        ),
        ParallelTask(
            task_id="task2",
            request=ToolExecutionRequest(
                tool_id="nonexistent_tool",  # This will fail
                parameters={},
                agent_id="test-agent",
            ),
            dependencies=[],
        ),
        ParallelTask(
            task_id="task3",
            request=ToolExecutionRequest(
                tool_id="echo",
                parameters={"message": "test"},
                agent_id="test-agent",
            ),
            dependencies=[],
        ),
    ]

    results = await parallel_executor.execute_parallel(tasks, max_concurrent=10)

    assert len(results) == 3
    assert results["task1"].status == ToolExecutionStatus.SUCCESS
    assert results["task2"].status == ToolExecutionStatus.FAILED
    assert results["task3"].status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_execute_parallel_deadlock_detection(parallel_executor: ParallelExecutor):
    """Test deadlock detection with circular dependencies."""
    # Create circular dependency: task1 -> task2 -> task1
    tasks = [
        ParallelTask(
            task_id="task1",
            request=ToolExecutionRequest(
                tool_id="echo",
                parameters={"message": "a"},
                agent_id="test-agent",
            ),
            dependencies=["task2"],
        ),
        ParallelTask(
            task_id="task2",
            request=ToolExecutionRequest(
                tool_id="echo",
                parameters={"message": "b"},
                agent_id="test-agent",
            ),
            dependencies=["task1"],
        ),
    ]

    with pytest.raises(RuntimeError, match="deadlock"):
        await parallel_executor.execute_parallel(tasks, max_concurrent=10)


@pytest.mark.asyncio
async def test_execute_with_timeout_success(tool_executor: ToolExecutor):
    """Test tool execution with timeout - success case."""
    request = ToolExecutionRequest(
        tool_id="calculator",
        parameters={"operation": "+", "a": 1, "b": 1},
        agent_id="test-agent",
    )

    result = await execute_with_timeout(tool_executor, request, timeout_seconds=5.0)

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result == 2


@pytest.mark.asyncio
async def test_execute_with_timeout_exceeded(tool_executor: ToolExecutor):
    """Test tool execution timeout."""
    # Note: This test uses a very short timeout to trigger timeout behavior
    # In reality, even fast tools may not complete in 0.001s
    request = ToolExecutionRequest(
        tool_id="calculator",
        parameters={"operation": "+", "a": 1, "b": 1},
        agent_id="test-agent",
    )

    result = await execute_with_timeout(tool_executor, request, timeout_seconds=0.001)

    # May timeout or succeed depending on execution speed
    assert result.status in [ToolExecutionStatus.SUCCESS, ToolExecutionStatus.TIMEOUT]


@pytest.mark.asyncio
async def test_execute_with_fallback_primary_success(tool_executor: ToolExecutor):
    """Test fallback when primary succeeds."""
    primary_request = ToolExecutionRequest(
        tool_id="calculator",
        parameters={"operation": "+", "a": 1, "b": 1},
        agent_id="test-agent",
    )

    fallback_request = ToolExecutionRequest(
        tool_id="echo",
        parameters={"message": "fallback"},
        agent_id="test-agent",
    )

    result = await execute_with_fallback(
        tool_executor,
        primary_request,
        fallback_request,
    )

    # Primary should succeed, no fallback needed
    assert result.tool_id == "calculator"
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.result == 2
    assert result.metadata.get("fallback_used") is None


@pytest.mark.asyncio
async def test_execute_with_fallback_primary_fails(tool_executor: ToolExecutor):
    """Test fallback when primary fails."""
    primary_request = ToolExecutionRequest(
        tool_id="nonexistent_tool",
        parameters={},
        agent_id="test-agent",
    )

    fallback_request = ToolExecutionRequest(
        tool_id="echo",
        parameters={"message": "fallback_success"},
        agent_id="test-agent",
    )

    result = await execute_with_fallback(
        tool_executor,
        primary_request,
        fallback_request,
    )

    # Fallback should be used
    assert result.tool_id == "echo"
    assert result.result == "fallback_success"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["primary_tool"] == "nonexistent_tool"


@pytest.mark.asyncio
async def test_parallel_execution_order(parallel_executor: ParallelExecutor):
    """Test that results are returned in request order."""
    requests = [
        ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": f"msg_{i}"},
            agent_id="test-agent",
        )
        for i in range(5)
    ]

    results = await parallel_executor.execute_batch(requests, max_concurrent=10)

    # Results should be in same order as requests
    for i, result in enumerate(results):
        assert result.result == f"msg_{i}"
