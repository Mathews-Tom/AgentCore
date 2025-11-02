"""Integration tests for tool executor with rate limiting and retry logic."""

import asyncio
from datetime import datetime, timezone

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionStatus,
    ToolParameter,
)
from agentcore.agent_runtime.services.parallel_executor import (
    ParallelExecutor,
    ParallelTask,
)
from agentcore.agent_runtime.services.rate_limiter import (
    RateLimitExceeded,
    RateLimiter,
)
from agentcore.agent_runtime.services.retry_handler import BackoffStrategy, RetryHandler
from agentcore.agent_runtime.services.tool_executor import ToolExecutor
from agentcore.agent_runtime.services.tool_executor_factory import create_tool_executor
from agentcore.agent_runtime.services.tool_registry import ToolRegistry


@pytest.fixture
async def redis_rate_limiter():
    """Create rate limiter with Redis container."""
    pytest.importorskip("testcontainers")
    from testcontainers.redis import RedisContainer

    with RedisContainer("redis:7-alpine") as redis:
        # Get connection details
        host = redis.get_container_host_ip()
        port = redis.get_exposed_port(6379)
        redis_url = f"redis://{host}:{port}/0"

        limiter = RateLimiter(redis_url=redis_url)
        await limiter.connect()
        yield limiter
        await limiter.disconnect()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create tool registry with test tools."""
    registry = ToolRegistry()

    # Simple success tool
    async def success_tool(message: str) -> str:
        return f"Success: {message}"

    success_def = ToolDefinition(
        tool_id="success_tool",
        name="Success Tool",
        description="Always succeeds",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "message": ToolParameter(
                name="message",
                type="string",
                description="Message to return",
                required=True,
            )
        },
        auth_method=AuthMethod.NONE,
        is_retryable=True,
        max_retries=3,
    )
    registry.register_tool(success_def, success_tool)

    # Flaky tool (fails first 2 times)
    call_counts = {}

    async def flaky_tool(task_id: str) -> str:
        count = call_counts.get(task_id, 0)
        call_counts[task_id] = count + 1

        if count < 2:
            raise ValueError(f"Temporary failure (attempt {count + 1})")
        return f"Success after {count + 1} attempts"

    flaky_def = ToolDefinition(
        tool_id="flaky_tool",
        name="Flaky Tool",
        description="Fails first 2 times, then succeeds",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "task_id": ToolParameter(
                name="task_id",
                type="string",
                description="Task identifier",
                required=True,
            )
        },
        auth_method=AuthMethod.NONE,
        is_retryable=True,
        max_retries=5,
    )
    registry.register_tool(flaky_def, flaky_tool)

    # Rate limited tool
    async def rate_limited_tool(counter: int) -> int:
        return counter * 2

    rate_limited_def = ToolDefinition(
        tool_id="rate_limited_tool",
        name="Rate Limited Tool",
        description="Tool with rate limiting",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "counter": ToolParameter(
                name="counter",
                type="number",
                description="Counter value",
                required=True,
            )
        },
        auth_method=AuthMethod.NONE,
        rate_limits={"requests_per_minute": 5},  # 5 requests per minute
    )
    registry.register_tool(rate_limited_def, rate_limited_tool)

    # Slow tool
    async def slow_tool(delay: float) -> str:
        await asyncio.sleep(delay)
        return f"Completed after {delay}s"

    slow_def = ToolDefinition(
        tool_id="slow_tool",
        name="Slow Tool",
        description="Takes time to execute",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "delay": ToolParameter(
                name="delay",
                type="number",
                description="Delay in seconds",
                required=True,
                min_value=0,
                max_value=10,
            )
        },
        auth_method=AuthMethod.NONE,
        timeout_seconds=5,
    )
    registry.register_tool(slow_def, slow_tool)

    return registry


@pytest.mark.asyncio
async def test_tool_executor_with_retry_handler(tool_registry: ToolRegistry):
    """Test tool executor with retry handler for flaky tools."""
    retry_handler = RetryHandler(
        max_retries=5,
        base_delay=0.1,
        strategy=BackoffStrategy.EXPONENTIAL,
        jitter=False,
    )

    executor = ToolExecutor(
        registry=tool_registry,
        retry_handler=retry_handler,
    )

    request = ToolExecutionRequest(
        tool_id="flaky_tool",
        parameters={"task_id": "test_task_1"},
        agent_id="test-agent",
    )

    result = await executor.execute(request)

    assert result.status == ToolExecutionStatus.SUCCESS
    assert "Success after 3 attempts" in result.result
    assert result.retry_count == 2  # 2 retries after initial attempt


@pytest.mark.asyncio
async def test_tool_executor_with_rate_limiter(
    tool_registry: ToolRegistry, redis_rate_limiter: RateLimiter
):
    """Test tool executor with rate limiting."""
    executor = ToolExecutor(
        registry=tool_registry,
        rate_limiter=redis_rate_limiter,
    )

    # Execute 5 requests (should succeed - within limit)
    requests = [
        ToolExecutionRequest(
            tool_id="rate_limited_tool",
            parameters={"counter": i},
            agent_id="test-agent",
        )
        for i in range(5)
    ]

    results = []
    for req in requests:
        result = await executor.execute(req)
        results.append(result)

    # All should succeed
    assert all(r.status == ToolExecutionStatus.SUCCESS for r in results)

    # 6th request should be rate limited
    extra_request = ToolExecutionRequest(
        tool_id="rate_limited_tool",
        parameters={"counter": 100},
        agent_id="test-agent",
    )

    result = await executor.execute(extra_request)
    assert result.status == ToolExecutionStatus.FAILED
    assert result.error_type == "RateLimitExceeded"
    assert "retry_after" in result.metadata


@pytest.mark.asyncio
async def test_tool_executor_timeout_handling(tool_registry: ToolRegistry):
    """Test tool executor handles timeouts correctly."""
    executor = ToolExecutor(registry=tool_registry)

    # Tool with 5s timeout, request takes 10s
    request = ToolExecutionRequest(
        tool_id="slow_tool",
        parameters={"delay": 10.0},
        agent_id="test-agent",
    )

    result = await executor.execute(request)

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_parallel_execution_with_executor(tool_registry: ToolRegistry):
    """Test parallel execution of multiple tools."""
    executor = ToolExecutor(registry=tool_registry)
    parallel_exec = ParallelExecutor(executor)

    # Create tasks without dependencies
    tasks = [
        ParallelTask(
            task_id=f"task_{i}",
            request=ToolExecutionRequest(
                tool_id="success_tool",
                parameters={"message": f"Message {i}"},
                agent_id="test-agent",
            ),
        )
        for i in range(10)
    ]

    results = await parallel_exec.execute_parallel(tasks, max_concurrent=5)

    assert len(results) == 10
    assert all(r.status == ToolExecutionStatus.SUCCESS for r in results.values())


@pytest.mark.asyncio
async def test_parallel_execution_with_dependencies(tool_registry: ToolRegistry):
    """Test parallel execution respects task dependencies."""
    executor = ToolExecutor(registry=tool_registry)
    parallel_exec = ParallelExecutor(executor)

    execution_order = []

    # Override success tool to track execution order
    async def tracking_tool(message: str) -> str:
        execution_order.append(message)
        await asyncio.sleep(0.1)
        return f"Success: {message}"

    success_def = tool_registry.get_tool("success_tool")
    tool_registry._executors["success_tool"] = tracking_tool

    # Create dependency chain: task1 -> task2 -> task3
    tasks = [
        ParallelTask(
            task_id="task1",
            request=ToolExecutionRequest(
                tool_id="success_tool",
                parameters={"message": "first"},
                agent_id="test-agent",
            ),
            dependencies=[],
        ),
        ParallelTask(
            task_id="task2",
            request=ToolExecutionRequest(
                tool_id="success_tool",
                parameters={"message": "second"},
                agent_id="test-agent",
            ),
            dependencies=["task1"],
        ),
        ParallelTask(
            task_id="task3",
            request=ToolExecutionRequest(
                tool_id="success_tool",
                parameters={"message": "third"},
                agent_id="test-agent",
            ),
            dependencies=["task2"],
        ),
    ]

    results = await parallel_exec.execute_parallel(tasks)

    # All tasks should succeed
    assert len(results) == 3
    assert all(r.status == ToolExecutionStatus.SUCCESS for r in results.values())

    # Verify execution order
    assert execution_order == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_tool_executor_factory_with_config(tool_registry: ToolRegistry):
    """Test tool executor factory creates properly configured executor."""
    config_override = {
        "tool_max_retries": 5,
        "tool_retry_strategy": "linear",
        "tool_retry_base_delay": 0.5,
        "rate_limiter_enabled": False,
    }

    executor = create_tool_executor(
        registry=tool_registry,
        settings_override=config_override,
    )

    # Test retry configuration works
    request = ToolExecutionRequest(
        tool_id="flaky_tool",
        parameters={"task_id": "factory_test"},
        agent_id="test-agent",
    )

    result = await executor.execute(request)
    assert result.status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_executor_with_hooks(tool_registry: ToolRegistry):
    """Test executor lifecycle hooks."""
    before_calls = []
    after_calls = []
    error_calls = []

    def before_hook(request: ToolExecutionRequest) -> None:
        before_calls.append(request.tool_id)

    def after_hook(result) -> None:
        after_calls.append(result.tool_id)

    def error_hook(request: ToolExecutionRequest, error: Exception) -> None:
        error_calls.append((request.tool_id, type(error).__name__))

    executor = ToolExecutor(registry=tool_registry)
    executor.add_before_hook(before_hook)
    executor.add_after_hook(after_hook)
    executor.add_error_hook(error_hook)

    # Successful execution
    request = ToolExecutionRequest(
        tool_id="success_tool",
        parameters={"message": "test"},
        agent_id="test-agent",
    )

    await executor.execute(request)

    assert "success_tool" in before_calls
    assert "success_tool" in after_calls
    assert len(error_calls) == 0


@pytest.mark.asyncio
async def test_combined_rate_limiting_and_retry(
    tool_registry: ToolRegistry, redis_rate_limiter: RateLimiter
):
    """Test executor with both rate limiting and retry logic."""
    retry_handler = RetryHandler(
        max_retries=3,
        base_delay=0.1,
        strategy=BackoffStrategy.EXPONENTIAL,
    )

    executor = ToolExecutor(
        registry=tool_registry,
        rate_limiter=redis_rate_limiter,
        retry_handler=retry_handler,
    )

    # Test flaky tool with rate limiting
    request = ToolExecutionRequest(
        tool_id="flaky_tool",
        parameters={"task_id": "combined_test"},
        agent_id="test-agent",
    )

    result = await executor.execute(request)

    # Should succeed after retries
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.retry_count >= 2


@pytest.mark.asyncio
async def test_parallel_execution_batch(tool_registry: ToolRegistry):
    """Test batch execution returns results in order."""
    executor = ToolExecutor(registry=tool_registry)
    parallel_exec = ParallelExecutor(executor)

    requests = [
        ToolExecutionRequest(
            tool_id="success_tool",
            parameters={"message": f"msg_{i}"},
            agent_id="test-agent",
        )
        for i in range(10)
    ]

    results = await parallel_exec.execute_batch(requests, max_concurrent=5)

    # Results should be in same order as requests
    assert len(results) == 10
    for i, result in enumerate(results):
        assert result.status == ToolExecutionStatus.SUCCESS
        assert f"msg_{i}" in result.result


@pytest.mark.asyncio
async def test_executor_metrics_tracking(tool_registry: ToolRegistry):
    """Test executor tracks execution metrics."""
    executor = ToolExecutor(registry=tool_registry, enable_metrics=True)

    request = ToolExecutionRequest(
        tool_id="success_tool",
        parameters={"message": "metrics_test"},
        agent_id="test-agent",
    )

    result = await executor.execute(request)

    # Verify metrics are recorded
    assert result.execution_time_ms > 0
    assert result.timestamp is not None
    assert isinstance(result.timestamp, datetime)
