"""Tests for tool executor Prometheus metrics integration."""

import pytest
from prometheus_client import CollectorRegistry

from agentcore.agent_runtime.models.tool_integration import (
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolResult,
)
from agentcore.agent_runtime.services.metrics_collector import MetricsCollector
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry


class SimpleTestTool(Tool):
    """Simple test tool for metrics testing."""

    def __init__(self):
        """Initialize simple test tool."""
        metadata = ToolDefinition(
            tool_id="simple_test_tool",
            name="Simple Test Tool",
            description="A simple test tool",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={},
            returns={},
        )
        super().__init__(metadata)

    async def execute(self, parameters: dict, context: ExecutionContext) -> ToolResult:
        """Execute the tool."""
        return ToolResult(
            request_id=context.request_id,
            tool_id="simple_test_tool",
            status=ToolExecutionStatus.SUCCESS,
            result={"result": "success"},
            execution_time_ms=10.0,
        )


class FailingTestTool(Tool):
    """Test tool that always fails."""

    def __init__(self):
        """Initialize failing test tool."""
        metadata = ToolDefinition(
            tool_id="failing_test_tool",
            name="Failing Test Tool",
            description="A tool that always fails",
            version="1.0.0",
            category=ToolCategory.UTILITY,
            parameters={},
            returns={},
        )
        super().__init__(metadata)

    async def execute(self, parameters: dict, context: ExecutionContext) -> ToolResult:
        """Execute the tool - always fails."""
        raise ValueError("Tool execution failed")


@pytest.fixture
def metrics_collector() -> MetricsCollector:
    """Fixture for metrics collector with isolated registry."""
    registry = CollectorRegistry()
    return MetricsCollector(registry=registry)


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Fixture for tool registry."""
    registry = ToolRegistry()
    registry.register(SimpleTestTool())
    registry.register(FailingTestTool())
    return registry


@pytest.fixture
def executor_with_metrics(
    tool_registry: ToolRegistry, metrics_collector: MetricsCollector
) -> ToolExecutor:
    """Fixture for executor with metrics collection enabled."""
    return ToolExecutor(
        registry=tool_registry,
        metrics_collector=metrics_collector,
    )


@pytest.mark.asyncio
async def test_executor_with_metrics_initialization(
    executor_with_metrics: ToolExecutor, metrics_collector: MetricsCollector
):
    """Test that executor initializes with metrics collector."""
    assert executor_with_metrics.metrics_collector is metrics_collector


@pytest.mark.asyncio
async def test_metrics_emitted_on_success(
    executor_with_metrics: ToolExecutor, metrics_collector: MetricsCollector
):
    """Test that metrics are emitted on successful tool execution."""
    context = ExecutionContext(
        request_id="test-request-1",
        user_id="test-user",
        agent_id="test-agent",
        trace_id="test-trace",
    )

    # Execute tool
    result = await executor_with_metrics.execute_tool(
        tool_id="simple_test_tool",
        parameters={},
        context=context,
    )

    # Verify execution succeeded
    assert result.status == ToolExecutionStatus.SUCCESS

    # Verify metrics were emitted
    # Get metric samples
    samples = list(metrics_collector._registry.collect())

    # Find tool execution counter
    execution_counter_samples = [
        s
        for metric in samples
        for s in metric.samples
        if s.name == "agentcore_tool_executions_total"
    ]

    # Should have one sample for this execution
    assert len(execution_counter_samples) >= 1

    # Find the sample for our tool
    success_samples = [
        s
        for s in execution_counter_samples
        if s.labels.get("tool_id") == "simple_test_tool"
        and s.labels.get("status") == "success"
    ]

    assert len(success_samples) == 1
    assert success_samples[0].value == 1.0

    # Check duration histogram was recorded
    duration_samples = [
        s
        for metric in samples
        for s in metric.samples
        if s.name == "agentcore_tool_execution_seconds_count"
        and s.labels.get("tool_id") == "simple_test_tool"
    ]

    assert len(duration_samples) == 1
    assert duration_samples[0].value == 1.0


@pytest.mark.asyncio
async def test_metrics_emitted_on_failure(
    executor_with_metrics: ToolExecutor, metrics_collector: MetricsCollector
):
    """Test that metrics are emitted on tool execution failure."""
    context = ExecutionContext(
        request_id="test-request-2",
        user_id="test-user",
        agent_id="test-agent",
        trace_id="test-trace",
    )

    # Execute failing tool
    result = await executor_with_metrics.execute_tool(
        tool_id="failing_test_tool",
        parameters={},
        context=context,
    )

    # Verify execution failed
    assert result.status == ToolExecutionStatus.FAILED

    # Verify metrics were emitted
    samples = list(metrics_collector._registry.collect())

    # Check execution counter for failure
    execution_counter_samples = [
        s
        for metric in samples
        for s in metric.samples
        if s.name == "agentcore_tool_executions_total"
        and s.labels.get("tool_id") == "failing_test_tool"
        and s.labels.get("status") == "failed"
    ]

    assert len(execution_counter_samples) == 1
    assert execution_counter_samples[0].value == 1.0

    # Check error counter was incremented
    error_samples = [
        s
        for metric in samples
        for s in metric.samples
        if s.name == "agentcore_tool_errors_total"
        and s.labels.get("tool_id") == "failing_test_tool"
    ]

    assert len(error_samples) >= 1
    # Should have at least one error recorded
    total_errors = sum(s.value for s in error_samples)
    assert total_errors >= 1.0


@pytest.mark.asyncio
async def test_metrics_not_emitted_without_collector(tool_registry: ToolRegistry):
    """Test that executor works without metrics collector (no metrics emitted)."""
    executor = ToolExecutor(registry=tool_registry)

    assert executor.metrics_collector is None

    context = ExecutionContext(
        request_id="test-request-3",
        user_id="test-user",
        agent_id="test-agent",
        trace_id="test-trace",
    )

    # Execute tool - should succeed without metrics
    result = await executor.execute_tool(
        tool_id="simple_test_tool",
        parameters={},
        context=context,
    )

    assert result.status == ToolExecutionStatus.SUCCESS


@pytest.mark.asyncio
async def test_multiple_executions_increment_metrics(
    executor_with_metrics: ToolExecutor, metrics_collector: MetricsCollector
):
    """Test that multiple executions correctly increment metrics."""
    context = ExecutionContext(
        request_id="test-request-4",
        user_id="test-user",
        agent_id="test-agent",
        trace_id="test-trace",
    )

    # Execute tool 3 times
    for i in range(3):
        result = await executor_with_metrics.execute_tool(
            tool_id="simple_test_tool",
            parameters={},
            context=context,
        )
        assert result.status == ToolExecutionStatus.SUCCESS

    # Verify metrics show 3 executions
    samples = list(metrics_collector._registry.collect())

    execution_counter_samples = [
        s
        for metric in samples
        for s in metric.samples
        if s.name == "agentcore_tool_executions_total"
        and s.labels.get("tool_id") == "simple_test_tool"
        and s.labels.get("status") == "success"
    ]

    assert len(execution_counter_samples) == 1
    assert execution_counter_samples[0].value == 3.0

    # Check duration histogram count
    duration_samples = [
        s
        for metric in samples
        for s in metric.samples
        if s.name == "agentcore_tool_execution_seconds_count"
        and s.labels.get("tool_id") == "simple_test_tool"
    ]

    assert len(duration_samples) == 1
    assert duration_samples[0].value == 3.0
