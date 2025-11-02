"""Tests for parallel execution JSON-RPC methods."""

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.agent_runtime.jsonrpc.tools_jsonrpc import (
    handle_tools_execute_batch,
    handle_tools_execute_parallel,
    handle_tools_execute_with_fallback,
)
from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from agentcore.agent_runtime.services.tool_registry import get_tool_registry


@pytest.fixture
def setup_tools():
    """Register test tools."""
    registry = get_tool_registry()

    # Success tool
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
    )
    registry.register_tool(success_def, success_tool)

    # Calculator tool
    async def calculator(operation: str, a: float, b: float) -> float:
        if operation == "+":
            return a + b
        elif operation == "-":
            return a - b
        elif operation == "*":
            return a * b
        elif operation == "/":
            if b == 0:
                raise ValueError("Division by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")

    calc_def = ToolDefinition(
        tool_id="calculator",
        name="calculator",
        description="Basic calculator",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "operation": ToolParameter(
                name="operation",
                type="string",
                description="Operation",
                required=True,
                enum=["+", "-", "*", "/"],
            ),
            "a": ToolParameter(
                name="a",
                type="number",
                description="First number",
                required=True,
            ),
            "b": ToolParameter(
                name="b",
                type="number",
                description="Second number",
                required=True,
            ),
        },
        auth_method=AuthMethod.NONE,
    )
    registry.register_tool(calc_def, calculator)

    # Failing tool
    async def failing_tool(message: str) -> str:
        raise ValueError(f"Tool failed: {message}")

    failing_def = ToolDefinition(
        tool_id="failing_tool",
        name="Failing Tool",
        description="Always fails",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "message": ToolParameter(
                name="message",
                type="string",
                description="Failure message",
                required=True,
            )
        },
        auth_method=AuthMethod.NONE,
    )
    registry.register_tool(failing_def, failing_tool)

    yield


@pytest.mark.asyncio
async def test_tools_execute_batch(setup_tools):
    """Test batch execution via JSON-RPC."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_batch",
        params={
            "requests": [
                {
                    "tool_id": "success_tool",
                    "parameters": {"message": "msg1"},
                    "agent_id": "test-agent",
                },
                {
                    "tool_id": "success_tool",
                    "parameters": {"message": "msg2"},
                    "agent_id": "test-agent",
                },
                {
                    "tool_id": "calculator",
                    "parameters": {"operation": "+", "a": 10, "b": 20},
                    "agent_id": "test-agent",
                },
            ],
            "max_concurrent": 5,
        },
        id="test-1",
    )

    result = await handle_tools_execute_batch(request)

    assert "results" in result
    assert len(result["results"]) == 3
    assert result["successful_count"] == 3
    assert result["failed_count"] == 0
    assert result["total_time_ms"] > 0

    # Verify results are in order
    assert "msg1" in result["results"][0]["result"]
    assert "msg2" in result["results"][1]["result"]
    assert result["results"][2]["result"] == 30


@pytest.mark.asyncio
async def test_tools_execute_batch_with_failures(setup_tools):
    """Test batch execution with some failures."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_batch",
        params={
            "requests": [
                {
                    "tool_id": "success_tool",
                    "parameters": {"message": "success"},
                    "agent_id": "test-agent",
                },
                {
                    "tool_id": "failing_tool",
                    "parameters": {"message": "failure"},
                    "agent_id": "test-agent",
                },
                {
                    "tool_id": "calculator",
                    "parameters": {"operation": "/", "a": 10, "b": 0},
                    "agent_id": "test-agent",
                },
            ],
        },
        id="test-2",
    )

    result = await handle_tools_execute_batch(request)

    assert len(result["results"]) == 3
    assert result["successful_count"] == 1
    assert result["failed_count"] == 2

    # First should succeed
    assert result["results"][0]["status"] == "success"

    # Second and third should fail
    assert result["results"][1]["status"] == "failed"
    assert result["results"][2]["status"] == "failed"


@pytest.mark.asyncio
async def test_tools_execute_parallel_with_dependencies(setup_tools):
    """Test parallel execution with dependencies."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_parallel",
        params={
            "tasks": [
                {
                    "task_id": "task1",
                    "tool_id": "calculator",
                    "parameters": {"operation": "+", "a": 5, "b": 3},
                    "agent_id": "test-agent",
                    "dependencies": [],
                },
                {
                    "task_id": "task2",
                    "tool_id": "calculator",
                    "parameters": {"operation": "*", "a": 2, "b": 4},
                    "agent_id": "test-agent",
                    "dependencies": ["task1"],
                },
                {
                    "task_id": "task3",
                    "tool_id": "success_tool",
                    "parameters": {"message": "final"},
                    "agent_id": "test-agent",
                    "dependencies": ["task2"],
                },
            ],
            "max_concurrent": 10,
        },
        id="test-3",
    )

    result = await handle_tools_execute_parallel(request)

    assert "results" in result
    assert len(result["results"]) == 3
    assert result["successful_count"] == 3
    assert result["failed_count"] == 0

    # Verify results
    assert result["results"]["task1"]["result"] == 8
    assert result["results"]["task2"]["result"] == 8
    assert "final" in result["results"]["task3"]["result"]

    # Verify execution order respects dependencies
    assert "execution_order" in result
    assert result["execution_order"] == ["task1", "task2", "task3"]


@pytest.mark.asyncio
async def test_tools_execute_parallel_diamond_dependencies(setup_tools):
    """Test parallel execution with diamond dependency pattern."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_parallel",
        params={
            "tasks": [
                {
                    "task_id": "task1",
                    "tool_id": "success_tool",
                    "parameters": {"message": "start"},
                    "agent_id": "test-agent",
                    "dependencies": [],
                },
                {
                    "task_id": "task2",
                    "tool_id": "calculator",
                    "parameters": {"operation": "+", "a": 1, "b": 1},
                    "agent_id": "test-agent",
                    "dependencies": ["task1"],
                },
                {
                    "task_id": "task3",
                    "tool_id": "calculator",
                    "parameters": {"operation": "+", "a": 2, "b": 2},
                    "agent_id": "test-agent",
                    "dependencies": ["task1"],
                },
                {
                    "task_id": "task4",
                    "tool_id": "success_tool",
                    "parameters": {"message": "end"},
                    "agent_id": "test-agent",
                    "dependencies": ["task2", "task3"],
                },
            ],
        },
        id="test-4",
    )

    result = await handle_tools_execute_parallel(request)

    assert len(result["results"]) == 4
    assert result["successful_count"] == 4
    assert result["failed_count"] == 0


@pytest.mark.asyncio
async def test_tools_execute_with_fallback_primary_succeeds(setup_tools):
    """Test fallback execution when primary succeeds."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_with_fallback",
        params={
            "primary": {
                "tool_id": "calculator",
                "parameters": {"operation": "+", "a": 10, "b": 20},
                "agent_id": "test-agent",
            },
            "fallback": {
                "tool_id": "success_tool",
                "parameters": {"message": "fallback"},
                "agent_id": "test-agent",
            },
        },
        id="test-5",
    )

    result = await handle_tools_execute_with_fallback(request)

    assert result["used_fallback"] is False
    assert result["primary_error"] is None
    assert result["result"]["tool_id"] == "calculator"
    assert result["result"]["result"] == 30
    assert result["result"]["status"] == "success"


@pytest.mark.asyncio
async def test_tools_execute_with_fallback_primary_fails(setup_tools):
    """Test fallback execution when primary fails."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_with_fallback",
        params={
            "primary": {
                "tool_id": "failing_tool",
                "parameters": {"message": "will fail"},
                "agent_id": "test-agent",
            },
            "fallback": {
                "tool_id": "success_tool",
                "parameters": {"message": "fallback success"},
                "agent_id": "test-agent",
            },
        },
        id="test-6",
    )

    result = await handle_tools_execute_with_fallback(request)

    assert result["used_fallback"] is True
    assert result["primary_error"] is not None
    assert result["result"]["tool_id"] == "success_tool"
    assert "fallback success" in result["result"]["result"]
    assert result["result"]["status"] == "success"


@pytest.mark.asyncio
async def test_tools_execute_batch_validation(setup_tools):
    """Test batch execution parameter validation."""
    # Missing requests
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_batch",
        params={},
        id="test-7",
    )

    with pytest.raises(ValueError, match="requests parameter required"):
        await handle_tools_execute_batch(request)

    # Empty requests list
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_batch",
        params={"requests": []},
        id="test-8",
    )

    with pytest.raises(ValueError, match="requests parameter required"):
        await handle_tools_execute_batch(request)


@pytest.mark.asyncio
async def test_tools_execute_parallel_validation(setup_tools):
    """Test parallel execution parameter validation."""
    # Missing tasks
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_parallel",
        params={},
        id="test-9",
    )

    with pytest.raises(ValueError, match="tasks parameter required"):
        await handle_tools_execute_parallel(request)

    # Missing task_id
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_parallel",
        params={
            "tasks": [
                {
                    "tool_id": "success_tool",
                    "parameters": {"message": "test"},
                    "agent_id": "test-agent",
                }
            ]
        },
        id="test-10",
    )

    with pytest.raises(ValueError, match="Each task must have task_id"):
        await handle_tools_execute_parallel(request)


@pytest.mark.asyncio
async def test_tools_execute_with_fallback_validation(setup_tools):
    """Test fallback execution parameter validation."""
    # Missing primary
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_with_fallback",
        params={
            "fallback": {
                "tool_id": "success_tool",
                "parameters": {"message": "test"},
                "agent_id": "test-agent",
            }
        },
        id="test-11",
    )

    with pytest.raises(ValueError, match="primary parameter required"):
        await handle_tools_execute_with_fallback(request)

    # Missing fallback
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_with_fallback",
        params={
            "primary": {
                "tool_id": "success_tool",
                "parameters": {"message": "test"},
                "agent_id": "test-agent",
            }
        },
        id="test-12",
    )

    with pytest.raises(ValueError, match="fallback parameter required"):
        await handle_tools_execute_with_fallback(request)
