"""Tests for parallel execution JSON-RPC methods."""

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.agent_runtime.jsonrpc.tools_jsonrpc import (
    handle_tools_execute_batch,
    handle_tools_execute_parallel,
    handle_tools_execute_with_fallback,
)


@pytest.fixture
def setup_tools():
    """Tests use built-in tools that are already registered (calculator, echo)."""
    # The calculator and echo tools are already registered via get_tool_registry()
    # in tools_jsonrpc.py, so we don't need to register anything here
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
                    "tool_id": "echo",
                    "parameters": {"message": "msg1"},
                    "agent_id": "test-agent",
                },
                {
                    "tool_id": "echo",
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
    # Echo tool returns a dict with 'echo' key
    assert result["results"][0]["result"]["echo"] == "msg1"
    assert result["results"][1]["result"]["echo"] == "msg2"
    # Calculator tool returns a dict with 'result' key
    assert result["results"][2]["result"]["result"] == 30.0


@pytest.mark.asyncio
async def test_tools_execute_batch_with_failures(setup_tools):
    """Test batch execution with some failures."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_batch",
        params={
            "requests": [
                {
                    "tool_id": "echo",
                    "parameters": {"message": "success"},
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

    assert len(result["results"]) == 2
    assert result["successful_count"] == 1
    assert result["failed_count"] == 1

    # First should succeed
    assert result["results"][0]["status"] == "success"

    # Second should fail
    assert result["results"][1]["status"] == "failed"


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
                    "tool_id": "echo",
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
    # Calculator returns dict with 'result' key
    assert result["results"]["task1"]["result"]["result"] == 8.0
    assert result["results"]["task2"]["result"]["result"] == 8.0
    # Echo tool returns dict with 'echo' key
    assert "final" in result["results"]["task3"]["result"]["echo"]

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
                    "tool_id": "echo",
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
                    "tool_id": "echo",
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
                "tool_id": "echo",
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
    # Calculator tool returns dict with 'result' key
    assert result["result"]["result"]["result"] == 30.0
    assert result["result"]["status"] == "success"


@pytest.mark.asyncio
async def test_tools_execute_with_fallback_primary_fails(setup_tools):
    """Test fallback execution when primary fails."""
    request = JsonRpcRequest(
        jsonrpc="2.0",
        method="tools.execute_with_fallback",
        params={
            "primary": {
                "tool_id": "calculator",
                "parameters": {"operation": "/", "a": 10, "b": 0},
                "agent_id": "test-agent",
            },
            "fallback": {
                "tool_id": "echo",
                "parameters": {"message": "fallback success"},
                "agent_id": "test-agent",
            },
        },
        id="test-6",
    )

    result = await handle_tools_execute_with_fallback(request)

    assert result["used_fallback"] is True
    assert result["primary_error"] is not None
    assert result["result"]["tool_id"] == "echo"
    # Echo tool returns dict with 'echo' key
    assert "fallback success" in result["result"]["result"]["echo"]
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
                    "tool_id": "echo",
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
                "tool_id": "echo",
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
                "tool_id": "echo",
                "parameters": {"message": "test"},
                "agent_id": "test-agent",
            }
        },
        id="test-12",
    )

    with pytest.raises(ValueError, match="fallback parameter required"):
        await handle_tools_execute_with_fallback(request)
