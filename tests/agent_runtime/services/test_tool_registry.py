"""Tests for tool registry."""

import pytest

from agentcore.agent_runtime.engines.react_models import ToolCall
from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from agentcore.agent_runtime.services.tool_registry import (
    ToolExecutionError,
    ToolRegistry,
    get_tool_registry,
)


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create fresh tool registry."""
    return ToolRegistry()


@pytest.mark.asyncio
async def test_register_tool(tool_registry: ToolRegistry) -> None:
    """Test tool registration."""
    async def test_tool(param: str) -> str:
        return f"Result: {param}"

    tool_def = ToolDefinition(
        tool_id="test_tool",
        name="test_tool",
        description="Test tool",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "param": ToolParameter(
                name="param",
                type="string",
                description="Test parameter",
                required=True,
            )
        },
    )

    tool_registry.register_tool(tool_def, test_tool)

    assert tool_registry.get_tool("test_tool") is not None
    assert len(tool_registry.list_tools()) == 1


@pytest.mark.asyncio
async def test_unregister_tool(tool_registry: ToolRegistry) -> None:
    """Test tool unregistration."""
    async def test_tool() -> str:
        return "test"

    tool_def = ToolDefinition(
        tool_id="test_tool",
        name="test_tool",
        description="Test",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={},
        auth_method=AuthMethod.NONE,
    )

    tool_registry.register_tool(tool_def, test_tool)
    assert tool_registry.get_tool("test_tool") is not None

    tool_registry.unregister_tool("test_tool")
    assert tool_registry.get_tool("test_tool") is None


@pytest.mark.asyncio
async def test_execute_tool_success(tool_registry: ToolRegistry) -> None:
    """Test successful tool execution."""
    async def add_tool(a: int, b: int) -> int:
        return a + b

    tool_def = ToolDefinition(
        tool_id="add",
        name="add",
        description="Add two numbers",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
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

    tool_registry.register_tool(tool_def, add_tool)

    tool_call = ToolCall(
        tool_name="add",
        parameters={"a": 5, "b": 3},
        call_id="test-call-1")

    result = await tool_registry.execute_tool(tool_call, "test-agent")

    assert result.success is True
    assert result.result == 8
    assert result.call_id == "test-call-1"


@pytest.mark.asyncio
async def test_execute_tool_not_found(tool_registry: ToolRegistry) -> None:
    """Test execution of non-existent tool."""
    tool_call = ToolCall(
        tool_name="nonexistent",
        parameters={},
        call_id="test-call-2")

    result = await tool_registry.execute_tool(tool_call, "test-agent")

    assert result.success is False
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_execute_tool_error(tool_registry: ToolRegistry) -> None:
    """Test tool execution with error."""
    async def error_tool() -> None:
        raise ValueError("Tool error")

    tool_def = ToolDefinition(
        tool_id="error_tool",
        name="error_tool",
        description="Tool that errors",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={},
        auth_method=AuthMethod.NONE,
    )

    tool_registry.register_tool(tool_def, error_tool)

    tool_call = ToolCall(
        tool_name="error_tool",
        parameters={},
        call_id="test-call-3")

    result = await tool_registry.execute_tool(tool_call, "test-agent")

    assert result.success is False
    assert "Tool error" in result.error


@pytest.mark.asyncio
async def test_get_tool_descriptions(tool_registry: ToolRegistry) -> None:
    """Test getting tool descriptions."""
    async def tool1(param: str) -> str:
        return param

    tool_def = ToolDefinition(
        tool_id="tool1",
        name="tool1",
        description="Test tool 1",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "param": ToolParameter(
                name="param",
                type="string",
                description="Parameter",
                required=True,
            )
        },
        auth_method=AuthMethod.NONE,
    )

    tool_registry.register_tool(tool_def, tool1)

    descriptions = tool_registry.get_tool_descriptions()

    assert "tool1" in descriptions
    assert "Test tool 1" in descriptions


@pytest.mark.asyncio
async def test_global_registry() -> None:
    """Test global tool registry."""
    registry = get_tool_registry()

    assert registry is not None
    # Should have built-in tools
    assert registry.get_tool("calculator") is not None
    assert registry.get_tool("get_current_time") is not None
    assert registry.get_tool("echo") is not None


@pytest.mark.asyncio
async def test_builtin_calculator_tool() -> None:
    """Test built-in calculator tool."""
    registry = get_tool_registry()

    # Test addition
    tool_call = ToolCall(
        tool_name="calculator",
        parameters={"operation": "+", "a": 10, "b": 5},
        call_id="calc-1")

    result = await registry.execute_tool(tool_call, "test-agent")

    assert result.success is True
    assert result.result == 15


@pytest.mark.asyncio
async def test_builtin_echo_tool() -> None:
    """Test built-in echo tool."""
    registry = get_tool_registry()

    tool_call = ToolCall(
        tool_name="echo",
        parameters={"message": "Hello, World!"},
        call_id="echo-1")

    result = await registry.execute_tool(tool_call, "test-agent")

    assert result.success is True
    assert result.result == "Hello, World!"


@pytest.mark.asyncio
async def test_tool_execution_time_tracking(tool_registry: ToolRegistry) -> None:
    """Test that execution time is tracked."""
    async def slow_tool() -> str:
        import asyncio
        await asyncio.sleep(0.01)
        return "done"

    tool_def = ToolDefinition(
        tool_id="slow_tool",
        name="slow_tool",
        description="Slow tool",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={},
        auth_method=AuthMethod.NONE,
    )

    tool_registry.register_tool(tool_def, slow_tool)

    tool_call = ToolCall(
        tool_name="slow_tool",
        parameters={},
        call_id="slow-1")

    result = await tool_registry.execute_tool(tool_call, "test-agent")

    assert result.success is True
    assert result.execution_time_ms > 0
