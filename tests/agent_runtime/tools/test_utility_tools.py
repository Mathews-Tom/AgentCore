"""Tests for utility tools (calculator, get_current_time, echo)."""

import time

import pytest

from agentcore.agent_runtime.models.tool_integration import (
    ToolExecutionRequest,
    ToolExecutionStatus,
)
from agentcore.agent_runtime.services.tool_executor import ToolExecutor
from agentcore.agent_runtime.services.tool_registry import ToolRegistry
from agentcore.agent_runtime.tools.utility_tools import register_utility_tools


@pytest.fixture
def registry() -> ToolRegistry:
    """Create tool registry with utility tools."""
    registry = ToolRegistry()
    register_utility_tools(registry)
    return registry


@pytest.fixture
def executor(registry: ToolRegistry) -> ToolExecutor:
    """Create tool executor with utility tools."""
    return ToolExecutor(registry=registry, enable_metrics=False)


class TestCalculatorTool:
    """Tests for calculator tool."""

    @pytest.mark.asyncio
    async def test_addition(self, executor: ToolExecutor):
        """Test addition operation."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "+", "a": 10, "b": 20},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 30
        assert result.result["expression"] == "10 + 20 = 30"

    @pytest.mark.asyncio
    async def test_subtraction(self, executor: ToolExecutor):
        """Test subtraction operation."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "-", "a": 50, "b": 30},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 20

    @pytest.mark.asyncio
    async def test_multiplication(self, executor: ToolExecutor):
        """Test multiplication operation."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "*", "a": 7, "b": 6},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 42

    @pytest.mark.asyncio
    async def test_division(self, executor: ToolExecutor):
        """Test division operation."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "/", "a": 100, "b": 4},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 25

    @pytest.mark.asyncio
    async def test_division_by_zero(self, executor: ToolExecutor):
        """Test division by zero error."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "/", "a": 10, "b": 0},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.FAILED
        assert "Division by zero" in result.error

    @pytest.mark.asyncio
    async def test_modulo(self, executor: ToolExecutor):
        """Test modulo operation."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "%", "a": 17, "b": 5},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 2

    @pytest.mark.asyncio
    async def test_modulo_by_zero(self, executor: ToolExecutor):
        """Test modulo by zero error."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "%", "a": 10, "b": 0},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.FAILED
        assert "Modulo by zero" in result.error

    @pytest.mark.asyncio
    async def test_power(self, executor: ToolExecutor):
        """Test power operation."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "**", "a": 2, "b": 8},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 256

    @pytest.mark.asyncio
    async def test_power_alternative_notation(self, executor: ToolExecutor):
        """Test power operation with ^ notation."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "^", "a": 3, "b": 3},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 27

    @pytest.mark.asyncio
    async def test_floating_point(self, executor: ToolExecutor):
        """Test floating point operations."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "+", "a": 1.5, "b": 2.3},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert abs(result.result["result"] - 3.8) < 0.0001

    @pytest.mark.asyncio
    async def test_negative_numbers(self, executor: ToolExecutor):
        """Test operations with negative numbers."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "*", "a": -5, "b": 4},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == -20

    @pytest.mark.asyncio
    async def test_invalid_operation(self, executor: ToolExecutor):
        """Test invalid operation error."""
        request = ToolExecutionRequest(
            tool_id="calculator",
            parameters={"operation": "&", "a": 10, "b": 20},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.FAILED
        # Parameter validation catches invalid enum values
        assert "must be one of" in result.error or "Invalid operation" in result.error


class TestGetCurrentTimeTool:
    """Tests for get_current_time tool."""

    @pytest.mark.asyncio
    async def test_get_time_iso_format(self, executor: ToolExecutor):
        """Test getting current time in ISO format."""
        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={"timezone": "UTC", "format": "iso"},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert "current_time" in result.result
        assert "T" in result.result["current_time"]  # ISO format has T
        assert result.result["timezone"] == "UTC"
        assert result.result["format"] == "iso"

    @pytest.mark.asyncio
    async def test_get_time_unix_format(self, executor: ToolExecutor):
        """Test getting current time in Unix timestamp format."""
        before = time.time()

        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={"timezone": "UTC", "format": "unix"},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        after = time.time()

        assert result.status == ToolExecutionStatus.SUCCESS
        assert isinstance(result.result["current_time"], (int, float))
        assert before <= result.result["current_time"] <= after

    @pytest.mark.asyncio
    async def test_get_time_human_format(self, executor: ToolExecutor):
        """Test getting current time in human-readable format."""
        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={"timezone": "UTC", "format": "human"},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert "UTC" in result.result["current_time"]
        assert "-" in result.result["current_time"]  # Date separator
        assert ":" in result.result["current_time"]  # Time separator

    @pytest.mark.asyncio
    async def test_get_time_defaults(self, executor: ToolExecutor):
        """Test getting current time with default parameters."""
        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["timezone"] == "UTC"
        assert result.result["format"] == "iso"

    @pytest.mark.asyncio
    async def test_get_time_includes_metadata(self, executor: ToolExecutor):
        """Test that result includes all metadata fields."""
        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert "current_time" in result.result
        assert "timezone" in result.result
        assert "format" in result.result
        assert "utc_timestamp" in result.result
        assert "iso_string" in result.result

    @pytest.mark.asyncio
    async def test_invalid_timezone(self, executor: ToolExecutor):
        """Test error with unsupported timezone."""
        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={"timezone": "America/New_York", "format": "iso"},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.FAILED
        assert "not supported" in result.error

    @pytest.mark.asyncio
    async def test_invalid_format(self, executor: ToolExecutor):
        """Test error with invalid format."""
        request = ToolExecutionRequest(
            tool_id="get_current_time",
            parameters={"timezone": "UTC", "format": "invalid"},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.FAILED
        # Parameter validation catches invalid enum values
        assert "must be one of" in result.error or "Invalid format" in result.error


class TestEchoTool:
    """Tests for echo tool."""

    @pytest.mark.asyncio
    async def test_echo_basic(self, executor: ToolExecutor):
        """Test basic echo functionality."""
        request = ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": "Hello, World!"},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["echo"] == "Hello, World!"
        assert result.result["original"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_echo_uppercase(self, executor: ToolExecutor):
        """Test echo with uppercase transformation."""
        request = ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": "hello world", "uppercase": True},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["echo"] == "HELLO WORLD"
        assert result.result["original"] == "hello world"
        assert result.result["uppercase"] is True

    @pytest.mark.asyncio
    async def test_echo_metadata(self, executor: ToolExecutor):
        """Test echo includes metadata."""
        request = ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": "The quick brown fox"},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["length"] == 19
        assert result.result["word_count"] == 4

    @pytest.mark.asyncio
    async def test_echo_empty_string(self, executor: ToolExecutor):
        """Test echo with empty string."""
        request = ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": ""},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["echo"] == ""
        assert result.result["length"] == 0
        # For empty strings, we should return 0 word count
        assert result.result["word_count"] == 0

    @pytest.mark.asyncio
    async def test_echo_special_characters(self, executor: ToolExecutor):
        """Test echo with special characters."""
        message = "Hello\nWorld\t!\n@#$%"
        request = ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": message},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["echo"] == message
        assert result.result["length"] == len(message)

    @pytest.mark.asyncio
    async def test_echo_unicode(self, executor: ToolExecutor):
        """Test echo with Unicode characters."""
        message = "Hello 世界 🌍"
        request = ToolExecutionRequest(
            tool_id="echo",
            parameters={"message": message},
            agent_id="test-agent",
        )

        result = await executor.execute(request)

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["echo"] == message


class TestUtilityToolsRegistration:
    """Tests for utility tools registration."""

    def test_tools_registered(self, registry: ToolRegistry):
        """Test that all utility tools are registered."""
        tools = registry.list_tools()
        tool_ids = {tool.tool_id for tool in tools}

        assert "calculator" in tool_ids
        assert "get_current_time" in tool_ids
        assert "echo" in tool_ids

    def test_calculator_definition(self, registry: ToolRegistry):
        """Test calculator tool definition."""
        tool = registry.get_tool("calculator")

        assert tool is not None
        assert tool.name == "Calculator"
        assert tool.category.value == "utility"
        assert "operation" in tool.parameters
        assert "a" in tool.parameters
        assert "b" in tool.parameters

    def test_get_current_time_definition(self, registry: ToolRegistry):
        """Test get_current_time tool definition."""
        tool = registry.get_tool("get_current_time")

        assert tool is not None
        assert tool.name == "Get Current Time"
        assert tool.category.value == "utility"
        assert "timezone" in tool.parameters
        assert "format" in tool.parameters

    def test_echo_definition(self, registry: ToolRegistry):
        """Test echo tool definition."""
        tool = registry.get_tool("echo")

        assert tool is not None
        assert tool.name == "Echo"
        assert tool.category.value == "utility"
        assert "message" in tool.parameters
        assert "uppercase" in tool.parameters
