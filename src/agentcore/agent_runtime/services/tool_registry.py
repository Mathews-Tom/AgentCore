"""Tool registry and execution service."""

import asyncio
import time
from typing import Any, Callable

import structlog

from ..models.tool_integration import ToolDefinition
from ..engines.react_models import ToolCall, ToolResult

logger = structlog.get_logger()


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""


class ToolRegistry:
    """Registry for managing and executing agent tools."""

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable[..., Any]] = {}

    def register_tool(
        self,
        tool: ToolDefinition,
        executor: Callable[..., Any],
    ) -> None:
        """
        Register a tool with its executor function.

        Args:
            tool: Tool definition
            executor: Async function to execute the tool
        """
        self._tools[tool.tool_id] = tool
        self._executors[tool.tool_id] = executor

        logger.info(
            "tool_registered",
            tool_id=tool.tool_id,
            tool_name=tool.name,
        )

    def unregister_tool(self, tool_id: str) -> None:
        """
        Unregister a tool.

        Args:
            tool_id: Tool identifier
        """
        if tool_id in self._tools:
            del self._tools[tool_id]
            del self._executors[tool_id]
            logger.info("tool_unregistered", tool_id=tool_id)

    def get_tool(self, tool_id: str) -> ToolDefinition | None:
        """
        Get tool definition by ID.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool definition or None if not found
        """
        return self._tools.get(tool_id)

    def list_tools(self) -> list[ToolDefinition]:
        """
        List all registered tools.

        Returns:
            List of tool definitions
        """
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools.

        Returns:
            Formatted tool descriptions for prompts
        """
        descriptions = []
        for tool in self._tools.values():
            param_str = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in tool.parameters.items()
            )
            descriptions.append(
                f"- {tool.name}({param_str}): {tool.description}"
            )
        return "\n".join(descriptions)

    async def execute_tool(
        self,
        tool_call: ToolCall,
        agent_id: str,
    ) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: Tool call specification
            agent_id: Agent making the call

        Returns:
            Tool execution result
        """
        start_time = time.time()

        try:
            # Validate tool exists
            tool = self.get_tool(tool_call.tool_name)
            if not tool:
                raise ToolExecutionError(f"Tool '{tool_call.tool_name}' not found")

            # Get executor
            executor = self._executors.get(tool_call.tool_name)
            if not executor:
                raise ToolExecutionError(
                    f"No executor for tool '{tool_call.tool_name}'"
                )

            # Execute tool
            logger.info(
                "tool_execution_start",
                tool_name=tool_call.tool_name,
                agent_id=agent_id,
                call_id=tool_call.call_id,
            )

            # Run executor (handle both sync and async)
            if asyncio.iscoroutinefunction(executor):
                result = await executor(**tool_call.parameters)
            else:
                result = executor(**tool_call.parameters)

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                "tool_execution_success",
                tool_name=tool_call.tool_name,
                agent_id=agent_id,
                execution_time_ms=execution_time,
            )

            return ToolResult(
                call_id=tool_call.call_id,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000

            logger.error(
                "tool_execution_failed",
                tool_name=tool_call.tool_name,
                agent_id=agent_id,
                error=str(e),
                execution_time_ms=execution_time,
            )

            return ToolResult(
                call_id=tool_call.call_id,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )


# Built-in example tools
async def calculator_tool(operation: str, a: float, b: float) -> float:
    """
    Perform basic calculator operations.

    Args:
        operation: Operation (+, -, *, /)
        a: First number
        b: Second number

    Returns:
        Result of operation
    """
    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    return operations[operation](a, b)


def get_current_time_tool() -> str:
    """
    Get current time.

    Returns:
        Current time as ISO string
    """
    from datetime import datetime
    return datetime.now().isoformat()


async def echo_tool(message: str) -> str:
    """
    Echo a message back.

    Args:
        message: Message to echo

    Returns:
        The same message
    """
    return message


# Global tool registry instance
_global_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get global tool registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        _register_builtin_tools(_global_registry)
    return _global_registry


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Register built-in example tools."""
    # Calculator tool
    calculator_def = ToolDefinition(
        tool_id="calculator",
        name="calculator",
        description="Perform basic arithmetic operations",
        parameters={
            "operation": {"type": "string", "enum": ["+", "-", "*", "/"]},
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
    )
    registry.register_tool(calculator_def, calculator_tool)

    # Current time tool
    time_def = ToolDefinition(
        tool_id="get_current_time",
        name="get_current_time",
        description="Get the current date and time",
        parameters={},
    )
    registry.register_tool(time_def, get_current_time_tool)

    # Echo tool
    echo_def = ToolDefinition(
        tool_id="echo",
        name="echo",
        description="Echo a message back",
        parameters={
            "message": {"type": "string"},
        },
    )
    registry.register_tool(echo_def, echo_tool)
