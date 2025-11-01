"""Tool registry and execution service."""

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from ..models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)

# Avoid circular import
if TYPE_CHECKING:
    from ..engines.react_models import ToolCall, ToolResult as ReactToolResult

logger = structlog.get_logger()


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""


class ToolNotFoundError(Exception):
    """Raised when tool is not found in registry."""


class ToolValidationError(Exception):
    """Raised when tool parameters fail validation."""


class ToolRegistry:
    """Registry for managing and executing agent tools with advanced search and discovery."""

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: dict[str, ToolDefinition] = {}
        self._executors: dict[str, Callable[..., Any]] = {}
        self._category_index: dict[ToolCategory, set[str]] = {
            category: set() for category in ToolCategory
        }
        self._capability_index: dict[str, set[str]] = {}
        self._tag_index: dict[str, set[str]] = {}

    def register_tool(
        self,
        tool: ToolDefinition,
        executor: Callable[..., Any],
    ) -> None:
        """
        Register a tool with its executor function and build search indexes.

        Args:
            tool: Tool definition
            executor: Async function to execute the tool
        """
        self._tools[tool.tool_id] = tool
        self._executors[tool.tool_id] = executor

        # Build category index
        self._category_index[tool.category].add(tool.tool_id)

        # Build capability index
        for capability in tool.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(tool.tool_id)

        # Build tag index
        for tag in tool.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(tool.tool_id)

        logger.info(
            "tool_registered",
            tool_id=tool.tool_id,
            tool_name=tool.name,
            category=tool.category.value,
            version=tool.version,
        )

    def unregister_tool(self, tool_id: str) -> None:
        """
        Unregister a tool and update indexes.

        Args:
            tool_id: Tool identifier
        """
        if tool_id not in self._tools:
            return

        tool = self._tools[tool_id]

        # Remove from category index
        self._category_index[tool.category].discard(tool_id)

        # Remove from capability index
        for capability in tool.capabilities:
            if capability in self._capability_index:
                self._capability_index[capability].discard(tool_id)

        # Remove from tag index
        for tag in tool.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(tool_id)

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

    def search_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """
        Search tools by category.

        Args:
            category: Tool category to filter by

        Returns:
            List of tools in the specified category
        """
        tool_ids = self._category_index.get(category, set())
        return [self._tools[tid] for tid in tool_ids if tid in self._tools]

    def search_by_capability(self, capability: str) -> list[ToolDefinition]:
        """
        Search tools by capability.

        Args:
            capability: Required capability

        Returns:
            List of tools with the specified capability
        """
        tool_ids = self._capability_index.get(capability, set())
        return [self._tools[tid] for tid in tool_ids if tid in self._tools]

    def search_by_tags(self, tags: list[str]) -> list[ToolDefinition]:
        """
        Search tools by tags (OR logic).

        Args:
            tags: List of tags to search for

        Returns:
            List of tools matching any of the tags
        """
        matching_ids: set[str] = set()
        for tag in tags:
            if tag in self._tag_index:
                matching_ids.update(self._tag_index[tag])
        return [self._tools[tid] for tid in matching_ids if tid in self._tools]

    def search_tools(
        self,
        name_query: str | None = None,
        category: ToolCategory | None = None,
        capabilities: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """
        Comprehensive tool search with multiple filters.

        Args:
            name_query: Filter by name (case-insensitive substring match)
            category: Filter by category
            capabilities: Filter by capabilities (AND logic - must have all)
            tags: Filter by tags (OR logic - must have at least one)

        Returns:
            List of tools matching all specified filters
        """
        # Start with all tools or filter by category
        if category:
            candidates = set(self._category_index.get(category, set()))
        else:
            candidates = set(self._tools.keys())

        # Filter by capabilities (AND logic)
        if capabilities:
            for capability in capabilities:
                cap_tools = self._capability_index.get(capability, set())
                candidates &= cap_tools

        # Filter by tags (OR logic)
        if tags:
            tag_tools: set[str] = set()
            for tag in tags:
                tag_tools.update(self._tag_index.get(tag, set()))
            candidates &= tag_tools

        # Filter by name query
        results = []
        for tool_id in candidates:
            if tool_id not in self._tools:
                continue
            tool = self._tools[tool_id]
            if name_query and name_query.lower() not in tool.name.lower():
                continue
            results.append(tool)

        return results

    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools for LLM prompts.

        Returns:
            Formatted tool descriptions with parameters
        """
        descriptions = []
        for tool in self._tools.values():
            # Format parameters
            param_list = []
            for param_name, param in tool.parameters.items():
                param_desc = f"{param_name}: {param.type}"
                if param.required:
                    param_desc += " (required)"
                param_list.append(param_desc)

            param_str = ", ".join(param_list) if param_list else "no parameters"
            descriptions.append(f"- {tool.name}({param_str}): {tool.description}")

        return "\n".join(descriptions)

    async def execute_tool(
        self,
        tool_call: "ToolCall",
        agent_id: str,
    ) -> "ReactToolResult":
        """
        Execute a tool call (ReAct engine compatibility).

        Args:
            tool_call: Tool call specification from ReAct engine
            agent_id: Agent making the call

        Returns:
            ReactToolResult for backward compatibility with ReAct engine
        """
        # Import at runtime to avoid circular dependency
        from ..engines.react_models import ToolResult as ReactToolResult

        start_time = time.time()

        try:
            # Validate tool exists
            tool = self.get_tool(tool_call.tool_name)
            if not tool:
                raise ToolNotFoundError(f"Tool '{tool_call.tool_name}' not found")

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

            return ReactToolResult(
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

            return ReactToolResult(
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
    return datetime.now(UTC).isoformat()


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
    """Register built-in example tools with enhanced definitions."""
    # Calculator tool
    calculator_def = ToolDefinition(
        tool_id="calculator",
        name="calculator",
        description="Perform basic arithmetic operations",
        version="1.0.0",
        category=ToolCategory.DATA_PROCESSING,
        parameters={
            "operation": ToolParameter(
                name="operation",
                type="string",
                description="Arithmetic operation to perform",
                required=True,
                enum=["+", "-", "*", "/"],
            ),
            "a": ToolParameter(
                name="a",
                type="number",
                description="First operand",
                required=True,
            ),
            "b": ToolParameter(
                name="b",
                type="number",
                description="Second operand",
                required=True,
            ),
        },
        capabilities=["sync_execution"],
        tags=["math", "arithmetic", "calculator"],
    )
    registry.register_tool(calculator_def, calculator_tool)

    # Current time tool
    time_def = ToolDefinition(
        tool_id="get_current_time",
        name="get_current_time",
        description="Get the current date and time in ISO format",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={},
        capabilities=["sync_execution", "no_side_effects"],
        tags=["time", "datetime", "utility"],
        is_idempotent=False,  # Returns different value each time
    )
    registry.register_tool(time_def, get_current_time_tool)

    # Echo tool
    echo_def = ToolDefinition(
        tool_id="echo",
        name="echo",
        description="Echo a message back (useful for testing)",
        version="1.0.0",
        category=ToolCategory.CUSTOM,
        parameters={
            "message": ToolParameter(
                name="message",
                type="string",
                description="Message to echo",
                required=True,
                max_length=1000,
            ),
        },
        capabilities=["sync_execution", "no_side_effects"],
        tags=["test", "utility", "echo"],
    )
    registry.register_tool(echo_def, echo_tool)
