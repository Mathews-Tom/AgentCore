"""JSON-RPC methods for tool integration."""

from typing import Any

import structlog

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.agent_runtime.models.tool_integration import (
    ToolCategory,
)

from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry
from agentcore.agent_runtime.tools.registration import register_native_builtin_tools

# Global registry and executor
_tool_registry: ToolRegistry | None = None
_tool_executor: ToolExecutor | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        register_native_builtin_tools(_tool_registry)
    return _tool_registry


def get_tool_executor() -> ToolExecutor:
    """Get or create global tool executor."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor(get_tool_registry())
    return _tool_executor

logger = structlog.get_logger()


@register_jsonrpc_method("tools.list")
async def handle_tools_list(request: JsonRpcRequest) -> dict[str, Any]:
    """
    List available tools with optional filtering.

    Method: tools.list
    Params:
        - category: string (optional) - Filter by tool category
        - capabilities: list[string] (optional) - Filter by capabilities (AND logic)
        - tags: list[string] (optional) - Filter by tags (OR logic)

    Returns:
        - tools: list of tool definitions
        - count: number of tools
    """
    params = request.params or {}
    category = params.get("category")
    capabilities = params.get("capabilities")
    tags = params.get("tags")

    return await tools_list(category, capabilities, tags)


async def tools_list(
    category: str | None = None,
    capabilities: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    List available tools with optional filtering.

    Args:
        category: Filter by tool category
        capabilities: Filter by capabilities (AND logic - currently not supported)
        tags: Filter by tags (OR logic - currently not supported)

    Returns:
        Dictionary with list of tools
    """
    registry = get_tool_registry()

    # Parse category if provided
    tool_category = None
    if category:
        try:
            tool_category = ToolCategory(category)
        except ValueError:
            raise ValueError(f"Invalid category: {category}")

    # Search tools (capabilities and tags filtering not yet implemented in new registry)
    if tool_category:
        tools = registry.list_by_category(tool_category)
    else:
        tools = registry.list_all()

    # Convert to JSON-serializable format
    tools_data = [
        {
            "tool_id": tool.metadata.tool_id,
            "name": tool.metadata.name,
            "description": tool.metadata.description,
            "version": tool.metadata.version,
            "category": tool.metadata.category.value,
            "parameters": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                }
                for param in tool.metadata.parameters
            ],
        }
        for tool in tools
    ]

    logger.info(
        "tools_list_called",
        count=len(tools_data),
        category=category,
    )

    return {
        "tools": tools_data,
        "count": len(tools_data),
    }


@register_jsonrpc_method("tools.get")
async def handle_tools_get(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get detailed information about a specific tool.

    Method: tools.get
    Params:
        - tool_id: string - Tool identifier

    Returns:
        Tool definition with all details
    """
    params = request.params or {}
    tool_id = params.get("tool_id")

    if not tool_id:
        raise ValueError("tool_id parameter required")

    return await tools_get(tool_id)


async def tools_get(tool_id: str) -> dict[str, Any]:
    """
    Get detailed information about a specific tool.

    Args:
        tool_id: Tool identifier

    Returns:
        Dictionary with tool details
    """
    registry = get_tool_registry()
    tool = registry.get(tool_id)

    if not tool:
        raise ValueError(f"Tool not found: {tool_id}")

    # Convert to JSON-serializable format
    tool_data = {
        "tool_id": tool.metadata.tool_id,
        "name": tool.metadata.name,
        "description": tool.metadata.description,
        "version": tool.metadata.version,
        "category": tool.metadata.category.value,
        "parameters": [
            {
                "name": param.name,
                "type": param.type,
                "description": param.description,
                "required": param.required,
            }
            for param in tool.metadata.parameters
        ],
    }

    logger.info("tools_get_called", tool_id=tool_id)

    return tool_data


@register_jsonrpc_method("tools.execute")
async def handle_tools_execute(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute a tool.

    Method: tools.execute
    Params:
        - tool_id: string - Tool to execute
        - parameters: dict - Tool parameters
        - agent_id: string - Requesting agent ID
        - execution_context: dict (optional) - Execution context (trace_id, session_id, etc.)
        - timeout_override: number (optional) - Timeout override in seconds

    Returns:
        Tool execution result with metadata
    """
    params = request.params or {}
    tool_id = params.get("tool_id")
    parameters = params.get("parameters", {})
    agent_id = params.get("agent_id")
    execution_context = params.get("execution_context")
    timeout_override = params.get("timeout_override")

    if not tool_id:
        raise ValueError("tool_id parameter required")

    if not agent_id:
        raise ValueError("agent_id parameter required")

    return await tools_execute(
        tool_id, parameters, agent_id, execution_context, timeout_override
    )


async def tools_execute(
    tool_id: str,
    parameters: dict[str, Any],
    agent_id: str,
    execution_context: dict[str, str] | None = None,
    timeout_override: int | None = None,
) -> dict[str, Any]:
    """
    Execute a tool.

    Args:
        tool_id: Tool to execute
        parameters: Tool parameters
        agent_id: Requesting agent ID
        execution_context: Optional execution context (trace_id, session_id, etc.)
        timeout_override: Optional timeout override

    Returns:
        Dictionary with execution result
    """
    executor = get_tool_executor()

    # Create execution context
    context = ExecutionContext(
        agent_id=agent_id,
        user_id=agent_id,
        trace_id=execution_context.get("trace_id") if execution_context else None,
    )

    # Execute tool
    result = await executor.execute_tool(tool_id, parameters, context)

    # Convert to JSON-serializable format
    result_data = {
        "tool_id": tool_id,
        "status": result.status.value,
        "result": result.result,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms,
        "timestamp": result.timestamp.isoformat() if result.timestamp else None,
    }

    logger.info(
        "tools_execute_called",
        tool_id=tool_id,
        agent_id=agent_id,
        status=result.status.value,
        execution_time_ms=result.execution_time_ms,
    )

    return result_data


@register_jsonrpc_method("tools.search")
async def handle_tools_search(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Search tools with comprehensive filters.

    Method: tools.search
    Params:
        - name_query: string (optional) - Search by name (substring match)
        - category: string (optional) - Filter by tool category
        - capabilities: list[string] (optional) - Filter by capabilities (AND logic)
        - tags: list[string] (optional) - Filter by tags (OR logic)

    Returns:
        - tools: list of matching tools
        - count: number of results
        - query: search parameters used
    """
    params = request.params or {}
    return await tools_search(
        name_query=params.get("name_query"),
        category=params.get("category"),
        capabilities=params.get("capabilities"),
        tags=params.get("tags"),
    )


async def tools_search(
    name_query: str | None = None,
    category: str | None = None,
    capabilities: list[str] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Search tools with comprehensive filters.

    Args:
        name_query: Search by name (substring match)
        category: Filter by category
        capabilities: Filter by capabilities (not yet implemented)
        tags: Filter by tags (not yet implemented)

    Returns:
        Dictionary with search results
    """
    registry = get_tool_registry()

    # Parse category if provided
    tool_category = None
    if category:
        try:
            tool_category = ToolCategory(category)
        except ValueError:
            raise ValueError(f"Invalid category: {category}")

    # Search tools (capabilities and tags not yet implemented)
    tools = registry.search(query=name_query, category=tool_category)

    # Convert to JSON-serializable format
    tools_data = [
        {
            "tool_id": tool.metadata.tool_id,
            "name": tool.metadata.name,
            "description": tool.metadata.description,
            "version": tool.metadata.version,
            "category": tool.metadata.category.value,
        }
        for tool in tools
    ]

    logger.info(
        "tools_search_called",
        count=len(tools_data),
        name_query=name_query,
        category=category,
    )

    return {
        "tools": tools_data,
        "count": len(tools_data),
        "query": {
            "name": name_query,
            "category": category,
            "capabilities": capabilities,
            "tags": tags,
        },
    }


