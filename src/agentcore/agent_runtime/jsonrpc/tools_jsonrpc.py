"""JSON-RPC methods for tool integration."""

from typing import Any
import asyncio
from datetime import datetime

import structlog

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.agent_runtime.models.tool_integration import (
    ToolCategory,
    ToolExecutionStatus,
)

from agentcore.agent_runtime.tools.base import ExecutionContext
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry
from agentcore.agent_runtime.tools.registration import register_native_builtin_tools

# Global registry and executor
_tool_registry: ToolRegistry | None = None
_tool_executor: ToolExecutor | None = None


def get_tool_registry() -> ToolRegistry:
    """Get or create global tool registry.

    Note: For production use, prefer initializing via startup.initialize_tool_system()
    during application lifespan. This lazy initialization is for backward compatibility.
    """
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        # Use legacy registration for backward compatibility
        # Production should use startup.register_builtin_tools() with config
        register_native_builtin_tools(_tool_registry)
    return _tool_registry


def set_tool_registry(registry: ToolRegistry) -> None:
    """Set the global tool registry.

    This should be called during application startup after initializing
    the tool system with proper configuration.

    Args:
        registry: Initialized ToolRegistry instance
    """
    global _tool_registry
    _tool_registry = registry


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


@register_jsonrpc_method("tools.execute_batch")
async def handle_tools_execute_batch(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute multiple tools in batch with concurrency control.

    Method: tools.execute_batch
    Params:
        - requests: list[dict] - List of tool execution requests
            Each request contains:
                - tool_id: string
                - parameters: dict
                - agent_id: string
        - max_concurrent: int (optional) - Maximum concurrent executions (default: 5)

    Returns:
        - results: list of execution results (preserves order)
        - successful_count: number of successful executions
        - failed_count: number of failed executions
        - total_time_ms: total execution time in milliseconds
    """
    params = request.params or {}
    requests = params.get("requests")
    max_concurrent = params.get("max_concurrent", 5)

    if not requests or not isinstance(requests, list) or len(requests) == 0:
        raise ValueError("requests parameter required and must be non-empty list")

    start_time = datetime.utcnow()
    executor = get_tool_executor()

    # Create execution contexts and tasks
    tasks = []
    for req in requests:
        tool_id = req.get("tool_id")
        parameters = req.get("parameters", {})
        agent_id = req.get("agent_id")

        if not tool_id or not agent_id:
            raise ValueError("Each request must have tool_id and agent_id")

        context = ExecutionContext(
            agent_id=agent_id,
            user_id=agent_id,
            trace_id=req.get("execution_context", {}).get("trace_id") if "execution_context" in req else None,
        )

        tasks.append((tool_id, parameters, context))

    # Execute with concurrency control using semaphore
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_with_semaphore(tool_id: str, parameters: dict[str, Any], context: ExecutionContext) -> dict[str, Any]:
        async with semaphore:
            try:
                result = await executor.execute_tool(tool_id, parameters, context)
                return {
                    "tool_id": tool_id,
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms,
                }
            except Exception as e:
                return {
                    "tool_id": tool_id,
                    "status": "failed",
                    "result": None,
                    "error": str(e),
                    "execution_time_ms": 0,
                }

    # Execute all tasks concurrently with semaphore control
    results = await asyncio.gather(
        *[execute_with_semaphore(tool_id, params, ctx) for tool_id, params, ctx in tasks]
    )

    # Calculate statistics
    successful_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - successful_count
    total_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        "tools_execute_batch_called",
        total_requests=len(requests),
        successful_count=successful_count,
        failed_count=failed_count,
        total_time_ms=total_time_ms,
    )

    return {
        "results": results,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "total_time_ms": total_time_ms,
    }


@register_jsonrpc_method("tools.execute_parallel")
async def handle_tools_execute_parallel(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute tools in parallel with dependency management.

    Method: tools.execute_parallel
    Params:
        - tasks: list[dict] - List of tasks with dependencies
            Each task contains:
                - task_id: string - Unique task identifier
                - tool_id: string
                - parameters: dict
                - agent_id: string
                - dependencies: list[string] - Task IDs this task depends on
        - max_concurrent: int (optional) - Maximum concurrent executions (default: 10)

    Returns:
        - results: dict[task_id -> execution result]
        - successful_count: number of successful executions
        - failed_count: number of failed executions
        - execution_order: list of task_ids in execution order
        - total_time_ms: total execution time in milliseconds
    """
    params = request.params or {}
    tasks = params.get("tasks")
    max_concurrent = params.get("max_concurrent", 10)

    if not tasks or not isinstance(tasks, list):
        raise ValueError("tasks parameter required and must be a list")

    # Validate task structure
    task_map = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            raise ValueError("Each task must have task_id")
        if not task.get("tool_id") or not task.get("agent_id"):
            raise ValueError(f"Task {task_id} must have tool_id and agent_id")
        task_map[task_id] = task

    start_time = datetime.utcnow()
    executor = get_tool_executor()

    # Track completed tasks and their results
    completed: dict[str, dict[str, Any]] = {}
    execution_order: list[str] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_task(task_id: str) -> None:
        """Execute a task after its dependencies complete."""
        task = task_map[task_id]
        dependencies = task.get("dependencies", [])

        # Wait for dependencies to complete
        while not all(dep_id in completed for dep_id in dependencies):
            await asyncio.sleep(0.01)  # Small delay to avoid busy waiting

        # Check if any dependency failed
        dependency_failed = any(
            completed[dep_id]["status"] == "failed" for dep_id in dependencies
        )

        if dependency_failed:
            completed[task_id] = {
                "tool_id": task["tool_id"],
                "status": "failed",
                "result": None,
                "error": "Dependency failed",
                "execution_time_ms": 0,
            }
            execution_order.append(task_id)
            return

        # Execute the tool
        async with semaphore:
            try:
                context = ExecutionContext(
                    agent_id=task["agent_id"],
                    user_id=task["agent_id"],
                    trace_id=task.get("execution_context", {}).get("trace_id") if "execution_context" in task else None,
                )

                result = await executor.execute_tool(
                    task["tool_id"],
                    task.get("parameters", {}),
                    context,
                )

                completed[task_id] = {
                    "tool_id": task["tool_id"],
                    "status": result.status.value,
                    "result": result.result,
                    "error": result.error,
                    "execution_time_ms": result.execution_time_ms,
                }
            except Exception as e:
                completed[task_id] = {
                    "tool_id": task["tool_id"],
                    "status": "failed",
                    "result": None,
                    "error": str(e),
                    "execution_time_ms": 0,
                }

            execution_order.append(task_id)

    # Execute all tasks concurrently (dependencies handled internally)
    await asyncio.gather(*[execute_task(task_id) for task_id in task_map.keys()])

    # Calculate statistics
    successful_count = sum(1 for r in completed.values() if r["status"] == "success")
    failed_count = len(completed) - successful_count
    total_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        "tools_execute_parallel_called",
        total_tasks=len(tasks),
        successful_count=successful_count,
        failed_count=failed_count,
        total_time_ms=total_time_ms,
    )

    return {
        "results": completed,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "execution_order": execution_order,
        "total_time_ms": total_time_ms,
    }


@register_jsonrpc_method("tools.execute_with_fallback")
async def handle_tools_execute_with_fallback(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute a tool with fallback on failure.

    Method: tools.execute_with_fallback
    Params:
        - primary: dict - Primary tool execution request
            - tool_id: string
            - parameters: dict
            - agent_id: string
        - fallback: dict - Fallback tool execution request (same structure as primary)

    Returns:
        - result: execution result (from primary or fallback)
        - used_fallback: boolean - Whether fallback was used
        - primary_error: string | null - Error from primary if it failed
        - total_time_ms: total execution time in milliseconds
    """
    params = request.params or {}
    primary = params.get("primary")
    fallback = params.get("fallback")

    if not primary:
        raise ValueError("primary parameter required")
    if not fallback:
        raise ValueError("fallback parameter required")

    start_time = datetime.utcnow()
    executor = get_tool_executor()

    # Try primary execution
    primary_error = None
    used_fallback = False
    result_data = None

    try:
        context = ExecutionContext(
            agent_id=primary["agent_id"],
            user_id=primary["agent_id"],
            trace_id=primary.get("execution_context", {}).get("trace_id") if "execution_context" in primary else None,
        )

        result = await executor.execute_tool(
            primary["tool_id"],
            primary.get("parameters", {}),
            context,
        )

        if result.status == ToolExecutionStatus.SUCCESS:
            result_data = {
                "tool_id": primary["tool_id"],
                "status": result.status.value,
                "result": result.result,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            }
        else:
            primary_error = result.error or "Primary execution failed"
            raise ValueError(primary_error)

    except Exception as e:
        primary_error = str(e)
        used_fallback = True

        # Execute fallback
        try:
            context = ExecutionContext(
                agent_id=fallback["agent_id"],
                user_id=fallback["agent_id"],
                trace_id=fallback.get("execution_context", {}).get("trace_id") if "execution_context" in fallback else None,
            )

            result = await executor.execute_tool(
                fallback["tool_id"],
                fallback.get("parameters", {}),
                context,
            )

            result_data = {
                "tool_id": fallback["tool_id"],
                "status": result.status.value,
                "result": result.result,
                "error": result.error,
                "execution_time_ms": result.execution_time_ms,
            }

        except Exception as fallback_error:
            result_data = {
                "tool_id": fallback["tool_id"],
                "status": "failed",
                "result": None,
                "error": str(fallback_error),
                "execution_time_ms": 0,
            }

    total_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

    logger.info(
        "tools_execute_with_fallback_called",
        used_fallback=used_fallback,
        primary_error=primary_error,
        total_time_ms=total_time_ms,
    )

    return {
        "result": result_data,
        "used_fallback": used_fallback,
        "primary_error": primary_error,
        "total_time_ms": total_time_ms,
    }


