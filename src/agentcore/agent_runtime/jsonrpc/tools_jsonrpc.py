"""JSON-RPC methods for tool integration."""

from typing import Any
import asyncio
from datetime import UTC, datetime

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
from agentcore.agent_runtime.services.rate_limiter import get_rate_limiter
from agentcore.agent_runtime.services.quota_manager import get_quota_manager

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

    # Get tools by category if specified
    if tool_category:
        tools = registry.list_by_category(tool_category)
    else:
        tools = registry.list_all()

    # Filter by capabilities (AND logic - all specified capabilities must be present)
    if capabilities:
        tools = [
            tool
            for tool in tools
            if all(
                cap in getattr(tool.metadata, "capabilities", [])
                for cap in capabilities
            )
        ]

    # Filter by tags (OR logic - at least one tag must match)
    if tags:
        tools = [
            tool
            for tool in tools
            if any(
                tag in getattr(tool.metadata, "tags", [])
                for tag in tags
            )
        ]

    # Convert to JSON-serializable format
    tools_data = [
        {
            "tool_id": tool.metadata.tool_id,
            "name": tool.metadata.name,
            "description": tool.metadata.description,
            "version": tool.metadata.version,
            "category": tool.metadata.category.value,
            "capabilities": getattr(tool.metadata, "capabilities", []),
            "tags": getattr(tool.metadata, "tags", []),
            "parameters": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                }
                for param in tool.metadata.parameters.values()
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
        "capabilities": getattr(tool.metadata, "capabilities", []),
        "tags": getattr(tool.metadata, "tags", []),
        "auth_method": tool.metadata.auth_method.value,
        "timeout_seconds": tool.metadata.timeout_seconds,
        "parameters": {
            param.name: {
                "name": param.name,
                "type": param.type,
                "description": param.description,
                "required": param.required,
            }
            for param in tool.metadata.parameters.values()
        },
    }

    logger.info("tools_get_called", tool_id=tool_id)

    return tool_data


@register_jsonrpc_method("tools.execute")
async def handle_tools_execute(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute a tool with A2A authentication integration.

    Method: tools.execute
    Params:
        - tool_id: string - Tool to execute
        - parameters: dict - Tool parameters
        - agent_id: string - Requesting agent ID (optional if A2A context provided)
        - execution_context: dict (optional) - Execution context (trace_id, session_id, etc.)
        - timeout_override: number (optional) - Timeout override in seconds

    A2A Context:
        If request.a2a_context is provided, it will be used for authentication:
        - source_agent: Requesting agent identifier
        - trace_id: Distributed tracing ID
        - session_id: Session identifier
        - target_agent: Target agent (if tool execution is delegated)

    Returns:
        Tool execution result with metadata
    """
    params = request.params or {}
    tool_id = params.get("tool_id")
    parameters = params.get("parameters", {})
    agent_id = params.get("agent_id")
    execution_context = params.get("execution_context")
    timeout_override = params.get("timeout_override")

    # Extract A2A context for authentication
    a2a_context = request.a2a_context

    # Use A2A context source_agent if agent_id not provided
    if not agent_id and a2a_context:
        agent_id = a2a_context.source_agent

    if not tool_id:
        raise ValueError("tool_id parameter required")

    if not agent_id:
        raise ValueError("agent_id parameter required or must be provided via A2A context")

    return await tools_execute(
        tool_id, parameters, agent_id, execution_context, timeout_override, a2a_context
    )


async def tools_execute(
    tool_id: str,
    parameters: dict[str, Any],
    agent_id: str,
    execution_context: dict[str, str] | None = None,
    timeout_override: int | None = None,
    a2a_context: Any = None,
) -> dict[str, Any]:
    """
    Execute a tool with A2A authentication support.

    Args:
        tool_id: Tool to execute
        parameters: Tool parameters
        agent_id: Requesting agent ID
        execution_context: Optional execution context (trace_id, session_id, etc.)
        timeout_override: Optional timeout override
        a2a_context: Optional A2A protocol context (A2AContext model)

    Returns:
        Dictionary with execution result
    """
    executor = get_tool_executor()

    # Build execution context from A2A context or legacy execution_context
    trace_id = None
    session_id = None

    if a2a_context:
        # Prefer A2A context for distributed tracing
        trace_id = a2a_context.trace_id
        session_id = a2a_context.session_id
        logger.info(
            "tools_execute_with_a2a_context",
            tool_id=tool_id,
            source_agent=a2a_context.source_agent,
            target_agent=a2a_context.target_agent,
            trace_id=trace_id,
            session_id=session_id,
        )
    elif execution_context:
        # Fall back to legacy execution_context
        trace_id = execution_context.get("trace_id")
        session_id = execution_context.get("session_id")

    # Create execution context
    context = ExecutionContext(
        agent_id=agent_id,
        user_id=agent_id,  # In A2A, agent acts as user
        trace_id=trace_id,
        session_id=session_id,
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

    # Search tools
    tools = registry.search(query=name_query, category=tool_category)

    # Apply capabilities filter (AND logic - all must match)
    if capabilities:
        tools = [
            tool
            for tool in tools
            if all(
                cap in getattr(tool.metadata, "capabilities", [])
                for cap in capabilities
            )
        ]

    # Apply tags filter (OR logic - any must match)
    if tags:
        tools = [
            tool
            for tool in tools
            if any(
                tag in getattr(tool.metadata, "tags", [])
                for tag in tags
            )
        ]

    # Convert to JSON-serializable format
    tools_data = [
        {
            "tool_id": tool.metadata.tool_id,
            "name": tool.metadata.name,
            "description": tool.metadata.description,
            "version": tool.metadata.version,
            "category": tool.metadata.category.value,
            "capabilities": getattr(tool.metadata, "capabilities", []),
            "tags": getattr(tool.metadata, "tags", []),
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
    Execute multiple tools in batch with concurrency control and A2A authentication.

    Method: tools.execute_batch
    Params:
        - requests: list[dict] - List of tool execution requests
            Each request contains:
                - tool_id: string
                - parameters: dict
                - agent_id: string (optional if A2A context provided)
        - max_concurrent: int (optional) - Maximum concurrent executions (default: 5)

    A2A Context:
        If request.a2a_context is provided, trace_id and session_id will be
        propagated to all batch executions for distributed tracing.

    Returns:
        - results: list of execution results (preserves order)
        - successful_count: number of successful executions
        - failed_count: number of failed executions
        - total_time_ms: total execution time in milliseconds
    """
    params = request.params or {}
    requests = params.get("requests")
    max_concurrent = params.get("max_concurrent", 5)
    a2a_context = request.a2a_context

    if not requests or not isinstance(requests, list) or len(requests) == 0:
        raise ValueError("requests parameter required and must be non-empty list")

    start_time = datetime.now(UTC)
    executor = get_tool_executor()

    # Extract A2A context values for reuse
    a2a_trace_id = a2a_context.trace_id if a2a_context else None
    a2a_session_id = a2a_context.session_id if a2a_context else None
    a2a_source_agent = a2a_context.source_agent if a2a_context else None

    # Create execution contexts and tasks
    tasks = []
    for req in requests:
        tool_id = req.get("tool_id")
        parameters = req.get("parameters", {})
        agent_id = req.get("agent_id")

        # Use A2A source_agent if agent_id not provided
        if not agent_id and a2a_source_agent:
            agent_id = a2a_source_agent

        if not tool_id or not agent_id:
            raise ValueError("Each request must have tool_id and agent_id (or A2A context with source_agent)")

        # Prefer A2A context trace_id, fall back to per-request execution_context
        trace_id = a2a_trace_id or (req.get("execution_context", {}).get("trace_id") if "execution_context" in req else None)
        session_id = a2a_session_id or (req.get("execution_context", {}).get("session_id") if "execution_context" in req else None)

        context = ExecutionContext(
            agent_id=agent_id,
            user_id=agent_id,
            trace_id=trace_id,
            session_id=session_id,
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
    total_time_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

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
    Execute tools in parallel with dependency management and A2A authentication.

    Method: tools.execute_parallel
    Params:
        - tasks: list[dict] - List of tasks with dependencies
            Each task contains:
                - task_id: string - Unique task identifier
                - tool_id: string
                - parameters: dict
                - agent_id: string (optional if A2A context provided)
                - dependencies: list[string] - Task IDs this task depends on
        - max_concurrent: int (optional) - Maximum concurrent executions (default: 10)

    A2A Context:
        If request.a2a_context is provided, trace_id and session_id will be
        propagated to all parallel executions for distributed tracing.

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
    a2a_context = request.a2a_context

    if not tasks or not isinstance(tasks, list):
        raise ValueError("tasks parameter required and must be a list")

    # Extract A2A context values for reuse
    a2a_trace_id = a2a_context.trace_id if a2a_context else None
    a2a_session_id = a2a_context.session_id if a2a_context else None
    a2a_source_agent = a2a_context.source_agent if a2a_context else None

    # Validate task structure
    task_map = {}
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            raise ValueError("Each task must have task_id")

        agent_id = task.get("agent_id")
        # Use A2A source_agent if agent_id not provided
        if not agent_id and a2a_source_agent:
            agent_id = a2a_source_agent
            task["agent_id"] = agent_id  # Update task with resolved agent_id

        if not task.get("tool_id") or not agent_id:
            raise ValueError(f"Task {task_id} must have tool_id and agent_id (or A2A context with source_agent)")

        task_map[task_id] = task

    start_time = datetime.now(UTC)
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
                # Prefer A2A context trace_id, fall back to per-task execution_context
                trace_id = a2a_trace_id or (task.get("execution_context", {}).get("trace_id") if "execution_context" in task else None)
                session_id = a2a_session_id or (task.get("execution_context", {}).get("session_id") if "execution_context" in task else None)

                context = ExecutionContext(
                    agent_id=task["agent_id"],
                    user_id=task["agent_id"],
                    trace_id=trace_id,
                    session_id=session_id,
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
    total_time_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

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
    Execute a tool with fallback on failure and A2A authentication.

    Method: tools.execute_with_fallback
    Params:
        - primary: dict - Primary tool execution request
            - tool_id: string
            - parameters: dict
            - agent_id: string (optional if A2A context provided)
        - fallback: dict - Fallback tool execution request (same structure as primary)

    A2A Context:
        If request.a2a_context is provided, trace_id and session_id will be
        propagated to both primary and fallback executions.

    Returns:
        - result: execution result (from primary or fallback)
        - used_fallback: boolean - Whether fallback was used
        - primary_error: string | null - Error from primary if it failed
        - total_time_ms: total execution time in milliseconds
    """
    params = request.params or {}
    primary = params.get("primary")
    fallback = params.get("fallback")
    a2a_context = request.a2a_context

    if not primary:
        raise ValueError("primary parameter required")
    if not fallback:
        raise ValueError("fallback parameter required")

    # Extract A2A context values for reuse
    a2a_trace_id = a2a_context.trace_id if a2a_context else None
    a2a_session_id = a2a_context.session_id if a2a_context else None
    a2a_source_agent = a2a_context.source_agent if a2a_context else None

    # Use A2A source_agent if agent_id not provided
    if not primary.get("agent_id") and a2a_source_agent:
        primary["agent_id"] = a2a_source_agent
    if not fallback.get("agent_id") and a2a_source_agent:
        fallback["agent_id"] = a2a_source_agent

    start_time = datetime.now(UTC)
    executor = get_tool_executor()

    # Try primary execution
    primary_error = None
    used_fallback = False
    result_data = None

    try:
        # Prefer A2A context trace_id, fall back to execution_context
        trace_id = a2a_trace_id or (primary.get("execution_context", {}).get("trace_id") if "execution_context" in primary else None)
        session_id = a2a_session_id or (primary.get("execution_context", {}).get("session_id") if "execution_context" in primary else None)

        context = ExecutionContext(
            agent_id=primary["agent_id"],
            user_id=primary["agent_id"],
            trace_id=trace_id,
            session_id=session_id,
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
            # Prefer A2A context trace_id, fall back to execution_context
            trace_id = a2a_trace_id or (fallback.get("execution_context", {}).get("trace_id") if "execution_context" in fallback else None)
            session_id = a2a_session_id or (fallback.get("execution_context", {}).get("session_id") if "execution_context" in fallback else None)

            context = ExecutionContext(
                agent_id=fallback["agent_id"],
                user_id=fallback["agent_id"],
                trace_id=trace_id,
                session_id=session_id,
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

    total_time_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

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


@register_jsonrpc_method("tools.get_rate_limit_status")
async def handle_tools_get_rate_limit_status(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get rate limit and quota status for a tool.

    Method: tools.get_rate_limit_status
    Params:
        - tool_id: string - Tool identifier
        - agent_id: string (optional) - Agent ID for per-user rate limits/quotas (optional if A2A context provided)

    A2A Context:
        If request.a2a_context is provided, source_agent will be used for per-user limits.

    Returns:
        - tool_id: string - Tool identifier
        - rate_limit: dict | null - Rate limit status
            - limit: int - Maximum requests per window
            - remaining: int - Remaining requests in current window
            - reset_at: string - When the window resets (ISO format)
            - window_seconds: int - Window duration in seconds
        - quota: dict | null - Quota status
            - daily_limit: int | null - Daily quota limit (null if unlimited)
            - daily_used: int - Used daily quota
            - daily_remaining: int | null - Remaining daily quota (null if unlimited)
            - daily_reset_at: string | null - When daily quota resets
            - monthly_limit: int | null - Monthly quota limit (null if unlimited)
            - monthly_used: int - Used monthly quota
            - monthly_remaining: int | null - Remaining monthly quota (null if unlimited)
            - monthly_reset_at: string | null - When monthly quota resets
    """
    params = request.params or {}
    tool_id = params.get("tool_id")
    agent_id = params.get("agent_id")
    a2a_context = request.a2a_context

    # Use A2A source_agent if agent_id not provided
    if not agent_id and a2a_context:
        agent_id = a2a_context.source_agent

    if not tool_id:
        raise ValueError("tool_id parameter required")

    return await tools_get_rate_limit_status(tool_id, agent_id)


async def tools_get_rate_limit_status(
    tool_id: str,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """
    Get rate limit and quota status for a tool.

    Args:
        tool_id: Tool identifier
        agent_id: Optional agent ID for per-user limits/quotas

    Returns:
        Dictionary with rate limit and quota status
    """
    registry = get_tool_registry()
    tool = registry.get(tool_id)

    if not tool:
        raise ValueError(f"Tool not found: {tool_id}")

    # Get rate limit status
    rate_limit_status = None
    if tool.metadata.rate_limits:
        # Check if there's a rate limit configuration
        # Format: {"calls_per_minute": 60} or {"requests_per_second": 10}
        rate_limiter = get_rate_limiter()

        # Extract rate limit config (prefer calls_per_minute)
        if "calls_per_minute" in tool.metadata.rate_limits:
            limit = tool.metadata.rate_limits["calls_per_minute"]
            window_seconds = 60
        elif "requests_per_second" in tool.metadata.rate_limits:
            limit = tool.metadata.rate_limits["requests_per_second"]
            window_seconds = 1
        elif "calls_per_hour" in tool.metadata.rate_limits:
            limit = tool.metadata.rate_limits["calls_per_hour"]
            window_seconds = 3600
        else:
            # Use first available rate limit config
            key = next(iter(tool.metadata.rate_limits))
            limit = tool.metadata.rate_limits[key]
            # Default to 60 second window
            window_seconds = 60

        rate_limit_status = await rate_limiter.get_remaining(
            tool_id=tool_id,
            limit=limit,
            window_seconds=window_seconds,
            identifier=agent_id,
        )

    # Get quota status
    quota_status = None
    if tool.metadata.daily_quota is not None or tool.metadata.monthly_quota is not None:
        quota_manager = get_quota_manager()
        quota_status = await quota_manager.get_quota_status(
            tool_id=tool_id,
            daily_quota=tool.metadata.daily_quota,
            monthly_quota=tool.metadata.monthly_quota,
            identifier=agent_id,
        )

    logger.info(
        "tools_get_rate_limit_status_called",
        tool_id=tool_id,
        agent_id=agent_id,
        has_rate_limit=rate_limit_status is not None,
        has_quota=quota_status is not None,
    )

    return {
        "tool_id": tool_id,
        "rate_limit": rate_limit_status,
        "quota": quota_status,
    }


