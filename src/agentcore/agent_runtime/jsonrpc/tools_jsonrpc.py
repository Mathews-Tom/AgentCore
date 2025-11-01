"""JSON-RPC methods for tool integration."""

from typing import Any

import structlog

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.agent_runtime.models.tool_integration import (
    ToolCategory,
    ToolExecutionRequest,
)
from agentcore.agent_runtime.services.parallel_executor import ParallelExecutor, ParallelTask
from agentcore.agent_runtime.services.tool_executor import get_tool_executor
from agentcore.agent_runtime.services.tool_registry import get_tool_registry

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
        capabilities: Filter by capabilities (AND logic)
        tags: Filter by tags (OR logic)

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

    # Search tools
    if category or capabilities or tags:
        tools = registry.search_tools(
            category=tool_category,
            capabilities=capabilities,
            tags=tags,
        )
    else:
        tools = registry.list_tools()

    # Convert to JSON-serializable format
    tools_data = [
        {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "description": tool.description,
            "version": tool.version,
            "category": tool.category.value,
            "parameters": {
                name: {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default,
                    "enum": param.enum,
                    "min_value": param.min_value,
                    "max_value": param.max_value,
                    "min_length": param.min_length,
                    "max_length": param.max_length,
                }
                for name, param in tool.parameters.items()
            },
            "auth_method": tool.auth_method.value,
            "timeout_seconds": tool.timeout_seconds,
            "is_retryable": tool.is_retryable,
            "is_idempotent": tool.is_idempotent,
            "max_retries": tool.max_retries,
            "capabilities": tool.capabilities,
            "tags": tool.tags,
            "requirements": tool.requirements,
            "metadata": tool.metadata,
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
    tool = registry.get_tool(tool_id)

    if not tool:
        raise ValueError(f"Tool not found: {tool_id}")

    # Convert to JSON-serializable format
    tool_data = {
        "tool_id": tool.tool_id,
        "name": tool.name,
        "description": tool.description,
        "version": tool.version,
        "category": tool.category.value,
        "parameters": {
            name: {
                "name": param.name,
                "type": param.type,
                "description": param.description,
                "required": param.required,
                "default": param.default,
                "enum": param.enum,
                "min_value": param.min_value,
                "max_value": param.max_value,
                "min_length": param.min_length,
                "max_length": param.max_length,
                "pattern": param.pattern,
            }
            for name, param in tool.parameters.items()
        },
        "auth_method": tool.auth_method.value,
        "auth_config": tool.auth_config,
        "timeout_seconds": tool.timeout_seconds,
        "is_retryable": tool.is_retryable,
        "is_idempotent": tool.is_idempotent,
        "max_retries": tool.max_retries,
        "rate_limits": tool.rate_limits,
        "cost_per_execution": tool.cost_per_execution,
        "capabilities": tool.capabilities,
        "tags": tool.tags,
        "requirements": tool.requirements,
        "security_requirements": tool.security_requirements,
        "metadata": tool.metadata,
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

    # Create execution request
    request = ToolExecutionRequest(
        tool_id=tool_id,
        parameters=parameters,
        agent_id=agent_id,
        execution_context=execution_context or {},
        timeout_override=timeout_override,
    )

    # Execute tool
    result = await executor.execute(request)

    # Convert to JSON-serializable format
    result_data = {
        "request_id": result.request_id,
        "tool_id": result.tool_id,
        "status": result.status.value,
        "result": result.result,
        "error": result.error,
        "error_type": result.error_type,
        "execution_time_ms": result.execution_time_ms,
        "timestamp": result.timestamp.isoformat(),
        "retry_count": result.retry_count,
        "memory_mb": result.memory_mb,
        "cpu_percent": result.cpu_percent,
        "metadata": result.metadata,
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
        capabilities: Filter by capabilities (AND logic)
        tags: Filter by tags (OR logic)

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
    tools = registry.search_tools(
        name_query=name_query,
        category=tool_category,
        capabilities=capabilities,
        tags=tags,
    )

    # Convert to JSON-serializable format
    tools_data = [
        {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "description": tool.description,
            "version": tool.version,
            "category": tool.category.value,
            "capabilities": tool.capabilities,
            "tags": tool.tags,
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
    Execute multiple tools in parallel without dependencies.

    Method: tools.execute_batch
    Params:
        - requests: list[dict] - List of tool execution requests
            Each request contains:
            - tool_id: string - Tool to execute
            - parameters: dict - Tool parameters
            - agent_id: string - Requesting agent ID
            - execution_context: dict (optional) - Execution context
            - timeout_override: number (optional) - Timeout override in seconds
        - max_concurrent: number (optional, default=10) - Maximum concurrent executions

    Returns:
        - results: list of tool execution results (in same order as requests)
        - total_time_ms: total execution time
        - successful_count: number of successful executions
        - failed_count: number of failed executions
    """
    params = request.params or {}
    requests_data = params.get("requests", [])
    max_concurrent = params.get("max_concurrent", 10)

    if not requests_data:
        raise ValueError("requests parameter required and must be non-empty list")

    if not isinstance(requests_data, list):
        raise ValueError("requests must be a list")

    return await tools_execute_batch(requests_data, max_concurrent)


async def tools_execute_batch(
    requests_data: list[dict[str, Any]],
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """
    Execute multiple tools in parallel without dependencies.

    Args:
        requests_data: List of tool execution request dictionaries
        max_concurrent: Maximum concurrent executions

    Returns:
        Dictionary with batch execution results
    """
    import time

    start_time = time.time()

    executor = get_tool_executor()
    parallel_exec = ParallelExecutor(executor)

    # Convert to ToolExecutionRequest objects
    requests = []
    for req_data in requests_data:
        tool_id = req_data.get("tool_id")
        parameters = req_data.get("parameters", {})
        agent_id = req_data.get("agent_id")
        execution_context = req_data.get("execution_context")
        timeout_override = req_data.get("timeout_override")

        if not tool_id:
            raise ValueError("Each request must have tool_id")
        if not agent_id:
            raise ValueError("Each request must have agent_id")

        requests.append(
            ToolExecutionRequest(
                tool_id=tool_id,
                parameters=parameters,
                agent_id=agent_id,
                execution_context=execution_context or {},
                timeout_override=timeout_override,
            )
        )

    # Execute batch
    results = await parallel_exec.execute_batch(requests, max_concurrent=max_concurrent)

    # Convert results to JSON-serializable format
    results_data = [
        {
            "request_id": result.request_id,
            "tool_id": result.tool_id,
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
            "error_type": result.error_type,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
            "retry_count": result.retry_count,
            "metadata": result.metadata,
        }
        for result in results
    ]

    # Calculate statistics
    total_time_ms = (time.time() - start_time) * 1000
    successful_count = sum(1 for r in results if r.status.value == "success")
    failed_count = len(results) - successful_count

    logger.info(
        "tools_execute_batch_called",
        total_requests=len(requests),
        max_concurrent=max_concurrent,
        successful_count=successful_count,
        failed_count=failed_count,
        total_time_ms=total_time_ms,
    )

    return {
        "results": results_data,
        "total_time_ms": total_time_ms,
        "successful_count": successful_count,
        "failed_count": failed_count,
    }


@register_jsonrpc_method("tools.execute_parallel")
async def handle_tools_execute_parallel(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute multiple tools in parallel with dependency management.

    Method: tools.execute_parallel
    Params:
        - tasks: list[dict] - List of parallel tasks
            Each task contains:
            - task_id: string - Unique task identifier
            - tool_id: string - Tool to execute
            - parameters: dict - Tool parameters
            - agent_id: string - Requesting agent ID
            - dependencies: list[string] (optional) - List of task_ids this task depends on
            - execution_context: dict (optional) - Execution context
            - timeout_override: number (optional) - Timeout override
        - max_concurrent: number (optional, default=10) - Maximum concurrent executions

    Returns:
        - results: dict mapping task_id to execution result
        - total_time_ms: total execution time
        - successful_count: number of successful executions
        - failed_count: number of failed executions
        - execution_order: list of task_ids in execution order
    """
    params = request.params or {}
    tasks_data = params.get("tasks", [])
    max_concurrent = params.get("max_concurrent", 10)

    if not tasks_data:
        raise ValueError("tasks parameter required and must be non-empty list")

    if not isinstance(tasks_data, list):
        raise ValueError("tasks must be a list")

    return await tools_execute_parallel(tasks_data, max_concurrent)


async def tools_execute_parallel(
    tasks_data: list[dict[str, Any]],
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """
    Execute multiple tools in parallel with dependency management.

    Args:
        tasks_data: List of task dictionaries with dependencies
        max_concurrent: Maximum concurrent executions

    Returns:
        Dictionary with parallel execution results
    """
    import time

    start_time = time.time()

    executor = get_tool_executor()
    parallel_exec = ParallelExecutor(executor)

    # Convert to ParallelTask objects
    tasks = []
    for task_data in tasks_data:
        task_id = task_data.get("task_id")
        tool_id = task_data.get("tool_id")
        parameters = task_data.get("parameters", {})
        agent_id = task_data.get("agent_id")
        dependencies = task_data.get("dependencies", [])
        execution_context = task_data.get("execution_context")
        timeout_override = task_data.get("timeout_override")

        if not task_id:
            raise ValueError("Each task must have task_id")
        if not tool_id:
            raise ValueError("Each task must have tool_id")
        if not agent_id:
            raise ValueError("Each task must have agent_id")

        tasks.append(
            ParallelTask(
                task_id=task_id,
                request=ToolExecutionRequest(
                    tool_id=tool_id,
                    parameters=parameters,
                    agent_id=agent_id,
                    execution_context=execution_context or {},
                    timeout_override=timeout_override,
                ),
                dependencies=dependencies,
            )
        )

    # Execute parallel
    results = await parallel_exec.execute_parallel(tasks, max_concurrent=max_concurrent)

    # Convert results to JSON-serializable format
    results_data = {
        task_id: {
            "request_id": result.request_id,
            "tool_id": result.tool_id,
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
            "error_type": result.error_type,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.timestamp.isoformat(),
            "retry_count": result.retry_count,
            "metadata": result.metadata,
        }
        for task_id, result in results.items()
    }

    # Track execution order
    execution_order = [task.task_id for task in tasks if task.status == "completed"]

    # Calculate statistics
    total_time_ms = (time.time() - start_time) * 1000
    successful_count = sum(
        1 for r in results.values() if r.status.value == "success"
    )
    failed_count = len(results) - successful_count

    logger.info(
        "tools_execute_parallel_called",
        total_tasks=len(tasks),
        max_concurrent=max_concurrent,
        successful_count=successful_count,
        failed_count=failed_count,
        total_time_ms=total_time_ms,
    )

    return {
        "results": results_data,
        "total_time_ms": total_time_ms,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "execution_order": execution_order,
    }


@register_jsonrpc_method("tools.execute_with_fallback")
async def handle_tools_execute_with_fallback(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Execute a tool with automatic fallback to alternative tool on failure.

    Method: tools.execute_with_fallback
    Params:
        - primary: dict - Primary tool execution request
            - tool_id: string - Primary tool to execute
            - parameters: dict - Tool parameters
            - agent_id: string - Requesting agent ID
            - execution_context: dict (optional) - Execution context
            - timeout_override: number (optional) - Timeout override
        - fallback: dict - Fallback tool execution request (same structure as primary)

    Returns:
        - result: tool execution result (from primary or fallback)
        - used_fallback: boolean - Whether fallback was used
        - primary_error: string (optional) - Error from primary if fallback was used
    """
    params = request.params or {}
    primary_data = params.get("primary")
    fallback_data = params.get("fallback")

    if not primary_data:
        raise ValueError("primary parameter required")
    if not fallback_data:
        raise ValueError("fallback parameter required")

    return await tools_execute_with_fallback(primary_data, fallback_data)


async def tools_execute_with_fallback(
    primary_data: dict[str, Any],
    fallback_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute a tool with automatic fallback.

    Args:
        primary_data: Primary tool execution request
        fallback_data: Fallback tool execution request

    Returns:
        Dictionary with execution result and fallback information
    """
    from agentcore.agent_runtime.services.parallel_executor import execute_with_fallback

    executor = get_tool_executor()

    # Create primary request
    primary_request = ToolExecutionRequest(
        tool_id=primary_data.get("tool_id"),
        parameters=primary_data.get("parameters", {}),
        agent_id=primary_data.get("agent_id"),
        execution_context=primary_data.get("execution_context", {}),
        timeout_override=primary_data.get("timeout_override"),
    )

    # Create fallback request
    fallback_request = ToolExecutionRequest(
        tool_id=fallback_data.get("tool_id"),
        parameters=fallback_data.get("parameters", {}),
        agent_id=fallback_data.get("agent_id"),
        execution_context=fallback_data.get("execution_context", {}),
        timeout_override=fallback_data.get("timeout_override"),
    )

    # Execute with fallback
    result = await execute_with_fallback(executor, primary_request, fallback_request)

    # Check if fallback was used
    used_fallback = bool(result.metadata and result.metadata.get("fallback_used", False))
    primary_error = None
    if used_fallback and result.metadata:
        primary_error = result.metadata.get("primary_error")

    # Convert result to JSON-serializable format
    result_data = {
        "request_id": result.request_id,
        "tool_id": result.tool_id,
        "status": result.status.value,
        "result": result.result,
        "error": result.error,
        "error_type": result.error_type,
        "execution_time_ms": result.execution_time_ms,
        "timestamp": result.timestamp.isoformat(),
        "retry_count": result.retry_count,
        "metadata": result.metadata,
    }

    logger.info(
        "tools_execute_with_fallback_called",
        primary_tool=primary_data.get("tool_id"),
        fallback_tool=fallback_data.get("tool_id"),
        used_fallback=used_fallback,
        status=result.status.value,
    )

    return {
        "result": result_data,
        "used_fallback": used_fallback,
        "primary_error": primary_error,
    }
