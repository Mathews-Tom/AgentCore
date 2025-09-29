"""
Task Management JSON-RPC Methods

A2A protocol compliant task management methods for creating, managing,
and monitoring task execution lifecycle.
"""

from typing import Any, Dict, List, Optional

import structlog
from datetime import datetime

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.task import (
    TaskDefinition,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskQuery,
    TaskQueryResponse,
    TaskStatus,
    TaskPriority,
    TaskArtifact
)
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.task_manager import task_manager

logger = structlog.get_logger()


@register_jsonrpc_method("task.create")
async def handle_task_create(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Create a new task.

    Method: task.create
    Params:
        - task_definition: TaskDefinition object
        - auto_assign: bool (optional, default true)
        - preferred_agent: string (optional)

    Returns:
        Task creation response with execution_id, task_id, status, assigned_agent
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: task_definition and optional auto_assign, preferred_agent")

        # Extract parameters
        task_definition_data = request.params.get("task_definition")
        if not task_definition_data:
            raise ValueError("Missing required parameter: task_definition")

        auto_assign = request.params.get("auto_assign", True)
        preferred_agent = request.params.get("preferred_agent")

        # Parse task definition
        task_def = TaskDefinition.model_validate(task_definition_data)

        # Create request
        task_request = TaskCreateRequest(
            task_definition=task_def,
            auto_assign=auto_assign,
            preferred_agent=preferred_agent
        )

        # Create task
        response = await task_manager.create_task(task_request)

        logger.info(
            "Task created via JSON-RPC",
            task_id=response.task_id,
            execution_id=response.execution_id,
            auto_assign=auto_assign,
            method="task.create"
        )

        return response.model_dump()

    except Exception as e:
        logger.error("Task creation failed", error=str(e))
        raise


@register_jsonrpc_method("task.get")
async def handle_task_get(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get task execution details.

    Method: task.get
    Params:
        - execution_id: string

    Returns:
        Task execution details
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: execution_id")

    execution_id = request.params.get("execution_id")
    if not execution_id:
        raise ValueError("Missing required parameter: execution_id")

    execution = await task_manager.get_task(execution_id)

    if not execution:
        raise ValueError(f"Task not found: {execution_id}")

    return execution.model_dump()


@register_jsonrpc_method("task.assign")
async def handle_task_assign(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Assign task to a specific agent.

    Method: task.assign
    Params:
        - execution_id: string
        - agent_id: string

    Returns:
        Assignment result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: execution_id, agent_id")

    execution_id = request.params.get("execution_id")
    agent_id = request.params.get("agent_id")

    if not execution_id or not agent_id:
        raise ValueError("Missing required parameters: execution_id and/or agent_id")

    success = await task_manager.assign_task(execution_id, agent_id)

    if not success:
        raise ValueError(f"Task assignment failed: {execution_id} to {agent_id}")

    logger.info("Task assigned via JSON-RPC", execution_id=execution_id, agent_id=agent_id, method="task.assign")

    return {
        "success": True,
        "execution_id": execution_id,
        "agent_id": agent_id,
        "message": "Task assigned successfully"
    }


@register_jsonrpc_method("task.start")
async def handle_task_start(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Start task execution.

    Method: task.start
    Params:
        - execution_id: string

    Returns:
        Start result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: execution_id")

    execution_id = request.params.get("execution_id")
    if not execution_id:
        raise ValueError("Missing required parameter: execution_id")

    success = await task_manager.start_task(execution_id)

    if not success:
        raise ValueError(f"Task start failed: {execution_id}")

    logger.info("Task started via JSON-RPC", execution_id=execution_id, method="task.start")

    return {
        "success": True,
        "execution_id": execution_id,
        "message": "Task started successfully"
    }


@register_jsonrpc_method("task.complete")
async def handle_task_complete(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Complete task execution.

    Method: task.complete
    Params:
        - execution_id: string
        - result_data: object
        - artifacts: array (optional)

    Returns:
        Completion result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: execution_id, result_data")

    execution_id = request.params.get("execution_id")
    result_data = request.params.get("result_data")
    artifacts_data = request.params.get("artifacts")

    if not execution_id or result_data is None:
        raise ValueError("Missing required parameters: execution_id and/or result_data")

    # Parse artifacts if provided
    parsed_artifacts = None
    if artifacts_data:
        parsed_artifacts = [TaskArtifact.model_validate(artifact) for artifact in artifacts_data]

    success = await task_manager.complete_task(execution_id, result_data, parsed_artifacts)

    if not success:
        raise ValueError(f"Task completion failed: {execution_id}")

    logger.info("Task completed via JSON-RPC", execution_id=execution_id, method="task.complete")

    return {
        "success": True,
        "execution_id": execution_id,
        "message": "Task completed successfully"
    }


@register_jsonrpc_method("task.fail")
async def handle_task_fail(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Mark task as failed.

    Method: task.fail
    Params:
        - execution_id: string
        - error_message: string
        - should_retry: bool (optional, default true)

    Returns:
        Failure result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: execution_id, error_message")

    execution_id = request.params.get("execution_id")
    error_message = request.params.get("error_message")
    should_retry = request.params.get("should_retry", True)

    if not execution_id or not error_message:
        raise ValueError("Missing required parameters: execution_id and/or error_message")

    success = await task_manager.fail_task(execution_id, error_message, should_retry)

    if not success:
        raise ValueError(f"Task failure recording failed: {execution_id}")

    logger.info("Task failed via JSON-RPC", execution_id=execution_id, error=error_message, method="task.fail")

    return {
        "success": True,
        "execution_id": execution_id,
        "error_message": error_message,
        "should_retry": should_retry,
        "message": "Task failure recorded successfully"
    }


@register_jsonrpc_method("task.cancel")
async def handle_task_cancel(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Cancel task execution.

    Method: task.cancel
    Params:
        - execution_id: string

    Returns:
        Cancellation result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: execution_id")

    execution_id = request.params.get("execution_id")
    if not execution_id:
        raise ValueError("Missing required parameter: execution_id")

    success = await task_manager.cancel_task(execution_id)

    if not success:
        raise ValueError(f"Task cancellation failed: {execution_id}")

    logger.info("Task cancelled via JSON-RPC", execution_id=execution_id, method="task.cancel")

    return {
        "success": True,
        "execution_id": execution_id,
        "message": "Task cancelled successfully"
    }


@register_jsonrpc_method("task.update_progress")
async def handle_task_update_progress(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Update task execution progress.

    Method: task.update_progress
    Params:
        - execution_id: string
        - percentage: number (0-100)
        - current_step: string (optional)

    Returns:
        Update result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: execution_id, percentage")

    execution_id = request.params.get("execution_id")
    percentage = request.params.get("percentage")
    current_step = request.params.get("current_step")

    if not execution_id or percentage is None:
        raise ValueError("Missing required parameters: execution_id and/or percentage")

    success = await task_manager.update_task_progress(execution_id, percentage, current_step)

    if not success:
        raise ValueError(f"Task progress update failed: {execution_id}")

    return {
        "success": True,
        "execution_id": execution_id,
        "percentage": percentage,
        "current_step": current_step,
        "message": "Task progress updated successfully"
    }


@register_jsonrpc_method("task.query")
async def handle_task_query(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Query tasks with filtering and pagination.

    Method: task.query
    Params:
        - status: string (optional)
        - task_type: string (optional)
        - assigned_agent: string (optional)
        - created_by: string (optional)
        - name_pattern: string (optional)
        - tags: array (optional)
        - priority: string (optional)
        - created_after: string ISO8601 (optional)
        - created_before: string ISO8601 (optional)
        - limit: number (optional, default 50, max 1000)
        - offset: number (optional, default 0)

    Returns:
        Query response with matching tasks
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    try:
        # Parse query parameters
        query_params = {}

        # Status filter
        if "status" in params:
            query_params["status"] = TaskStatus(params["status"])

        # String filters
        for field in ["task_type", "assigned_agent", "created_by", "name_pattern"]:
            if field in params:
                query_params[field] = params[field]

        # List filters
        if "tags" in params:
            query_params["tags"] = params["tags"] if isinstance(params["tags"], list) else [params["tags"]]

        # Priority filter
        if "priority" in params:
            query_params["priority"] = TaskPriority(params["priority"])

        # Time filters
        for field in ["created_after", "created_before"]:
            if field in params:
                query_params[field] = datetime.fromisoformat(params[field])

        # Pagination
        query_params["limit"] = min(params.get("limit", 50), 1000)
        query_params["offset"] = max(params.get("offset", 0), 0)

        # Create query
        query = TaskQuery.model_validate(query_params)

        # Execute query
        response = await task_manager.query_tasks(query)

        logger.debug("Tasks queried via JSON-RPC", filters=query_params, count=len(response.tasks), method="task.query")

        return response.model_dump()

    except Exception as e:
        logger.error("Task query failed", error=str(e))
        raise


@register_jsonrpc_method("task.dependencies")
async def handle_task_dependencies(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get task dependency information.

    Method: task.dependencies
    Params:
        - task_id: string

    Returns:
        Task dependencies (prerequisites and dependents)
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: task_id")

    task_id = request.params.get("task_id")
    if not task_id:
        raise ValueError("Missing required parameter: task_id")

    dependencies = await task_manager.get_task_dependencies(task_id)

    logger.debug("Task dependencies queried via JSON-RPC", task_id=task_id, method="task.dependencies")

    return {
        "task_id": task_id,
        "dependencies": dependencies
    }


@register_jsonrpc_method("task.ready")
async def handle_task_ready(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get tasks that are ready to be assigned.

    Method: task.ready
    Params: none

    Returns:
        List of ready task execution IDs
    """
    ready_tasks = await task_manager.get_ready_tasks()

    logger.debug("Ready tasks queried via JSON-RPC", count=len(ready_tasks), method="task.ready")

    return {
        "ready_tasks": ready_tasks,
        "count": len(ready_tasks)
    }


@register_jsonrpc_method("task.cleanup")
async def handle_task_cleanup(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Cleanup old completed/failed tasks.

    Method: task.cleanup
    Params:
        - max_age_days: number (optional, default 30)

    Returns:
        Cleanup result
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    max_age_days = params.get("max_age_days", 30)

    try:
        cleanup_count = await task_manager.cleanup_old_tasks(max_age_days)

        logger.info("Tasks cleaned up via JSON-RPC", count=cleanup_count, max_age_days=max_age_days, method="task.cleanup")

        return {
            "success": True,
            "cleanup_count": cleanup_count,
            "max_age_days": max_age_days,
            "message": f"Cleaned up {cleanup_count} old tasks"
        }

    except Exception as e:
        logger.error("Task cleanup failed", error=str(e))
        raise


@register_jsonrpc_method("task.summary")
async def handle_task_summary(request: JsonRpcRequest) -> Dict[str, Any]:
    """
    Get task execution summary.

    Method: task.summary
    Params:
        - execution_id: string

    Returns:
        Task execution summary
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: execution_id")

    execution_id = request.params.get("execution_id")
    if not execution_id:
        raise ValueError("Missing required parameter: execution_id")

    execution = await task_manager.get_task(execution_id)

    if not execution:
        raise ValueError(f"Task not found: {execution_id}")

    summary = execution.to_summary()

    logger.debug("Task summary requested via JSON-RPC", execution_id=execution_id, method="task.summary")

    return summary


# Log registration on import
logger.info("Task JSON-RPC methods registered",
           methods=[
               "task.create", "task.get", "task.assign", "task.start",
               "task.complete", "task.fail", "task.cancel", "task.update_progress",
               "task.query", "task.dependencies", "task.ready", "task.cleanup", "task.summary"
           ])