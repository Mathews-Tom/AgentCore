"""
Workflow Management JSON-RPC Methods

A2A protocol compliant workflow management methods for running, listing,
retrieving, and stopping workflow executions.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog

from agentcore.a2a_protocol.database import get_session
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.orchestration.state.models import WorkflowStatus
from agentcore.orchestration.state.repository import WorkflowStateRepository

logger = structlog.get_logger()


def _status_to_api(status: WorkflowStatus) -> str:
    """Convert database status enum to API status string."""
    status_map = {
        WorkflowStatus.PENDING: "pending",
        WorkflowStatus.PLANNING: "planning",
        WorkflowStatus.EXECUTING: "running",
        WorkflowStatus.PAUSED: "paused",
        WorkflowStatus.COMPLETED: "completed",
        WorkflowStatus.FAILED: "failed",
        WorkflowStatus.COMPENSATING: "compensating",
        WorkflowStatus.COMPENSATED: "compensated",
        WorkflowStatus.COMPENSATION_FAILED: "compensation_failed",
        WorkflowStatus.CANCELLED: "cancelled",
    }
    return status_map.get(status, str(status.value))


def _api_to_status(api_status: str) -> WorkflowStatus | None:
    """Convert API status string to database status enum."""
    status_map = {
        "pending": WorkflowStatus.PENDING,
        "planning": WorkflowStatus.PLANNING,
        "running": WorkflowStatus.EXECUTING,
        "executing": WorkflowStatus.EXECUTING,
        "paused": WorkflowStatus.PAUSED,
        "completed": WorkflowStatus.COMPLETED,
        "failed": WorkflowStatus.FAILED,
        "cancelled": WorkflowStatus.CANCELLED,
        "canceled": WorkflowStatus.CANCELLED,
    }
    return status_map.get(api_status.lower())


def _execution_to_dict(execution: Any) -> dict[str, Any]:
    """Convert workflow execution DB model to API response dict."""
    return {
        "workflow_id": execution.execution_id,
        "name": execution.workflow_name,
        "status": _status_to_api(execution.status),
        "orchestration_pattern": execution.orchestration_pattern,
        "created_at": execution.created_at.isoformat() if execution.created_at else None,
        "updated_at": execution.updated_at.isoformat() if execution.updated_at else None,
        "started_at": execution.started_at.isoformat() if execution.started_at else None,
        "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        "duration_seconds": execution.duration_seconds,
        "definition": execution.workflow_definition,
        "parameters": execution.input_data,
        "total_tasks": execution.total_tasks,
        "completed_tasks": execution.completed_task_count,
        "failed_tasks": execution.failed_task_count,
        "error_message": execution.error_message,
    }


@register_jsonrpc_method("workflow.run")
async def handle_workflow_run(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Run a new workflow from definition.

    Method: workflow.run
    Params:
        - definition: dict - Workflow definition with 'name' and 'steps'
        - parameters: dict (optional) - Input parameters for the workflow

    Returns:
        workflow_id: string - The ID of the created workflow execution
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: definition")

        definition = request.params.get("definition")
        if not definition:
            raise ValueError("Missing required parameter: definition")

        if not isinstance(definition, dict):
            raise ValueError("Definition must be a dictionary")

        if "name" not in definition:
            raise ValueError("Workflow definition must have a 'name' field")

        if "steps" not in definition:
            raise ValueError("Workflow definition must have a 'steps' field")

        parameters = request.params.get("parameters", {})

        # Generate execution ID
        execution_id = f"wf-{uuid4().hex[:12]}"
        workflow_id = definition.get("id", f"wfdef-{uuid4().hex[:8]}")
        workflow_name = definition["name"]
        orchestration_pattern = definition.get("pattern", "sequential")

        async with get_session() as session:
            execution = await WorkflowStateRepository.create_execution(
                session=session,
                execution_id=execution_id,
                workflow_id=workflow_id,
                workflow_name=workflow_name,
                orchestration_pattern=orchestration_pattern,
                workflow_definition=definition,
                input_data=parameters,
                tags=definition.get("tags", []),
                metadata=definition.get("metadata", {}),
            )

            # Start the workflow (set to executing status)
            await WorkflowStateRepository.update_execution_status(
                session=session,
                execution_id=execution_id,
                status=WorkflowStatus.EXECUTING,
            )

            await session.commit()

            logger.info(
                "Workflow started via JSON-RPC",
                workflow_id=execution_id,
                workflow_name=workflow_name,
                method="workflow.run",
            )

            return {"workflow_id": execution_id}

    except Exception as e:
        logger.error("Workflow run failed", error=str(e))
        raise


@register_jsonrpc_method("workflow.list")
async def handle_workflow_list(request: JsonRpcRequest) -> dict[str, Any]:
    """
    List workflow executions with optional filtering.

    Method: workflow.list
    Params:
        - status: string (optional) - Filter by status (running, completed, failed, cancelled)
        - limit: int (optional, default 100) - Maximum results
        - offset: int (optional, default 0) - Results offset

    Returns:
        workflows: list - List of workflow execution summaries
    """
    try:
        params = request.params or {}

        status_str = params.get("status")
        limit = params.get("limit", 100)
        offset = params.get("offset", 0)

        # Convert API status to DB status
        db_status = None
        if status_str:
            db_status = _api_to_status(status_str)
            if db_status is None:
                raise ValueError(
                    f"Invalid status: {status_str}. "
                    "Must be one of: pending, running, completed, failed, cancelled"
                )

        async with get_session() as session:
            executions = await WorkflowStateRepository.list_executions(
                session=session,
                status=db_status,
                limit=limit,
                offset=offset,
            )

            workflows = [_execution_to_dict(ex) for ex in executions]

            logger.info(
                "Listed workflows via JSON-RPC",
                count=len(workflows),
                status=status_str,
                method="workflow.list",
            )

            return {"workflows": workflows}

    except Exception as e:
        logger.error("Workflow list failed", error=str(e))
        raise


@register_jsonrpc_method("workflow.get")
async def handle_workflow_get(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get workflow execution details.

    Method: workflow.get
    Params:
        - workflow_id: string - The workflow execution ID

    Returns:
        workflow: dict - Workflow execution details
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameter required: workflow_id")

        workflow_id = request.params.get("workflow_id")
        if not workflow_id:
            raise ValueError("Missing required parameter: workflow_id")

        async with get_session() as session:
            execution = await WorkflowStateRepository.get_execution(
                session=session,
                execution_id=workflow_id,
            )

            if not execution:
                raise ValueError(f"Workflow not found: {workflow_id}")

            workflow = _execution_to_dict(execution)

            # Add additional details for single workflow view
            workflow["execution_state"] = execution.execution_state
            workflow["allocated_agents"] = execution.allocated_agents
            workflow["task_states"] = execution.task_states
            workflow["output_data"] = execution.output_data

            logger.info(
                "Retrieved workflow via JSON-RPC",
                workflow_id=workflow_id,
                method="workflow.get",
            )

            return {"workflow": workflow}

    except Exception as e:
        logger.error("Workflow get failed", error=str(e))
        raise


@register_jsonrpc_method("workflow.stop")
async def handle_workflow_stop(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Stop a running workflow.

    Method: workflow.stop
    Params:
        - workflow_id: string - The workflow execution ID
        - force: bool (optional, default False) - Force stop even if in critical state

    Returns:
        success: bool - Whether the workflow was successfully stopped
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameter required: workflow_id")

        workflow_id = request.params.get("workflow_id")
        if not workflow_id:
            raise ValueError("Missing required parameter: workflow_id")

        force = request.params.get("force", False)

        async with get_session() as session:
            execution = await WorkflowStateRepository.get_execution(
                session=session,
                execution_id=workflow_id,
            )

            if not execution:
                raise ValueError(f"Workflow not found: {workflow_id}")

            # Check if workflow can be stopped
            stoppable_statuses = {
                WorkflowStatus.PENDING,
                WorkflowStatus.PLANNING,
                WorkflowStatus.EXECUTING,
                WorkflowStatus.PAUSED,
            }

            if execution.status not in stoppable_statuses:
                if not force:
                    raise ValueError(
                        f"Workflow cannot be stopped in status: {execution.status.value}. "
                        "Use force=true to override."
                    )

            # Update status to cancelled
            await WorkflowStateRepository.update_execution_status(
                session=session,
                execution_id=workflow_id,
                status=WorkflowStatus.CANCELLED,
            )

            await session.commit()

            logger.info(
                "Workflow stopped via JSON-RPC",
                workflow_id=workflow_id,
                force=force,
                method="workflow.stop",
            )

            return {"success": True}

    except Exception as e:
        logger.error("Workflow stop failed", error=str(e))
        raise
