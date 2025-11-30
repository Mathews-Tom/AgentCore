"""
Session Management JSON-RPC Methods

A2A protocol compliant session management methods for creating, managing,
and monitoring long-running workflow sessions.
"""

from datetime import datetime
from typing import Any

import structlog

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.session import (
    SessionCreateRequest,
    SessionCreateResponse,
    SessionPriority,
    SessionQuery,
    SessionQueryResponse,
    SessionState,
)
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.session_manager import session_manager

logger = structlog.get_logger()


@register_jsonrpc_method("session.create")
async def handle_session_create(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Create a new session.

    Method: session.create
    Params:
        - name: string (required)
        - description: string (optional)
        - owner_agent: string (required)
        - priority: string (optional, default "normal")
        - timeout_seconds: number (optional, default 3600)
        - max_idle_seconds: number (optional, default 300)
        - tags: array (optional)
        - initial_context: object (optional)

    Returns:
        Session creation response
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: name, owner_agent")

    name = request.params.get("name")
    owner_agent = request.params.get("owner_agent")

    if not name or not owner_agent:
        raise ValueError("Missing required parameters: name and/or owner_agent")

    # Build create request
    create_params = {
        "name": name,
        "owner_agent": owner_agent,
    }

    # Optional parameters
    if "description" in request.params:
        create_params["description"] = request.params["description"]

    if "priority" in request.params:
        create_params["priority"] = SessionPriority(request.params["priority"])

    if "timeout_seconds" in request.params:
        create_params["timeout_seconds"] = request.params["timeout_seconds"]

    if "max_idle_seconds" in request.params:
        create_params["max_idle_seconds"] = request.params["max_idle_seconds"]

    if "tags" in request.params:
        create_params["tags"] = request.params["tags"]

    if "initial_context" in request.params:
        create_params["initial_context"] = request.params["initial_context"]

    # Create session
    session_request = SessionCreateRequest.model_validate(create_params)
    response = await session_manager.create_session(session_request)

    logger.info(
        "Session created via JSON-RPC",
        session_id=response.session_id,
        name=name,
        owner=owner_agent,
        method="session.create",
    )

    return response.model_dump()


@register_jsonrpc_method("session.get")
async def handle_session_get(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get session details.

    Method: session.get
    Params:
        - session_id: string (required)

    Returns:
        Session snapshot
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    session = await session_manager.get_session(session_id)

    if not session:
        raise ValueError(f"Session not found: {session_id}")

    return session.model_dump(mode="json")


@register_jsonrpc_method("session.pause")
async def handle_session_pause(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Pause active session.

    Method: session.pause
    Params:
        - session_id: string (required)

    Returns:
        Pause result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    success = await session_manager.pause_session(session_id)

    if not success:
        raise ValueError(f"Session pause failed: {session_id}")

    logger.info(
        "Session paused via JSON-RPC", session_id=session_id, method="session.pause"
    )

    return {
        "success": True,
        "session_id": session_id,
        "message": "Session paused successfully",
    }


@register_jsonrpc_method("session.resume")
async def handle_session_resume(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Resume paused or suspended session.

    Method: session.resume
    Params:
        - session_id: string (required)

    Returns:
        Resume result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    success = await session_manager.resume_session(session_id)

    if not success:
        raise ValueError(f"Session resume failed: {session_id}")

    logger.info(
        "Session resumed via JSON-RPC", session_id=session_id, method="session.resume"
    )

    return {
        "success": True,
        "session_id": session_id,
        "message": "Session resumed successfully",
    }


@register_jsonrpc_method("session.suspend")
async def handle_session_suspend(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Suspend session for later resumption.

    Method: session.suspend
    Params:
        - session_id: string (required)

    Returns:
        Suspend result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    success = await session_manager.suspend_session(session_id)

    if not success:
        raise ValueError(f"Session suspend failed: {session_id}")

    logger.info(
        "Session suspended via JSON-RPC",
        session_id=session_id,
        method="session.suspend",
    )

    return {
        "success": True,
        "session_id": session_id,
        "message": "Session suspended successfully",
    }


@register_jsonrpc_method("session.complete")
async def handle_session_complete(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Mark session as completed.

    Method: session.complete
    Params:
        - session_id: string (required)

    Returns:
        Completion result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    success = await session_manager.complete_session(session_id)

    if not success:
        raise ValueError(f"Session completion failed: {session_id}")

    logger.info(
        "Session completed via JSON-RPC",
        session_id=session_id,
        method="session.complete",
    )

    return {
        "success": True,
        "session_id": session_id,
        "message": "Session completed successfully",
    }


@register_jsonrpc_method("session.fail")
async def handle_session_fail(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Mark session as failed.

    Method: session.fail
    Params:
        - session_id: string (required)
        - reason: string (optional)

    Returns:
        Failure result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    reason = request.params.get("reason")

    success = await session_manager.fail_session(session_id, reason)

    if not success:
        raise ValueError(f"Session failure recording failed: {session_id}")

    logger.info(
        "Session failed via JSON-RPC",
        session_id=session_id,
        reason=reason,
        method="session.fail",
    )

    return {
        "success": True,
        "session_id": session_id,
        "reason": reason,
        "message": "Session failure recorded successfully",
    }


@register_jsonrpc_method("session.update_context")
async def handle_session_update_context(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Update session context variables.

    Method: session.update_context
    Params:
        - session_id: string (required)
        - updates: object (required)

    Returns:
        Update result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: session_id, updates")

    session_id = request.params.get("session_id")
    updates = request.params.get("updates")

    if not session_id or updates is None:
        raise ValueError("Missing required parameters: session_id and/or updates")

    success = await session_manager.update_context(session_id, updates)

    if not success:
        raise ValueError(f"Context update failed: {session_id}")

    return {
        "success": True,
        "session_id": session_id,
        "message": "Context updated successfully",
    }


@register_jsonrpc_method("session.set_agent_state")
async def handle_session_set_agent_state(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Set state for a specific agent in session.

    Method: session.set_agent_state
    Params:
        - session_id: string (required)
        - agent_id: string (required)
        - state: object (required)

    Returns:
        Update result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: session_id, agent_id, state")

    session_id = request.params.get("session_id")
    agent_id = request.params.get("agent_id")
    state = request.params.get("state")

    if not session_id or not agent_id or state is None:
        raise ValueError(
            "Missing required parameters: session_id, agent_id, and/or state"
        )

    success = await session_manager.set_agent_state(session_id, agent_id, state)

    if not success:
        raise ValueError(f"Agent state update failed: {session_id}")

    return {
        "success": True,
        "session_id": session_id,
        "agent_id": agent_id,
        "message": "Agent state updated successfully",
    }


@register_jsonrpc_method("session.get_agent_state")
async def handle_session_get_agent_state(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get state for a specific agent in session.

    Method: session.get_agent_state
    Params:
        - session_id: string (required)
        - agent_id: string (required)

    Returns:
        Agent state
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: session_id, agent_id")

    session_id = request.params.get("session_id")
    agent_id = request.params.get("agent_id")

    if not session_id or not agent_id:
        raise ValueError("Missing required parameters: session_id and/or agent_id")

    state = await session_manager.get_agent_state(session_id, agent_id)

    if state is None:
        raise ValueError(f"Agent state not found: {agent_id} in session {session_id}")

    return {"session_id": session_id, "agent_id": agent_id, "state": state}


@register_jsonrpc_method("session.add_task")
async def handle_session_add_task(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Add task to session.

    Method: session.add_task
    Params:
        - session_id: string (required)
        - task_id: string (required)

    Returns:
        Add result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: session_id, task_id")

    session_id = request.params.get("session_id")
    task_id = request.params.get("task_id")

    if not session_id or not task_id:
        raise ValueError("Missing required parameters: session_id and/or task_id")

    success = await session_manager.add_task(session_id, task_id)

    if not success:
        raise ValueError(f"Task addition failed: {session_id}")

    return {
        "success": True,
        "session_id": session_id,
        "task_id": task_id,
        "message": "Task added to session successfully",
    }


@register_jsonrpc_method("session.record_event")
async def handle_session_record_event(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Record execution event in session history.

    Method: session.record_event
    Params:
        - session_id: string (required)
        - event_type: string (required)
        - event_data: object (required)

    Returns:
        Record result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: session_id, event_type, event_data")

    session_id = request.params.get("session_id")
    event_type = request.params.get("event_type")
    event_data = request.params.get("event_data")

    if not session_id or not event_type or event_data is None:
        raise ValueError(
            "Missing required parameters: session_id, event_type, and/or event_data"
        )

    success = await session_manager.record_event(session_id, event_type, event_data)

    if not success:
        raise ValueError(f"Event recording failed: {session_id}")

    return {
        "success": True,
        "session_id": session_id,
        "event_type": event_type,
        "message": "Event recorded successfully",
    }


@register_jsonrpc_method("session.checkpoint")
async def handle_session_checkpoint(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Create checkpoint for session.

    Method: session.checkpoint
    Params:
        - session_id: string (required)

    Returns:
        Checkpoint result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    success = await session_manager.create_checkpoint(session_id)

    if not success:
        raise ValueError(f"Checkpoint creation failed: {session_id}")

    logger.info(
        "Session checkpoint created via JSON-RPC",
        session_id=session_id,
        method="session.checkpoint",
    )

    return {
        "success": True,
        "session_id": session_id,
        "message": "Checkpoint created successfully",
    }


@register_jsonrpc_method("session.query")
async def handle_session_query(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Query sessions with filtering and pagination.

    Method: session.query
    Params:
        - state: string (optional)
        - owner_agent: string (optional)
        - participant_agent: string (optional)
        - priority: string (optional)
        - tags: array (optional)
        - created_after: string ISO8601 (optional)
        - created_before: string ISO8601 (optional)
        - include_expired: bool (optional, default false)
        - limit: number (optional, default 50, max 1000)
        - offset: number (optional, default 0)

    Returns:
        Query response with matching sessions
    """
    params = request.params or {}
    if not isinstance(params, dict):
        params = {}

    try:
        # Parse query parameters
        query_params = {}

        # State filter
        if "state" in params:
            query_params["state"] = SessionState(params["state"])

        # String filters
        for field in ["owner_agent", "participant_agent"]:
            if field in params:
                query_params[field] = params[field]

        # Priority filter
        if "priority" in params:
            query_params["priority"] = SessionPriority(params["priority"])

        # List filters
        if "tags" in params:
            query_params["tags"] = (
                params["tags"] if isinstance(params["tags"], list) else [params["tags"]]
            )

        # Time filters
        for field in ["created_after", "created_before"]:
            if field in params:
                query_params[field] = datetime.fromisoformat(params[field])

        # Boolean filters
        if "include_expired" in params:
            query_params["include_expired"] = params["include_expired"]

        # Pagination
        query_params["limit"] = min(params.get("limit", 50), 1000)
        query_params["offset"] = max(params.get("offset", 0), 0)

        # Create query
        query = SessionQuery.model_validate(query_params)

        # Execute query
        response = await session_manager.query_sessions(query)

        logger.debug(
            "Sessions queried via JSON-RPC",
            filters=query_params,
            count=len(response.sessions),
            method="session.query",
        )

        return response.model_dump()

    except Exception as e:
        logger.error("Session query failed", error=str(e))
        raise


@register_jsonrpc_method("session.cleanup_expired")
async def handle_session_cleanup_expired(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Cleanup expired sessions.

    Method: session.cleanup_expired
    Params: none

    Returns:
        Cleanup result
    """
    try:
        cleanup_count = await session_manager.cleanup_expired_sessions()

        logger.info(
            "Expired sessions cleaned up via JSON-RPC",
            count=cleanup_count,
            method="session.cleanup_expired",
        )

        return {
            "success": True,
            "cleanup_count": cleanup_count,
            "message": f"Cleaned up {cleanup_count} expired sessions",
        }

    except Exception as e:
        logger.error("Session cleanup failed", error=str(e))
        raise


@register_jsonrpc_method("session.cleanup_idle")
async def handle_session_cleanup_idle(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Cleanup idle sessions.

    Method: session.cleanup_idle
    Params: none

    Returns:
        Cleanup result
    """
    try:
        cleanup_count = await session_manager.cleanup_idle_sessions()

        logger.info(
            "Idle sessions suspended via JSON-RPC",
            count=cleanup_count,
            method="session.cleanup_idle",
        )

        return {
            "success": True,
            "suspended_count": cleanup_count,
            "message": f"Suspended {cleanup_count} idle sessions",
        }

    except Exception as e:
        logger.error("Idle session cleanup failed", error=str(e))
        raise


@register_jsonrpc_method("session.delete")
async def handle_session_delete(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Delete session (soft or hard delete).

    Method: session.delete
    Params:
        - session_id: string (required)
        - hard_delete: bool (optional, default false) - Permanent deletion vs soft delete

    Returns:
        Deletion result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    hard_delete = request.params.get("hard_delete", False)

    success = await session_manager.delete_session(session_id, hard_delete)

    if not success:
        raise ValueError(f"Session deletion failed: {session_id}")

    delete_type = "hard" if hard_delete else "soft"
    logger.info(
        f"Session {delete_type} deleted via JSON-RPC",
        session_id=session_id,
        method="session.delete",
    )

    return {
        "success": True,
        "session_id": session_id,
        "delete_type": delete_type,
        "message": f"Session {delete_type} deleted successfully",
    }


@register_jsonrpc_method("session.export")
async def handle_session_export(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Export session to JSON format.

    Method: session.export
    Params:
        - session_id: string (required)
        - include_history: bool (optional, default true)

    Returns:
        JSON export data
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_id")

    session_id = request.params.get("session_id")
    if not session_id:
        raise ValueError("Missing required parameter: session_id")

    include_history = request.params.get("include_history", True)

    json_data = await session_manager.export_session(session_id, include_history)

    if not json_data:
        raise ValueError(f"Session not found: {session_id}")

    logger.info(
        "Session exported via JSON-RPC", session_id=session_id, method="session.export"
    )

    return {
        "session_id": session_id,
        "json_data": json_data,
        "size_bytes": len(json_data),
    }


@register_jsonrpc_method("session.import")
async def handle_session_import(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Import session from JSON format.

    Method: session.import
    Params:
        - json_data: string (required) - JSON export data
        - overwrite: bool (optional, default false)

    Returns:
        Import result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: json_data")

    json_data = request.params.get("json_data")
    if not json_data:
        raise ValueError("Missing required parameter: json_data")

    overwrite = request.params.get("overwrite", False)

    try:
        session_id = await session_manager.import_session(json_data, overwrite)

        logger.info(
            "Session imported via JSON-RPC",
            session_id=session_id,
            overwrite=overwrite,
            method="session.import",
        )

        return {
            "success": True,
            "session_id": session_id,
            "message": "Session imported successfully",
        }

    except Exception as e:
        logger.error("Session import failed", error=str(e))
        raise


@register_jsonrpc_method("session.export_batch")
async def handle_session_export_batch(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Export multiple sessions to JSON format.

    Method: session.export_batch
    Params:
        - session_ids: array (required) - List of session IDs
        - include_history: bool (optional, default true)

    Returns:
        Batch export data
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: session_ids")

    session_ids = request.params.get("session_ids")
    if not session_ids or not isinstance(session_ids, list):
        raise ValueError("Missing or invalid parameter: session_ids (must be array)")

    include_history = request.params.get("include_history", True)

    json_data = await session_manager.export_sessions_batch(
        session_ids, include_history
    )

    logger.info(
        "Sessions batch exported via JSON-RPC",
        count=len(session_ids),
        method="session.export_batch",
    )

    return {
        "count": len(session_ids),
        "json_data": json_data,
        "size_bytes": len(json_data),
    }


@register_jsonrpc_method("session.import_batch")
async def handle_session_import_batch(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Import multiple sessions from JSON format.

    Method: session.import_batch
    Params:
        - json_data: string (required) - JSON batch export data
        - overwrite: bool (optional, default false)

    Returns:
        Batch import results
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: json_data")

    json_data = request.params.get("json_data")
    if not json_data:
        raise ValueError("Missing required parameter: json_data")

    overwrite = request.params.get("overwrite", False)

    try:
        results = await session_manager.import_sessions_batch(json_data, overwrite)

        logger.info(
            "Sessions batch imported via JSON-RPC",
            total=results["total"],
            imported=results["imported"],
            skipped=results["skipped"],
            failed=results["failed"],
            method="session.import_batch",
        )

        return {
            "success": True,
            "results": results,
            "message": f"Imported {results['imported']} of {results['total']} sessions",
        }

    except Exception as e:
        logger.error("Batch import failed", error=str(e))
        raise


# Log registration on import
logger.info(
    "Session JSON-RPC methods registered",
    methods=[
        "session.create",
        "session.get",
        "session.pause",
        "session.resume",
        "session.suspend",
        "session.complete",
        "session.fail",
        "session.delete",
        "session.update_context",
        "session.set_agent_state",
        "session.get_agent_state",
        "session.add_task",
        "session.record_event",
        "session.checkpoint",
        "session.query",
        "session.cleanup_expired",
        "session.cleanup_idle",
        "session.export",
        "session.import",
        "session.export_batch",
        "session.import_batch",
    ],
)
