"""
Real-time Communication Routes

WebSocket and Server-Sent Events (SSE) endpoints for real-time agent communication.
"""

from __future__ import annotations

from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from sse_starlette.sse import EventSourceResponse

from gateway.auth.dependencies import get_current_user
from gateway.auth.models import User
from gateway.realtime.connection_pool import connection_pool
from gateway.realtime.event_bus import event_bus
from gateway.realtime.sse import SSEManager
from gateway.realtime.subscriptions import subscription_manager
from gateway.realtime.websocket import WebSocketConnectionManager

logger = structlog.get_logger()

router = APIRouter(prefix="/realtime")

# Initialize managers
websocket_manager = WebSocketConnectionManager(
    connection_pool=connection_pool,
    event_bus=event_bus,
    subscription_manager=subscription_manager,
)

sse_manager = SSEManager(
    connection_pool=connection_pool,
    event_bus=event_bus,
    subscription_manager=subscription_manager,
)


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str | None = Query(None, description="JWT access token for authentication"),
) -> None:
    """
    WebSocket endpoint for real-time bidirectional communication.

    Supports:
    - Real-time event streaming
    - Topic-based subscriptions
    - Event filtering
    - Connection health monitoring

    Message Format:
    ```json
    // Subscribe to topics and event types
    {
        "type": "subscribe",
        "topics": ["agent.123", "workflow.456"],
        "event_types": ["task.created", "task.completed"],
        "filters": {
            "agent_ids": ["agent-123"],
            "task_ids": ["task-456"],
            "user_id": "user-789"
        }
    }

    // Unsubscribe
    {
        "type": "unsubscribe",
        "subscription_id": "sub-123"
    }

    // Ping/Pong for heartbeat
    {
        "type": "ping"
    }
    ```

    Events are sent in this format:
    ```json
    {
        "type": "event",
        "event": {
            "event_id": "evt-123",
            "event_type": "task.created",
            "topic": "agent.123",
            "payload": {...},
            "timestamp": "2025-01-01T00:00:00Z",
            "source": "agent-service",
            "metadata": {...}
        }
    }
    ```
    """
    connection_id = None

    try:
        # Connect and authenticate
        connection_id = await websocket_manager.connect(websocket, token=token)

        if not connection_id:
            logger.warning("WebSocket connection rejected")
            return

        # Handle messages
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()

                # Handle message
                await websocket_manager.handle_message(connection_id, message)

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected", connection_id=connection_id)
                break

    except Exception as e:
        logger.error(
            "WebSocket error",
            connection_id=connection_id,
            error=str(e),
            exc_info=True,
        )
    finally:
        # Cleanup connection
        if connection_id:
            await websocket_manager.disconnect(connection_id)


@router.get("/events")
async def sse_endpoint(
    request: Request,
    token: Annotated[str | None, Query(description="JWT access token for authentication")] = None,
    topics: Annotated[list[str] | None, Query(description="Topics to subscribe to")] = None,
    event_types: Annotated[list[str] | None, Query(description="Event types to subscribe to")] = None,
    agent_ids: Annotated[list[str] | None, Query(description="Filter by agent IDs")] = None,
    task_ids: Annotated[list[str] | None, Query(description="Filter by task IDs")] = None,
    workflow_ids: Annotated[list[str] | None, Query(description="Filter by workflow IDs")] = None,
    user_id: Annotated[str | None, Query(description="Filter by user ID")] = None,
) -> EventSourceResponse:
    """
    Server-Sent Events (SSE) endpoint for real-time event streaming.

    Supports:
    - One-way server-to-client event streaming
    - Topic-based subscriptions
    - Event type filtering
    - Automatic keepalive
    - Reconnection support with Last-Event-ID

    Query Parameters:
    - token: JWT access token for authentication (optional)
    - topics: List of topics to subscribe to (optional)
    - event_types: List of event types to subscribe to (optional)
    - agent_ids: Filter events by agent IDs (optional)
    - task_ids: Filter events by task IDs (optional)
    - workflow_ids: Filter events by workflow IDs (optional)
    - user_id: Filter events by user ID (optional)

    Event Format (Server-Sent Events standard):
    ```
    event: task.created
    id: evt-123
    data: {"event_id": "evt-123", "event_type": "task.created", ...}

    event: keepalive
    data: {"timestamp": "2025-01-01T00:00:00Z"}
    ```

    Example:
    ```bash
    curl -N -H "Accept: text/event-stream" \
      "http://localhost:8080/realtime/events?topics=agent.123&event_types=task.created&token=..."
    ```
    """
    # Build filters
    filters = {}
    if agent_ids:
        filters["agent_ids"] = agent_ids
    if task_ids:
        filters["task_ids"] = task_ids
    if workflow_ids:
        filters["workflow_ids"] = workflow_ids
    if user_id:
        filters["user_id"] = user_id

    # Create event stream
    return await sse_manager.create_event_stream(
        request=request,
        token=token,
        topics=topics,
        event_types=event_types,
        filters=filters if filters else None,
    )


@router.get("/stats")
async def get_realtime_stats() -> dict[str, Any]:
    """
    Get real-time communication statistics.

    Returns connection pool, event bus, and subscription statistics.
    """
    return {
        "connection_pool": connection_pool.get_stats(),
        "event_bus": event_bus.get_stats(),
        "subscriptions": subscription_manager.get_stats(),
    }


@router.get("/health")
async def realtime_health() -> dict[str, Any]:
    """
    Real-time communication health check.

    Returns health status of connection pool, event bus, and subscriptions.
    """
    stats = connection_pool.get_stats()

    # Check health status
    is_healthy = True
    issues = []

    # Check connection pool capacity
    if stats["utilization"] > 0.9:
        is_healthy = False
        issues.append("Connection pool near capacity")

    # Check event bus queue
    event_bus_stats = event_bus.get_stats()
    if event_bus_stats["queue_size"] > 8000:
        is_healthy = False
        issues.append("Event queue high")

    return {
        "status": "healthy" if is_healthy else "degraded",
        "issues": issues,
        "connection_pool": stats,
        "event_bus": event_bus_stats,
        "subscriptions": subscription_manager.get_stats(),
    }
