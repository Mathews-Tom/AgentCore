"""
WebSocket Router

WebSocket endpoint for real-time event notifications.
"""

import asyncio
from typing import Optional

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from agentcore.a2a_protocol.services.event_manager import event_manager

logger = structlog.get_logger()
router = APIRouter()


@router.websocket("/ws/events")
async def websocket_events_endpoint(
    websocket: WebSocket,
    subscriber_id: str = Query(..., description="Subscriber identifier"),
):
    """
    WebSocket endpoint for real-time event notifications.

    Clients connect with a subscriber_id and receive events based on their subscriptions.

    Args:
        websocket: WebSocket connection
        subscriber_id: Subscriber identifier (typically agent_id)
    """
    connection_id: Optional[str] = None

    try:
        # Accept connection
        await websocket.accept()

        # Register connection
        connection_id = await event_manager.register_websocket(websocket, subscriber_id)

        logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            subscriber_id=subscriber_id,
        )

        # Send connection confirmation
        await websocket.send_json(
            {
                "message_type": "connected",
                "payload": {
                    "connection_id": connection_id,
                    "subscriber_id": subscriber_id,
                    "message": "WebSocket connection established",
                },
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()

                # Handle different message types
                message_type = data.get("message_type")

                if message_type == "ping":
                    # Respond to ping
                    await websocket.send_json(
                        {
                            "message_type": "pong",
                            "payload": {
                                "timestamp": data.get("payload", {}).get("timestamp")
                            },
                        }
                    )

                elif message_type == "subscribe":
                    # Handle subscription request
                    logger.info(
                        "Subscription request via WebSocket",
                        connection_id=connection_id,
                        subscriber_id=subscriber_id,
                    )
                    # Subscriptions are handled via JSON-RPC methods

                else:
                    logger.warning(
                        "Unknown WebSocket message type",
                        message_type=message_type,
                        connection_id=connection_id,
                    )

            except asyncio.TimeoutError:
                # Timeout waiting for message - send ping
                try:
                    await websocket.send_json({"message_type": "ping", "payload": {}})
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected",
            connection_id=connection_id,
            subscriber_id=subscriber_id,
        )

    except Exception as e:
        logger.error(
            "WebSocket error",
            connection_id=connection_id,
            subscriber_id=subscriber_id,
            error=str(e),
        )

    finally:
        # Clean up connection
        if connection_id:
            await event_manager.close_connection(connection_id)

        logger.info(
            "WebSocket connection closed",
            connection_id=connection_id,
            subscriber_id=subscriber_id,
        )
