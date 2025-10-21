"""
WebSocket Connection Manager

High-performance WebSocket connection management supporting 10,000+ concurrent connections.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from jose import JWTError

from gateway.auth.jwt import jwt_manager
from gateway.realtime.connection_pool import ConnectionPool, ConnectionType
from gateway.realtime.event_bus import EventBus, EventMessage, EventType
from gateway.realtime.subscriptions import (
    Subscription,
    SubscriptionFilter,
    SubscriptionManager,
)

logger = structlog.get_logger()


class WebSocketConnectionManager:
    """
    WebSocket connection manager.

    Manages WebSocket connections with authentication, subscription management,
    and event broadcasting. Optimized for 10,000+ concurrent connections.
    """

    def __init__(
        self,
        connection_pool: ConnectionPool,
        event_bus: EventBus,
        subscription_manager: SubscriptionManager,
        heartbeat_interval: int = 30,
    ) -> None:
        """
        Initialize WebSocket connection manager.

        Args:
            connection_pool: Connection pool instance
            event_bus: Event bus instance
            subscription_manager: Subscription manager instance
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.connection_pool = connection_pool
        self.event_bus = event_bus
        self.subscription_manager = subscription_manager
        self.heartbeat_interval = heartbeat_interval

        # Active WebSocket connections: connection_id -> WebSocket
        self._connections: dict[str, WebSocket] = {}

        # Heartbeat tasks: connection_id -> Task
        self._heartbeat_tasks: dict[str, asyncio.Task[None]] = {}

        logger.info(
            "WebSocket connection manager initialized",
            heartbeat_interval=heartbeat_interval,
        )

    async def connect(
        self,
        websocket: WebSocket,
        token: str | None = None,
        client_id: str | None = None,
    ) -> str | None:
        """
        Accept and authenticate WebSocket connection.

        Args:
            websocket: WebSocket connection
            token: JWT authentication token (optional)
            client_id: Client identifier (optional, will be generated if not provided)

        Returns:
            Connection ID if successful, None if rejected
        """
        # Generate connection ID
        connection_id = str(uuid4())

        # Authenticate if token provided
        user_id = None
        if token:
            try:
                payload = jwt_manager.validate_access_token(token)
                user_id = payload.sub
                logger.debug(
                    "WebSocket authenticated",
                    connection_id=connection_id,
                    user_id=user_id,
                )
            except JWTError as e:
                logger.warning(
                    "WebSocket authentication failed",
                    connection_id=connection_id,
                    error=str(e),
                )
                await websocket.close(code=1008, reason="Authentication failed")
                return None

        # Generate client ID if not provided
        if not client_id:
            client_id = str(uuid4())

        # Accept connection
        await websocket.accept()

        # Get client info
        remote_addr = None
        user_agent = None
        if websocket.client:
            remote_addr = f"{websocket.client.host}:{websocket.client.port}"
        if "user-agent" in websocket.headers:
            user_agent = websocket.headers["user-agent"]

        # Add to connection pool
        conn_info = self.connection_pool.add_connection(
            connection_id=connection_id,
            connection_type=ConnectionType.WEBSOCKET,
            client_id=client_id,
            user_id=user_id,
            remote_addr=remote_addr,
            user_agent=user_agent,
        )

        if not conn_info:
            logger.warning(
                "Connection pool full",
                connection_id=connection_id,
            )
            await websocket.close(code=1008, reason="Server capacity reached")
            return None

        # Store WebSocket connection
        self._connections[connection_id] = websocket

        # Start heartbeat monitoring
        heartbeat_task = asyncio.create_task(
            self._heartbeat_monitor(connection_id, websocket)
        )
        self._heartbeat_tasks[connection_id] = heartbeat_task

        # Send connection success message
        await self._send_message(
            connection_id,
            {
                "type": "connection",
                "status": "connected",
                "connection_id": connection_id,
                "client_id": client_id,
            },
        )

        logger.info(
            "WebSocket connected",
            connection_id=connection_id,
            client_id=client_id,
            user_id=user_id,
        )

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect WebSocket connection.

        Args:
            connection_id: Connection identifier
        """
        # Cancel heartbeat task
        if connection_id in self._heartbeat_tasks:
            self._heartbeat_tasks[connection_id].cancel()
            try:
                await self._heartbeat_tasks[connection_id]
            except asyncio.CancelledError:
                pass
            del self._heartbeat_tasks[connection_id]

        # Remove all subscriptions
        conn_info = self.connection_pool.get_connection(connection_id)
        if conn_info:
            self.subscription_manager.remove_client_subscriptions(conn_info.client_id)

        # Remove from connection pool
        self.connection_pool.remove_connection(connection_id)

        # Remove WebSocket connection
        if connection_id in self._connections:
            del self._connections[connection_id]

        logger.info("WebSocket disconnected", connection_id=connection_id)

    async def handle_message(
        self,
        connection_id: str,
        message: str,
    ) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            connection_id: Connection identifier
            message: Message string
        """
        try:
            # Parse message
            data = json.loads(message)
            message_type = data.get("type")

            # Record message received
            self.connection_pool.record_message_received(connection_id, len(message))

            # Handle different message types
            if message_type == "subscribe":
                await self._handle_subscribe(connection_id, data)
            elif message_type == "unsubscribe":
                await self._handle_unsubscribe(connection_id, data)
            elif message_type == "ping":
                await self._handle_ping(connection_id)
            elif message_type == "pong":
                await self._handle_pong(connection_id)
            else:
                logger.warning(
                    "Unknown message type",
                    connection_id=connection_id,
                    message_type=message_type,
                )
                await self._send_error(
                    connection_id, f"Unknown message type: {message_type}"
                )

        except json.JSONDecodeError as e:
            logger.warning(
                "Invalid JSON message",
                connection_id=connection_id,
                error=str(e),
            )
            await self._send_error(connection_id, "Invalid JSON message")
        except Exception as e:
            logger.error(
                "Error handling message",
                connection_id=connection_id,
                error=str(e),
                exc_info=True,
            )
            await self._send_error(connection_id, "Internal server error")

    async def broadcast_event(self, event: EventMessage) -> None:
        """
        Broadcast event to subscribed WebSocket connections.

        Args:
            event: Event message to broadcast
        """
        # Get connection info for this connection type
        websocket_connections = [
            conn_id
            for conn_id, conn_info in self.connection_pool._connections.items()
            if conn_info.connection_type == ConnectionType.WEBSOCKET
        ]

        if not websocket_connections:
            return

        # Get all subscriptions
        broadcast_count = 0
        for connection_id in websocket_connections:
            conn_info = self.connection_pool.get_connection(connection_id)
            if not conn_info:
                continue

            # Get client subscriptions
            subscriptions = self.subscription_manager.get_client_subscriptions(
                conn_info.client_id
            )

            # Check if any subscription matches the event
            for subscription in subscriptions:
                if self._should_send_event(subscription, event):
                    await self._send_event(connection_id, event)
                    broadcast_count += 1
                    break

        logger.debug(
            "Event broadcasted to WebSocket connections",
            event_id=event.event_id,
            event_type=event.event_type.value,
            broadcast_count=broadcast_count,
        )

    async def _handle_subscribe(
        self,
        connection_id: str,
        data: dict[str, Any],
    ) -> None:
        """Handle subscription request."""
        conn_info = self.connection_pool.get_connection(connection_id)
        if not conn_info:
            return

        # Parse subscription request
        topics = set(data.get("topics", []))
        event_types_str = data.get("event_types", [])
        event_types = {EventType(et) for et in event_types_str}

        # Parse filters
        filters_data = data.get("filters", {})
        filters = SubscriptionFilter(
            agent_ids=set(filters_data.get("agent_ids", [])),
            task_ids=set(filters_data.get("task_ids", [])),
            workflow_ids=set(filters_data.get("workflow_ids", [])),
            user_id=filters_data.get("user_id"),
            metadata_filters=filters_data.get("metadata_filters", {}),
        )

        # Create subscription
        subscription_id = str(uuid4())
        subscription = self.subscription_manager.add_subscription(
            subscription_id=subscription_id,
            client_id=conn_info.client_id,
            topics=topics,
            event_types=event_types,
            filters=filters,
        )

        # Subscribe to event bus
        self.event_bus.subscribe(
            handler=lambda event: asyncio.create_task(self.broadcast_event(event)),
            topics=topics,
            event_types=event_types,
        )

        # Send subscription confirmation
        await self._send_message(
            connection_id,
            {
                "type": "subscribed",
                "subscription_id": subscription_id,
                "topics": list(topics),
                "event_types": [et.value for et in event_types],
            },
        )

        logger.debug(
            "Subscription created",
            connection_id=connection_id,
            subscription_id=subscription_id,
        )

    async def _handle_unsubscribe(
        self,
        connection_id: str,
        data: dict[str, Any],
    ) -> None:
        """Handle unsubscribe request."""
        subscription_id = data.get("subscription_id")
        if not subscription_id:
            await self._send_error(connection_id, "Missing subscription_id")
            return

        # Remove subscription
        removed = self.subscription_manager.remove_subscription(subscription_id)

        if removed:
            await self._send_message(
                connection_id,
                {
                    "type": "unsubscribed",
                    "subscription_id": subscription_id,
                },
            )
            logger.debug(
                "Subscription removed",
                connection_id=connection_id,
                subscription_id=subscription_id,
            )
        else:
            await self._send_error(connection_id, "Subscription not found")

    async def _handle_ping(self, connection_id: str) -> None:
        """Handle ping message."""
        self.connection_pool.record_ping(connection_id)
        await self._send_message(connection_id, {"type": "pong"})

    async def _handle_pong(self, connection_id: str) -> None:
        """Handle pong message."""
        self.connection_pool.record_pong(connection_id)

    async def _heartbeat_monitor(
        self,
        connection_id: str,
        websocket: WebSocket,
    ) -> None:
        """
        Monitor connection health with heartbeat.

        Args:
            connection_id: Connection identifier
            websocket: WebSocket connection
        """
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Send ping
                await websocket.send_json({"type": "ping"})
                self.connection_pool.record_ping(connection_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Heartbeat error",
                    connection_id=connection_id,
                    error=str(e),
                )
                break

    def _should_send_event(
        self,
        subscription: Subscription,
        event: EventMessage,
    ) -> bool:
        """
        Check if event should be sent based on subscription.

        Args:
            subscription: Subscription
            event: Event message

        Returns:
            True if event matches subscription, False otherwise
        """
        # Check topic match
        if subscription.topics and event.topic not in subscription.topics:
            return False

        # Check event type match
        if (
            subscription.event_types
            and event.event_type not in subscription.event_types
        ):
            return False

        # Check filters
        if not subscription.filters.matches(
            {**event.payload, **(event.metadata or {})}
        ):
            return False

        return True

    async def _send_event(
        self,
        connection_id: str,
        event: EventMessage,
    ) -> None:
        """Send event to WebSocket connection."""
        message = {
            "type": "event",
            "event": event.to_dict(),
        }
        await self._send_message(connection_id, message)

    async def _send_message(
        self,
        connection_id: str,
        message: dict[str, Any],
    ) -> bool:
        """
        Send message to WebSocket connection.

        Args:
            connection_id: Connection identifier
            message: Message dictionary

        Returns:
            True if sent successfully, False otherwise
        """
        if connection_id not in self._connections:
            return False

        try:
            websocket = self._connections[connection_id]
            message_str = json.dumps(message)
            await websocket.send_text(message_str)

            # Record message sent
            self.connection_pool.record_message_sent(connection_id, len(message_str))

            return True
        except Exception as e:
            logger.error(
                "Error sending message",
                connection_id=connection_id,
                error=str(e),
            )
            return False

    async def _send_error(
        self,
        connection_id: str,
        error: str,
    ) -> None:
        """Send error message to WebSocket connection."""
        await self._send_message(
            connection_id,
            {
                "type": "error",
                "error": error,
            },
        )
