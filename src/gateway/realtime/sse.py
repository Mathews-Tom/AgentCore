"""
Server-Sent Events (SSE) Manager

High-performance SSE implementation for real-time event streaming.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator
from uuid import uuid4

import structlog
from fastapi import Request
from jose import JWTError
from sse_starlette.sse import EventSourceResponse

from gateway.auth.jwt import jwt_manager
from gateway.realtime.connection_pool import ConnectionPool, ConnectionType
from gateway.realtime.event_bus import EventBus, EventMessage, EventType
from gateway.realtime.subscriptions import (
    Subscription,
    SubscriptionFilter,
    SubscriptionManager,
)

logger = structlog.get_logger()


class SSEManager:
    """
    Server-Sent Events (SSE) manager.

    Manages SSE connections with authentication, subscription management,
    and event streaming. Optimized for high-throughput event delivery.
    """

    def __init__(
        self,
        connection_pool: ConnectionPool,
        event_bus: EventBus,
        subscription_manager: SubscriptionManager,
        keepalive_interval: int = 30,
    ) -> None:
        """
        Initialize SSE manager.

        Args:
            connection_pool: Connection pool instance
            event_bus: Event bus instance
            subscription_manager: Subscription manager instance
            keepalive_interval: Keepalive interval in seconds
        """
        self.connection_pool = connection_pool
        self.event_bus = event_bus
        self.subscription_manager = subscription_manager
        self.keepalive_interval = keepalive_interval

        # Event queues for SSE connections: connection_id -> Queue
        self._event_queues: dict[str, asyncio.Queue[EventMessage | None]] = {}

        logger.info(
            "SSE manager initialized",
            keepalive_interval=keepalive_interval,
        )

    async def create_event_stream(
        self,
        request: Request,
        token: str | None = None,
        topics: list[str] | None = None,
        event_types: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> EventSourceResponse:
        """
        Create SSE event stream.

        Args:
            request: FastAPI request
            token: JWT authentication token (optional)
            topics: Topics to subscribe to (optional)
            event_types: Event types to subscribe to (optional)
            filters: Event filters (optional)

        Returns:
            EventSourceResponse for SSE streaming
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
                    "SSE authenticated",
                    connection_id=connection_id,
                    user_id=user_id,
                )
            except JWTError as e:
                logger.warning(
                    "SSE authentication failed",
                    connection_id=connection_id,
                    error=str(e),
                )
                # Return error event
                async def error_generator() -> AsyncGenerator[dict[str, str], None]:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": "Authentication failed"}),
                    }
                return EventSourceResponse(error_generator())

        # Generate client ID
        client_id = str(uuid4())

        # Get client info
        remote_addr = None
        user_agent = None
        if request.client:
            remote_addr = f"{request.client.host}:{request.client.port}"
        if "user-agent" in request.headers:
            user_agent = request.headers["user-agent"]

        # Add to connection pool
        conn_info = self.connection_pool.add_connection(
            connection_id=connection_id,
            connection_type=ConnectionType.SSE,
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
            # Return error event
            async def capacity_error_generator() -> AsyncGenerator[dict[str, str], None]:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": "Server capacity reached"}),
                }
            return EventSourceResponse(capacity_error_generator())

        # Create event queue for this connection
        event_queue: asyncio.Queue[EventMessage | None] = asyncio.Queue(maxsize=1000)
        self._event_queues[connection_id] = event_queue

        # Parse subscription parameters
        topics_set = set(topics or [])
        event_types_set = {EventType(et) for et in (event_types or [])}

        # Parse filters
        filters_obj = SubscriptionFilter()
        if filters:
            filters_obj = SubscriptionFilter(
                agent_ids=set(filters.get("agent_ids", [])),
                task_ids=set(filters.get("task_ids", [])),
                workflow_ids=set(filters.get("workflow_ids", [])),
                user_id=filters.get("user_id"),
                metadata_filters=filters.get("metadata_filters", {}),
            )

        # Create subscription
        subscription_id = str(uuid4())
        subscription = self.subscription_manager.add_subscription(
            subscription_id=subscription_id,
            client_id=client_id,
            topics=topics_set,
            event_types=event_types_set,
            filters=filters_obj,
        )

        # Subscribe to event bus
        event_bus_subscription_id = self.event_bus.subscribe(
            handler=lambda event: self._queue_event(connection_id, event, subscription),
            topics=topics_set,
            event_types=event_types_set,
        )

        logger.info(
            "SSE connection established",
            connection_id=connection_id,
            client_id=client_id,
            user_id=user_id,
            subscription_id=subscription_id,
        )

        # Create event generator
        async def event_generator() -> AsyncGenerator[dict[str, str], None]:
            try:
                # Send connection established event
                yield {
                    "event": "connected",
                    "data": json.dumps({
                        "connection_id": connection_id,
                        "client_id": client_id,
                        "subscription_id": subscription_id,
                    }),
                }

                # Start keepalive task
                keepalive_task = asyncio.create_task(
                    self._keepalive_monitor(connection_id, event_queue)
                )

                try:
                    while True:
                        # Check if client disconnected
                        if await request.is_disconnected():
                            logger.info(
                                "SSE client disconnected",
                                connection_id=connection_id,
                            )
                            break

                        # Wait for event with timeout
                        try:
                            event = await asyncio.wait_for(
                                event_queue.get(),
                                timeout=1.0,
                            )

                            if event is None:
                                # Keepalive message
                                yield {
                                    "event": "keepalive",
                                    "data": json.dumps({"timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else ""}),
                                }
                            else:
                                # Send event
                                yield {
                                    "event": event.event_type.value,
                                    "data": json.dumps(event.to_dict()),
                                    "id": event.event_id,
                                }

                                # Record message sent
                                self.connection_pool.record_message_sent(
                                    connection_id,
                                    len(json.dumps(event.to_dict())),
                                )

                        except asyncio.TimeoutError:
                            # Continue to check for disconnect
                            continue

                finally:
                    # Cancel keepalive task
                    keepalive_task.cancel()
                    try:
                        await keepalive_task
                    except asyncio.CancelledError:
                        pass

            finally:
                # Cleanup
                await self._cleanup_connection(
                    connection_id,
                    client_id,
                    subscription_id,
                    event_bus_subscription_id,
                )

        return EventSourceResponse(event_generator())

    def _queue_event(
        self,
        connection_id: str,
        event: EventMessage,
        subscription: Subscription,
    ) -> None:
        """
        Queue event for SSE connection.

        Args:
            connection_id: Connection identifier
            event: Event message
            subscription: Subscription
        """
        if connection_id not in self._event_queues:
            return

        # Check if event matches subscription filters
        if not subscription.filters.matches({**event.payload, **(event.metadata or {})}):
            return

        # Queue event
        try:
            self._event_queues[connection_id].put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "SSE event queue full, dropping event",
                connection_id=connection_id,
                event_id=event.event_id,
            )

    async def _keepalive_monitor(
        self,
        connection_id: str,
        event_queue: asyncio.Queue[EventMessage | None],
    ) -> None:
        """
        Monitor connection and send keepalive messages.

        Args:
            connection_id: Connection identifier
            event_queue: Event queue
        """
        while True:
            try:
                await asyncio.sleep(self.keepalive_interval)

                # Send keepalive (None event)
                try:
                    await event_queue.put(None)
                    self.connection_pool.update_activity(connection_id)
                except asyncio.QueueFull:
                    logger.warning(
                        "SSE event queue full for keepalive",
                        connection_id=connection_id,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Keepalive error",
                    connection_id=connection_id,
                    error=str(e),
                )
                break

    async def _cleanup_connection(
        self,
        connection_id: str,
        client_id: str,
        subscription_id: str,
        event_bus_subscription_id: str,
    ) -> None:
        """
        Cleanup SSE connection.

        Args:
            connection_id: Connection identifier
            client_id: Client identifier
            subscription_id: Subscription identifier
            event_bus_subscription_id: Event bus subscription ID
        """
        # Unsubscribe from event bus
        self.event_bus.unsubscribe(event_bus_subscription_id)

        # Remove subscription
        self.subscription_manager.remove_subscription(subscription_id)

        # Remove from connection pool
        self.connection_pool.remove_connection(connection_id)

        # Remove event queue
        if connection_id in self._event_queues:
            del self._event_queues[connection_id]

        logger.info(
            "SSE connection cleaned up",
            connection_id=connection_id,
            client_id=client_id,
        )


# Global SSE manager instance
sse_manager = SSEManager(
    connection_pool=None,  # Will be set during initialization
    event_bus=None,  # Will be set during initialization
    subscription_manager=None,  # Will be set during initialization
)
