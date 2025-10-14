"""
Event Manager Service

Manages event publishing, subscriptions, and WebSocket connections for real-time notifications.
"""

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import structlog
from fastapi import WebSocket

from agentcore.a2a_protocol.models.events import (
    DeadLetterMessage,
    Event,
    EventPriority,
    EventPublishRequest,
    EventPublishResponse,
    EventSubscribeRequest,
    EventSubscribeResponse,
    EventSubscription,
    EventType,
    WebSocketMessage,
)

logger = structlog.get_logger()


class WebSocketConnection:
    """WebSocket connection wrapper."""

    def __init__(self, connection_id: str, websocket: WebSocket, subscriber_id: str):
        self.connection_id = connection_id
        self.websocket = websocket
        self.subscriber_id = subscriber_id
        self.connected_at = datetime.now(UTC)
        self.last_ping = datetime.now(UTC)
        self.is_alive = True

    async def send_event(self, event: Event) -> bool:
        """
        Send event to WebSocket client.

        Args:
            event: Event to send

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = WebSocketMessage(
                message_type="event", payload=event.to_notification()
            )
            await self.websocket.send_json(message.model_dump(mode="json"))
            return True
        except Exception as e:
            logger.error(
                "Failed to send event via WebSocket",
                connection_id=self.connection_id,
                error=str(e),
            )
            self.is_alive = False
            return False

    async def send_ping(self) -> bool:
        """Send ping to keep connection alive."""
        try:
            message = WebSocketMessage(
                message_type="ping",
                payload={"timestamp": datetime.now(UTC).isoformat()},
            )
            await self.websocket.send_json(message.model_dump(mode="json"))
            self.last_ping = datetime.now(UTC)
            return True
        except Exception:
            self.is_alive = False
            return False

    async def close(self) -> None:
        """Close WebSocket connection."""
        try:
            await self.websocket.close()
        except Exception:
            pass
        self.is_alive = False


class EventManager:
    """
    Event manager for publishing and subscribing to events.

    Manages event subscriptions, WebSocket connections, and dead letter queue.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

        # Subscriptions
        self._subscriptions: dict[str, EventSubscription] = {}
        self._subscriptions_by_type: dict[EventType, set[str]] = defaultdict(set)
        self._subscriptions_by_subscriber: dict[str, set[str]] = defaultdict(set)

        # WebSocket connections
        self._websocket_connections: dict[str, WebSocketConnection] = {}
        self._connections_by_subscriber: dict[str, set[str]] = defaultdict(set)

        # Dead letter queue
        self._dead_letter_queue: deque[DeadLetterMessage] = deque(maxlen=10000)

        # Event history (for replay)
        self._event_history: deque[Event] = deque(maxlen=1000)

        # Event statistics
        self._event_stats = {
            "total_published": 0,
            "total_delivered": 0,
            "total_failed": 0,
            "dead_letter_messages": 0,
            "active_subscriptions": 0,
            "active_websockets": 0,
        }

        # Event hooks for custom processing
        self._event_hooks: dict[EventType, list[Callable]] = defaultdict(list)

    # ==================== Event Publishing ====================

    async def publish_event(
        self,
        event_type: EventType,
        source: str,
        data: dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> EventPublishResponse:
        """
        Publish an event to all matching subscribers.

        Args:
            event_type: Type of event
            source: Event source identifier
            data: Event data
            priority: Event priority
            metadata: Event metadata
            correlation_id: Correlation ID

        Returns:
            Event publish response
        """
        # Create event
        event = Event(
            event_type=event_type,
            source=source,
            data=data,
            priority=priority,
            metadata=metadata or {},
            correlation_id=correlation_id,
        )

        # Add to history
        self._event_history.append(event)
        self._event_stats["total_published"] += 1

        # Execute event hooks
        await self._execute_hooks(event)

        # Find matching subscriptions
        matching_subscriptions = self._find_matching_subscriptions(event)

        # Notify subscribers
        subscribers_notified = await self._notify_subscribers(
            event, matching_subscriptions
        )

        self.logger.info(
            "Event published",
            event_id=event.event_id,
            event_type=event_type.value,
            source=source,
            subscribers_notified=subscribers_notified,
        )

        return EventPublishResponse(
            success=True,
            event_id=event.event_id,
            subscribers_notified=subscribers_notified,
        )

    def _find_matching_subscriptions(self, event: Event) -> list[EventSubscription]:
        """Find subscriptions that match the event."""
        matching = []

        # Get subscriptions for this event type
        subscription_ids = self._subscriptions_by_type.get(event.event_type, set())

        for sub_id in subscription_ids:
            subscription = self._subscriptions.get(sub_id)
            if not subscription:
                continue

            # Check if subscription is active and not expired
            if not subscription.active or subscription.is_expired():
                continue

            # Check if event matches subscription filters
            if subscription.matches_event(event):
                matching.append(subscription)

        return matching

    async def _notify_subscribers(
        self, event: Event, subscriptions: list[EventSubscription]
    ) -> int:
        """
        Notify subscribers about event.

        Args:
            event: Event to notify about
            subscriptions: Matching subscriptions

        Returns:
            Number of successfully notified subscribers
        """
        notified_count = 0

        for subscription in subscriptions:
            subscriber_id = subscription.subscriber_id

            # Get WebSocket connections for subscriber
            connection_ids = self._connections_by_subscriber.get(subscriber_id, set())

            if not connection_ids:
                # No active connections - add to dead letter queue
                await self._add_to_dead_letter_queue(
                    event, subscriber_id, "No active WebSocket connections"
                )
                continue

            # Send to all connections
            success = False
            for conn_id in list(connection_ids):
                connection = self._websocket_connections.get(conn_id)
                if not connection or not connection.is_alive:
                    # Clean up dead connection
                    await self._remove_connection(conn_id)
                    continue

                if await connection.send_event(event):
                    success = True
                    notified_count += 1
                    self._event_stats["total_delivered"] += 1
                else:
                    # Connection failed - remove it
                    await self._remove_connection(conn_id)

            if not success:
                # All connections failed - add to dead letter queue
                await self._add_to_dead_letter_queue(
                    event, subscriber_id, "All WebSocket connections failed"
                )

        return notified_count

    async def _execute_hooks(self, event: Event) -> None:
        """Execute registered hooks for event type."""
        hooks = self._event_hooks.get(event.event_type, [])
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(event)
                else:
                    hook(event)
            except Exception as e:
                self.logger.error(
                    "Event hook failed", event_type=event.event_type.value, error=str(e)
                )

    def register_hook(self, event_type: EventType, hook: Callable) -> None:
        """Register a hook for event type."""
        self._event_hooks[event_type].append(hook)
        self.logger.info("Event hook registered", event_type=event_type.value)

    # ==================== Subscription Management ====================

    async def subscribe(
        self,
        subscriber_id: str,
        event_types: list[EventType],
        filters: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> EventSubscribeResponse:
        """
        Create event subscription.

        Args:
            subscriber_id: Subscriber identifier
            event_types: Event types to subscribe to
            filters: Event filters
            ttl_seconds: Subscription TTL in seconds

        Returns:
            Subscription response
        """
        # Calculate expiration if TTL provided
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)

        # Create subscription
        subscription = EventSubscription(
            subscriber_id=subscriber_id,
            event_types=event_types,
            filters=filters or {},
            expires_at=expires_at,
        )

        # Store subscription
        self._subscriptions[subscription.subscription_id] = subscription

        # Update indexes
        for event_type in event_types:
            self._subscriptions_by_type[event_type].add(subscription.subscription_id)

        self._subscriptions_by_subscriber[subscriber_id].add(
            subscription.subscription_id
        )

        self._event_stats["active_subscriptions"] = len(self._subscriptions)

        self.logger.info(
            "Subscription created",
            subscription_id=subscription.subscription_id,
            subscriber_id=subscriber_id,
            event_types=[et.value for et in event_types],
        )

        return EventSubscribeResponse(
            success=True, subscription_id=subscription.subscription_id
        )

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Remove subscription.

        Args:
            subscription_id: Subscription ID to remove

        Returns:
            True if removed, False if not found
        """
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return False

        # Remove from indexes
        for event_type in subscription.event_types:
            self._subscriptions_by_type[event_type].discard(subscription_id)

        self._subscriptions_by_subscriber[subscription.subscriber_id].discard(
            subscription_id
        )

        # Remove subscription
        del self._subscriptions[subscription_id]

        self._event_stats["active_subscriptions"] = len(self._subscriptions)

        self.logger.info(
            "Subscription removed",
            subscription_id=subscription_id,
            subscriber_id=subscription.subscriber_id,
        )

        return True

    def get_subscriptions(
        self, subscriber_id: str | None = None
    ) -> list[EventSubscription]:
        """
        Get subscriptions.

        Args:
            subscriber_id: Optional filter by subscriber

        Returns:
            List of subscriptions
        """
        if subscriber_id:
            sub_ids = self._subscriptions_by_subscriber.get(subscriber_id, set())
            return [
                self._subscriptions[sid]
                for sid in sub_ids
                if sid in self._subscriptions
            ]

        return list(self._subscriptions.values())

    # ==================== WebSocket Connection Management ====================

    async def register_websocket(self, websocket: WebSocket, subscriber_id: str) -> str:
        """
        Register WebSocket connection.

        Args:
            websocket: WebSocket connection
            subscriber_id: Subscriber identifier

        Returns:
            Connection ID
        """
        connection_id = str(uuid4())

        connection = WebSocketConnection(
            connection_id=connection_id,
            websocket=websocket,
            subscriber_id=subscriber_id,
        )

        self._websocket_connections[connection_id] = connection
        self._connections_by_subscriber[subscriber_id].add(connection_id)

        self._event_stats["active_websockets"] = len(self._websocket_connections)

        self.logger.info(
            "WebSocket registered",
            connection_id=connection_id,
            subscriber_id=subscriber_id,
        )

        return connection_id

    async def _remove_connection(self, connection_id: str) -> None:
        """Remove WebSocket connection."""
        connection = self._websocket_connections.get(connection_id)
        if not connection:
            return

        subscriber_id = connection.subscriber_id

        # Remove from indexes
        self._connections_by_subscriber[subscriber_id].discard(connection_id)

        # Close connection
        await connection.close()

        # Remove connection
        del self._websocket_connections[connection_id]

        self._event_stats["active_websockets"] = len(self._websocket_connections)

        self.logger.info(
            "WebSocket removed",
            connection_id=connection_id,
            subscriber_id=subscriber_id,
        )

    async def close_connection(self, connection_id: str) -> None:
        """Close WebSocket connection by ID."""
        await self._remove_connection(connection_id)

    # ==================== Dead Letter Queue ====================

    async def _add_to_dead_letter_queue(
        self, event: Event, subscriber_id: str, failure_reason: str
    ) -> None:
        """Add failed delivery to dead letter queue."""
        message = DeadLetterMessage(
            event=event, subscriber_id=subscriber_id, failure_reason=failure_reason
        )

        self._dead_letter_queue.append(message)

        self._event_stats["total_failed"] += 1
        self._event_stats["dead_letter_messages"] = len(self._dead_letter_queue)

        self.logger.warning(
            "Event added to dead letter queue",
            event_id=event.event_id,
            subscriber_id=subscriber_id,
            reason=failure_reason,
        )

    def get_dead_letter_messages(self, limit: int = 100) -> list[DeadLetterMessage]:
        """Get messages from dead letter queue."""
        return list(self._dead_letter_queue)[:limit]

    async def retry_dead_letter_message(self, message_id: str) -> bool:
        """
        Retry delivery of dead letter message.

        Args:
            message_id: Dead letter message ID

        Returns:
            True if retry successful, False otherwise
        """
        # Find message in queue
        message = None
        for msg in self._dead_letter_queue:
            if msg.message_id == message_id:
                message = msg
                break

        if not message:
            return False

        if not message.can_retry():
            self.logger.warning(
                "Dead letter message exceeded max retries", message_id=message_id
            )
            return False

        # Increment retry counter
        message.increment_retry()

        # Find matching subscriptions for the event
        matching_subscriptions = self._find_matching_subscriptions(message.event)

        # Filter to only subscriber that failed
        subscriber_subscriptions = [
            sub
            for sub in matching_subscriptions
            if sub.subscriber_id == message.subscriber_id
        ]

        if not subscriber_subscriptions:
            return False

        # Attempt delivery
        notified = await self._notify_subscribers(
            message.event, subscriber_subscriptions
        )

        if notified > 0:
            # Success - remove from dead letter queue
            self._dead_letter_queue.remove(message)
            self._event_stats["dead_letter_messages"] = len(self._dead_letter_queue)
            self.logger.info(
                "Dead letter message retry successful", message_id=message_id
            )
            return True

        return False

    # ==================== Event History & Replay ====================

    def get_event_history(
        self, event_type: EventType | None = None, limit: int = 100
    ) -> list[Event]:
        """
        Get event history.

        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        events = list(self._event_history)

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    # ==================== Cleanup & Statistics ====================

    async def cleanup_expired_subscriptions(self) -> int:
        """Remove expired subscriptions."""
        expired_ids = []

        for sub_id, subscription in self._subscriptions.items():
            if subscription.is_expired():
                expired_ids.append(sub_id)

        for sub_id in expired_ids:
            await self.unsubscribe(sub_id)

        if expired_ids:
            self.logger.info("Expired subscriptions cleaned up", count=len(expired_ids))

        return len(expired_ids)

    async def cleanup_dead_connections(self) -> int:
        """Remove dead WebSocket connections."""
        dead_ids = []

        for conn_id, connection in list(self._websocket_connections.items()):
            if not connection.is_alive:
                dead_ids.append(conn_id)

        for conn_id in dead_ids:
            await self._remove_connection(conn_id)

        if dead_ids:
            self.logger.info("Dead connections cleaned up", count=len(dead_ids))

        return len(dead_ids)

    def get_statistics(self) -> dict[str, Any]:
        """Get event system statistics."""
        return {
            **self._event_stats,
            "event_history_size": len(self._event_history),
            "subscriptions_by_type": {
                et.value: len(subs) for et, subs in self._subscriptions_by_type.items()
            },
        }


# Global event manager instance
event_manager = EventManager()
