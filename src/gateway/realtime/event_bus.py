"""
Event Bus Implementation

Publish/subscribe event bus for real-time event distribution to WebSocket and SSE clients.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    """Event type enumeration."""

    AGENT_REGISTERED = "agent.registered"
    AGENT_UPDATED = "agent.updated"
    AGENT_DELETED = "agent.deleted"
    AGENT_STATUS_CHANGED = "agent.status.changed"

    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"

    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"

    SYSTEM_ALERT = "system.alert"
    SYSTEM_NOTIFICATION = "system.notification"


@dataclass
class EventMessage:
    """Event message structure."""

    event_id: str
    event_type: EventType
    topic: str
    payload: dict[str, Any]
    timestamp: datetime
    source: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        event_type: EventType,
        topic: str,
        payload: dict[str, Any],
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventMessage:
        """Create a new event message."""
        return cls(
            event_id=str(uuid4()),
            event_type=event_type,
            topic=topic,
            payload=payload,
            timestamp=datetime.now(datetime.UTC),
            source=source,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


# Type alias for event handler
EventHandler = Callable[[EventMessage], None]


class EventBus:
    """
    Event bus for publish/subscribe pattern.

    Supports topic-based subscriptions with efficient event distribution.
    Optimized for high-throughput event broadcasting to multiple subscribers.
    """

    def __init__(self) -> None:
        """Initialize event bus."""
        # Topic subscribers: topic -> set of handler ids
        self._topic_subscribers: dict[str, set[str]] = defaultdict(set)

        # Event type subscribers: event_type -> set of handler ids
        self._event_type_subscribers: dict[EventType, set[str]] = defaultdict(set)

        # Handler registry: handler_id -> (handler, topics, event_types)
        self._handlers: dict[str, tuple[EventHandler, set[str], set[EventType]]] = {}

        # Event queue for async processing
        self._event_queue: asyncio.Queue[EventMessage] = asyncio.Queue(maxsize=10000)

        # Event processing task
        self._processing_task: asyncio.Task[None] | None = None

        # Statistics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "subscribers_count": 0,
            "queue_size": 0,
        }

        logger.info("Event bus initialized")

    async def start(self) -> None:
        """Start event processing."""
        if self._processing_task is not None:
            logger.warning("Event bus already started")
            return

        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop event processing."""
        if self._processing_task is None:
            return

        # Cancel processing task
        self._processing_task.cancel()
        try:
            await self._processing_task
        except asyncio.CancelledError:
            pass

        self._processing_task = None
        logger.info("Event bus stopped")

    def subscribe(
        self,
        handler: EventHandler,
        topics: set[str] | None = None,
        event_types: set[EventType] | None = None,
    ) -> str:
        """
        Subscribe to events.

        Args:
            handler: Event handler function
            topics: Set of topics to subscribe to (optional)
            event_types: Set of event types to subscribe to (optional)

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid4())

        topics = topics or set()
        event_types = event_types or set()

        # Register handler
        self._handlers[subscription_id] = (handler, topics, event_types)

        # Add to topic subscribers
        for topic in topics:
            self._topic_subscribers[topic].add(subscription_id)

        # Add to event type subscribers
        for event_type in event_types:
            self._event_type_subscribers[event_type].add(subscription_id)

        self._stats["subscribers_count"] = len(self._handlers)

        logger.debug(
            "Subscription created",
            subscription_id=subscription_id,
            topics=list(topics),
            event_types=[et.value for et in event_types],
        )

        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID from subscribe()

        Returns:
            True if unsubscribed, False if subscription not found
        """
        if subscription_id not in self._handlers:
            return False

        # Get handler info
        _, topics, event_types = self._handlers[subscription_id]

        # Remove from topic subscribers
        for topic in topics:
            self._topic_subscribers[topic].discard(subscription_id)
            if not self._topic_subscribers[topic]:
                del self._topic_subscribers[topic]

        # Remove from event type subscribers
        for event_type in event_types:
            self._event_type_subscribers[event_type].discard(subscription_id)
            if not self._event_type_subscribers[event_type]:
                del self._event_type_subscribers[event_type]

        # Remove handler
        del self._handlers[subscription_id]

        self._stats["subscribers_count"] = len(self._handlers)

        logger.debug("Subscription removed", subscription_id=subscription_id)

        return True

    async def publish(self, event: EventMessage) -> None:
        """
        Publish event to subscribers.

        Args:
            event: Event message to publish
        """
        try:
            await self._event_queue.put(event)
            self._stats["events_published"] += 1
            self._stats["queue_size"] = self._event_queue.qsize()

            logger.debug(
                "Event published",
                event_id=event.event_id,
                event_type=event.event_type.value,
                topic=event.topic,
            )
        except asyncio.QueueFull:
            logger.error(
                "Event queue full, dropping event",
                event_id=event.event_id,
                event_type=event.event_type.value,
            )

    async def _process_events(self) -> None:
        """Process events from queue and distribute to subscribers."""
        logger.info("Event processing started")

        while True:
            try:
                # Get event from queue
                event = await self._event_queue.get()

                # Find matching subscribers
                subscriber_ids = self._get_matching_subscribers(event)

                # Distribute to subscribers
                if subscriber_ids:
                    await self._distribute_event(event, subscriber_ids)

                self._stats["events_processed"] += 1
                self._stats["queue_size"] = self._event_queue.qsize()

            except asyncio.CancelledError:
                logger.info("Event processing cancelled")
                break
            except Exception as e:
                logger.error("Error processing event", error=str(e), exc_info=True)

    def _get_matching_subscribers(self, event: EventMessage) -> set[str]:
        """
        Get subscriber IDs matching the event.

        Args:
            event: Event message

        Returns:
            Set of matching subscriber IDs
        """
        matching_subscribers: set[str] = set()

        # Get topic subscribers
        topic_subs = self._topic_subscribers.get(event.topic, set())
        matching_subscribers.update(topic_subs)

        # Get event type subscribers
        event_type_subs = self._event_type_subscribers.get(event.event_type, set())
        matching_subscribers.update(event_type_subs)

        return matching_subscribers

    async def _distribute_event(
        self,
        event: EventMessage,
        subscriber_ids: set[str],
    ) -> None:
        """
        Distribute event to subscribers.

        Args:
            event: Event message
            subscriber_ids: Set of subscriber IDs to notify
        """
        # Execute handlers concurrently
        tasks = []
        for subscriber_id in subscriber_ids:
            if subscriber_id in self._handlers:
                handler, _, _ = self._handlers[subscriber_id]
                task = asyncio.create_task(self._execute_handler(handler, event))
                tasks.append(task)

        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_handler(
        self,
        handler: EventHandler,
        event: EventMessage,
    ) -> None:
        """
        Execute event handler.

        Args:
            handler: Event handler function
            event: Event message
        """
        try:
            # Check if handler is async or sync
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Run sync handler in thread pool
                await asyncio.get_event_loop().run_in_executor(None, handler, event)
        except Exception as e:
            logger.error(
                "Error executing event handler",
                error=str(e),
                event_id=event.event_id,
                exc_info=True,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self._stats,
            "queue_size": self._event_queue.qsize(),
            "topics_count": len(self._topic_subscribers),
            "event_types_count": len(self._event_type_subscribers),
        }


# Global event bus instance
event_bus = EventBus()
