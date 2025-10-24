"""Event publishing and subscription management."""

import asyncio
import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from .config import WebhookConfig
from .exceptions import EventPublishError
from .models import EventPayload, EventSubscription, WebhookEvent

logger = logging.getLogger(__name__)


class EventPublisher:
    """Event publishing system with pub/sub pattern.

    Manages event publishing and routing to subscribed webhooks.
    Supports async event processing with batching and rate limiting.
    """

    def __init__(self, config: WebhookConfig | None = None) -> None:
        """Initialize event publisher.

        Args:
            config: Webhook configuration
        """
        self.config = config or WebhookConfig()

        # Event queue for async processing
        self._event_queue: asyncio.Queue[EventPayload] = asyncio.Queue(
            maxsize=self.config.event_queue_size
        )

        # Subscriptions: event_type -> set of webhook_ids
        self._subscriptions: dict[WebhookEvent, set[UUID]] = defaultdict(set)

        # Subscription metadata
        self._subscription_metadata: dict[UUID, EventSubscription] = {}

        # Processing state
        self._running = False
        self._processor_task: asyncio.Task[None] | None = None

        # Locks
        self._subscription_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start event processing loop."""
        if self._running:
            logger.warning("Event publisher already running")
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("Event publisher started")

    async def stop(self) -> None:
        """Stop event processing loop."""
        if not self._running:
            return

        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        logger.info("Event publisher stopped")

    async def publish(
        self,
        event_type: WebhookEvent,
        data: dict[str, Any],
        source: str,
        tenant_id: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventPayload:
        """Publish an event to subscribed webhooks.

        Args:
            event_type: Type of event
            data: Event-specific data
            source: Event source identifier
            tenant_id: Tenant identifier
            correlation_id: Correlation ID for tracing
            metadata: Additional metadata

        Returns:
            Created event payload

        Raises:
            EventPublishError: If publishing fails
        """
        # Create event payload
        event = EventPayload(
            event_type=event_type,
            data=data,
            source=source,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        try:
            # Add to queue for async processing
            await asyncio.wait_for(
                self._event_queue.put(event),
                timeout=5.0,
            )

            logger.debug(
                f"Published event {event.event_id} (type: {event_type})",
                extra={
                    "event_id": str(event.event_id),
                    "event_type": event_type.value,
                    "source": source,
                },
            )

            return event

        except asyncio.TimeoutError:
            raise EventPublishError(
                event_type.value, "Event queue full - timeout waiting to enqueue"
            )
        except Exception as e:
            raise EventPublishError(event_type.value, str(e))

    async def subscribe(
        self,
        webhook_id: UUID,
        event_types: list[WebhookEvent],
        tenant_id: str | None = None,
    ) -> EventSubscription:
        """Subscribe webhook to events.

        Args:
            webhook_id: Webhook identifier
            event_types: List of event types to subscribe to
            tenant_id: Tenant identifier

        Returns:
            Created subscription
        """
        subscription = EventSubscription(
            event_types=event_types,
            webhook_id=webhook_id,
            tenant_id=tenant_id,
        )

        async with self._subscription_lock:
            # Add webhook to subscriptions for each event type
            for event_type in event_types:
                self._subscriptions[event_type].add(webhook_id)

            # Store subscription metadata
            self._subscription_metadata[subscription.subscription_id] = subscription

        logger.info(
            f"Webhook {webhook_id} subscribed to {len(event_types)} event types"
        )

        return subscription

    async def unsubscribe(
        self,
        webhook_id: UUID,
        event_types: list[WebhookEvent] | None = None,
    ) -> None:
        """Unsubscribe webhook from events.

        Args:
            webhook_id: Webhook identifier
            event_types: Specific event types to unsubscribe from (None = all)
        """
        async with self._subscription_lock:
            if event_types is None:
                # Unsubscribe from all events
                for event_subscriptions in self._subscriptions.values():
                    event_subscriptions.discard(webhook_id)

                # Remove subscription metadata
                to_remove = [
                    sub_id
                    for sub_id, sub in self._subscription_metadata.items()
                    if sub.webhook_id == webhook_id
                ]
                for sub_id in to_remove:
                    del self._subscription_metadata[sub_id]

            else:
                # Unsubscribe from specific events
                for event_type in event_types:
                    self._subscriptions[event_type].discard(webhook_id)

        logger.info(f"Webhook {webhook_id} unsubscribed from events")

    async def get_subscribers(
        self,
        event_type: WebhookEvent,
        tenant_id: str | None = None,
    ) -> list[UUID]:
        """Get webhooks subscribed to an event type.

        Args:
            event_type: Event type
            tenant_id: Optional tenant filter

        Returns:
            List of subscribed webhook IDs
        """
        async with self._subscription_lock:
            webhook_ids = list(self._subscriptions.get(event_type, set()))

            # Filter by tenant if specified
            if tenant_id is not None:
                webhook_ids = [
                    wid
                    for wid in webhook_ids
                    if any(
                        sub.webhook_id == wid and sub.tenant_id == tenant_id
                        for sub in self._subscription_metadata.values()
                    )
                ]

            return webhook_ids

    async def _process_events(self) -> None:
        """Background event processing loop."""
        logger.info("Event processing loop started")

        while self._running:
            try:
                # Collect batch of events
                events = []
                timeout = self.config.event_processing_interval_seconds

                try:
                    # Get first event with timeout
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=timeout,
                    )
                    events.append(event)

                    # Collect more events up to batch size (non-blocking)
                    while len(events) < self.config.event_batch_size:
                        try:
                            event = self._event_queue.get_nowait()
                            events.append(event)
                        except asyncio.QueueEmpty:
                            break

                except asyncio.TimeoutError:
                    # No events available, continue loop
                    continue

                # Process batch
                await self._process_event_batch(events)

            except asyncio.CancelledError:
                logger.info("Event processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Back off on error

        logger.info("Event processing loop stopped")

    async def _process_event_batch(self, events: list[EventPayload]) -> None:
        """Process a batch of events.

        Args:
            events: List of events to process
        """
        logger.debug(f"Processing batch of {len(events)} events")

        # Group events by subscribers
        deliveries: dict[UUID, list[EventPayload]] = defaultdict(list)

        async with self._subscription_lock:
            for event in events:
                # Get subscribers for this event type
                webhook_ids = self._subscriptions.get(event.event_type, set())

                # Filter by tenant if applicable
                for webhook_id in webhook_ids:
                    # Check tenant match
                    matching_subs = [
                        sub
                        for sub in self._subscription_metadata.values()
                        if sub.webhook_id == webhook_id
                        and (
                            event.tenant_id is None
                            or sub.tenant_id is None
                            or sub.tenant_id == event.tenant_id
                        )
                    ]

                    if matching_subs:
                        deliveries[webhook_id].append(event)

        # Log delivery summary
        total_deliveries = sum(len(events) for events in deliveries.values())
        logger.debug(
            f"Routing {len(events)} events to {len(deliveries)} webhooks "
            f"({total_deliveries} total deliveries)"
        )

        # NOTE: Actual delivery is handled by the DeliveryService
        # This just prepares the routing information
        # In a full implementation, we would call delivery_service.schedule() here

    async def get_queue_size(self) -> int:
        """Get current event queue size."""
        return self._event_queue.qsize()

    async def get_subscription_count(self) -> int:
        """Get total number of subscriptions."""
        async with self._subscription_lock:
            return len(self._subscription_metadata)
