"""
Tests for Event Bus

Tests event publishing, subscribing, and distribution.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.realtime.event_bus import EventBus, EventMessage, EventType


@pytest.fixture
async def event_bus():
    """Create event bus instance for testing."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
class TestEventBus:
    """Event bus tests."""

    async def test_event_message_creation(self):
        """Test event message creation."""
        event = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-456"},
            source="test-service",
        )

        assert event.event_id is not None
        assert event.event_type == EventType.TASK_CREATED
        assert event.topic == "agent.123"
        assert event.payload == {"task_id": "task-456"}
        assert event.source == "test-service"
        assert event.timestamp is not None

    async def test_subscribe_and_publish(self, event_bus):
        """Test subscribing and publishing events."""
        received_events = []

        async def handler(event: EventMessage) -> None:
            received_events.append(event)

        # Subscribe to topic
        subscription_id = event_bus.subscribe(
            handler=handler,
            topics={"agent.123"},
        )

        # Publish event
        event = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-456"},
        )
        await event_bus.publish(event)

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify event received
        assert len(received_events) == 1
        assert received_events[0].event_id == event.event_id
        assert received_events[0].topic == "agent.123"

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    async def test_subscribe_by_event_type(self, event_bus):
        """Test subscribing by event type."""
        received_events = []

        async def handler(event: EventMessage) -> None:
            received_events.append(event)

        # Subscribe to event type
        subscription_id = event_bus.subscribe(
            handler=handler,
            event_types={EventType.TASK_CREATED},
        )

        # Publish events with different event types
        event1 = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-1"},
        )
        event2 = EventMessage.create(
            event_type=EventType.TASK_COMPLETED,
            topic="agent.123",
            payload={"task_id": "task-2"},
        )

        await event_bus.publish(event1)
        await event_bus.publish(event2)

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify only TASK_CREATED event received
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.TASK_CREATED

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers receiving same event."""
        received_events_1 = []
        received_events_2 = []

        async def handler1(event: EventMessage) -> None:
            received_events_1.append(event)

        async def handler2(event: EventMessage) -> None:
            received_events_2.append(event)

        # Subscribe multiple handlers
        sub1 = event_bus.subscribe(handler=handler1, topics={"agent.123"})
        sub2 = event_bus.subscribe(handler=handler2, topics={"agent.123"})

        # Publish event
        event = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-456"},
        )
        await event_bus.publish(event)

        # Wait for event processing
        await asyncio.sleep(0.1)

        # Verify both handlers received event
        assert len(received_events_1) == 1
        assert len(received_events_2) == 1
        assert received_events_1[0].event_id == event.event_id
        assert received_events_2[0].event_id == event.event_id

        # Cleanup
        event_bus.unsubscribe(sub1)
        event_bus.unsubscribe(sub2)

    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        received_events = []

        async def handler(event: EventMessage) -> None:
            received_events.append(event)

        # Subscribe
        subscription_id = event_bus.subscribe(
            handler=handler,
            topics={"agent.123"},
        )

        # Publish first event
        event1 = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-1"},
        )
        await event_bus.publish(event1)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Unsubscribe
        result = event_bus.unsubscribe(subscription_id)
        assert result is True

        # Publish second event
        event2 = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-2"},
        )
        await event_bus.publish(event2)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify only first event received
        assert len(received_events) == 1
        assert received_events[0].event_id == event1.event_id

    async def test_topic_filtering(self, event_bus):
        """Test topic-based filtering."""
        received_events = []

        async def handler(event: EventMessage) -> None:
            received_events.append(event)

        # Subscribe to specific topic
        subscription_id = event_bus.subscribe(
            handler=handler,
            topics={"agent.123"},
        )

        # Publish events to different topics
        event1 = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-1"},
        )
        event2 = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.456",
            payload={"task_id": "task-2"},
        )

        await event_bus.publish(event1)
        await event_bus.publish(event2)

        # Wait for processing
        await asyncio.sleep(0.1)

        # Verify only matching topic event received
        assert len(received_events) == 1
        assert received_events[0].topic == "agent.123"

        # Cleanup
        event_bus.unsubscribe(subscription_id)

    async def test_get_stats(self, event_bus):
        """Test getting event bus statistics."""
        stats = event_bus.get_stats()

        assert "events_published" in stats
        assert "events_processed" in stats
        assert "subscribers_count" in stats
        assert "queue_size" in stats
        assert "topics_count" in stats
        assert "event_types_count" in stats

    async def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = EventMessage.create(
            event_type=EventType.TASK_CREATED,
            topic="agent.123",
            payload={"task_id": "task-456"},
            source="test-service",
            metadata={"key": "value"},
        )

        event_dict = event.to_dict()

        assert event_dict["event_id"] == event.event_id
        assert event_dict["event_type"] == EventType.TASK_CREATED.value
        assert event_dict["topic"] == "agent.123"
        assert event_dict["payload"] == {"task_id": "task-456"}
        assert event_dict["source"] == "test-service"
        assert event_dict["metadata"] == {"key": "value"}
        assert "timestamp" in event_dict
