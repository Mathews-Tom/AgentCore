"""
Comprehensive test suite for EventManager service.

Tests event publishing, subscriptions, WebSocket connections, dead letter queue,
event history, and cleanup operations.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

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
from agentcore.a2a_protocol.services.event_manager import (
    EventManager,
    WebSocketConnection,
    event_manager,
)


@pytest.fixture
def manager():
    """Create fresh EventManager instance for each test."""
    return EventManager()


@pytest.fixture
def sample_event():
    """Create sample event for testing."""
    return Event(
        event_type=EventType.AGENT_REGISTERED,
        source="test-agent",
        data={"status": "active"},
        priority=EventPriority.NORMAL,
    )


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


# ==================== Event Publishing Tests ====================


class TestEventPublishing:
    """Test event publishing functionality."""

    @pytest.mark.asyncio
    async def test_publish_event_success(self, manager, sample_event):
        """Test publishing event successfully."""
        response = await manager.publish_event(
            event_type=EventType.AGENT_REGISTERED,
            source="test-agent",
            data={"status": "active"},
        )

        assert isinstance(response, EventPublishResponse)
        assert response.success is True
        assert response.event_id is not None
        assert response.subscribers_notified == 0  # No subscribers yet
        assert manager._event_stats["total_published"] == 1

    @pytest.mark.asyncio
    async def test_publish_event_with_priority(self, manager):
        """Test publishing event with high priority."""
        response = await manager.publish_event(
            event_type=EventType.TASK_CREATED,
            source="test-source",
            data={"task_id": "123"},
            priority=EventPriority.HIGH,
        )

        assert response.success is True
        event = manager._event_history[-1]
        assert event.priority == EventPriority.HIGH

    @pytest.mark.asyncio
    async def test_publish_event_with_correlation_id(self, manager):
        """Test publishing event with correlation ID."""
        correlation_id = str(uuid4())

        response = await manager.publish_event(
            event_type=EventType.TASK_COMPLETED,
            source="test-source",
            data={"result": "success"},
            correlation_id=correlation_id,
        )

        assert response.success is True
        event = manager._event_history[-1]
        assert event.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_publish_event_with_metadata(self, manager):
        """Test publishing event with metadata."""
        metadata = {"version": "1.0", "environment": "test"}

        response = await manager.publish_event(
            event_type=EventType.SYSTEM_STARTUP,
            source="system",
            data={},
            metadata=metadata,
        )

        assert response.success is True
        event = manager._event_history[-1]
        assert event.metadata == metadata

    @pytest.mark.asyncio
    async def test_publish_event_adds_to_history(self, manager):
        """Test event is added to history."""
        initial_count = len(manager._event_history)

        await manager.publish_event(
            event_type=EventType.MESSAGE_ROUTED,
            source="router",
            data={"target": "agent-1"},
        )

        assert len(manager._event_history) == initial_count + 1

    @pytest.mark.asyncio
    async def test_publish_event_increments_stats(self, manager):
        """Test event publishing increments statistics."""
        initial_published = manager._event_stats["total_published"]

        await manager.publish_event(
            event_type=EventType.AGENT_STATUS_CHANGED,
            source="agent-1",
            data={"old_status": "active", "new_status": "idle"},
        )

        assert manager._event_stats["total_published"] == initial_published + 1

    @pytest.mark.asyncio
    async def test_publish_event_notifies_subscribers(self, manager, mock_websocket):
        """Test event notification to subscribers."""
        # Create subscription
        await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.TASK_STARTED]
        )

        # Register WebSocket
        await manager.register_websocket(mock_websocket, "sub-1")

        # Publish event
        response = await manager.publish_event(
            event_type=EventType.TASK_STARTED,
            source="task-manager",
            data={"task_id": "123"},
        )

        assert response.subscribers_notified == 1
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_event_no_matching_subscribers(self, manager):
        """Test publishing event with no matching subscribers."""
        # Create subscription for different event type
        await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.AGENT_REGISTERED]
        )

        # Publish different event type
        response = await manager.publish_event(
            event_type=EventType.TASK_CREATED, source="test", data={}
        )

        assert response.subscribers_notified == 0

    @pytest.mark.asyncio
    async def test_publish_event_executes_hooks(self, manager):
        """Test event hooks are executed during publishing."""
        hook_called = []

        def sync_hook(event: Event):
            hook_called.append("sync")

        async def async_hook(event: Event):
            hook_called.append("async")

        # Register hooks
        manager.register_hook(EventType.SYSTEM_STARTUP, sync_hook)
        manager.register_hook(EventType.SYSTEM_STARTUP, async_hook)

        # Publish event
        await manager.publish_event(
            event_type=EventType.SYSTEM_STARTUP, source="system", data={}
        )

        assert "sync" in hook_called
        assert "async" in hook_called

    @pytest.mark.asyncio
    async def test_publish_event_hook_failure_does_not_block(self, manager):
        """Test failed hook does not block event publishing."""

        def failing_hook(event: Event):
            raise ValueError("Hook failure")

        manager.register_hook(EventType.AGENT_REGISTERED, failing_hook)

        # Should not raise exception
        response = await manager.publish_event(
            event_type=EventType.AGENT_REGISTERED, source="test", data={}
        )

        assert response.success is True


# ==================== Subscription Management Tests ====================


class TestSubscriptionManagement:
    """Test subscription management functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_success(self, manager):
        """Test creating subscription successfully."""
        response = await manager.subscribe(
            subscriber_id="sub-1",
            event_types=[EventType.AGENT_REGISTERED, EventType.AGENT_UNREGISTERED],
        )

        assert isinstance(response, EventSubscribeResponse)
        assert response.success is True
        assert response.subscription_id is not None
        assert len(manager._subscriptions) == 1

    @pytest.mark.asyncio
    async def test_subscribe_with_filters(self, manager):
        """Test subscription with event filters."""
        filters = {"status": "active", "priority": "high"}

        response = await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.TASK_CREATED], filters=filters
        )

        assert response.success is True
        subscription = manager._subscriptions[response.subscription_id]
        assert subscription.filters == filters

    @pytest.mark.asyncio
    async def test_subscribe_with_ttl(self, manager):
        """Test subscription with TTL."""
        ttl_seconds = 300

        response = await manager.subscribe(
            subscriber_id="sub-1",
            event_types=[EventType.MESSAGE_DELIVERED],
            ttl_seconds=ttl_seconds,
        )

        subscription = manager._subscriptions[response.subscription_id]
        assert subscription.expires_at is not None
        assert subscription.expires_at > datetime.now(UTC)

    @pytest.mark.asyncio
    async def test_subscribe_updates_indexes(self, manager):
        """Test subscription updates internal indexes."""
        response = await manager.subscribe(
            subscriber_id="sub-1",
            event_types=[EventType.AGENT_REGISTERED, EventType.TASK_CREATED],
        )

        # Check by-type index
        assert (
            response.subscription_id
            in manager._subscriptions_by_type[EventType.AGENT_REGISTERED]
        )
        assert (
            response.subscription_id
            in manager._subscriptions_by_type[EventType.TASK_CREATED]
        )

        # Check by-subscriber index
        assert response.subscription_id in manager._subscriptions_by_subscriber["sub-1"]

    @pytest.mark.asyncio
    async def test_subscribe_updates_stats(self, manager):
        """Test subscription updates statistics."""
        initial_count = manager._event_stats["active_subscriptions"]

        await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.SYSTEM_ERROR]
        )

        assert manager._event_stats["active_subscriptions"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_unsubscribe_success(self, manager):
        """Test unsubscribing successfully."""
        # Create subscription
        response = await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.AGENT_REGISTERED]
        )

        # Unsubscribe
        result = await manager.unsubscribe(response.subscription_id)

        assert result is True
        assert response.subscription_id not in manager._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, manager):
        """Test unsubscribing nonexistent subscription."""
        result = await manager.unsubscribe("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_from_indexes(self, manager):
        """Test unsubscribe removes from all indexes."""
        # Create subscription
        response = await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.TASK_COMPLETED]
        )

        # Unsubscribe
        await manager.unsubscribe(response.subscription_id)

        # Check indexes
        assert (
            response.subscription_id
            not in manager._subscriptions_by_type[EventType.TASK_COMPLETED]
        )
        assert (
            response.subscription_id
            not in manager._subscriptions_by_subscriber["sub-1"]
        )

    @pytest.mark.asyncio
    async def test_get_subscriptions_all(self, manager):
        """Test getting all subscriptions."""
        await manager.subscribe("sub-1", [EventType.AGENT_REGISTERED])
        await manager.subscribe("sub-2", [EventType.TASK_CREATED])

        subscriptions = manager.get_subscriptions()
        assert len(subscriptions) == 2

    @pytest.mark.asyncio
    async def test_get_subscriptions_by_subscriber(self, manager):
        """Test getting subscriptions for specific subscriber."""
        await manager.subscribe("sub-1", [EventType.AGENT_REGISTERED])
        await manager.subscribe("sub-1", [EventType.TASK_CREATED])
        await manager.subscribe("sub-2", [EventType.MESSAGE_ROUTED])

        subscriptions = manager.get_subscriptions(subscriber_id="sub-1")
        assert len(subscriptions) == 2
        assert all(sub.subscriber_id == "sub-1" for sub in subscriptions)


# ==================== WebSocket Connection Tests ====================


class TestWebSocketConnections:
    """Test WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_register_websocket_success(self, manager, mock_websocket):
        """Test registering WebSocket connection."""
        connection_id = await manager.register_websocket(
            mock_websocket, subscriber_id="sub-1"
        )

        assert connection_id is not None
        assert connection_id in manager._websocket_connections
        assert manager._event_stats["active_websockets"] == 1

    @pytest.mark.asyncio
    async def test_register_websocket_updates_indexes(self, manager, mock_websocket):
        """Test WebSocket registration updates indexes."""
        connection_id = await manager.register_websocket(
            mock_websocket, subscriber_id="sub-1"
        )

        assert connection_id in manager._connections_by_subscriber["sub-1"]

    @pytest.mark.asyncio
    async def test_websocket_connection_send_event(self, mock_websocket):
        """Test sending event through WebSocket connection."""
        connection = WebSocketConnection(
            connection_id="conn-1", websocket=mock_websocket, subscriber_id="sub-1"
        )

        event = Event(
            event_type=EventType.TASK_STARTED, source="test", data={"task_id": "123"}
        )

        result = await connection.send_event(event)

        assert result is True
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_connection_send_event_failure(self):
        """Test WebSocket send failure marks connection as dead."""
        failing_websocket = AsyncMock()
        failing_websocket.send_json = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        connection = WebSocketConnection(
            connection_id="conn-1", websocket=failing_websocket, subscriber_id="sub-1"
        )

        event = Event(event_type=EventType.SYSTEM_ERROR, source="test", data={})

        result = await connection.send_event(event)

        assert result is False
        assert connection.is_alive is False

    @pytest.mark.asyncio
    async def test_websocket_connection_send_ping(self, mock_websocket):
        """Test sending ping through WebSocket."""
        connection = WebSocketConnection(
            connection_id="conn-1", websocket=mock_websocket, subscriber_id="sub-1"
        )

        initial_ping = connection.last_ping
        await asyncio.sleep(0.01)  # Small delay

        result = await connection.send_ping()

        assert result is True
        assert connection.last_ping > initial_ping
        mock_websocket.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_connection_send_ping_failure(self):
        """Test ping failure marks connection as dead."""
        failing_websocket = AsyncMock()
        failing_websocket.send_json = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        connection = WebSocketConnection(
            connection_id="conn-1", websocket=failing_websocket, subscriber_id="sub-1"
        )

        result = await connection.send_ping()

        assert result is False
        assert connection.is_alive is False

    @pytest.mark.asyncio
    async def test_websocket_connection_close(self, mock_websocket):
        """Test closing WebSocket connection."""
        connection = WebSocketConnection(
            connection_id="conn-1", websocket=mock_websocket, subscriber_id="sub-1"
        )

        await connection.close()

        assert connection.is_alive is False
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_connection(self, manager, mock_websocket):
        """Test closing connection by ID."""
        connection_id = await manager.register_websocket(mock_websocket, "sub-1")

        await manager.close_connection(connection_id)

        assert connection_id not in manager._websocket_connections
        assert manager._event_stats["active_websockets"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_dead_connections(self, manager):
        """Test cleaning up dead connections."""
        # Register connections
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        conn_id1 = await manager.register_websocket(ws1, "sub-1")
        conn_id2 = await manager.register_websocket(ws2, "sub-2")

        # Mark one as dead
        manager._websocket_connections[conn_id1].is_alive = False

        # Cleanup
        removed_count = await manager.cleanup_dead_connections()

        assert removed_count == 1
        assert conn_id1 not in manager._websocket_connections
        assert conn_id2 in manager._websocket_connections


# ==================== Dead Letter Queue Tests ====================


class TestDeadLetterQueue:
    """Test dead letter queue functionality."""

    @pytest.mark.asyncio
    async def test_add_to_dead_letter_queue(self, manager, sample_event):
        """Test adding message to dead letter queue."""
        await manager._add_to_dead_letter_queue(
            event=sample_event,
            subscriber_id="sub-1",
            failure_reason="No active connections",
        )

        assert len(manager._dead_letter_queue) == 1
        assert manager._event_stats["total_failed"] == 1
        assert manager._event_stats["dead_letter_messages"] == 1

    @pytest.mark.asyncio
    async def test_dead_letter_queue_for_offline_agent(self, manager):
        """Test DLQ when subscriber has no active connections."""
        # Create subscription without WebSocket
        await manager.subscribe(
            subscriber_id="offline-sub", event_types=[EventType.TASK_CREATED]
        )

        # Publish event
        response = await manager.publish_event(
            event_type=EventType.TASK_CREATED, source="test", data={}
        )

        # Should be added to DLQ
        assert len(manager._dead_letter_queue) > 0
        assert response.subscribers_notified == 0

    @pytest.mark.asyncio
    async def test_get_dead_letter_messages(self, manager, sample_event):
        """Test retrieving dead letter messages."""
        # Add multiple messages
        for i in range(5):
            await manager._add_to_dead_letter_queue(
                event=sample_event,
                subscriber_id=f"sub-{i}",
                failure_reason="Test failure",
            )

        messages = manager.get_dead_letter_messages(limit=3)
        assert len(messages) == 3

    @pytest.mark.asyncio
    async def test_retry_dead_letter_message_success(self, manager, mock_websocket):
        """Test successful retry of dead letter message."""
        # Create subscription and register WebSocket
        await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.AGENT_REGISTERED]
        )
        await manager.register_websocket(mock_websocket, "sub-1")

        # Add message to DLQ
        event = Event(event_type=EventType.AGENT_REGISTERED, source="test", data={})
        await manager._add_to_dead_letter_queue(
            event=event, subscriber_id="sub-1", failure_reason="Temporary failure"
        )

        # Get message ID
        message = manager._dead_letter_queue[0]
        message_id = message.message_id

        # Retry
        result = await manager.retry_dead_letter_message(message_id)

        assert result is True
        assert len(manager._dead_letter_queue) == 0

    @pytest.mark.asyncio
    async def test_retry_dead_letter_message_not_found(self, manager):
        """Test retrying nonexistent message."""
        result = await manager.retry_dead_letter_message("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_dead_letter_message_max_retries_exceeded(
        self, manager, sample_event
    ):
        """Test retry fails when max retries exceeded."""
        # Add message to DLQ
        await manager._add_to_dead_letter_queue(
            event=sample_event, subscriber_id="sub-1", failure_reason="Test"
        )

        message = manager._dead_letter_queue[0]
        message.retry_count = message.max_retries  # Exceed limit

        result = await manager.retry_dead_letter_message(message.message_id)
        assert result is False

    @pytest.mark.asyncio
    async def test_retry_dead_letter_message_no_subscription(
        self, manager, sample_event
    ):
        """Test retry fails when no matching subscription."""
        # Add message without subscription
        await manager._add_to_dead_letter_queue(
            event=sample_event, subscriber_id="sub-1", failure_reason="Test"
        )

        message = manager._dead_letter_queue[0]
        result = await manager.retry_dead_letter_message(message.message_id)

        assert result is False


# ==================== Event History Tests ====================


class TestEventHistory:
    """Test event history functionality."""

    @pytest.mark.asyncio
    async def test_get_event_history_all(self, manager):
        """Test getting all event history."""
        # Publish multiple events
        for i in range(5):
            await manager.publish_event(
                event_type=EventType.AGENT_REGISTERED, source=f"agent-{i}", data={}
            )

        history = manager.get_event_history()
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_get_event_history_filtered_by_type(self, manager):
        """Test filtering event history by type."""
        # Publish different event types
        await manager.publish_event(
            event_type=EventType.AGENT_REGISTERED, source="agent-1", data={}
        )
        await manager.publish_event(
            event_type=EventType.TASK_CREATED, source="task-manager", data={}
        )
        await manager.publish_event(
            event_type=EventType.AGENT_REGISTERED, source="agent-2", data={}
        )

        history = manager.get_event_history(event_type=EventType.AGENT_REGISTERED)
        assert len(history) == 2
        assert all(e.event_type == EventType.AGENT_REGISTERED for e in history)

    @pytest.mark.asyncio
    async def test_get_event_history_with_limit(self, manager):
        """Test limiting event history results."""
        # Publish many events
        for i in range(20):
            await manager.publish_event(
                event_type=EventType.SYSTEM_STARTUP, source="system", data={}
            )

        history = manager.get_event_history(limit=10)
        assert len(history) == 10

    @pytest.mark.asyncio
    async def test_event_history_bounded_size(self, manager):
        """Test event history maintains maximum size."""
        # The deque has maxlen=1000
        # Publish more than maxlen events
        for i in range(1100):
            await manager.publish_event(
                event_type=EventType.MESSAGE_ROUTED, source="router", data={"index": i}
            )

        assert len(manager._event_history) == 1000

    @pytest.mark.asyncio
    async def test_event_history_order(self, manager):
        """Test event history maintains chronological order."""
        event_ids = []
        for i in range(5):
            response = await manager.publish_event(
                event_type=EventType.TASK_CREATED, source="test", data={"index": i}
            )
            event_ids.append(response.event_id)

        history = manager.get_event_history()
        history_ids = [e.event_id for e in history]

        assert history_ids == event_ids


# ==================== Cleanup Operations Tests ====================


class TestCleanupOperations:
    """Test cleanup and maintenance operations."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_subscriptions(self, manager):
        """Test cleaning up expired subscriptions."""
        # Create subscription that is already expired
        expired_time = datetime.now(UTC) - timedelta(seconds=10)

        subscription = EventSubscription(
            subscriber_id="sub-1",
            event_types=[EventType.AGENT_REGISTERED],
            expires_at=expired_time,
        )

        manager._subscriptions[subscription.subscription_id] = subscription
        manager._subscriptions_by_type[EventType.AGENT_REGISTERED].add(
            subscription.subscription_id
        )
        manager._subscriptions_by_subscriber["sub-1"].add(subscription.subscription_id)

        # Cleanup
        removed_count = await manager.cleanup_expired_subscriptions()

        assert removed_count == 1
        assert subscription.subscription_id not in manager._subscriptions

    @pytest.mark.asyncio
    async def test_cleanup_expired_subscriptions_keeps_active(self, manager):
        """Test cleanup preserves active subscriptions."""
        # Create active subscription
        await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.TASK_COMPLETED]
        )

        initial_count = len(manager._subscriptions)
        removed_count = await manager.cleanup_expired_subscriptions()

        assert removed_count == 0
        assert len(manager._subscriptions) == initial_count

    @pytest.mark.asyncio
    async def test_cleanup_expired_subscriptions_with_ttl(self, manager):
        """Test cleanup of TTL-based subscriptions."""
        # Create subscription with very short TTL
        response = await manager.subscribe(
            subscriber_id="sub-1",
            event_types=[EventType.MESSAGE_DELIVERED],
            ttl_seconds=1,  # 1 second TTL
        )

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Cleanup
        removed_count = await manager.cleanup_expired_subscriptions()
        assert removed_count == 1


# ==================== Statistics Tests ====================


class TestStatistics:
    """Test statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, manager):
        """Test getting event system statistics."""
        # Publish some events
        await manager.publish_event(
            event_type=EventType.AGENT_REGISTERED, source="test", data={}
        )

        # Create subscription
        await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.AGENT_REGISTERED]
        )

        stats = manager.get_statistics()

        assert "total_published" in stats
        assert "total_delivered" in stats
        assert "active_subscriptions" in stats
        assert "active_websockets" in stats
        assert "event_history_size" in stats
        assert stats["total_published"] >= 1

    @pytest.mark.asyncio
    async def test_statistics_subscriptions_by_type(self, manager):
        """Test statistics include subscriptions by type."""
        # Create subscriptions for different types
        await manager.subscribe(
            subscriber_id="sub-1",
            event_types=[EventType.AGENT_REGISTERED, EventType.TASK_CREATED],
        )
        await manager.subscribe(
            subscriber_id="sub-2", event_types=[EventType.AGENT_REGISTERED]
        )

        stats = manager.get_statistics()

        assert "subscriptions_by_type" in stats
        assert EventType.AGENT_REGISTERED.value in stats["subscriptions_by_type"]

    @pytest.mark.asyncio
    async def test_statistics_delivered_count(self, manager, mock_websocket):
        """Test statistics track delivered events."""
        # Setup subscription and WebSocket
        await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.TASK_STARTED]
        )
        await manager.register_websocket(mock_websocket, "sub-1")

        initial_delivered = manager._event_stats["total_delivered"]

        # Publish event
        await manager.publish_event(
            event_type=EventType.TASK_STARTED, source="test", data={}
        )

        assert manager._event_stats["total_delivered"] == initial_delivered + 1

    @pytest.mark.asyncio
    async def test_statistics_failed_count(self, manager, sample_event):
        """Test statistics track failed deliveries."""
        initial_failed = manager._event_stats["total_failed"]

        # Add to DLQ
        await manager._add_to_dead_letter_queue(
            event=sample_event, subscriber_id="sub-1", failure_reason="Test failure"
        )

        assert manager._event_stats["total_failed"] == initial_failed + 1


# ==================== Event Matching Tests ====================


class TestEventMatching:
    """Test event matching with filters."""

    @pytest.mark.asyncio
    async def test_subscription_matches_event_type(self, manager):
        """Test subscription matches by event type."""
        subscription = EventSubscription(
            subscriber_id="sub-1", event_types=[EventType.AGENT_REGISTERED]
        )

        event = Event(event_type=EventType.AGENT_REGISTERED, source="test", data={})

        assert subscription.matches_event(event) is True

    @pytest.mark.asyncio
    async def test_subscription_does_not_match_different_type(self, manager):
        """Test subscription does not match different event type."""
        subscription = EventSubscription(
            subscriber_id="sub-1", event_types=[EventType.AGENT_REGISTERED]
        )

        event = Event(event_type=EventType.TASK_CREATED, source="test", data={})

        assert subscription.matches_event(event) is False

    @pytest.mark.asyncio
    async def test_subscription_matches_with_filters(self, manager):
        """Test subscription matches with data filters."""
        subscription = EventSubscription(
            subscriber_id="sub-1",
            event_types=[EventType.AGENT_STATUS_CHANGED],
            filters={"new_status": "active"},
        )

        event = Event(
            event_type=EventType.AGENT_STATUS_CHANGED,
            source="agent-1",
            data={"old_status": "idle", "new_status": "active"},
        )

        assert subscription.matches_event(event) is True

    @pytest.mark.asyncio
    async def test_subscription_does_not_match_filters(self, manager):
        """Test subscription does not match when filters don't match."""
        subscription = EventSubscription(
            subscriber_id="sub-1",
            event_types=[EventType.AGENT_STATUS_CHANGED],
            filters={"new_status": "active"},
        )

        event = Event(
            event_type=EventType.AGENT_STATUS_CHANGED,
            source="agent-1",
            data={"old_status": "idle", "new_status": "error"},
        )

        assert subscription.matches_event(event) is False

    @pytest.mark.asyncio
    async def test_find_matching_subscriptions_filters_inactive(self, manager):
        """Test find_matching_subscriptions filters out inactive subscriptions."""
        # Create active and inactive subscriptions
        response1 = await manager.subscribe(
            subscriber_id="sub-1", event_types=[EventType.TASK_CREATED]
        )
        response2 = await manager.subscribe(
            subscriber_id="sub-2", event_types=[EventType.TASK_CREATED]
        )

        # Mark one as inactive
        manager._subscriptions[response2.subscription_id].active = False

        event = Event(event_type=EventType.TASK_CREATED, source="test", data={})

        matching = manager._find_matching_subscriptions(event)
        assert len(matching) == 1
        assert matching[0].subscription_id == response1.subscription_id


# ==================== Global Instance Test ====================


class TestGlobalInstance:
    """Test global event manager instance."""

    def test_global_instance_exists(self):
        """Test global event_manager instance exists."""
        assert event_manager is not None
        assert isinstance(event_manager, EventManager)

    def test_global_instance_is_singleton(self):
        """Test global instance behaves like singleton."""
        from agentcore.a2a_protocol.services.event_manager import event_manager as em1
        from agentcore.a2a_protocol.services.event_manager import event_manager as em2

        assert em1 is em2
