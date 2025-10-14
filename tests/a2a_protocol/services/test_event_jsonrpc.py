"""
Unit tests for Event System JSON-RPC Service.

Tests for event JSON-RPC method handlers covering event publishing,
subscriptions, history, and dead letter queue management.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.a2a_protocol.models.events import (
    DeadLetterMessage,
    Event,
    EventPriority,
    EventPublishResponse,
    EventSubscribeResponse,
    EventSubscription,
    EventType,
)
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest


class TestEventPublish:
    """Test event.publish JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_publish_event_success(self, mock_manager):
        """Test successful event publishing."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_publish_event

        publish_response = EventPublishResponse(
            success=True,
            event_id="event-123",
            subscribers_notified=5,
        )
        mock_manager.publish_event = AsyncMock(return_value=publish_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.publish",
            params={
                "event_type": "agent.status_changed",
                "source": "test-agent",
                "data": {"status": "active"},
            },
            id="1",
        )

        result = await handle_publish_event(request)

        assert result["event_id"] == "event-123"
        assert result["subscribers_notified"] == 5
        mock_manager.publish_event.assert_called_once()

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_publish_event_with_priority(self, mock_manager):
        """Test publishing event with priority."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_publish_event

        publish_response = EventPublishResponse(
            success=True,
            event_id="event-123",
            subscribers_notified=3,
        )
        mock_manager.publish_event = AsyncMock(return_value=publish_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.publish",
            params={
                "event_type": "task.created",
                "source": "task-service",
                "data": {"task_id": "task-123"},
                "priority": "high",
                "correlation_id": "corr-123",
            },
            id="1",
        )

        result = await handle_publish_event(request)

        assert result["event_id"] == "event-123"

    @pytest.mark.asyncio
    async def test_publish_event_missing_params(self):
        """Test event publishing with missing parameters."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_publish_event

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.publish",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_publish_event(request)

    @pytest.mark.asyncio
    async def test_publish_event_missing_required_fields(self):
        """Test event publishing with missing required fields."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_publish_event

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.publish",
            params={"event_type": "agent.status_changed"},
            id="1",
        )

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_publish_event(request)


class TestEventSubscribe:
    """Test event.subscribe JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_subscribe_success(self, mock_manager):
        """Test successful event subscription."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_subscribe

        subscribe_response = EventSubscribeResponse(
            success=True,
            subscription_id="sub-123",
        )
        mock_manager.subscribe = AsyncMock(return_value=subscribe_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.subscribe",
            params={
                "subscriber_id": "subscriber-123",
                "event_types": ["agent.status_changed"],
            },
            id="1",
        )

        result = await handle_subscribe(request)

        assert result["subscription_id"] == "sub-123"
        assert result["success"] is True
        mock_manager.subscribe.assert_called_once()

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_subscribe_with_filters(self, mock_manager):
        """Test event subscription with filters."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_subscribe

        subscribe_response = EventSubscribeResponse(
            success=True,
            subscription_id="sub-123",
        )
        mock_manager.subscribe = AsyncMock(return_value=subscribe_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.subscribe",
            params={
                "subscriber_id": "subscriber-123",
                "event_types": ["task.created", "task.completed"],
                "filters": {"status": "active"},
                "ttl_seconds": 3600,
            },
            id="1",
        )

        result = await handle_subscribe(request)

        assert result["subscription_id"] == "sub-123"

    @pytest.mark.asyncio
    async def test_subscribe_missing_params(self):
        """Test event subscription with missing parameters."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_subscribe

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.subscribe",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_subscribe(request)

    @pytest.mark.asyncio
    async def test_subscribe_missing_subscriber_id(self):
        """Test event subscription with missing subscriber_id."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_subscribe

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.subscribe",
            params={"event_types": ["agent.status_changed"]},
            id="1",
        )

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_subscribe(request)


class TestEventUnsubscribe:
    """Test event.unsubscribe JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_unsubscribe_success(self, mock_manager):
        """Test successful unsubscription."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_unsubscribe

        mock_manager.unsubscribe = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.unsubscribe",
            params={"subscription_id": "sub-123"},
            id="1",
        )

        result = await handle_unsubscribe(request)

        assert result["success"] is True
        assert result["subscription_id"] == "sub-123"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_unsubscribe_not_found(self, mock_manager):
        """Test unsubscription when subscription not found."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_unsubscribe

        mock_manager.unsubscribe = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.unsubscribe",
            params={"subscription_id": "nonexistent"},
            id="1",
        )

        with pytest.raises(ValueError, match="Subscription not found"):
            await handle_unsubscribe(request)

    @pytest.mark.asyncio
    async def test_unsubscribe_missing_params(self):
        """Test unsubscription with missing parameters."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_unsubscribe

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.unsubscribe",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required"):
            await handle_unsubscribe(request)


class TestListSubscriptions:
    """Test event.list_subscriptions JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_list_subscriptions_all(self, mock_manager):
        """Test listing all subscriptions."""
        from agentcore.a2a_protocol.services.event_jsonrpc import (
            handle_list_subscriptions,
        )

        subscriptions = [
            EventSubscription(
                subscription_id="sub-1",
                subscriber_id="subscriber-1",
                event_types=[EventType.AGENT_STATUS_CHANGED],
            ),
            EventSubscription(
                subscription_id="sub-2",
                subscriber_id="subscriber-2",
                event_types=[EventType.TASK_CREATED],
            ),
        ]
        mock_manager.get_subscriptions = Mock(return_value=subscriptions)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.list_subscriptions",
            params={},
            id="1",
        )

        result = await handle_list_subscriptions(request)

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["subscriptions"]) == 2

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_list_subscriptions_by_subscriber(self, mock_manager):
        """Test listing subscriptions for specific subscriber."""
        from agentcore.a2a_protocol.services.event_jsonrpc import (
            handle_list_subscriptions,
        )

        subscriptions = [
            EventSubscription(
                subscription_id="sub-1",
                subscriber_id="subscriber-1",
                event_types=[EventType.AGENT_STATUS_CHANGED],
            ),
        ]
        mock_manager.get_subscriptions = Mock(return_value=subscriptions)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.list_subscriptions",
            params={"subscriber_id": "subscriber-1"},
            id="1",
        )

        result = await handle_list_subscriptions(request)

        assert result["count"] == 1
        mock_manager.get_subscriptions.assert_called_once_with("subscriber-1")


class TestEventHistory:
    """Test event.get_history JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_get_history_all(self, mock_manager):
        """Test getting all event history."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_get_history

        events = [
            Event(
                event_id="event-1",
                event_type=EventType.AGENT_STATUS_CHANGED,
                source="agent-1",
                data={"status": "active"},
                timestamp=datetime.now(UTC),
            ),
        ]
        mock_manager.get_event_history = Mock(return_value=events)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.get_history",
            params={},
            id="1",
        )

        result = await handle_get_history(request)

        assert result["success"] is True
        assert result["count"] == 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_get_history_filtered(self, mock_manager):
        """Test getting filtered event history."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_get_history

        events = [
            Event(
                event_id="event-1",
                event_type=EventType.TASK_CREATED,
                source="task-service",
                data={"task_id": "task-123"},
                timestamp=datetime.now(UTC),
            ),
        ]
        mock_manager.get_event_history = Mock(return_value=events)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.get_history",
            params={"event_type": "task.created", "limit": 50},
            id="1",
        )

        result = await handle_get_history(request)

        assert result["count"] == 1


class TestDeadLetterQueue:
    """Test dead letter queue JSON-RPC methods."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_get_dead_letter_queue(self, mock_manager):
        """Test getting dead letter queue messages."""
        from agentcore.a2a_protocol.services.event_jsonrpc import (
            handle_get_dead_letter_queue,
        )

        messages = [
            DeadLetterMessage(
                message_id="msg-1",
                event=Event(
                    event_id="event-1",
                    event_type=EventType.AGENT_STATUS_CHANGED,
                    source="agent-1",
                    data={"status": "error"},
                    timestamp=datetime.now(UTC),
                ),
                subscriber_id="subscriber-1",
                failure_reason="Connection timeout",
                retry_count=3,
                failed_at=datetime.now(UTC),
            ),
        ]
        mock_manager.get_dead_letter_messages = Mock(return_value=messages)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.get_dead_letter_queue",
            params={"limit": 50},
            id="1",
        )

        result = await handle_get_dead_letter_queue(request)

        assert result["success"] is True
        assert result["count"] == 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_retry_dead_letter_success(self, mock_manager):
        """Test successful retry of dead letter message."""
        from agentcore.a2a_protocol.services.event_jsonrpc import (
            handle_retry_dead_letter,
        )

        mock_manager.retry_dead_letter_message = AsyncMock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.retry_dead_letter",
            params={"message_id": "msg-123"},
            id="1",
        )

        result = await handle_retry_dead_letter(request)

        assert result["success"] is True
        assert result["message_id"] == "msg-123"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_retry_dead_letter_failure(self, mock_manager):
        """Test failed retry of dead letter message."""
        from agentcore.a2a_protocol.services.event_jsonrpc import (
            handle_retry_dead_letter,
        )

        mock_manager.retry_dead_letter_message = AsyncMock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.retry_dead_letter",
            params={"message_id": "nonexistent"},
            id="1",
        )

        with pytest.raises(ValueError, match="Retry failed"):
            await handle_retry_dead_letter(request)

    @pytest.mark.asyncio
    async def test_retry_dead_letter_missing_params(self):
        """Test retry dead letter with missing parameters."""
        from agentcore.a2a_protocol.services.event_jsonrpc import (
            handle_retry_dead_letter,
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.retry_dead_letter",
            params=None,
            id="1",
        )

        with pytest.raises(ValueError, match="Parameter required"):
            await handle_retry_dead_letter(request)


class TestEventStats:
    """Test event.get_stats JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_get_event_stats(self, mock_manager):
        """Test getting event statistics."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_get_event_stats

        stats = {
            "total_events": 1000,
            "active_subscriptions": 25,
            "dead_letter_count": 5,
        }
        mock_manager.get_statistics = Mock(return_value=stats)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.get_stats",
            params={},
            id="1",
        )

        result = await handle_get_event_stats(request)

        assert result["success"] is True
        assert result["stats"] == stats
        assert "timestamp" in result


class TestEventCleanup:
    """Test event.cleanup_expired JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.event_jsonrpc.event_manager")
    async def test_cleanup_expired(self, mock_manager):
        """Test cleaning up expired subscriptions and connections."""
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_cleanup_expired

        mock_manager.cleanup_expired_subscriptions = AsyncMock(return_value=5)
        mock_manager.cleanup_dead_connections = AsyncMock(return_value=3)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.cleanup_expired",
            params={},
            id="1",
        )

        result = await handle_cleanup_expired(request)

        assert result["success"] is True
        assert result["subscriptions_removed"] == 5
        assert result["connections_removed"] == 3
        from agentcore.a2a_protocol.services.event_jsonrpc import handle_cleanup_expired

        mock_manager.cleanup_expired_subscriptions = AsyncMock(return_value=5)
        mock_manager.cleanup_dead_connections = AsyncMock(return_value=3)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.cleanup_expired",
            params={},
            id="1",
        )

        result = await handle_cleanup_expired(request)

        assert result["success"] is True
        assert result["subscriptions_removed"] == 5
        assert result["connections_removed"] == 3

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="event.cleanup_expired",
            params={},
            id="1",
        )

        result = await handle_cleanup_expired(request)

        assert result["success"] is True
        assert result["subscriptions_removed"] == 5
        assert result["connections_removed"] == 3
