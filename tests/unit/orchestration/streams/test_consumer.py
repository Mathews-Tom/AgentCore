"""
Unit tests for Stream Consumer.

Tests event consumption, handler registration, and error handling using mocks.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, call
from uuid import uuid4

import pytest
import redis.asyncio as aioredis

from agentcore.orchestration.streams.client import RedisStreamsClient
from agentcore.orchestration.streams.config import StreamConfig
from agentcore.orchestration.streams.consumer import ConsumerGroup, StreamConsumer
from agentcore.orchestration.streams.models import EventType, TaskCreatedEvent


class TestConsumerGroup:
    """Test suite for ConsumerGroup."""

    def test_create_consumer_group(self) -> None:
        """Test creating consumer group."""
        group = ConsumerGroup("test-group", "consumer-1")

        assert group.group_name == "test-group"
        assert group.consumer_name == "consumer-1"


class TestStreamConsumer:
    """Test suite for StreamConsumer."""

    @pytest.fixture
    def config(self) -> StreamConfig:
        """Create test configuration."""
        return StreamConfig(
            stream_name="test:events",
            consumer_group_name="test-group",
            consumer_name="test-consumer",
            count=10,
            block_ms=1000,
            enable_auto_claim=True,
            auto_claim_idle_ms=30000)

    @pytest.fixture
    def consumer_group(self) -> ConsumerGroup:
        """Create consumer group."""
        return ConsumerGroup("test-group", "consumer-1")

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create mock Redis client."""
        mock = AsyncMock(spec=RedisStreamsClient)
        mock.client = AsyncMock()
        mock.create_consumer_group = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def consumer(
        self,
        mock_client: AsyncMock,
        consumer_group: ConsumerGroup,
        config: StreamConfig) -> StreamConsumer:
        """Create stream consumer instance."""
        return StreamConsumer(mock_client, consumer_group, config)

    def test_register_handler(
        self, consumer: StreamConsumer
    ) -> None:
        """Test registering event handler."""
        handler_called = []

        def handler(event):
            handler_called.append(event)

        consumer.register_handler(EventType.TASK_CREATED, handler)

        assert EventType.TASK_CREATED in consumer._handlers
        assert len(consumer._handlers[EventType.TASK_CREATED]) == 1

    def test_register_multiple_handlers(
        self, consumer: StreamConsumer
    ) -> None:
        """Test registering multiple handlers for same event type."""
        handler1 = lambda e: None
        handler2 = lambda e: None

        consumer.register_handler(EventType.TASK_CREATED, handler1)
        consumer.register_handler(EventType.TASK_CREATED, handler2)

        assert len(consumer._handlers[EventType.TASK_CREATED]) == 2

    @pytest.mark.asyncio
    async def test_start_creates_consumer_group(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test that start creates consumer group."""
        # Mock empty stream reads and make xreadgroup raise CancelledError after first call
        call_count = 0

        async def xreadgroup_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError()
            return []

        mock_client.client.xreadgroup = AsyncMock(side_effect=xreadgroup_side_effect)
        mock_client.client.xautoclaim = AsyncMock(return_value=(b"0-0", []))

        # Start consumer
        consume_task = asyncio.create_task(consumer.start())

        # Wait for first iteration then cancel
        await asyncio.sleep(0.2)
        consume_task.cancel()

        try:
            await consume_task
        except asyncio.CancelledError:
            pass

        # Verify consumer group was created
        mock_client.create_consumer_group.assert_called_once_with(
            "test:events", "test-group", start_id="$"
        )

    @pytest.mark.asyncio
    async def test_deserialize_event(self, consumer: StreamConsumer) -> None:
        """Test event deserialization."""
        task_id = uuid4()
        event_id = uuid4()

        fields = {
            b"event_id": json.dumps(str(event_id)).encode(),
            b"event_type": json.dumps("task_created").encode(),
            b"task_id": json.dumps(str(task_id)).encode(),
            b"task_type": json.dumps("test_task").encode(),
            b"agent_id": json.dumps(None).encode(),
            b"input_data": json.dumps({}).encode(),
            b"timeout_seconds": json.dumps(300).encode(),
            b"timestamp": json.dumps("2024-01-01T00:00:00Z").encode(),
            b"trace_id": json.dumps(None).encode(),
            b"source_agent_id": json.dumps(None).encode(),
            b"workflow_id": json.dumps(None).encode(),
            b"metadata": json.dumps({}).encode(),
        }

        event = consumer._deserialize_event(fields)

        assert event.event_type == EventType.TASK_CREATED
        assert event.event_id == event_id

    @pytest.mark.asyncio
    async def test_read_messages(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test reading messages from stream."""
        # Mock xreadgroup response
        mock_client.client.xreadgroup = AsyncMock(
            return_value=[
                [
                    b"test:events",
                    [
                        (b"msg-1", {b"event_type": b'"task_created"'}),
                        (b"msg-2", {b"event_type": b'"task_completed"'}),
                    ],
                ]
            ]
        )

        messages = await consumer._read_messages("test:events")

        assert len(messages) == 2
        assert messages[0][0] == b"msg-1"
        assert messages[1][0] == b"msg-2"

    @pytest.mark.asyncio
    async def test_read_messages_empty(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test reading when no messages available."""
        mock_client.client.xreadgroup = AsyncMock(return_value=[])

        messages = await consumer._read_messages("test:events")

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_read_messages_error(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test handling read errors."""
        mock_client.client.xreadgroup = AsyncMock(
            side_effect=aioredis.ResponseError("Stream not found")
        )

        messages = await consumer._read_messages("test:events")

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_ack_message(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test acknowledging message."""
        mock_client.client.xack = AsyncMock(return_value=1)

        await consumer._ack_message("test:events", b"msg-1")

        mock_client.client.xack.assert_called_once_with(
            "test:events", "test-group", b"msg-1"
        )

    @pytest.mark.asyncio
    async def test_process_message_with_handler(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test processing message with registered handler."""
        received_events = []

        def handler(event):
            received_events.append(event)

        consumer.register_handler(EventType.TASK_CREATED, handler)

        task_id = uuid4()
        event_id = uuid4()
        fields = {
            b"event_id": json.dumps(str(event_id)).encode(),
            b"event_type": json.dumps("task_created").encode(),
            b"task_id": json.dumps(str(task_id)).encode(),
            b"task_type": json.dumps("test").encode(),
            b"agent_id": json.dumps(None).encode(),
            b"input_data": json.dumps({}).encode(),
            b"timeout_seconds": json.dumps(300).encode(),
            b"timestamp": json.dumps("2024-01-01T00:00:00Z").encode(),
            b"trace_id": json.dumps(None).encode(),
            b"source_agent_id": json.dumps(None).encode(),
            b"workflow_id": json.dumps(None).encode(),
            b"metadata": json.dumps({}).encode(),
        }

        mock_client.client.xack = AsyncMock()

        await consumer._process_message("test:events", b"msg-1", fields)

        # Verify handler was called
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.TASK_CREATED

        # Verify message was acknowledged
        mock_client.client.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_with_async_handler(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test processing message with async handler."""
        received_events = []

        async def async_handler(event):
            await asyncio.sleep(0.01)
            received_events.append(event)

        consumer.register_handler(EventType.TASK_CREATED, async_handler)

        task_id = uuid4()
        event_id = uuid4()
        fields = {
            b"event_id": json.dumps(str(event_id)).encode(),
            b"event_type": json.dumps("task_created").encode(),
            b"task_id": json.dumps(str(task_id)).encode(),
            b"task_type": json.dumps("test").encode(),
            b"agent_id": json.dumps(None).encode(),
            b"input_data": json.dumps({}).encode(),
            b"timeout_seconds": json.dumps(300).encode(),
            b"timestamp": json.dumps("2024-01-01T00:00:00Z").encode(),
            b"trace_id": json.dumps(None).encode(),
            b"source_agent_id": json.dumps(None).encode(),
            b"workflow_id": json.dumps(None).encode(),
            b"metadata": json.dumps({}).encode(),
        }

        mock_client.client.xack = AsyncMock()

        await consumer._process_message("test:events", b"msg-1", fields)

        # Verify async handler was called
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_process_message_without_handler(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test processing message without registered handler."""
        # No handler registered
        task_id = uuid4()
        event_id = uuid4()
        fields = {
            b"event_id": json.dumps(str(event_id)).encode(),
            b"event_type": json.dumps("task_created").encode(),
            b"task_id": json.dumps(str(task_id)).encode(),
            b"task_type": json.dumps("test").encode(),
            b"agent_id": json.dumps(None).encode(),
            b"input_data": json.dumps({}).encode(),
            b"timeout_seconds": json.dumps(300).encode(),
            b"timestamp": json.dumps("2024-01-01T00:00:00Z").encode(),
            b"trace_id": json.dumps(None).encode(),
            b"source_agent_id": json.dumps(None).encode(),
            b"workflow_id": json.dumps(None).encode(),
            b"metadata": json.dumps({}).encode(),
        }

        mock_client.client.xack = AsyncMock()

        # Should not raise error
        await consumer._process_message("test:events", b"msg-1", fields)

        # Message should still be acknowledged
        mock_client.client.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_handler_error(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test processing message when handler raises error."""

        def failing_handler(event):
            raise ValueError("Handler failed")

        consumer.register_handler(EventType.TASK_CREATED, failing_handler)

        task_id = uuid4()
        event_id = uuid4()
        fields = {
            b"event_id": json.dumps(str(event_id)).encode(),
            b"event_type": json.dumps("task_created").encode(),
            b"task_id": json.dumps(str(task_id)).encode(),
            b"task_type": json.dumps("test").encode(),
            b"agent_id": json.dumps(None).encode(),
            b"input_data": json.dumps({}).encode(),
            b"timeout_seconds": json.dumps(300).encode(),
            b"timestamp": json.dumps("2024-01-01T00:00:00Z").encode(),
            b"trace_id": json.dumps(None).encode(),
            b"source_agent_id": json.dumps(None).encode(),
            b"workflow_id": json.dumps(None).encode(),
            b"metadata": json.dumps({}).encode(),
        }

        mock_client.client.xack = AsyncMock()

        # Should not raise error, message should be ack'd
        await consumer._process_message("test:events", b"msg-1", fields)

        # Verify message was acknowledged despite error
        mock_client.client.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_claim_pending(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test auto-claiming pending messages."""
        task_id = uuid4()
        event_id = uuid4()
        fields = {
            b"event_id": json.dumps(str(event_id)).encode(),
            b"event_type": json.dumps("task_created").encode(),
            b"task_id": json.dumps(str(task_id)).encode(),
            b"task_type": json.dumps("test").encode(),
            b"agent_id": json.dumps(None).encode(),
            b"input_data": json.dumps({}).encode(),
            b"timeout_seconds": json.dumps(300).encode(),
            b"timestamp": json.dumps("2024-01-01T00:00:00Z").encode(),
            b"trace_id": json.dumps(None).encode(),
            b"source_agent_id": json.dumps(None).encode(),
            b"workflow_id": json.dumps(None).encode(),
            b"metadata": json.dumps({}).encode(),
        }

        # Mock autoclaim response
        mock_client.client.xautoclaim = AsyncMock(
            return_value=["0-0", [(b"msg-1", fields)]]
        )
        mock_client.client.xack = AsyncMock()

        received_events = []

        def handler(event):
            received_events.append(event)

        consumer.register_handler(EventType.TASK_CREATED, handler)

        await consumer._auto_claim_pending("test:events")

        # Verify claimed message was processed
        assert len(received_events) == 1
        mock_client.client.xack.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_claim_not_supported(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test auto-claim when not supported."""
        mock_client.client.xautoclaim = AsyncMock(
            side_effect=aioredis.ResponseError("NOSCRIPT")
        )

        # Should not raise error
        await consumer._auto_claim_pending("test:events")

    @pytest.mark.asyncio
    async def test_get_pending_messages(
        self, consumer: StreamConsumer, mock_client: AsyncMock
    ) -> None:
        """Test getting pending messages."""
        mock_client.client.xpending_range = AsyncMock(
            return_value=[
                {
                    "message_id": b"msg-1",
                    "consumer": b"consumer-1",
                    "time_since_delivered": 1000,
                    "times_delivered": 2,
                },
                {
                    "message_id": b"msg-2",
                    "consumer": b"consumer-1",
                    "time_since_delivered": 2000,
                    "times_delivered": 1,
                },
            ]
        )

        pending = await consumer.get_pending_messages("test:events")

        assert len(pending) == 2
        assert pending[0]["message_id"] == "msg-1"
        assert pending[0]["times_delivered"] == 2
        assert pending[1]["message_id"] == "msg-2"

    @pytest.mark.asyncio
    async def test_stop_consumer(self, consumer: StreamConsumer) -> None:
        """Test stopping consumer."""
        consumer._running = True

        await consumer.stop()

        assert consumer._running is False
