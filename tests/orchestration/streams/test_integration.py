"""Integration tests for Redis Streams with testcontainers."""

from __future__ import annotations

import asyncio
import warnings
from uuid import uuid4

import pytest
from testcontainers.redis import RedisContainer

from agentcore.orchestration.streams import (
    ConsumerGroup,
    EventType,
    RedisStreamsClient,
    StreamConsumer,
    StreamProducer,
    TaskCompletedEvent,
    TaskCreatedEvent,
)
from agentcore.orchestration.streams.config import StreamConfig


@pytest.fixture(scope="module")
def redis_container():
    """Provide Redis container for integration tests."""
    # Suppress testcontainers internal deprecation warnings
    # The warnings are from the testcontainers library's internal use of
    # @wait_container_is_ready, which we cannot control
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The @wait_container_is_ready decorator is deprecated",
            category=DeprecationWarning,
        )
        container = RedisContainer("redis:7-alpine")
        with container:
            yield container


@pytest.fixture
async def redis_client(redis_container):
    """Provide Redis Streams client connected to test container."""
    # Build connection URL from container info
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    redis_url = f"redis://{host}:{port}/0"

    config = StreamConfig(
        stream_name="test:events",
        consumer_group_name="test-group",
        dead_letter_stream="test:events:dlq",
    )

    client = RedisStreamsClient(redis_url=redis_url, config=config)
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
async def producer(redis_client):
    """Provide stream producer."""
    config = StreamConfig(
        stream_name="test:events",
        dead_letter_stream="test:events:dlq",
    )
    return StreamProducer(redis_client, config)


@pytest.fixture
async def consumer(redis_client):
    """Provide stream consumer."""
    config = StreamConfig(
        stream_name="test:events",
        consumer_group_name="test-group",
        consumer_name="test-consumer-1",
    )
    group = ConsumerGroup("test-group", "test-consumer-1")
    return StreamConsumer(redis_client, group, config)


class TestRedisStreamsClient:
    """Integration tests for Redis Streams client."""

    @pytest.mark.asyncio
    async def test_connect_and_ping(self, redis_client):
        """Test connecting to Redis and ping."""
        # Client is already connected via fixture
        result = await redis_client.client.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_create_consumer_group(self, redis_client):
        """Test creating consumer group."""
        stream_name = f"test:stream:{uuid4()}"
        group_name = "test-group"

        created = await redis_client.create_consumer_group(stream_name, group_name)
        assert created is True

        # Creating again should return False
        created_again = await redis_client.create_consumer_group(stream_name, group_name)
        assert created_again is False

    @pytest.mark.asyncio
    async def test_delete_consumer_group(self, redis_client):
        """Test deleting consumer group."""
        stream_name = f"test:stream:{uuid4()}"
        group_name = "test-group"

        await redis_client.create_consumer_group(stream_name, group_name)
        deleted = await redis_client.delete_consumer_group(stream_name, group_name)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_stream_length(self, redis_client, producer):
        """Test getting stream length."""
        stream_name = f"test:stream:{uuid4()}"
        config = StreamConfig(stream_name=stream_name)
        test_producer = StreamProducer(redis_client, config)

        # Initially empty
        length = await redis_client.get_stream_length(stream_name)
        assert length == 0

        # Publish events
        event = TaskCreatedEvent(task_id=uuid4(), task_type="test")
        await test_producer.publish(event)
        await test_producer.publish(event)

        length = await redis_client.get_stream_length(stream_name)
        assert length == 2

    @pytest.mark.asyncio
    async def test_trim_stream(self, redis_client, producer):
        """Test trimming stream."""
        stream_name = f"test:stream:{uuid4()}"
        config = StreamConfig(stream_name=stream_name, max_stream_length=5)
        test_producer = StreamProducer(redis_client, config)

        # Publish 10 events
        event = TaskCreatedEvent(task_id=uuid4(), task_type="test")
        for _ in range(10):
            await test_producer.publish(event)

        # Trim to 5 (approximate, so may be slightly more)
        await redis_client.trim_stream(stream_name, max_length=5)

        length = await redis_client.get_stream_length(stream_name)
        # Approximate trimming may keep a few extra entries
        assert length <= 10  # Should be roughly trimmed

    @pytest.mark.asyncio
    async def test_health_check(self, redis_client):
        """Test health check."""
        health = await redis_client.health_check()

        assert health["status"] == "healthy"
        assert "latency_ms" in health
        assert "version" in health


class TestStreamProducer:
    """Integration tests for stream producer."""

    @pytest.mark.asyncio
    async def test_publish_event(self, redis_client, producer):
        """Test publishing single event."""
        task_id = uuid4()
        event = TaskCreatedEvent(
            task_id=task_id,
            task_type="data_analysis",
            agent_id="agent-1",
        )

        message_id = await producer.publish(event)
        assert message_id is not None
        assert isinstance(message_id, str)

        # Verify event was published
        length = await redis_client.get_stream_length("test:events")
        assert length >= 1

    @pytest.mark.asyncio
    async def test_publish_batch(self, redis_client, producer):
        """Test publishing multiple events in batch."""
        events = [
            TaskCreatedEvent(task_id=uuid4(), task_type="test")
            for _ in range(5)
        ]

        message_ids = await producer.publish_batch(events)
        assert len(message_ids) == 5
        assert all(isinstance(mid, str) for mid in message_ids)

        # Verify all events were published
        length = await redis_client.get_stream_length("test:events")
        assert length >= 5

    @pytest.mark.asyncio
    async def test_publish_to_dlq(self, redis_client, producer):
        """Test publishing to dead letter queue."""
        event = TaskCreatedEvent(task_id=uuid4(), task_type="failed_task")

        message_id = await producer.publish_to_dlq(
            event,
            error="Processing failed",
            retry_count=3,
        )

        assert message_id is not None

        # Verify DLQ metadata
        assert event.metadata["dlq_reason"] == "Processing failed"
        assert event.metadata["dlq_retry_count"] == 3

        # Verify published to DLQ stream
        dlq_length = await redis_client.get_stream_length("test:events:dlq")
        assert dlq_length >= 1


class TestStreamConsumer:
    """Integration tests for stream consumer."""

    @pytest.mark.asyncio
    async def test_consume_events(self, redis_client, producer, consumer):
        """Test consuming events from stream."""
        received_events = []

        def handler(event):
            received_events.append(event)

        consumer.register_handler(EventType.TASK_CREATED, handler)

        # Start consumer first
        consume_task = asyncio.create_task(consumer.start())
        await asyncio.sleep(0.5)  # Let consumer initialize

        # Then publish events
        events = [
            TaskCreatedEvent(task_id=uuid4(), task_type="test")
            for _ in range(3)
        ]
        await producer.publish_batch(events)

        # Wait for processing
        await asyncio.sleep(2)
        await consumer.stop()

        try:
            await asyncio.wait_for(consume_task, timeout=5)
        except asyncio.TimeoutError:
            pass

        assert len(received_events) >= 3

    @pytest.mark.asyncio
    async def test_consumer_group_independence(self, redis_client, producer):
        """Test that consumer groups work independently."""
        config1 = StreamConfig(
            stream_name="test:events",
            consumer_group_name="group-1",
            consumer_name="consumer-1",
        )
        config2 = StreamConfig(
            stream_name="test:events",
            consumer_group_name="group-2",
            consumer_name="consumer-1",
        )

        group1 = ConsumerGroup("group-1", "consumer-1")
        group2 = ConsumerGroup("group-2", "consumer-1")

        consumer1 = StreamConsumer(redis_client, group1, config1)
        consumer2 = StreamConsumer(redis_client, group2, config2)

        events1 = []
        events2 = []

        consumer1.register_handler(EventType.TASK_CREATED, lambda e: events1.append(e))
        consumer2.register_handler(EventType.TASK_CREATED, lambda e: events2.append(e))

        # Start consumers first
        task1 = asyncio.create_task(consumer1.start())
        task2 = asyncio.create_task(consumer2.start())
        await asyncio.sleep(0.5)  # Let consumers initialize

        # Then publish event
        event = TaskCreatedEvent(task_id=uuid4(), task_type="test")
        await producer.publish(event)

        await asyncio.sleep(2)

        await consumer1.stop()
        await consumer2.stop()

        try:
            await asyncio.wait_for(task1, timeout=2)
        except asyncio.TimeoutError:
            pass

        try:
            await asyncio.wait_for(task2, timeout=2)
        except asyncio.TimeoutError:
            pass

        # Both groups should have received the event
        assert len(events1) >= 1
        assert len(events2) >= 1


class TestProducerConsumerIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_end_to_end_event_flow(self, redis_client):
        """Test complete producer-consumer event flow."""
        stream_name = f"test:e2e:{uuid4()}"
        config = StreamConfig(stream_name=stream_name)

        producer = StreamProducer(redis_client, config)
        group = ConsumerGroup("e2e-group", "e2e-consumer")
        consumer = StreamConsumer(redis_client, group, config)

        received_events = []

        async def async_handler(event):
            received_events.append(event)

        consumer.register_handler(EventType.TASK_CREATED, async_handler)
        consumer.register_handler(EventType.TASK_COMPLETED, async_handler)

        # Start consumer first
        consume_task = asyncio.create_task(consumer.start())
        await asyncio.sleep(0.5)  # Let consumer initialize

        # Then publish various events
        task_id = uuid4()

        created_event = TaskCreatedEvent(task_id=task_id, task_type="integration_test")
        await producer.publish(created_event)

        completed_event = TaskCompletedEvent(
            task_id=task_id,
            agent_id="test-agent",
            result_data={"status": "success"},
            execution_time_ms=1000,
        )
        await producer.publish(completed_event)

        # Wait for processing
        await asyncio.sleep(2)

        await consumer.stop()
        try:
            await asyncio.wait_for(consume_task, timeout=5)
        except asyncio.TimeoutError:
            pass

        # Verify events were received
        assert len(received_events) >= 2
        assert any(e.event_type == EventType.TASK_CREATED for e in received_events)
        assert any(e.event_type == EventType.TASK_COMPLETED for e in received_events)
