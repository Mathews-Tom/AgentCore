"""
Stream Producer

Event producer for publishing events to Redis Streams.
"""

from __future__ import annotations

import asyncio
import json

import redis.asyncio as aioredis

from .client import RedisStreamsClient
from .config import StreamConfig
from .models import OrchestrationEvent


class StreamProducer:
    """
    Event producer for publishing orchestration events to Redis Streams.

    Handles serialization, retry logic, and stream trimming.
    """

    def __init__(
        self, client: RedisStreamsClient, config: StreamConfig | None = None
    ) -> None:
        """
        Initialize stream producer.

        Args:
            client: Redis Streams client instance
            config: Stream configuration
        """
        self.client = client
        self.config = config or StreamConfig()

    async def publish(
        self,
        event: OrchestrationEvent,
        stream_name: str | None = None,
    ) -> str:
        """
        Publish event to Redis Stream.

        Args:
            event: Event to publish
            stream_name: Stream name (uses config default if None)

        Returns:
            Message ID from Redis

        Raises:
            ConnectionError: If Redis connection fails
            ValueError: If event serialization fails
        """
        stream = stream_name or self.config.stream_name

        # Serialize event to JSON
        event_data = self._serialize_event(event)

        # Publish with retry
        message_id = await self._publish_with_retry(stream, event_data)

        # Trim stream if needed
        await self._trim_stream_if_needed(stream)

        return message_id

    async def publish_batch(
        self,
        events: list[OrchestrationEvent],
        stream_name: str | None = None,
    ) -> list[str]:
        """
        Publish multiple events in batch.

        Args:
            events: List of events to publish
            stream_name: Stream name (uses config default if None)

        Returns:
            List of message IDs from Redis
        """
        stream = stream_name or self.config.stream_name

        # Use pipeline for batch publishing
        async with self.client.client.pipeline(transaction=False) as pipe:
            for event in events:
                event_data = self._serialize_event(event)
                pipe.xadd(name=stream, fields=event_data)

            results = await pipe.execute()

        # Trim stream after batch
        await self._trim_stream_if_needed(stream)

        # Decode bytes to strings
        return [r.decode() if isinstance(r, bytes) else str(r) for r in results]

    def _serialize_event(self, event: OrchestrationEvent) -> dict[bytes, bytes]:
        """
        Serialize event to Redis Stream format.

        Args:
            event: Event to serialize

        Returns:
            Dictionary with byte keys and values for Redis
        """
        # Convert Pydantic model to dict with JSON-serializable types
        event_dict = event.model_dump(mode="json")

        # Create fields dict with byte keys/values
        fields: dict[bytes, bytes] = {}
        for key, value in event_dict.items():
            # Convert value to JSON string
            json_value = json.dumps(value)
            fields[key.encode()] = json_value.encode()

        return fields

    async def _publish_with_retry(
        self, stream: str, event_data: dict[bytes, bytes]
    ) -> str:
        """
        Publish event with exponential backoff retry.

        Args:
            stream: Stream name
            event_data: Serialized event data

        Returns:
            Message ID

        Raises:
            ConnectionError: If all retries fail
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                message_id = await self.client.client.xadd(
                    name=stream, fields=event_data
                )
                # Decode bytes to string
                return (
                    message_id.decode()
                    if isinstance(message_id, bytes)
                    else str(message_id)
                )
            except (aioredis.ConnectionError, aioredis.TimeoutError) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    delay_ms = self.config.retry_backoff_ms * (2**attempt)
                    await asyncio.sleep(delay_ms / 1000.0)
                continue

        raise ConnectionError(
            f"Failed to publish event after {self.config.max_retries} attempts"
        ) from last_error

    async def _trim_stream_if_needed(self, stream: str) -> None:
        """
        Trim stream if it exceeds maximum length.

        Args:
            stream: Stream name
        """
        try:
            length = await self.client.get_stream_length(stream)
            if length > self.config.max_stream_length:
                await self.client.trim_stream(stream, self.config.max_stream_length)
        except Exception:
            # Don't fail publishing if trim fails
            pass

    async def publish_to_dlq(
        self, event: OrchestrationEvent, error: str, retry_count: int
    ) -> str:
        """
        Publish failed event to dead letter queue.

        Args:
            event: Original event that failed
            error: Error message
            retry_count: Number of retry attempts

        Returns:
            Message ID in DLQ
        """
        # Add failure metadata to event
        event.metadata["dlq_reason"] = error
        event.metadata["dlq_retry_count"] = retry_count
        event.metadata["dlq_original_stream"] = self.config.stream_name

        return await self.publish(event, stream_name=self.config.dead_letter_stream)
