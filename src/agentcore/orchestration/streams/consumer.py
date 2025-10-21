"""
Stream Consumer

Event consumer for reading events from Redis Streams with consumer groups.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import redis.asyncio as aioredis

from .client import RedisStreamsClient
from .config import StreamConfig
from .models import EventType, OrchestrationEvent


class ConsumerGroup:
    """Consumer group information."""

    def __init__(self, group_name: str, consumer_name: str) -> None:
        """
        Initialize consumer group.

        Args:
            group_name: Name of the consumer group
            consumer_name: Unique consumer name within the group
        """
        self.group_name = group_name
        self.consumer_name = consumer_name


class StreamConsumer:
    """
    Event consumer for reading orchestration events from Redis Streams.

    Supports consumer groups, automatic message claiming, and error handling.
    """

    def __init__(
        self,
        client: RedisStreamsClient,
        consumer_group: ConsumerGroup,
        config: StreamConfig | None = None,
    ) -> None:
        """
        Initialize stream consumer.

        Args:
            client: Redis Streams client instance
            consumer_group: Consumer group information
            config: Stream configuration
        """
        self.client = client
        self.consumer_group = consumer_group
        self.config = config or StreamConfig()
        self._running = False
        self._handlers: dict[EventType, list[Callable[[OrchestrationEvent], Any]]] = {}

    def register_handler(
        self, event_type: EventType, handler: Callable[[OrchestrationEvent], Any]
    ) -> None:
        """
        Register event handler for specific event type.

        Args:
            event_type: Type of event to handle
            handler: Async or sync callable to handle the event
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def start(self, stream_name: str | None = None) -> None:
        """
        Start consuming events from stream.

        Args:
            stream_name: Stream name (uses config default if None)
        """
        stream = stream_name or self.config.stream_name

        # Ensure consumer group exists
        # Use "$" to start reading from new messages only
        await self.client.create_consumer_group(
            stream, self.consumer_group.group_name, start_id="$"
        )

        self._running = True

        try:
            await self._consume_loop(stream)
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop consuming events."""
        self._running = False

    async def consume(self, count: int = 10) -> list[OrchestrationEvent]:
        """
        Consume a specific number of events (for testing).

        Args:
            count: Number of events to consume

        Returns:
            List of consumed events
        """
        events: list[OrchestrationEvent] = []
        stream = self.config.stream_name

        # Read messages from stream
        messages = await self.client.client.xread(
            {stream: "0"},  # Read from beginning
            count=count,
            block=100,
        )

        if messages:
            for _, message_list in messages:
                for message_id, fields in message_list:
                    # Decode and parse event
                    event_data = {}
                    for key, value in fields.items():
                        key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                        value_str = (
                            value.decode("utf-8") if isinstance(value, bytes) else value
                        )
                        # Parse JSON value (producer serializes each value as JSON)
                        try:
                            import json

                            event_data[key_str] = json.loads(value_str)
                        except (json.JSONDecodeError, TypeError):
                            event_data[key_str] = value_str

                    # Create OrchestrationEvent from data
                    if "event_type" in event_data:
                        event = OrchestrationEvent(**event_data)
                        events.append(event)

        return events

    async def _consume_loop(self, stream: str) -> None:
        """
        Main consumption loop.

        Args:
            stream: Stream name
        """
        while self._running:
            try:
                # Auto-claim pending messages first
                if self.config.enable_auto_claim:
                    await self._auto_claim_pending(stream)

                # Read new messages
                messages = await self._read_messages(stream)

                # Process messages
                for message_id, fields in messages:
                    await self._process_message(stream, message_id, fields)

            except asyncio.CancelledError:
                break
            except Exception:
                # Continue consuming on error
                await asyncio.sleep(1)

    async def _read_messages(
        self, stream: str
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """
        Read messages from stream using consumer group.

        Args:
            stream: Stream name

        Returns:
            List of (message_id, fields) tuples
        """
        try:
            result = await self.client.client.xreadgroup(
                groupname=self.consumer_group.group_name,
                consumername=self.consumer_group.consumer_name,
                streams={stream: ">"},
                count=self.config.count,
                block=self.config.block_ms,
            )

            if not result:
                return []

            # Result format: [(stream_name, [(message_id, fields)])]
            messages = result[0][1] if result else []
            return messages

        except aioredis.ResponseError:
            # Stream or group doesn't exist
            return []

    async def _auto_claim_pending(self, stream: str) -> None:
        """
        Automatically claim idle pending messages.

        Args:
            stream: Stream name
        """
        try:
            claimed = await self.client.client.xautoclaim(
                name=stream,
                groupname=self.consumer_group.group_name,
                consumername=self.consumer_group.consumer_name,
                min_idle_time=self.config.auto_claim_idle_ms,
                count=self.config.count,
                start_id="0-0",
            )

            # Process claimed messages
            if claimed and len(claimed) > 1:
                messages = claimed[1]
                for message_id, fields in messages:
                    await self._process_message(stream, message_id, fields)

        except aioredis.ResponseError:
            # Auto-claim not supported or error
            pass

    async def _process_message(
        self, stream: str, message_id: bytes, fields: dict[bytes, bytes]
    ) -> None:
        """
        Process single message.

        Args:
            stream: Stream name
            message_id: Message ID
            fields: Message fields
        """
        try:
            # Deserialize event
            event = self._deserialize_event(fields)

            # Call registered handlers
            if event.event_type in self._handlers:
                handlers = self._handlers[event.event_type]
                for handler in handlers:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)

            # Acknowledge message
            await self._ack_message(stream, message_id)

        except Exception as e:
            # Handle processing error
            await self._handle_processing_error(stream, message_id, fields, e)

    def _deserialize_event(self, fields: dict[bytes, bytes]) -> OrchestrationEvent:
        """
        Deserialize event from Redis Stream format.

        Args:
            fields: Message fields from Redis

        Returns:
            Deserialized event

        Raises:
            ValueError: If deserialization fails
        """
        # Decode bytes to strings and parse JSON
        event_dict: dict[str, Any] = {}
        for key, value in fields.items():
            key_str = key.decode()
            value_str = value.decode()
            event_dict[key_str] = json.loads(value_str)

        # Reconstruct event based on type
        event_type = EventType(event_dict["event_type"])

        # Use base OrchestrationEvent for now
        # In production, you'd map to specific event classes
        return OrchestrationEvent.model_validate(event_dict)

    async def _ack_message(self, stream: str, message_id: bytes) -> None:
        """
        Acknowledge message processing.

        Args:
            stream: Stream name
            message_id: Message ID
        """
        await self.client.client.xack(
            stream, self.consumer_group.group_name, message_id
        )

    async def _handle_processing_error(
        self,
        stream: str,
        message_id: bytes,
        fields: dict[bytes, bytes],
        error: Exception,
    ) -> None:
        """
        Handle message processing error.

        Args:
            stream: Stream name
            message_id: Message ID
            fields: Message fields
            error: Exception that occurred
        """
        # For now, acknowledge to prevent infinite retry
        # In production, implement retry logic with DLQ
        await self._ack_message(stream, message_id)

    async def get_pending_messages(self, stream: str) -> list[dict[str, Any]]:
        """
        Get list of pending messages for this consumer.

        Args:
            stream: Stream name

        Returns:
            List of pending message information
        """
        pending = await self.client.client.xpending_range(
            name=stream,
            groupname=self.consumer_group.group_name,
            consumername=self.consumer_group.consumer_name,
            min="-",
            max="+",
            count=100,
        )

        return [
            {
                "message_id": (
                    p["message_id"].decode("utf-8")
                    if isinstance(p["message_id"], bytes)
                    else str(p["message_id"])
                ),
                "consumer": p["consumer"],
                "time_since_delivered": p["time_since_delivered"],
                "times_delivered": p["times_delivered"],
            }
            for p in pending
        ]
