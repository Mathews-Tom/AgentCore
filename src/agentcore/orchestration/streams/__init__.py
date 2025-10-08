"""
Redis Streams Integration

Event-driven orchestration using Redis Streams for high-throughput message processing.
"""

from __future__ import annotations

from .client import RedisStreamsClient
from .consumer import StreamConsumer, ConsumerGroup
from .models import (
    OrchestrationEvent,
    EventType,
    TaskCreatedEvent,
    TaskCompletedEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
)
from .producer import StreamProducer

__all__ = [
    "RedisStreamsClient",
    "StreamProducer",
    "StreamConsumer",
    "ConsumerGroup",
    "OrchestrationEvent",
    "EventType",
    "TaskCreatedEvent",
    "TaskCompletedEvent",
    "AgentStartedEvent",
    "AgentStoppedEvent",
]
