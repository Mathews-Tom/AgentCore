"""
Redis Streams Integration

Event-driven orchestration using Redis Streams for high-throughput message processing.
"""

from __future__ import annotations

from .client import RedisStreamsClient
from .config import StreamConfig
from .consumer import StreamConsumer, ConsumerGroup
from .models import (
    OrchestrationEvent,
    EventType,
    TaskCreatedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    AgentStartedEvent,
    AgentStoppedEvent,
    WorkflowCreatedEvent,
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    WorkflowFailedEvent,
)
from .producer import StreamProducer

__all__ = [
    "RedisStreamsClient",
    "StreamConfig",
    "StreamProducer",
    "StreamConsumer",
    "ConsumerGroup",
    "OrchestrationEvent",
    "EventType",
    "TaskCreatedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "AgentStartedEvent",
    "AgentStoppedEvent",
    "WorkflowCreatedEvent",
    "WorkflowStartedEvent",
    "WorkflowCompletedEvent",
    "WorkflowFailedEvent",
]
