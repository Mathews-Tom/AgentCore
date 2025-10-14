"""
Event System Models

Data models for event publishing, subscriptions, and notifications.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Event type enumeration."""

    # Agent events
    AGENT_REGISTERED = "agent.registered"
    AGENT_UNREGISTERED = "agent.unregistered"
    AGENT_STATUS_CHANGED = "agent.status_changed"
    AGENT_HEALTH_CHANGED = "agent.health_changed"

    # Task events
    TASK_CREATED = "task.created"
    TASK_ASSIGNED = "task.assigned"
    TASK_STARTED = "task.started"
    TASK_PROGRESS_UPDATED = "task.progress_updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"

    # Routing events
    MESSAGE_ROUTED = "message.routed"
    MESSAGE_QUEUED = "message.queued"
    MESSAGE_DELIVERED = "message.delivered"
    MESSAGE_FAILED = "message.failed"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"


class EventPriority(str, Enum):
    """Event priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Event(BaseModel):
    """Base event model."""

    event_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique event identifier"
    )
    event_type: EventType = Field(..., description="Event type")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )
    source: str = Field(..., description="Event source (agent_id, service_name, etc.)")
    priority: EventPriority = Field(
        default=EventPriority.NORMAL, description="Event priority"
    )
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Event metadata")
    correlation_id: str | None = Field(
        None, description="Correlation ID for related events"
    )
    parent_event_id: str | None = Field(
        None, description="Parent event ID for hierarchical events"
    )

    def to_notification(self) -> dict[str, Any]:
        """Convert event to notification format."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "priority": self.priority.value,
            "data": self.data,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "parent_event_id": self.parent_event_id,
        }


class EventSubscription(BaseModel):
    """Event subscription configuration."""

    subscription_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Subscription ID"
    )
    subscriber_id: str = Field(
        ..., description="Subscriber identifier (agent_id, connection_id)"
    )
    event_types: list[EventType] = Field(..., description="Event types to subscribe to")
    filters: dict[str, Any] = Field(default_factory=dict, description="Event filters")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Subscription creation time"
    )
    expires_at: datetime | None = Field(
        None, description="Subscription expiration time"
    )
    active: bool = Field(default=True, description="Subscription active status")

    def matches_event(self, event: Event) -> bool:
        """
        Check if event matches subscription criteria.

        Args:
            event: Event to check

        Returns:
            True if event matches subscription
        """
        # Check event type
        if event.event_type not in self.event_types:
            return False

        # Check filters
        for key, value in self.filters.items():
            # Support nested keys with dot notation
            event_value = self._get_nested_value(event.data, key)
            if event_value != value:
                return False

        return True

    @staticmethod
    def _get_nested_value(data: dict[str, Any], key: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = key.split(".")
        value = data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value

    def is_expired(self) -> bool:
        """Check if subscription is expired."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at


class DeadLetterMessage(BaseModel):
    """Dead letter queue message for failed event delivery."""

    message_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Message ID"
    )
    event: Event = Field(..., description="Failed event")
    subscriber_id: str = Field(..., description="Subscriber that failed to receive")
    failure_reason: str = Field(..., description="Failure reason")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    failed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Initial failure timestamp"
    )
    last_retry_at: datetime | None = Field(
        None, description="Last retry attempt timestamp"
    )
    next_retry_at: datetime | None = Field(
        None, description="Next scheduled retry timestamp"
    )

    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry counter and update timestamps."""
        self.retry_count += 1
        self.last_retry_at = datetime.now(UTC)


class EventPublishRequest(BaseModel):
    """Event publish request."""

    event_type: EventType = Field(..., description="Event type")
    source: str = Field(..., description="Event source")
    data: dict[str, Any] = Field(..., description="Event data")
    priority: EventPriority = Field(
        default=EventPriority.NORMAL, description="Event priority"
    )
    metadata: dict[str, Any] | None = Field(None, description="Event metadata")
    correlation_id: str | None = Field(None, description="Correlation ID")


class EventPublishResponse(BaseModel):
    """Event publish response."""

    success: bool = Field(..., description="Publish success status")
    event_id: str = Field(..., description="Published event ID")
    subscribers_notified: int = Field(..., description="Number of subscribers notified")
    message: str | None = Field(None, description="Response message")


class EventSubscribeRequest(BaseModel):
    """Event subscription request."""

    subscriber_id: str = Field(..., description="Subscriber identifier")
    event_types: list[EventType] = Field(..., description="Event types to subscribe to")
    filters: dict[str, Any] | None = Field(None, description="Event filters")
    ttl_seconds: int | None = Field(None, description="Subscription TTL in seconds")


class EventSubscribeResponse(BaseModel):
    """Event subscription response."""

    success: bool = Field(..., description="Subscription success status")
    subscription_id: str = Field(..., description="Subscription ID")
    message: str | None = Field(None, description="Response message")


class EventUnsubscribeRequest(BaseModel):
    """Event unsubscribe request."""

    subscription_id: str = Field(..., description="Subscription ID to cancel")


class EventUnsubscribeResponse(BaseModel):
    """Event unsubscribe response."""

    success: bool = Field(..., description="Unsubscribe success status")
    message: str | None = Field(None, description="Response message")


class WebSocketMessage(BaseModel):
    """WebSocket message format."""

    message_type: str = Field(..., description="Message type (event, ping, pong, etc.)")
    payload: dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Message timestamp"
    )
    message_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Message ID"
    )
