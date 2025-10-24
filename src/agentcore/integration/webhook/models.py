"""Webhook and event models for integration layer.

This module defines Pydantic models for webhook registration, event publishing,
and delivery tracking with guaranteed delivery semantics.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


class WebhookEvent(str, Enum):
    """Webhook event types."""

    # Integration events
    INTEGRATION_CREATED = "integration.created"
    INTEGRATION_UPDATED = "integration.updated"
    INTEGRATION_DELETED = "integration.deleted"
    INTEGRATION_FAILED = "integration.failed"

    # LLM provider events
    PROVIDER_HEALTH_CHANGED = "provider.health.changed"
    PROVIDER_QUOTA_EXCEEDED = "provider.quota.exceeded"
    PROVIDER_COST_ALERT = "provider.cost.alert"

    # Task events
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"

    # Agent events
    AGENT_REGISTERED = "agent.registered"
    AGENT_HEALTH_CHANGED = "agent.health.changed"

    # Custom events
    CUSTOM = "custom"


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXHAUSTED = "exhausted"  # Max retries exceeded


class WebhookRegistration(BaseModel):
    """Webhook registration configuration."""

    id: UUID = Field(default_factory=uuid4, description="Unique webhook identifier")
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable name")
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: list[WebhookEvent] = Field(..., min_length=1, description="Subscribed events")
    secret: str = Field(..., min_length=32, description="Webhook signing secret")
    active: bool = Field(default=True, description="Whether webhook is active")

    # Delivery configuration
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(default=60, ge=1, description="Initial retry delay")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Request timeout")

    # Metadata
    tenant_id: str | None = Field(None, description="Tenant identifier for multi-tenancy")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("secret")
    @classmethod
    def validate_secret(cls, v: str) -> str:
        """Validate webhook secret strength."""
        if len(v) < 32:
            raise ValueError("Secret must be at least 32 characters")
        return v


class EventPayload(BaseModel):
    """Event payload for webhook delivery."""

    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    event_type: WebhookEvent = Field(..., description="Event type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Event data
    data: dict[str, Any] = Field(..., description="Event-specific data")

    # Context
    source: str = Field(..., description="Event source (e.g., 'integration.portkey')")
    tenant_id: str | None = Field(None, description="Tenant identifier")
    correlation_id: str | None = Field(None, description="Correlation ID for tracing")

    # Metadata
    version: str = Field(default="1.0", description="Event schema version")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class WebhookDelivery(BaseModel):
    """Webhook delivery tracking."""

    id: UUID = Field(default_factory=uuid4, description="Unique delivery identifier")
    webhook_id: UUID = Field(..., description="Associated webhook ID")
    event_id: UUID = Field(..., description="Associated event ID")

    # Delivery state
    status: DeliveryStatus = Field(default=DeliveryStatus.PENDING)
    attempt: int = Field(default=0, ge=0, description="Current attempt number")
    max_retries: int = Field(..., ge=0, description="Maximum retry attempts")

    # Timing
    scheduled_at: datetime = Field(default_factory=datetime.utcnow)
    delivered_at: datetime | None = None
    next_retry_at: datetime | None = None

    # Response tracking
    status_code: int | None = Field(None, description="HTTP response status code")
    response_body: str | None = Field(None, description="Response body (truncated)")
    error_message: str | None = Field(None, description="Error message if failed")

    # Payload snapshot
    payload: EventPayload = Field(..., description="Event payload snapshot")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WebhookDeliveryResult(BaseModel):
    """Result of webhook delivery attempt."""

    delivery_id: UUID
    webhook_id: UUID
    event_id: UUID
    status: DeliveryStatus
    attempt: int
    status_code: int | None
    response_body: str | None
    error_message: str | None
    delivered_at: datetime | None
    next_retry_at: datetime | None


class WebhookStats(BaseModel):
    """Webhook statistics and health metrics."""

    webhook_id: UUID
    total_deliveries: int
    successful_deliveries: int
    failed_deliveries: int
    pending_deliveries: int
    average_latency_ms: float
    success_rate: float
    last_delivery_at: datetime | None
    last_failure_at: datetime | None


class EventSubscription(BaseModel):
    """Event subscription for pub/sub pattern."""

    subscription_id: UUID = Field(default_factory=uuid4)
    event_types: list[WebhookEvent]
    webhook_id: UUID
    tenant_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
