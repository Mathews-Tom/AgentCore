"""Webhook and event system for integration layer.

This module provides webhook registration, event publishing, and guaranteed
delivery with retry logic for external integrations.

Example usage:
    >>> from agentcore.integration.webhook import (
    ...     WebhookManager,
    ...     EventPublisher,
    ...     DeliveryService,
    ...     WebhookEvent,
    ... )
    >>>
    >>> # Initialize components
    >>> manager = WebhookManager()
    >>> publisher = EventPublisher()
    >>> delivery = DeliveryService()
    >>>
    >>> # Register webhook
    >>> webhook = await manager.register(
    ...     name="My Webhook",
    ...     url="https://api.example.com/webhooks",
    ...     events=[WebhookEvent.TASK_COMPLETED],
    ... )
    >>>
    >>> # Subscribe to events
    >>> await publisher.subscribe(webhook.id, [WebhookEvent.TASK_COMPLETED])
    >>>
    >>> # Publish event
    >>> event = await publisher.publish(
    ...     event_type=WebhookEvent.TASK_COMPLETED,
    ...     data={"task_id": "123", "status": "completed"},
    ...     source="agentcore.tasks",
    ... )
    >>>
    >>> # Schedule delivery
    >>> delivery_record = await delivery.schedule(webhook, event)
"""

from .config import WebhookConfig
from .delivery import DeliveryService
from .exceptions import (
    DeliveryExhaustedError,
    EventPublishError,
    SignatureVerificationError,
    WebhookDeliveryError,
    WebhookError,
    WebhookNotFoundError,
    WebhookValidationError,
)
from .manager import WebhookManager
from .models import (
    DeliveryStatus,
    EventPayload,
    EventSubscription,
    WebhookDelivery,
    WebhookDeliveryResult,
    WebhookEvent,
    WebhookRegistration,
    WebhookStats,
)
from .publisher import EventPublisher
from .validator import WebhookValidator

__all__ = [
    # Core classes
    "WebhookManager",
    "EventPublisher",
    "DeliveryService",
    "WebhookValidator",
    "WebhookConfig",
    # Models
    "WebhookRegistration",
    "EventPayload",
    "WebhookDelivery",
    "WebhookDeliveryResult",
    "EventSubscription",
    "WebhookStats",
    # Enums
    "WebhookEvent",
    "DeliveryStatus",
    # Exceptions
    "WebhookError",
    "WebhookNotFoundError",
    "WebhookValidationError",
    "WebhookDeliveryError",
    "EventPublishError",
    "DeliveryExhaustedError",
    "SignatureVerificationError",
]
