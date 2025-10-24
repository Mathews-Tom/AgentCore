"""Webhook and event exceptions."""


class WebhookError(Exception):
    """Base exception for webhook operations."""

    pass


class WebhookNotFoundError(WebhookError):
    """Webhook not found."""

    def __init__(self, webhook_id: str) -> None:
        super().__init__(f"Webhook not found: {webhook_id}")
        self.webhook_id = webhook_id


class WebhookValidationError(WebhookError):
    """Webhook validation failed."""

    pass


class WebhookDeliveryError(WebhookError):
    """Webhook delivery failed."""

    def __init__(self, webhook_id: str, reason: str) -> None:
        super().__init__(f"Webhook delivery failed for {webhook_id}: {reason}")
        self.webhook_id = webhook_id
        self.reason = reason


class EventPublishError(WebhookError):
    """Event publishing failed."""

    def __init__(self, event_type: str, reason: str) -> None:
        super().__init__(f"Failed to publish event {event_type}: {reason}")
        self.event_type = event_type
        self.reason = reason


class DeliveryExhaustedError(WebhookError):
    """Maximum delivery retries exhausted."""

    def __init__(self, delivery_id: str, attempts: int) -> None:
        super().__init__(
            f"Delivery {delivery_id} exhausted after {attempts} attempts"
        )
        self.delivery_id = delivery_id
        self.attempts = attempts


class SignatureVerificationError(WebhookError):
    """Webhook signature verification failed."""

    pass
