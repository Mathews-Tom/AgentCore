"""Webhook registration and management."""

from __future__ import annotations

import asyncio
import secrets
from datetime import datetime
from typing import Any
from uuid import UUID

from .config import WebhookConfig
from .exceptions import WebhookNotFoundError, WebhookValidationError
from .models import WebhookEvent, WebhookRegistration, WebhookStats


class WebhookManager:
    """Manages webhook registration, validation, and lifecycle.

    Provides CRUD operations for webhooks with validation and health tracking.
    Thread-safe for concurrent operations.
    """

    def __init__(self, config: WebhookConfig | None = None) -> None:
        """Initialize webhook manager.

        Args:
            config: Webhook configuration (defaults to WebhookConfig())
        """
        self.config = config or WebhookConfig()
        self._webhooks: dict[UUID, WebhookRegistration] = {}
        self._lock = asyncio.Lock()

        # Stats tracking
        self._stats: dict[UUID, dict[str, Any]] = {}

    async def register(
        self,
        name: str,
        url: str,
        events: list[WebhookEvent],
        secret: str | None = None,
        max_retries: int | None = None,
        retry_delay_seconds: int | None = None,
        timeout_seconds: int | None = None,
        tenant_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WebhookRegistration:
        """Register a new webhook.

        Args:
            name: Human-readable webhook name
            url: Webhook endpoint URL
            events: List of events to subscribe to
            secret: Signing secret (auto-generated if not provided)
            max_retries: Maximum retry attempts (uses config default if None)
            retry_delay_seconds: Initial retry delay (uses config default if None)
            timeout_seconds: Request timeout (uses config default if None)
            tenant_id: Tenant identifier for multi-tenancy
            metadata: Additional metadata

        Returns:
            WebhookRegistration: Created webhook

        Raises:
            WebhookValidationError: If validation fails
        """
        # Generate secret if not provided
        if secret is None:
            secret = secrets.token_urlsafe(self.config.min_secret_length)

        # Use config defaults if not specified
        if max_retries is None:
            max_retries = self.config.default_max_retries
        if retry_delay_seconds is None:
            retry_delay_seconds = self.config.default_retry_delay_seconds
        if timeout_seconds is None:
            timeout_seconds = self.config.default_timeout_seconds

        # Validate inputs
        if not events:
            raise WebhookValidationError("At least one event must be specified")

        if len(secret) < self.config.min_secret_length:
            raise WebhookValidationError(
                f"Secret must be at least {self.config.min_secret_length} characters"
            )

        # Create webhook
        webhook = WebhookRegistration(
            name=name,
            url=url,
            events=events,
            secret=secret,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            timeout_seconds=timeout_seconds,
            tenant_id=tenant_id,
            metadata=metadata or {},
        )

        # Store webhook
        async with self._lock:
            self._webhooks[webhook.id] = webhook
            self._stats[webhook.id] = {
                "total_deliveries": 0,
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "pending_deliveries": 0,
                "last_delivery_at": None,
                "last_failure_at": None,
            }

        return webhook

    async def get(self, webhook_id: UUID) -> WebhookRegistration:
        """Get webhook by ID.

        Args:
            webhook_id: Webhook identifier

        Returns:
            WebhookRegistration: Webhook configuration

        Raises:
            WebhookNotFoundError: If webhook not found
        """
        async with self._lock:
            webhook = self._webhooks.get(webhook_id)
            if webhook is None:
                raise WebhookNotFoundError(str(webhook_id))
            return webhook

    async def list(
        self,
        tenant_id: str | None = None,
        event_type: WebhookEvent | None = None,
        active_only: bool = True,
    ) -> list[WebhookRegistration]:
        """List webhooks with optional filtering.

        Args:
            tenant_id: Filter by tenant ID
            event_type: Filter by subscribed event type
            active_only: Only return active webhooks

        Returns:
            List of matching webhooks
        """
        async with self._lock:
            webhooks = list(self._webhooks.values())

        # Apply filters
        if tenant_id is not None:
            webhooks = [w for w in webhooks if w.tenant_id == tenant_id]

        if event_type is not None:
            webhooks = [w for w in webhooks if event_type in w.events]

        if active_only:
            webhooks = [w for w in webhooks if w.active]

        return webhooks

    async def update(
        self,
        webhook_id: UUID,
        name: str | None = None,
        url: str | None = None,
        events: list[WebhookEvent] | None = None,
        active: bool | None = None,
        max_retries: int | None = None,
        retry_delay_seconds: int | None = None,
        timeout_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WebhookRegistration:
        """Update webhook configuration.

        Args:
            webhook_id: Webhook identifier
            name: New name (optional)
            url: New URL (optional)
            events: New event list (optional)
            active: Active status (optional)
            max_retries: New max retries (optional)
            retry_delay_seconds: New retry delay (optional)
            timeout_seconds: New timeout (optional)
            metadata: New metadata (optional)

        Returns:
            Updated webhook

        Raises:
            WebhookNotFoundError: If webhook not found
        """
        async with self._lock:
            webhook = self._webhooks.get(webhook_id)
            if webhook is None:
                raise WebhookNotFoundError(str(webhook_id))

            # Update fields if provided
            if name is not None:
                webhook.name = name
            if url is not None:
                webhook.url = url
            if events is not None:
                if not events:
                    raise WebhookValidationError("At least one event must be specified")
                webhook.events = events
            if active is not None:
                webhook.active = active
            if max_retries is not None:
                webhook.max_retries = max_retries
            if retry_delay_seconds is not None:
                webhook.retry_delay_seconds = retry_delay_seconds
            if timeout_seconds is not None:
                webhook.timeout_seconds = timeout_seconds
            if metadata is not None:
                webhook.metadata = metadata

            webhook.updated_at = datetime.utcnow()
            self._webhooks[webhook_id] = webhook

        return webhook

    async def delete(self, webhook_id: UUID) -> None:
        """Delete webhook.

        Args:
            webhook_id: Webhook identifier

        Raises:
            WebhookNotFoundError: If webhook not found
        """
        async with self._lock:
            if webhook_id not in self._webhooks:
                raise WebhookNotFoundError(str(webhook_id))

            del self._webhooks[webhook_id]
            if webhook_id in self._stats:
                del self._stats[webhook_id]

    async def get_stats(self, webhook_id: UUID) -> WebhookStats:
        """Get webhook statistics.

        Args:
            webhook_id: Webhook identifier

        Returns:
            Webhook statistics

        Raises:
            WebhookNotFoundError: If webhook not found
        """
        async with self._lock:
            if webhook_id not in self._webhooks:
                raise WebhookNotFoundError(str(webhook_id))

            stats = self._stats.get(webhook_id, {})
            total = stats.get("total_deliveries", 0)
            successful = stats.get("successful_deliveries", 0)
            success_rate = (successful / total * 100) if total > 0 else 0.0

            return WebhookStats(
                webhook_id=webhook_id,
                total_deliveries=total,
                successful_deliveries=successful,
                failed_deliveries=stats.get("failed_deliveries", 0),
                pending_deliveries=stats.get("pending_deliveries", 0),
                average_latency_ms=stats.get("average_latency_ms", 0.0),
                success_rate=success_rate,
                last_delivery_at=stats.get("last_delivery_at"),
                last_failure_at=stats.get("last_failure_at"),
            )

    async def update_stats(
        self,
        webhook_id: UUID,
        success: bool,
        latency_ms: float | None = None,
    ) -> None:
        """Update webhook statistics after delivery attempt.

        Args:
            webhook_id: Webhook identifier
            success: Whether delivery was successful
            latency_ms: Delivery latency in milliseconds
        """
        async with self._lock:
            if webhook_id not in self._stats:
                return

            stats = self._stats[webhook_id]
            stats["total_deliveries"] += 1

            if success:
                stats["successful_deliveries"] += 1
                stats["last_delivery_at"] = datetime.utcnow()
            else:
                stats["failed_deliveries"] += 1
                stats["last_failure_at"] = datetime.utcnow()

            # Update average latency
            if latency_ms is not None:
                current_avg = stats.get("average_latency_ms", 0.0)
                total = stats["total_deliveries"]
                stats["average_latency_ms"] = (
                    current_avg * (total - 1) + latency_ms
                ) / total
