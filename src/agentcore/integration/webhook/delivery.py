"""Webhook delivery service with guaranteed delivery and retries."""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import httpx

from .config import WebhookConfig
from .exceptions import DeliveryExhaustedError, WebhookDeliveryError
from .models import (
    DeliveryStatus,
    EventPayload,
    WebhookDelivery,
    WebhookDeliveryResult,
    WebhookRegistration,
)
from .validator import WebhookValidator

logger = logging.getLogger(__name__)


class DeliveryService:
    """Webhook delivery service with retries and guarantees.

    Handles webhook delivery with:
    - Guaranteed delivery with configurable retries
    - Exponential backoff with jitter
    - Concurrent delivery with rate limiting
    - Delivery tracking and history
    """

    def __init__(
        self,
        config: WebhookConfig | None = None,
        validator: WebhookValidator | None = None,
    ) -> None:
        """Initialize delivery service.

        Args:
            config: Webhook configuration
            validator: Webhook validator for signature generation
        """
        self.config = config or WebhookConfig()
        self.validator = validator or WebhookValidator(self.config)

        # Delivery tracking
        self._deliveries: dict[UUID, WebhookDelivery] = {}
        self._lock = asyncio.Lock()

        # Rate limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_deliveries)

        # HTTP client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.default_timeout_seconds)
        )

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self._client.aclose()

    async def schedule(
        self,
        webhook: WebhookRegistration,
        event: EventPayload,
    ) -> WebhookDelivery:
        """Schedule event delivery to webhook.

        Args:
            webhook: Webhook configuration
            event: Event to deliver

        Returns:
            Created delivery record
        """
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event.event_id,
            max_retries=webhook.max_retries,
            payload=event,
        )

        async with self._lock:
            self._deliveries[delivery.id] = delivery

        logger.info(
            f"Scheduled delivery {delivery.id} for webhook {webhook.id}",
            extra={
                "delivery_id": str(delivery.id),
                "webhook_id": str(webhook.id),
                "event_id": str(event.event_id),
            },
        )

        # Start delivery async (don't await)
        asyncio.create_task(self._deliver(webhook, delivery))

        return delivery

    async def get_delivery(self, delivery_id: UUID) -> WebhookDelivery | None:
        """Get delivery by ID.

        Args:
            delivery_id: Delivery identifier

        Returns:
            Delivery record or None if not found
        """
        async with self._lock:
            return self._deliveries.get(delivery_id)

    async def list_deliveries(
        self,
        webhook_id: UUID | None = None,
        status: DeliveryStatus | None = None,
        limit: int = 100,
    ) -> list[WebhookDelivery]:
        """List deliveries with optional filtering.

        Args:
            webhook_id: Filter by webhook ID
            status: Filter by delivery status
            limit: Maximum number of results

        Returns:
            List of matching deliveries
        """
        async with self._lock:
            deliveries = list(self._deliveries.values())

        # Apply filters
        if webhook_id is not None:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        if status is not None:
            deliveries = [d for d in deliveries if d.status == status]

        # Sort by scheduled time (newest first) and limit
        deliveries.sort(key=lambda d: d.scheduled_at, reverse=True)
        return deliveries[:limit]

    async def _deliver(
        self,
        webhook: WebhookRegistration,
        delivery: WebhookDelivery,
    ) -> WebhookDeliveryResult:
        """Execute webhook delivery with retries.

        Args:
            webhook: Webhook configuration
            delivery: Delivery record

        Returns:
            Delivery result
        """
        # Check if webhook is active
        if not webhook.active:
            await self._update_delivery(
                delivery,
                DeliveryStatus.FAILED,
                error_message="Webhook is not active",
            )
            return self._create_result(delivery)

        # Attempt delivery with retries
        for attempt in range(webhook.max_retries + 1):
            delivery.attempt = attempt

            try:
                # Update status
                await self._update_delivery(delivery, DeliveryStatus.DELIVERING)

                # Execute delivery
                result = await self._execute_delivery(webhook, delivery)

                # Check if successful
                if result.status == DeliveryStatus.DELIVERED:
                    logger.info(
                        f"Delivery {delivery.id} succeeded (attempt {attempt + 1})",
                        extra={
                            "delivery_id": str(delivery.id),
                            "webhook_id": str(webhook.id),
                            "attempt": attempt + 1,
                        },
                    )
                    return result

                # Delivery failed, check if we should retry
                if attempt < webhook.max_retries:
                    # Calculate retry delay with exponential backoff
                    retry_delay = self._calculate_retry_delay(
                        webhook.retry_delay_seconds,
                        attempt,
                    )

                    next_retry_at = datetime.utcnow() + timedelta(seconds=retry_delay)

                    await self._update_delivery(
                        delivery,
                        DeliveryStatus.RETRYING,
                        next_retry_at=next_retry_at,
                        error_message=result.error_message,
                    )

                    logger.warning(
                        f"Delivery {delivery.id} failed, retrying in {retry_delay}s "
                        f"(attempt {attempt + 1}/{webhook.max_retries + 1})",
                        extra={
                            "delivery_id": str(delivery.id),
                            "retry_delay": retry_delay,
                            "attempt": attempt + 1,
                        },
                    )

                    # Wait for retry delay
                    await asyncio.sleep(retry_delay)

                else:
                    # Max retries exhausted
                    await self._update_delivery(
                        delivery,
                        DeliveryStatus.EXHAUSTED,
                        error_message=f"Max retries ({webhook.max_retries}) exhausted",
                    )

                    logger.error(
                        f"Delivery {delivery.id} exhausted after {attempt + 1} attempts",
                        extra={
                            "delivery_id": str(delivery.id),
                            "webhook_id": str(webhook.id),
                        },
                    )

                    raise DeliveryExhaustedError(str(delivery.id), attempt + 1)

            except DeliveryExhaustedError:
                # Retries exhausted
                return self._create_result(delivery)

            except Exception as e:
                logger.error(
                    f"Unexpected error in delivery {delivery.id}: {e}",
                    exc_info=True,
                    extra={"delivery_id": str(delivery.id)},
                )

                await self._update_delivery(
                    delivery,
                    DeliveryStatus.FAILED,
                    error_message=f"Unexpected error: {str(e)}",
                )

                return self._create_result(delivery)

        # Should not reach here, but return failed result
        return self._create_result(delivery)

    async def _execute_delivery(
        self,
        webhook: WebhookRegistration,
        delivery: WebhookDelivery,
    ) -> WebhookDeliveryResult:
        """Execute single delivery attempt.

        Args:
            webhook: Webhook configuration
            delivery: Delivery record

        Returns:
            Delivery result
        """
        start_time = datetime.utcnow()

        try:
            # Acquire semaphore for rate limiting
            async with self._semaphore:
                # Prepare request
                headers = self._prepare_headers(webhook, delivery.payload)
                payload_dict = delivery.payload.model_dump(mode="json")

                # Send request
                response = await self._client.post(
                    str(webhook.url),
                    json=payload_dict,
                    headers=headers,
                    timeout=webhook.timeout_seconds,
                )

                # Calculate latency
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                # Check response status
                if 200 <= response.status_code < 300:
                    # Success
                    await self._update_delivery(
                        delivery,
                        DeliveryStatus.DELIVERED,
                        status_code=response.status_code,
                        response_body=response.text[:1000],  # Truncate
                        delivered_at=datetime.utcnow(),
                    )

                    logger.debug(
                        f"Delivery {delivery.id} HTTP {response.status_code} "
                        f"({latency_ms:.1f}ms)"
                    )

                    return self._create_result(delivery)

                else:
                    # HTTP error
                    error_msg = (
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )

                    await self._update_delivery(
                        delivery,
                        DeliveryStatus.FAILED,
                        status_code=response.status_code,
                        response_body=response.text[:1000],
                        error_message=error_msg,
                    )

                    return self._create_result(delivery)

        except httpx.TimeoutException:
            error_msg = f"Request timeout after {webhook.timeout_seconds}s"
            await self._update_delivery(
                delivery,
                DeliveryStatus.FAILED,
                error_message=error_msg,
            )
            return self._create_result(delivery)

        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            await self._update_delivery(
                delivery,
                DeliveryStatus.FAILED,
                error_message=error_msg,
            )
            return self._create_result(delivery)

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            await self._update_delivery(
                delivery,
                DeliveryStatus.FAILED,
                error_message=error_msg,
            )
            return self._create_result(delivery)

    def _prepare_headers(
        self,
        webhook: WebhookRegistration,
        payload: EventPayload,
    ) -> dict[str, str]:
        """Prepare HTTP headers for webhook request.

        Args:
            webhook: Webhook configuration
            payload: Event payload

        Returns:
            Request headers
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentCore-Webhook/1.0",
            "X-Webhook-ID": str(webhook.id),
            "X-Event-ID": str(payload.event_id),
            "X-Event-Type": payload.event_type.value,
            "X-Event-Timestamp": payload.timestamp.isoformat(),
        }

        # Add signature if required
        if self.config.require_signature:
            signature = self.validator.generate_signature(
                webhook.secret,
                payload.model_dump(mode="json"),
            )
            headers["X-Webhook-Signature"] = signature

        # Add tenant ID if present
        if payload.tenant_id:
            headers["X-Tenant-ID"] = payload.tenant_id

        return headers

    def _calculate_retry_delay(
        self,
        base_delay: int,
        attempt: int,
    ) -> int:
        """Calculate retry delay with exponential backoff and jitter.

        Args:
            base_delay: Base delay in seconds
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base * multiplier^attempt
        delay = base_delay * (self.config.retry_backoff_multiplier**attempt)

        # Cap at maximum
        delay = min(delay, self.config.max_retry_delay_seconds)

        # Add jitter if enabled (±20%)
        if self.config.retry_backoff_jitter:
            jitter = delay * 0.2
            delay = delay + random.uniform(-jitter, jitter)

        return int(delay)

    async def _update_delivery(
        self,
        delivery: WebhookDelivery,
        status: DeliveryStatus,
        status_code: int | None = None,
        response_body: str | None = None,
        error_message: str | None = None,
        delivered_at: datetime | None = None,
        next_retry_at: datetime | None = None,
    ) -> None:
        """Update delivery record.

        Args:
            delivery: Delivery to update
            status: New status
            status_code: HTTP status code
            response_body: Response body
            error_message: Error message
            delivered_at: Delivery timestamp
            next_retry_at: Next retry timestamp
        """
        async with self._lock:
            delivery.status = status
            delivery.updated_at = datetime.utcnow()

            if status_code is not None:
                delivery.status_code = status_code
            if response_body is not None:
                delivery.response_body = response_body
            if error_message is not None:
                delivery.error_message = error_message
            if delivered_at is not None:
                delivery.delivered_at = delivered_at
            if next_retry_at is not None:
                delivery.next_retry_at = next_retry_at

            self._deliveries[delivery.id] = delivery

    def _create_result(self, delivery: WebhookDelivery) -> WebhookDeliveryResult:
        """Create delivery result from delivery record.

        Args:
            delivery: Delivery record

        Returns:
            Delivery result
        """
        return WebhookDeliveryResult(
            delivery_id=delivery.id,
            webhook_id=delivery.webhook_id,
            event_id=delivery.event_id,
            status=delivery.status,
            attempt=delivery.attempt,
            status_code=delivery.status_code,
            response_body=delivery.response_body,
            error_message=delivery.error_message,
            delivered_at=delivery.delivered_at,
            next_retry_at=delivery.next_retry_at,
        )
