"""Integration tests for webhook system."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import httpx
import pytest

from agentcore.integration.webhook import (
    DeliveryService,
    DeliveryStatus,
    EventPayload,
    EventPublisher,
    WebhookConfig,
    WebhookDeliveryError,
    WebhookEvent,
    WebhookManager,
    WebhookNotFoundError,
    WebhookRegistration,
    WebhookValidationError,
    WebhookValidator,
)


@pytest.fixture
def webhook_config():
    """Create test webhook configuration."""
    return WebhookConfig(
        default_max_retries=2,
        default_retry_delay_seconds=1,
        default_timeout_seconds=5,
        max_concurrent_deliveries=10,
    )


@pytest.fixture
async def webhook_manager(webhook_config):
    """Create webhook manager instance."""
    return WebhookManager(config=webhook_config)


@pytest.fixture
async def event_publisher(webhook_config):
    """Create event publisher instance."""
    publisher = EventPublisher(config=webhook_config)
    await publisher.start()
    yield publisher
    await publisher.stop()


@pytest.fixture
async def delivery_service(webhook_config):
    """Create delivery service instance."""
    service = DeliveryService(config=webhook_config)
    yield service
    await service.close()


class TestWebhookManager:
    """Test webhook registration and management."""

    @pytest.mark.asyncio
    async def test_register_webhook(self, webhook_manager):
        """Test webhook registration."""
        webhook = await webhook_manager.register(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )

        assert webhook.name == "Test Webhook"
        assert str(webhook.url) == "https://api.example.com/webhook"
        assert WebhookEvent.TASK_COMPLETED in webhook.events
        assert webhook.active is True
        assert len(webhook.secret) >= 32

    @pytest.mark.asyncio
    async def test_register_webhook_custom_secret(self, webhook_manager):
        """Test webhook registration with custom secret."""
        secret = "a" * 32  # Minimum length
        webhook = await webhook_manager.register(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret=secret,
        )

        assert webhook.secret == secret

    @pytest.mark.asyncio
    async def test_register_webhook_validation_error(self, webhook_manager):
        """Test webhook registration validation."""
        with pytest.raises(WebhookValidationError, match="At least one event"):
            await webhook_manager.register(
                name="Test Webhook",
                url="https://api.example.com/webhook",
                events=[],  # Empty events list
            )

    @pytest.mark.asyncio
    async def test_get_webhook(self, webhook_manager):
        """Test getting webhook by ID."""
        webhook = await webhook_manager.register(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )

        retrieved = await webhook_manager.get(webhook.id)
        assert retrieved.id == webhook.id
        assert retrieved.name == webhook.name

    @pytest.mark.asyncio
    async def test_get_webhook_not_found(self, webhook_manager):
        """Test getting non-existent webhook."""
        with pytest.raises(WebhookNotFoundError):
            await webhook_manager.get(uuid4())

    @pytest.mark.asyncio
    async def test_list_webhooks(self, webhook_manager):
        """Test listing webhooks."""
        webhook1 = await webhook_manager.register(
            name="Webhook 1",
            url="https://api.example.com/webhook1",
            events=[WebhookEvent.TASK_COMPLETED],
        )

        webhook2 = await webhook_manager.register(
            name="Webhook 2",
            url="https://api.example.com/webhook2",
            events=[WebhookEvent.TASK_STARTED],
        )

        webhooks = await webhook_manager.list()
        assert len(webhooks) == 2
        assert any(w.id == webhook1.id for w in webhooks)
        assert any(w.id == webhook2.id for w in webhooks)

    @pytest.mark.asyncio
    async def test_list_webhooks_filtered(self, webhook_manager):
        """Test listing webhooks with filters."""
        await webhook_manager.register(
            name="Webhook 1",
            url="https://api.example.com/webhook1",
            events=[WebhookEvent.TASK_COMPLETED],
        )

        webhook2 = await webhook_manager.register(
            name="Webhook 2",
            url="https://api.example.com/webhook2",
            events=[WebhookEvent.TASK_STARTED],
        )

        # Filter by event type
        webhooks = await webhook_manager.list(
            event_type=WebhookEvent.TASK_STARTED
        )
        assert len(webhooks) == 1
        assert webhooks[0].id == webhook2.id

    @pytest.mark.asyncio
    async def test_update_webhook(self, webhook_manager):
        """Test updating webhook."""
        webhook = await webhook_manager.register(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )

        updated = await webhook_manager.update(
            webhook.id,
            name="Updated Webhook",
            active=False,
        )

        assert updated.name == "Updated Webhook"
        assert updated.active is False

    @pytest.mark.asyncio
    async def test_delete_webhook(self, webhook_manager):
        """Test deleting webhook."""
        webhook = await webhook_manager.register(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )

        await webhook_manager.delete(webhook.id)

        with pytest.raises(WebhookNotFoundError):
            await webhook_manager.get(webhook.id)

    @pytest.mark.asyncio
    async def test_webhook_stats(self, webhook_manager):
        """Test webhook statistics."""
        webhook = await webhook_manager.register(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
        )

        # Update stats
        await webhook_manager.update_stats(webhook.id, success=True, latency_ms=100)
        await webhook_manager.update_stats(webhook.id, success=True, latency_ms=200)
        await webhook_manager.update_stats(webhook.id, success=False)

        stats = await webhook_manager.get_stats(webhook.id)
        assert stats.total_deliveries == 3
        assert stats.successful_deliveries == 2
        assert stats.failed_deliveries == 1
        assert stats.success_rate == pytest.approx(66.67, rel=0.1)


class TestEventPublisher:
    """Test event publishing and subscription."""

    @pytest.mark.asyncio
    async def test_publish_event(self, event_publisher):
        """Test publishing an event."""
        event = await event_publisher.publish(
            event_type=WebhookEvent.TASK_COMPLETED,
            data={"task_id": "123"},
            source="test",
        )

        assert event.event_type == WebhookEvent.TASK_COMPLETED
        assert event.data["task_id"] == "123"
        assert event.source == "test"

    @pytest.mark.asyncio
    async def test_subscribe_webhook(self, event_publisher):
        """Test webhook subscription."""
        webhook_id = uuid4()
        subscription = await event_publisher.subscribe(
            webhook_id,
            [WebhookEvent.TASK_COMPLETED, WebhookEvent.TASK_STARTED],
        )

        assert subscription.webhook_id == webhook_id
        assert len(subscription.event_types) == 2

    @pytest.mark.asyncio
    async def test_get_subscribers(self, event_publisher):
        """Test getting event subscribers."""
        webhook_id1 = uuid4()
        webhook_id2 = uuid4()

        await event_publisher.subscribe(
            webhook_id1, [WebhookEvent.TASK_COMPLETED]
        )
        await event_publisher.subscribe(
            webhook_id2, [WebhookEvent.TASK_COMPLETED]
        )
        await event_publisher.subscribe(
            webhook_id2, [WebhookEvent.TASK_STARTED]
        )

        subscribers = await event_publisher.get_subscribers(
            WebhookEvent.TASK_COMPLETED
        )
        assert len(subscribers) == 2
        assert webhook_id1 in subscribers
        assert webhook_id2 in subscribers

        subscribers = await event_publisher.get_subscribers(
            WebhookEvent.TASK_STARTED
        )
        assert len(subscribers) == 1
        assert webhook_id2 in subscribers

    @pytest.mark.asyncio
    async def test_unsubscribe_webhook(self, event_publisher):
        """Test webhook unsubscription."""
        webhook_id = uuid4()

        await event_publisher.subscribe(
            webhook_id, [WebhookEvent.TASK_COMPLETED]
        )

        subscribers = await event_publisher.get_subscribers(
            WebhookEvent.TASK_COMPLETED
        )
        assert webhook_id in subscribers

        await event_publisher.unsubscribe(webhook_id)

        subscribers = await event_publisher.get_subscribers(
            WebhookEvent.TASK_COMPLETED
        )
        assert webhook_id not in subscribers


class TestDeliveryService:
    """Test webhook delivery with retries."""

    @pytest.mark.asyncio
    async def test_successful_delivery(self, delivery_service):
        """Test successful webhook delivery."""
        webhook = WebhookRegistration(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret="a" * 32,
        )

        event = EventPayload(
            event_type=WebhookEvent.TASK_COMPLETED,
            data={"task_id": "123"},
            source="test",
        )

        # Mock HTTP client
        with patch.object(delivery_service, "_client") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            mock_client.post = AsyncMock(return_value=mock_response)

            delivery = await delivery_service.schedule(webhook, event)

            # Wait for delivery to complete
            await asyncio.sleep(0.5)

            result = await delivery_service.get_delivery(delivery.id)
            assert result is not None
            assert result.status == DeliveryStatus.DELIVERED
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_delivery_retry_on_failure(self, delivery_service):
        """Test delivery retry on HTTP error."""
        webhook = WebhookRegistration(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret="a" * 32,
            max_retries=2,
            retry_delay_seconds=1,
        )

        event = EventPayload(
            event_type=WebhookEvent.TASK_COMPLETED,
            data={"task_id": "123"},
            source="test",
        )

        # Mock HTTP client to fail once, then succeed
        with patch.object(delivery_service, "_client") as mock_client:
            mock_response_fail = Mock()
            mock_response_fail.status_code = 500
            mock_response_fail.text = "Internal Server Error"

            mock_response_success = Mock()
            mock_response_success.status_code = 200
            mock_response_success.text = "OK"

            mock_client.post = AsyncMock(
                side_effect=[mock_response_fail, mock_response_success]
            )

            delivery = await delivery_service.schedule(webhook, event)

            # Wait for retries
            await asyncio.sleep(2.5)

            result = await delivery_service.get_delivery(delivery.id)
            assert result is not None
            assert result.status == DeliveryStatus.DELIVERED
            assert result.attempt >= 1  # At least one retry

    @pytest.mark.asyncio
    async def test_delivery_exhausted(self, delivery_service):
        """Test delivery exhaustion after max retries."""
        webhook = WebhookRegistration(
            name="Test Webhook",
            url="https://api.example.com/webhook",
            events=[WebhookEvent.TASK_COMPLETED],
            secret="a" * 32,
            max_retries=1,
            retry_delay_seconds=1,
        )

        event = EventPayload(
            event_type=WebhookEvent.TASK_COMPLETED,
            data={"task_id": "123"},
            source="test",
        )

        # Mock HTTP client to always fail
        with patch.object(delivery_service, "_client") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_client.post = AsyncMock(return_value=mock_response)

            delivery = await delivery_service.schedule(webhook, event)

            # Wait for all retries
            await asyncio.sleep(3.5)

            result = await delivery_service.get_delivery(delivery.id)
            assert result is not None
            assert result.status == DeliveryStatus.EXHAUSTED


class TestWebhookValidator:
    """Test webhook validation and security."""

    def test_generate_signature(self):
        """Test signature generation."""
        validator = WebhookValidator()
        secret = "test_secret_key"
        payload = {"event_id": "123", "data": {"test": "value"}}

        signature = validator.generate_signature(secret, payload)

        assert signature.startswith("sha256=")
        assert len(signature) > 10

    def test_verify_signature_valid(self):
        """Test signature verification with valid signature."""
        validator = WebhookValidator()
        secret = "test_secret_key"
        payload = {"event_id": "123", "data": {"test": "value"}}

        signature = validator.generate_signature(secret, payload)
        result = validator.verify_signature(secret, payload, signature)

        assert result is True

    def test_verify_signature_invalid(self):
        """Test signature verification with invalid signature."""
        from agentcore.integration.webhook.exceptions import SignatureVerificationError

        validator = WebhookValidator()
        secret = "test_secret_key"
        payload = {"event_id": "123", "data": {"test": "value"}}

        invalid_signature = "sha256=invalid"

        with pytest.raises(SignatureVerificationError):
            validator.verify_signature(secret, payload, invalid_signature)

    def test_validate_secret_short(self):
        """Test short secret validation."""
        validator = WebhookValidator()

        with pytest.raises(WebhookValidationError, match="at least"):
            validator.validate_secret("short")

    def test_validate_payload(self):
        """Test payload validation."""
        validator = WebhookValidator()

        valid_payload = {
            "event_id": "123",
            "event_type": "task.completed",
            "timestamp": "2025-01-01T00:00:00Z",
            "data": {"test": "value"},
            "source": "test",
        }

        # Should not raise
        validator.validate_payload(valid_payload)

    def test_validate_payload_missing_field(self):
        """Test payload validation with missing field."""
        validator = WebhookValidator()

        invalid_payload = {
            "event_id": "123",
            # Missing required fields
        }

        with pytest.raises(WebhookValidationError, match="Missing required field"):
            validator.validate_payload(invalid_payload)


@pytest.mark.asyncio
async def test_end_to_end_workflow(
    webhook_manager,
    event_publisher,
    delivery_service,
):
    """Test complete webhook workflow end-to-end."""
    # 1. Register webhook
    webhook = await webhook_manager.register(
        name="Test Webhook",
        url="https://api.example.com/webhook",
        events=[WebhookEvent.TASK_COMPLETED],
    )

    # 2. Subscribe to events
    await event_publisher.subscribe(webhook.id, [WebhookEvent.TASK_COMPLETED])

    # 3. Publish event
    event = await event_publisher.publish(
        event_type=WebhookEvent.TASK_COMPLETED,
        data={"task_id": "123", "status": "completed"},
        source="agentcore.tasks",
    )

    # 4. Mock delivery
    with patch.object(delivery_service, "_client") as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "OK"
        mock_client.post = AsyncMock(return_value=mock_response)

        # 5. Schedule delivery
        delivery = await delivery_service.schedule(webhook, event)

        # Wait for delivery
        await asyncio.sleep(0.5)

        # 6. Verify delivery
        result = await delivery_service.get_delivery(delivery.id)
        assert result is not None
        assert result.status == DeliveryStatus.DELIVERED

        # 7. Check webhook stats
        await webhook_manager.update_stats(webhook.id, success=True, latency_ms=100)
        stats = await webhook_manager.get_stats(webhook.id)
        assert stats.successful_deliveries == 1
