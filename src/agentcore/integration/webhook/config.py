"""Webhook configuration."""

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class WebhookConfig(BaseSettings):
    """Webhook system configuration."""

    # Delivery settings
    default_max_retries: int = Field(default=3, ge=0, le=10)
    default_retry_delay_seconds: int = Field(default=60, ge=1)
    default_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_retry_delay_seconds: int = Field(default=3600, ge=1)

    # Retry backoff
    retry_backoff_multiplier: float = Field(default=2.0, ge=1.0)
    retry_backoff_jitter: bool = Field(default=True)

    # Rate limiting
    max_concurrent_deliveries: int = Field(default=100, ge=1)
    delivery_rate_limit: int = Field(default=1000, ge=1, description="Deliveries per minute")

    # Security
    require_signature: bool = Field(default=True)
    min_secret_length: int = Field(default=32, ge=16)

    # Storage
    max_delivery_history_days: int = Field(default=30, ge=1)
    cleanup_interval_hours: int = Field(default=24, ge=1)

    # Event queue
    event_queue_size: int = Field(default=10000, ge=100)
    event_batch_size: int = Field(default=100, ge=1)
    event_processing_interval_seconds: int = Field(default=1, ge=1)

    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_interval_seconds: int = Field(default=60, ge=1)

    model_config = ConfigDict(
        env_prefix="WEBHOOK_",
        case_sensitive=False,
    )
