"""
Stream Configuration

Configuration settings for Redis Streams integration.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StreamConfig(BaseModel):
    """Configuration for Redis Streams."""

    stream_name: str = Field(
        default="orchestration:events",
        description="Name of the Redis stream for orchestration events",
    )
    consumer_group_name: str = Field(
        default="orchestration-workers",
        description="Name of the consumer group",
    )
    consumer_name: str = Field(
        default="worker-1",
        description="Unique consumer name within the group",
    )
    block_ms: int = Field(
        default=5000,
        description="Time to block when reading from stream (milliseconds)",
    )
    count: int = Field(
        default=10,
        description="Maximum number of messages to read per operation",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for message processing",
    )
    retry_backoff_ms: int = Field(
        default=1000,
        description="Initial backoff delay for retries (milliseconds)",
    )
    max_stream_length: int = Field(
        default=10000,
        description="Maximum stream length (for XTRIM)",
    )
    trim_strategy: Literal["MAXLEN", "MINID"] = Field(
        default="MAXLEN",
        description="Stream trimming strategy",
    )
    dead_letter_stream: str = Field(
        default="orchestration:events:dlq",
        description="Dead letter queue stream name for failed messages",
    )
    enable_auto_claim: bool = Field(
        default=True,
        description="Enable automatic claiming of pending messages",
    )
    auto_claim_idle_ms: int = Field(
        default=60000,
        description="Time before idle messages can be claimed (milliseconds)",
    )
    ack_timeout_ms: int = Field(
        default=30000,
        description="Timeout for message acknowledgment (milliseconds)",
    )

    model_config = {
        "frozen": False,
    }
