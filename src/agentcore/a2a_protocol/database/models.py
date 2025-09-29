"""
Database Models

SQLAlchemy ORM models for agents, tasks, and related entities.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from agentcore.a2a_protocol.database.connection import Base
from agentcore.a2a_protocol.models.agent import AgentStatus
from agentcore.a2a_protocol.models.task import TaskStatus


class AgentDB(Base):
    """Agent database model."""

    __tablename__ = "agents"

    id = Column(String(255), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    status = Column(SQLEnum(AgentStatus), nullable=False, default=AgentStatus.INACTIVE, index=True)
    description = Column(Text, nullable=True)

    # Capabilities stored as JSON array
    capabilities = Column(JSON, nullable=False, default=list)

    # Requirements stored as JSON object
    requirements = Column(JSON, nullable=True)

    # Metadata stored as JSON object
    agent_metadata = Column(JSON, nullable=True)

    # Contact information
    endpoint = Column(String(512), nullable=True)

    # Load tracking
    current_load = Column(Integer, nullable=False, default=0)
    max_load = Column(Integer, nullable=False, default=10)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_seen = Column(DateTime, nullable=True)

    # Relationships
    tasks = relationship("TaskDB", back_populates="assigned_agent", foreign_keys="TaskDB.assigned_agent_id")
    health_metrics = relationship("AgentHealthMetricDB", back_populates="agent", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_agent_status_load", "status", "current_load"),
        Index("idx_agent_capabilities", "capabilities", postgresql_using="gin"),
    )


class TaskDB(Base):
    """Task database model."""

    __tablename__ = "tasks"

    id = Column(String(255), primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.PENDING, index=True)
    priority = Column(Integer, nullable=False, default=5, index=True)

    # Task requirements
    required_capabilities = Column(JSON, nullable=False, default=list)
    parameters = Column(JSON, nullable=True)

    # Agent assignment
    assigned_agent_id = Column(String(255), ForeignKey("agents.id"), nullable=True, index=True)
    assigned_agent = relationship("AgentDB", back_populates="tasks", foreign_keys=[assigned_agent_id])

    # Task dependencies
    depends_on = Column(JSON, nullable=True)  # Array of task IDs

    # Results and errors
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    # Timing
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Task metadata
    task_metadata = Column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_task_status_priority", "status", "priority"),
        Index("idx_task_agent_status", "assigned_agent_id", "status"),
    )


class AgentHealthMetricDB(Base):
    """Agent health metrics database model."""

    __tablename__ = "agent_health_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    agent_id = Column(String(255), ForeignKey("agents.id", ondelete="CASCADE"), nullable=False, index=True)
    agent = relationship("AgentDB", back_populates="health_metrics")

    # Health status
    is_healthy = Column(Boolean, nullable=False, default=True)
    status_code = Column(Integer, nullable=True)  # HTTP status code or equivalent

    # Response metrics
    response_time_ms = Column(Float, nullable=True)  # Response time in milliseconds

    # Error tracking
    error_message = Column(Text, nullable=True)
    consecutive_failures = Column(Integer, nullable=False, default=0)

    # Resource metrics
    cpu_percent = Column(Float, nullable=True)
    memory_mb = Column(Float, nullable=True)

    # Timestamp
    checked_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_health_agent_time", "agent_id", "checked_at"),
        Index("idx_health_status", "is_healthy", "checked_at"),
    )


class MessageQueueDB(Base):
    """Message queue database model for persistent message storage."""

    __tablename__ = "message_queue"

    id = Column(String(255), primary_key=True, index=True)

    # Message routing
    target_agent_id = Column(String(255), nullable=False, index=True)
    required_capabilities = Column(JSON, nullable=False, default=list)

    # Message content
    message_data = Column(JSON, nullable=False)  # MessageEnvelope serialized

    # Priority and TTL
    priority = Column(Integer, nullable=False, default=5, index=True)
    ttl_seconds = Column(Integer, nullable=True)
    expires_at = Column(DateTime, nullable=True, index=True)

    # Status tracking
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    last_error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_queue_target_priority", "target_agent_id", "priority"),
        Index("idx_queue_expires", "expires_at"),
    )


class EventSubscriptionDB(Base):
    """Event subscription database model for persistent subscriptions."""

    __tablename__ = "event_subscriptions"

    id = Column(String(255), primary_key=True, index=True)
    subscriber_id = Column(String(255), nullable=False, index=True)

    # Event filtering
    event_types = Column(JSON, nullable=False)  # Array of EventType values
    filters = Column(JSON, nullable=True)  # Optional filter criteria

    # Subscription lifecycle
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True, index=True)

    # Status
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    __table_args__ = (
        Index("idx_subscription_subscriber_active", "subscriber_id", "is_active"),
        Index("idx_subscription_expires", "expires_at"),
    )


class SecurityTokenDB(Base):
    """Security token database model for token tracking and revocation."""

    __tablename__ = "security_tokens"

    jti = Column(String(255), primary_key=True, index=True)  # JWT ID
    agent_id = Column(String(255), nullable=False, index=True)
    subject = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    token_type = Column(String(50), nullable=False)

    # Token lifecycle
    issued_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=False, index=True)

    # Revocation
    is_revoked = Column(Boolean, nullable=False, default=False, index=True)
    revoked_at = Column(DateTime, nullable=True)
    revocation_reason = Column(String(255), nullable=True)

    __table_args__ = (
        Index("idx_token_agent_active", "agent_id", "is_revoked"),
        Index("idx_token_expires", "expires_at"),
    )


class RateLimitDB(Base):
    """Rate limit tracking database model."""

    __tablename__ = "rate_limits"

    agent_id = Column(String(255), primary_key=True, index=True)

    # Rate limiting
    request_count = Column(Integer, nullable=False, default=0)
    window_start = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Configuration
    max_requests = Column(Integer, nullable=False, default=1000)
    window_seconds = Column(Integer, nullable=False, default=60)

    # Violation tracking
    total_violations = Column(Integer, nullable=False, default=0)
    last_violation = Column(DateTime, nullable=True)

    # Timestamps
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentPublicKeyDB(Base):
    """Agent public key storage for request signing verification."""

    __tablename__ = "agent_public_keys"

    agent_id = Column(String(255), primary_key=True, index=True)
    public_key_pem = Column(Text, nullable=False)

    # Key metadata
    algorithm = Column(String(50), nullable=False, default="RSA-2048")
    registered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    # Key rotation
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    replaced_by = Column(String(255), nullable=True)  # New key fingerprint

    __table_args__ = (
        Index("idx_public_key_active", "agent_id", "is_active"),
    )