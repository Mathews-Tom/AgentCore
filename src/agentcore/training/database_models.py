"""
SQLAlchemy ORM models for training infrastructure.

Defines database schema for training_jobs, trajectories, and policy_checkpoints tables.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    DECIMAL,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSON as JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from agentcore.a2a_protocol.database.connection import Base


class TrainingJobDB(Base):
    """SQLAlchemy model for training_jobs table."""

    __tablename__ = "training_jobs"

    job_id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        index=True,
    )
    agent_id = Column(String(255), nullable=False, index=True)
    status = Column(
        String(50),
        nullable=False,
        index=True,
        default="queued",
    )
    config = Column(JSONB, nullable=False)
    training_data = Column(JSONB, nullable=False)
    current_iteration = Column(Integer, nullable=False, default=0)
    total_iterations = Column(Integer, nullable=False)
    metrics = Column(JSONB, nullable=False, default=lambda: {})
    cost_usd = Column(DECIMAL(10, 2), nullable=False, default=Decimal("0.00"))
    budget_usd = Column(DECIMAL(10, 2), nullable=False)
    best_checkpoint_id = Column(PG_UUID(as_uuid=True), nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String(1000), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="check_training_job_status",
        ),
        Index("idx_training_jobs_agent", "agent_id"),
        Index("idx_training_jobs_status", "status"),
        Index(
            "idx_training_jobs_created",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
    )


class TrajectoryDB(Base):
    """SQLAlchemy model for trajectories table."""

    __tablename__ = "trajectories"

    trajectory_id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        index=True,
    )
    job_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("training_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_id = Column(String(255), nullable=False, index=True)
    query = Column(Text, nullable=False)
    steps = Column(JSONB, nullable=False)
    reward = Column(Float, nullable=False, default=0.0)
    normalized_reward = Column(Float, nullable=True, default=0.0)
    advantage = Column(Float, nullable=True, default=0.0)
    execution_time_ms = Column(Integer, nullable=True)
    success = Column(Boolean, nullable=True, index=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    __table_args__ = (
        Index("idx_trajectories_job", "job_id"),
        Index("idx_trajectories_agent", "agent_id"),
        Index(
            "idx_trajectories_created",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index("idx_trajectories_success", "success"),
        Index("idx_trajectories_steps_gin", "steps", postgresql_using="gin"),
    )


class PolicyCheckpointDB(Base):
    """SQLAlchemy model for policy_checkpoints table."""

    __tablename__ = "policy_checkpoints"

    checkpoint_id = Column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        nullable=False,
        index=True,
    )
    agent_id = Column(String(255), nullable=False, index=True)
    job_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("training_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    iteration = Column(Integer, nullable=False)
    policy_data = Column(JSONB, nullable=True)
    policy_s3_path = Column(String(500), nullable=True)
    validation_score = Column(Float, nullable=False, default=0.0)
    metrics = Column(JSONB, nullable=False, default=lambda: {})
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    __table_args__ = (
        Index("idx_policy_checkpoints_agent", "agent_id"),
        Index("idx_policy_checkpoints_job", "job_id"),
        Index(
            "idx_policy_checkpoints_created",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index("idx_policy_checkpoints_validation", "validation_score"),
    )
