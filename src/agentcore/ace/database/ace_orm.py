"""
ACE SQLAlchemy ORM Models

Database models for ACE (Agentic Context Engineering) system.
Maps to tables created by ACE migration (c03db99da40b).
Cross-database compatible (PostgreSQL, SQLite).
"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator, CHAR
import uuid as uuid_lib

# Cross-database UUID type
class UUID(TypeDecorator):
    """Platform-independent UUID type.

    Uses PostgreSQL's UUID type when available, otherwise uses CHAR(36)
    storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PostgreSQL_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid_lib.UUID):
                return str(value)
            return str(uuid_lib.UUID(value))

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid_lib.UUID):
                return value
            return uuid_lib.UUID(value)

from agentcore.a2a_protocol.database.connection import Base
from agentcore.ace.models.ace_models import EvolutionStatusType


class ContextPlaybookDB(Base):
    """Context Playbook ORM model.

    Stores agent context with versioning and metadata.
    """

    __tablename__ = "context_playbooks"

    playbook_id = Column(
        UUID(),
        primary_key=True,
        default=uuid4,
    )
    agent_id = Column(
        String(255),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    context = Column(JSON, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    playbook_metadata = Column(
        "metadata", JSON, nullable=False, default=dict
    )

    # Relationships
    deltas = relationship(
        "ContextDeltaDB",
        back_populates="playbook",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_playbooks_agent", "agent_id"),
        Index("idx_playbooks_updated", "updated_at"),
    )


class ContextDeltaDB(Base):
    """Context Delta ORM model.

    Stores LLM-generated context improvements with confidence scores.
    """

    __tablename__ = "context_deltas"

    delta_id = Column(
        UUID(),
        primary_key=True,
        default=uuid4,
    )
    playbook_id = Column(
        UUID(),
        ForeignKey("context_playbooks.playbook_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    changes = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    applied = Column(Boolean, nullable=False, default=False)
    applied_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    playbook = relationship("ContextPlaybookDB", back_populates="deltas")

    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="check_confidence_range",
        ),
        Index("idx_deltas_playbook", "playbook_id"),
        Index("idx_deltas_confidence", "confidence"),
        Index("idx_deltas_applied", "applied", "applied_at"),
    )


class ExecutionTraceDB(Base):
    """Execution Trace ORM model.

    Captures agent performance data for LLM analysis.
    """

    __tablename__ = "execution_traces"

    trace_id = Column(
        UUID(),
        primary_key=True,
        default=uuid4,
    )
    agent_id = Column(
        String(255),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id = Column(String(255), nullable=True)
    execution_time = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    output_quality = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    trace_metadata = Column(
        "metadata", JSON, nullable=False, default=dict
    )
    captured_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        CheckConstraint(
            "output_quality IS NULL OR (output_quality >= 0.0 AND output_quality <= 1.0)",
            name="check_output_quality_range",
        ),
        Index("idx_traces_agent", "agent_id", "captured_at"),
        Index("idx_traces_success", "success", "captured_at"),
    )


class EvolutionStatusDB(Base):
    """Evolution Status ORM model.

    Tracks context evolution progress and costs per agent.
    """

    __tablename__ = "evolution_status"

    agent_id = Column(
        String(255),
        ForeignKey("agents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    last_evolution = Column(DateTime(timezone=True), nullable=True)
    pending_traces = Column(Integer, nullable=False, default=0)
    deltas_generated = Column(Integer, nullable=False, default=0)
    deltas_applied = Column(Integer, nullable=False, default=0)
    total_cost = Column(Float, nullable=False, default=0.0)
    # Use String for cross-database compatibility (CheckConstraint enforces values)
    status = Column(
        String(50),
        nullable=False,
        default="idle",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('idle', 'processing', 'failed')",
            name="check_status_values",
        ),
        Index("idx_evolution_status", "status", "last_evolution"),
    )


class PerformanceMetricsDB(Base):
    """Performance Metrics ORM model (COMPASS ACE-1).

    Stores stage-aware performance metrics for agents.
    Optimized for TimescaleDB hypertable storage.
    """

    __tablename__ = "performance_metrics"

    metric_id = Column(
        UUID(),
        primary_key=True,
        default=uuid4,
    )
    task_id = Column(UUID(), nullable=False, index=True)
    agent_id = Column(String(255), nullable=False, index=True)
    stage = Column(String(50), nullable=False, index=True)

    # Stage-specific metrics
    stage_success_rate = Column(Float, nullable=False)
    stage_error_rate = Column(Float, nullable=False)
    stage_duration_ms = Column(Integer, nullable=False)
    stage_action_count = Column(Integer, nullable=False)

    # Cross-stage metrics
    overall_progress_velocity = Column(Float, nullable=False)
    error_accumulation_rate = Column(Float, nullable=False)
    context_staleness_score = Column(Float, nullable=False)
    intervention_effectiveness = Column(Float, nullable=True)

    # Baseline comparison
    baseline_delta = Column(
        JSON, nullable=False, default=dict
    )

    recorded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    __table_args__ = (
        CheckConstraint(
            "stage_success_rate >= 0.0 AND stage_success_rate <= 1.0",
            name="check_success_rate_range",
        ),
        CheckConstraint(
            "stage_error_rate >= 0.0 AND stage_error_rate <= 1.0",
            name="check_error_rate_range",
        ),
        CheckConstraint(
            "context_staleness_score >= 0.0 AND context_staleness_score <= 1.0",
            name="check_staleness_range",
        ),
        CheckConstraint(
            "intervention_effectiveness IS NULL OR "
            "(intervention_effectiveness >= 0.0 AND intervention_effectiveness <= 1.0)",
            name="check_intervention_effectiveness_range",
        ),
        CheckConstraint(
            "stage IN ('planning', 'execution', 'reflection', 'verification')",
            name="check_stage_values",
        ),
        Index("idx_metrics_task", "task_id"),
        Index("idx_metrics_agent", "agent_id"),
        Index("idx_metrics_stage", "stage"),
        Index("idx_metrics_recorded", "recorded_at"),
        Index("idx_metrics_agent_stage", "agent_id", "stage", "recorded_at"),
    )


class InterventionRecordDB(Base):
    """Intervention Record ORM model (COMPASS ACE-2).

    Stores strategic intervention records for tracking effectiveness.
    """

    __tablename__ = "intervention_records"

    intervention_id = Column(
        UUID(),
        primary_key=True,
        default=uuid4,
    )
    task_id = Column(UUID(), nullable=False, index=True)
    agent_id = Column(
        String(255),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Trigger information
    trigger_type = Column(String(50), nullable=False)
    trigger_signals = Column(JSON, nullable=False)
    trigger_metric_id = Column(
        UUID(),
        ForeignKey("performance_metrics.metric_id", ondelete="SET NULL"),
        nullable=True,
    )

    # Decision information
    intervention_type = Column(String(50), nullable=False)
    intervention_rationale = Column(Text, nullable=False)
    decision_confidence = Column(Float, nullable=False)

    # Execution information
    executed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    execution_duration_ms = Column(Integer, nullable=False, default=0)
    execution_status = Column(String(20), nullable=False, default="pending")
    execution_error = Column(Text, nullable=True)

    # Outcome tracking
    pre_metric_id = Column(
        UUID(),
        ForeignKey("performance_metrics.metric_id", ondelete="SET NULL"),
        nullable=True,
    )
    post_metric_id = Column(
        UUID(),
        ForeignKey("performance_metrics.metric_id", ondelete="SET NULL"),
        nullable=True,
    )
    effectiveness_delta = Column(Float, nullable=True)

    # Metadata
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        CheckConstraint(
            "decision_confidence >= 0.0 AND decision_confidence <= 1.0",
            name="check_decision_confidence_range",
        ),
        CheckConstraint(
            "effectiveness_delta IS NULL OR (effectiveness_delta >= -1.0 AND effectiveness_delta <= 1.0)",
            name="check_effectiveness_delta_range",
        ),
        CheckConstraint(
            "trigger_type IN ('performance_degradation', 'error_accumulation', 'context_staleness', 'capability_mismatch')",
            name="check_trigger_type_values",
        ),
        CheckConstraint(
            "intervention_type IN ('context_refresh', 'replan', 'reflect', 'capability_switch')",
            name="check_intervention_type_values",
        ),
        CheckConstraint(
            "execution_status IN ('success', 'failure', 'partial', 'pending')",
            name="check_execution_status_values",
        ),
        Index("idx_intervention_task", "task_id"),
        Index("idx_intervention_agent", "agent_id"),
        Index("idx_intervention_trigger_type", "trigger_type"),
        Index("idx_intervention_type", "intervention_type"),
        Index("idx_intervention_executed", "executed_at"),
        Index("idx_intervention_agent_task", "agent_id", "task_id", "executed_at"),
    )
