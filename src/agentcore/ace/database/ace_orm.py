"""
ACE SQLAlchemy ORM Models

Database models for ACE (Agentic Context Engineering) system.
Maps to tables created by ACE migration (2b034c2a4021).
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
    String,
    Text,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from agentcore.a2a_protocol.database.connection import Base
from agentcore.ace.models.ace_models import EvolutionStatusType


class ContextPlaybookDB(Base):
    """Context Playbook ORM model.

    Stores agent context with versioning and metadata.
    """

    __tablename__ = "context_playbooks"

    playbook_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default="gen_random_uuid()",
    )
    agent_id = Column(
        String(255),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    context = Column(JSONB, nullable=False)
    version = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="NOW()",
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default="NOW()",
    )
    playbook_metadata = Column(
        "metadata", JSONB, nullable=False, default=dict, server_default="'{}'::jsonb"
    )

    # Relationships
    deltas = relationship(
        "ContextDeltaDB",
        back_populates="playbook",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_playbooks_agent", "agent_id"),
        Index(
            "idx_playbooks_updated",
            "updated_at",
            postgresql_ops={"updated_at": "DESC"},
        ),
    )


class ContextDeltaDB(Base):
    """Context Delta ORM model.

    Stores LLM-generated context improvements with confidence scores.
    """

    __tablename__ = "context_deltas"

    delta_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default="gen_random_uuid()",
    )
    playbook_id = Column(
        UUID(as_uuid=True),
        ForeignKey("context_playbooks.playbook_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    changes = Column(JSONB, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text, nullable=False)
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="NOW()",
    )
    applied = Column(Boolean, nullable=False, default=False, server_default="false")
    applied_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    playbook = relationship("ContextPlaybookDB", back_populates="deltas")

    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="check_confidence_range",
        ),
        Index("idx_deltas_playbook", "playbook_id"),
        Index(
            "idx_deltas_confidence",
            "confidence",
            postgresql_ops={"confidence": "DESC"},
        ),
        Index(
            "idx_deltas_applied",
            "applied",
            "applied_at",
            postgresql_ops={"applied_at": "DESC"},
        ),
    )


class ExecutionTraceDB(Base):
    """Execution Trace ORM model.

    Captures agent performance data for LLM analysis.
    """

    __tablename__ = "execution_traces"

    trace_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default="gen_random_uuid()",
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
        "metadata", JSONB, nullable=False, default=dict, server_default="'{}'::jsonb"
    )
    captured_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="NOW()",
    )

    __table_args__ = (
        CheckConstraint(
            "output_quality IS NULL OR (output_quality >= 0.0 AND output_quality <= 1.0)",
            name="check_output_quality_range",
        ),
        Index(
            "idx_traces_agent",
            "agent_id",
            "captured_at",
            postgresql_ops={"captured_at": "DESC"},
        ),
        Index(
            "idx_traces_success",
            "success",
            "captured_at",
            postgresql_ops={"captured_at": "DESC"},
        ),
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
    pending_traces = Column(Integer, nullable=False, default=0, server_default="0")
    deltas_generated = Column(Integer, nullable=False, default=0, server_default="0")
    deltas_applied = Column(Integer, nullable=False, default=0, server_default="0")
    total_cost = Column(Float, nullable=False, default=0.0, server_default="0.0")
    status = Column(
        SQLEnum(EvolutionStatusType, name="evolution_status_type", create_type=True),
        nullable=False,
        default=EvolutionStatusType.IDLE,
        server_default="'idle'",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('idle', 'processing', 'failed')",
            name="check_status_values",
        ),
        Index(
            "idx_evolution_status",
            "status",
            "last_evolution",
            postgresql_ops={"last_evolution": "DESC"},
        ),
    )
