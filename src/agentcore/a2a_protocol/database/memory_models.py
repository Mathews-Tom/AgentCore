"""
Memory System Database Models

SQLAlchemy ORM models for the hybrid memory system (Mem0 + COMPASS + Graph).

Implements:
- MemoryModel: Main memory storage with vector embeddings
- StageMemoryModel: COMPASS stage compression summaries
- TaskContextModel: Progressive task context tracking
- ErrorModel: Error tracking with pattern detection
- CompressionMetricsModel: Cost tracking for compression operations

Component ID: MEM-006
Ticket: MEM-006 (Implement SQLAlchemy ORM Models)
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from agentcore.a2a_protocol.database.connection import Base
from agentcore.a2a_protocol.models.memory import (
    ErrorRecord,
    ErrorType,
    MemoryLayer,
    MemoryRecord,
    StageMemory,
    StageType,
    TaskContext,
)


class MemoryModel(Base):
    """
    Memory database model for all memory layers.

    Stores memories across episodic, semantic, and procedural layers.
    Working memory stored in Redis cache separately.
    Uses pgvector for semantic similarity search.
    """

    __tablename__ = "memories"

    # Primary key
    memory_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Memory classification
    memory_layer: Mapped[str] = mapped_column(
        SQLEnum(MemoryLayer, name="memorylayer", create_type=False),
        nullable=False,
        index=True,
    )

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        ARRAY(Float), nullable=True
    )  # Vector type handled by migration

    # Scope (at least one required)
    agent_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )
    session_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )
    user_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )
    task_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )

    # Metadata
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    entities: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    facts: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    keywords: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )

    # Relationships
    related_memory_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False, server_default="{}"
    )
    parent_memory_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True
    )

    # Tracking
    relevance_score: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="1.0"
    )
    access_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    last_accessed: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # COMPASS enhancements
    stage_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )
    is_critical: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    criticality_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Procedural memory fields
    actions: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    outcome: Mapped[str | None] = mapped_column(Text, nullable=True)
    success: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Audit
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="CURRENT_TIMESTAMP",
    )

    __table_args__ = (
        # Composite indexes for efficient queries
        Index("idx_memories_agent_session", "agent_id", "session_id"),
        Index("idx_memories_task_stage", "task_id", "stage_id"),
        Index("idx_memories_layer_agent", "memory_layer", "agent_id"),
    )

    def to_pydantic(self) -> MemoryRecord:
        """Convert ORM model to Pydantic model."""
        return MemoryRecord(
            memory_id=str(self.memory_id),
            memory_layer=MemoryLayer(self.memory_layer),
            content=self.content,
            summary=self.summary,
            embedding=self.embedding or [],
            agent_id=str(self.agent_id) if self.agent_id else None,
            session_id=str(self.session_id) if self.session_id else None,
            user_id=str(self.user_id) if self.user_id else None,
            task_id=str(self.task_id) if self.task_id else None,
            timestamp=self.timestamp,
            entities=self.entities or [],
            facts=self.facts or [],
            keywords=self.keywords or [],
            related_memory_ids=[str(mid) for mid in (self.related_memory_ids or [])],
            parent_memory_id=str(self.parent_memory_id)
            if self.parent_memory_id
            else None,
            relevance_score=self.relevance_score if self.relevance_score is not None else 1.0,
            access_count=self.access_count if self.access_count is not None else 0,
            last_accessed=self.last_accessed,
            stage_id=str(self.stage_id) if self.stage_id else None,
            is_critical=self.is_critical if self.is_critical is not None else False,
            criticality_reason=self.criticality_reason,
            actions=self.actions or [],
            outcome=self.outcome,
            success=self.success,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<MemoryModel(memory_id={self.memory_id}, "
            f"layer={self.memory_layer}, "
            f"agent_id={self.agent_id}, "
            f"is_critical={self.is_critical})>"
        )


class StageMemoryModel(Base):
    """
    Stage memory database model for COMPASS hierarchical organization.

    Stores compressed summaries of raw memories per reasoning stage.
    Implements 10:1 compression ratio target.
    """

    __tablename__ = "stage_memories"

    # Primary key
    stage_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Identifiers
    task_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False, index=True
    )
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False, index=True
    )

    # Stage classification
    stage_type: Mapped[str] = mapped_column(
        SQLEnum(StageType, name="stagetype", create_type=False),
        nullable=False,
        index=True,
    )

    # Content
    stage_summary: Mapped[str] = mapped_column(Text, nullable=False)
    stage_insights: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )

    # Raw memory references (JSONB for flexibility)
    raw_memory_refs: Mapped[list[UUID]] = mapped_column(
        ARRAY(PGUUID(as_uuid=True)), nullable=False
    )

    # Tracking
    relevance_score: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="1.0"
    )

    # Compression metrics
    compression_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    compression_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Quality metrics stored as JSONB for flexibility
    quality_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="CURRENT_TIMESTAMP",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="CURRENT_TIMESTAMP",
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        # Composite index for task + stage type queries
        Index("idx_stage_memories_task_type", "task_id", "stage_type"),
    )

    def to_pydantic(self) -> StageMemory:
        """Convert ORM model to Pydantic model."""
        return StageMemory(
            stage_id=str(self.stage_id),
            task_id=str(self.task_id),
            agent_id=str(self.agent_id),
            stage_type=StageType(self.stage_type),
            stage_summary=self.stage_summary,
            stage_insights=self.stage_insights or [],
            raw_memory_refs=[str(ref) for ref in (self.raw_memory_refs or [])],
            relevance_score=self.relevance_score if self.relevance_score is not None else 1.0,
            compression_ratio=self.compression_ratio or 1.0,
            compression_model=self.compression_model or "unknown",
            quality_score=self.quality_metrics.get("quality_score", 1.0)
            if self.quality_metrics
            else 1.0,
            created_at=self.created_at or datetime.now(UTC),
            updated_at=self.updated_at or datetime.now(UTC),
            completed_at=self.completed_at,
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<StageMemoryModel(stage_id={self.stage_id}, "
            f"task_id={self.task_id}, "
            f"type={self.stage_type}, "
            f"compression={self.compression_ratio})>"
        )


class TaskContextModel(Base):
    """
    Task context database model for COMPASS progressive summarization.

    Stores progressive task summary from stage summaries.
    Implements 5:1 compression ratio target.
    """

    __tablename__ = "task_contexts"

    # Primary key
    task_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Identifiers
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False, index=True
    )

    # Task information
    task_goal: Mapped[str] = mapped_column(Text, nullable=False)
    current_stage_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )

    # Progressive summary
    task_progress_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    critical_constraints: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )

    # Performance metrics (JSONB for flexibility)
    performance_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="CURRENT_TIMESTAMP",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="CURRENT_TIMESTAMP",
        onupdate=lambda: datetime.now(UTC),
    )

    def to_pydantic(self) -> TaskContext:
        """Convert ORM model to Pydantic model."""
        return TaskContext(
            task_id=str(self.task_id),
            agent_id=str(self.agent_id),
            task_goal=self.task_goal,
            current_stage_id=str(self.current_stage_id)
            if self.current_stage_id
            else None,
            task_progress_summary=self.task_progress_summary or "",
            critical_constraints=self.critical_constraints or [],
            performance_metrics=self.performance_metrics or {},
            created_at=self.created_at or datetime.now(UTC),
            updated_at=self.updated_at or datetime.now(UTC),
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<TaskContextModel(task_id={self.task_id}, "
            f"agent_id={self.agent_id}, "
            f"current_stage={self.current_stage_id})>"
        )


class ErrorModel(Base):
    """
    Error record database model for COMPASS error tracking.

    Captures errors explicitly for pattern detection and learning.
    Severity scored on 0-1 scale (0=minor, 1=critical).
    """

    __tablename__ = "error_records"

    # Primary key
    error_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # Identifiers
    task_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False, index=True
    )
    stage_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )
    agent_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), nullable=False, index=True
    )

    # Error classification
    error_type: Mapped[str] = mapped_column(
        SQLEnum(ErrorType, name="errortype", create_type=False),
        nullable=False,
        index=True,
    )

    # Error details
    error_description: Mapped[str] = mapped_column(Text, nullable=False)
    context_when_occurred: Mapped[str | None] = mapped_column(Text, nullable=True)
    recovery_action: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Severity
    error_severity: Mapped[float] = mapped_column(Float, nullable=False)

    # Timestamp
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="CURRENT_TIMESTAMP",
        index=True,
    )

    __table_args__ = (
        # Constraint validation for severity range
        CheckConstraint(
            "error_severity >= 0.0 AND error_severity <= 1.0",
            name="error_severity_range",
        ),
        # Composite index for task + error type queries
        Index("idx_error_records_task_type", "task_id", "error_type"),
    )

    def to_pydantic(self) -> ErrorRecord:
        """Convert ORM model to Pydantic model."""
        return ErrorRecord(
            error_id=str(self.error_id),
            task_id=str(self.task_id),
            stage_id=str(self.stage_id) if self.stage_id else None,
            agent_id=str(self.agent_id),
            error_type=ErrorType(self.error_type),
            error_description=self.error_description,
            context_when_occurred=self.context_when_occurred or "",
            recovery_action=self.recovery_action,
            error_severity=self.error_severity,
            recorded_at=self.recorded_at or datetime.now(UTC),
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<ErrorModel(error_id={self.error_id}, "
            f"type={self.error_type}, "
            f"task_id={self.task_id}, "
            f"severity={self.error_severity})>"
        )


class CompressionMetricsModel(Base):
    """
    Compression metrics database model for COMPASS cost tracking.

    Tracks compression quality, ratio, and cost for test-time scaling.
    Used for monitoring 70-80% cost reduction target.
    """

    __tablename__ = "compression_metrics"

    # Primary key
    metric_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid4
    )

    # References
    stage_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )
    task_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), nullable=True, index=True
    )

    # Compression type
    compression_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )

    # Token metrics
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    # Quality metrics
    compression_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    critical_fact_retention_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    coherence_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Cost tracking
    cost_usd: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Timestamp
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default="CURRENT_TIMESTAMP",
        index=True,
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<CompressionMetricsModel(metric_id={self.metric_id}, "
            f"type={self.compression_type}, "
            f"ratio={self.compression_ratio}, "
            f"cost=${self.cost_usd})>"
        )


# Export all models
__all__ = [
    "MemoryModel",
    "StageMemoryModel",
    "TaskContextModel",
    "ErrorModel",
    "CompressionMetricsModel",
]
