"""
Workflow State Database Models

PostgreSQL models for persistent workflow state management with JSONB optimization.
"""

from datetime import UTC, datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from agentcore.a2a_protocol.database.connection import Base

# Cross-database JSON type (JSONB for PostgreSQL, JSON for others)
JSONType = JSON().with_variant(JSONB(), "postgresql")


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"
    CANCELLED = "cancelled"


class WorkflowExecutionDB(Base):
    """
    Workflow execution tracking with PostgreSQL persistence.

    Stores workflow execution metadata, state snapshots, and performance metrics.
    Uses JSONB for efficient querying of workflow state and configuration.
    """

    __tablename__ = "workflow_executions"

    # Primary key
    execution_id = Column(String(255), primary_key=True, index=True)

    # Workflow identification
    workflow_id = Column(String(255), nullable=False, index=True)
    workflow_name = Column(String(255), nullable=False)
    workflow_version = Column(String(50), nullable=False, default="1.0")

    # Orchestration pattern
    orchestration_pattern = Column(
        String(50), nullable=False, index=True
    )  # supervisor, hierarchical, saga, etc.

    # Status tracking
    status = Column(
        SQLEnum(WorkflowStatus, native_enum=False),
        nullable=False,
        default=WorkflowStatus.PENDING,
        index=True,
    )

    # Workflow definition (JSONB for PostgreSQL, JSON for SQLite)
    workflow_definition = Column(
        JSONType, nullable=False
    )  # Original workflow definition

    # Current execution state
    execution_state = Column(
        JSONType, nullable=False, default=dict
    )  # Current state snapshot

    # Agent allocation
    allocated_agents = Column(
        JSONType, nullable=False, default=dict
    )  # agent_role -> agent_id mapping

    # Task tracking
    task_states = Column(
        JSONType, nullable=False, default=dict
    )  # task_id -> task state mapping
    completed_tasks = Column(
        JSONType, nullable=False, default=list
    )  # List of completed task IDs
    failed_tasks = Column(JSONType, nullable=False, default=list)  # List of failed task IDs

    # Checkpointing
    checkpoint_data = Column(JSONType, nullable=True)  # Latest checkpoint data
    checkpoint_count = Column(Integer, nullable=False, default=0)
    last_checkpoint_at = Column(DateTime, nullable=True)

    # Performance metrics
    coordination_overhead_ms = Column(Integer, nullable=True)
    total_tasks = Column(Integer, nullable=False, default=0)
    completed_task_count = Column(Integer, nullable=False, default=0)
    failed_task_count = Column(Integer, nullable=False, default=0)

    # Timing
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    updated_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_stack_trace = Column(Text, nullable=True)
    compensation_errors = Column(JSON, nullable=False, default=list)

    # Metadata
    input_data = Column(JSONType, nullable=True)  # Input parameters
    output_data = Column(JSONType, nullable=True)  # Final output
    tags = Column(JSON, nullable=False, default=list)  # Tags for categorization
    workflow_metadata = Column(JSONType, nullable=False, default=dict)  # Additional metadata

    # Relationships
    state_history = relationship(
        "WorkflowStateDB", back_populates="execution", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Status and timing indexes for filtering
        Index("idx_workflow_status_created", "status", "created_at"),
        Index("idx_workflow_name_status", "workflow_name", "status"),
        Index("idx_workflow_pattern", "orchestration_pattern"),
        # JSONB indexes for efficient querying
        Index(
            "idx_workflow_execution_state", "execution_state", postgresql_using="gin"
        ),
        Index("idx_workflow_task_states", "task_states", postgresql_using="gin"),
        Index("idx_workflow_metadata", "workflow_metadata", postgresql_using="gin"),
        Index("idx_workflow_tags", "tags", postgresql_using="gin"),
        # Performance query index
        Index(
            "idx_workflow_completed_performance",
            "status",
            "duration_seconds",
            postgresql_where=(
                Column("status").in_(["completed", "failed", "compensated"])
            ),
        ),
    )


class WorkflowStateDB(Base):
    """
    Workflow state history for versioning and audit trails.

    Stores immutable snapshots of workflow state at different points in time.
    Enables state recovery and temporal queries.
    """

    __tablename__ = "workflow_state_history"

    # Primary key (Integer for SQLite compatibility with autoincrement)
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Reference to execution
    execution_id = Column(
        String(255),
        ForeignKey("workflow_executions.execution_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    execution = relationship("WorkflowExecutionDB", back_populates="state_history")

    # State versioning
    version = Column(Integer, nullable=False)  # Incremental version number
    state_type = Column(
        String(50), nullable=False, index=True
    )  # checkpoint, event, snapshot

    # State snapshot (JSONB for efficient storage)
    state_snapshot = Column(JSONType, nullable=False)  # Complete state at this point

    # Change tracking
    changed_fields = Column(
        JSON, nullable=True
    )  # List of fields changed from previous version
    change_reason = Column(String(255), nullable=True)  # Reason for state change

    # Timestamp
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    # Metadata
    created_by = Column(String(255), nullable=True)  # Agent or system that created state
    state_metadata = Column(JSONType, nullable=False, default=dict)

    __table_args__ = (
        # Composite index for efficient version queries
        Index("idx_state_execution_version", "execution_id", "version", unique=True),
        Index("idx_state_type_created", "state_type", "created_at"),
        # JSONB index for state queries
        Index("idx_state_snapshot", "state_snapshot", postgresql_using="gin"),
    )


class WorkflowStateVersion(Base):
    """
    Workflow state schema version tracking for migrations.

    Tracks schema versions to enable safe state migration when workflow
    definitions or state structures change.
    """

    __tablename__ = "workflow_state_versions"

    # Primary key
    version_id = Column(String(255), primary_key=True)

    # Version information
    schema_version = Column(Integer, nullable=False, unique=True, index=True)
    workflow_type = Column(
        String(50), nullable=False, index=True
    )  # Type of workflow (saga, supervisor, etc.)

    # Schema definition
    state_schema = Column(JSONType, nullable=False)  # JSON schema for state validation
    migration_script = Column(
        Text, nullable=True
    )  # SQL or Python script for migration

    # Version metadata
    description = Column(Text, nullable=True)
    is_active = Column(Integer, nullable=False, default=True, index=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    deprecated_at = Column(DateTime, nullable=True)

    # Migration tracking
    applied_to_executions = Column(
        JSON, nullable=False, default=list
    )  # List of execution IDs migrated

    __table_args__ = (
        Index("idx_version_type_active", "workflow_type", "is_active"),
    )
