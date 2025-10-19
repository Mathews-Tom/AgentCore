"""
Read Models Module

Denormalized data structures optimized for query operations.
Read models are updated by projections from event store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from agentcore.a2a_protocol.database.connection import Base


class WorkflowReadModel(Base):
    """
    Read model for workflow data.

    Denormalized view optimized for workflow queries.
    """

    __tablename__ = "workflow_read_model"

    workflow_id = Column(String(36), primary_key=True)
    workflow_name = Column(String(255), nullable=False, index=True)
    orchestration_pattern = Column(String(50), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)

    # Denormalized data
    agent_requirements = Column(JSONB, nullable=True)
    task_definitions = Column(JSONB, nullable=True)
    workflow_config = Column(JSONB, nullable=True)

    # Metadata
    created_by = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    updated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    deleted_at = Column(DateTime, nullable=True)

    # Computed fields
    total_executions = Column(Integer, nullable=False, default=0)
    successful_executions = Column(Integer, nullable=False, default=0)
    failed_executions = Column(Integer, nullable=False, default=0)
    average_execution_time_ms = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_workflow_status_created", "status", "created_at"),
        Index("idx_workflow_pattern_status", "orchestration_pattern", "status"),
    )


class ExecutionReadModel(Base):
    """
    Read model for workflow execution data.

    Denormalized view optimized for execution queries.
    """

    __tablename__ = "execution_read_model"

    execution_id = Column(String(36), primary_key=True)
    workflow_id = Column(String(36), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)

    # Input/Output
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)

    # Execution options
    execution_options = Column(JSONB, nullable=True)

    # Metadata
    started_by = Column(String(255), nullable=True)
    started_at = Column(DateTime, nullable=True, index=True)
    completed_at = Column(DateTime, nullable=True)
    paused_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    # Performance metrics
    execution_time_ms = Column(Integer, nullable=True)
    tasks_total = Column(Integer, nullable=False, default=0)
    tasks_completed = Column(Integer, nullable=False, default=0)
    tasks_failed = Column(Integer, nullable=False, default=0)
    tasks_pending = Column(Integer, nullable=False, default=0)

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_type = Column(String(255), nullable=True)
    failed_task_id = Column(String(36), nullable=True)

    __table_args__ = (
        Index("idx_execution_workflow_status", "workflow_id", "status"),
        Index("idx_execution_started_at", "started_at"),
        Index("idx_execution_workflow_started", "workflow_id", "started_at"),
    )


class AgentAssignmentReadModel(Base):
    """
    Read model for agent assignments.

    Denormalized view optimized for agent assignment queries.
    """

    __tablename__ = "agent_assignment_read_model"

    # Composite primary key
    workflow_id = Column(String(36), primary_key=True)
    agent_id = Column(String(255), primary_key=True)
    agent_role = Column(String(100), primary_key=True)

    # Agent details
    capabilities = Column(JSONB, nullable=True)

    # Assignment metadata
    assigned_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    unassigned_at = Column(DateTime, nullable=True)
    is_active = Column(Integer, nullable=False, default=1, index=True)  # Boolean as int

    # Performance tracking
    tasks_completed = Column(Integer, nullable=False, default=0)
    tasks_failed = Column(Integer, nullable=False, default=0)
    average_task_time_ms = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_assignment_agent_active", "agent_id", "is_active"),
        Index("idx_assignment_workflow_active", "workflow_id", "is_active"),
        Index("idx_assignment_role", "agent_role"),
    )


class TaskReadModel(Base):
    """
    Read model for task data.

    Denormalized view optimized for task queries.
    """

    __tablename__ = "task_read_model"

    task_id = Column(String(36), primary_key=True)
    workflow_id = Column(String(36), nullable=False, index=True)
    execution_id = Column(String(36), nullable=True, index=True)
    task_type = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)

    # Agent assignment
    agent_id = Column(String(255), nullable=True, index=True)

    # Task data
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)

    # Dependencies
    dependencies = Column(JSONB, nullable=True)  # Array of task IDs

    # Execution metadata
    scheduled_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    failed_at = Column(DateTime, nullable=True)

    # Performance
    execution_time_ms = Column(Integer, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_type = Column(String(255), nullable=True)

    __table_args__ = (
        Index("idx_task_workflow_status", "workflow_id", "status"),
        Index("idx_task_execution_status", "execution_id", "status"),
        Index("idx_task_agent_status", "agent_id", "status"),
        Index("idx_task_started_at", "started_at"),
    )


class WorkflowMetricsReadModel(Base):
    """
    Read model for workflow performance metrics.

    Aggregated metrics for workflow analysis.
    """

    __tablename__ = "workflow_metrics_read_model"

    workflow_id = Column(String(36), primary_key=True)
    execution_id = Column(String(36), primary_key=True)

    # Timing metrics
    total_execution_time_ms = Column(Integer, nullable=False)
    coordination_overhead_ms = Column(Integer, nullable=False, default=0)
    average_task_time_ms = Column(Float, nullable=True)

    # Resource metrics
    agents_allocated = Column(Integer, nullable=False)
    max_concurrent_tasks = Column(Integer, nullable=False, default=0)
    total_tasks = Column(Integer, nullable=False)

    # Success metrics
    tasks_completed_successfully = Column(Integer, nullable=False, default=0)
    tasks_failed = Column(Integer, nullable=False, default=0)
    tasks_retried = Column(Integer, nullable=False, default=0)

    # Efficiency metrics
    success_rate = Column(Float, nullable=True)  # 0.0 - 1.0
    retry_rate = Column(Float, nullable=True)  # 0.0 - 1.0
    throughput_tasks_per_second = Column(Float, nullable=True)

    # Timestamp
    computed_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    __table_args__ = (Index("idx_metrics_workflow", "workflow_id", "computed_at"),)
