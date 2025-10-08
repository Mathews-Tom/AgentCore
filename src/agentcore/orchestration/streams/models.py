"""
Stream Event Models

Pydantic models for orchestration events transmitted through Redis Streams.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of orchestration events."""

    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_FAILED = "agent_failed"
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"


class OrchestrationEvent(BaseModel):
    """Base event model for all orchestration events."""

    event_id: UUID = Field(
        default_factory=uuid4,
        description="Unique event identifier",
    )
    event_type: EventType = Field(
        description="Type of orchestration event",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event creation timestamp",
    )
    trace_id: UUID | None = Field(
        default=None,
        description="Distributed trace identifier for correlation",
    )
    source_agent_id: str | None = Field(
        default=None,
        description="ID of the agent that triggered this event",
    )
    workflow_id: UUID | None = Field(
        default=None,
        description="ID of the workflow this event belongs to",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event metadata",
    )

    model_config = {
        "frozen": False,
    }


class TaskCreatedEvent(OrchestrationEvent):
    """Event emitted when a task is created."""

    event_type: EventType = Field(default=EventType.TASK_CREATED, frozen=True)
    task_id: UUID = Field(
        description="Unique task identifier",
    )
    task_type: str = Field(
        description="Type of task to be executed",
    )
    agent_id: str | None = Field(
        default=None,
        description="Assigned agent ID",
    )
    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Task input data",
    )
    timeout_seconds: int = Field(
        default=300,
        description="Task execution timeout",
    )


class TaskCompletedEvent(OrchestrationEvent):
    """Event emitted when a task completes successfully."""

    event_type: EventType = Field(default=EventType.TASK_COMPLETED, frozen=True)
    task_id: UUID = Field(
        description="Unique task identifier",
    )
    agent_id: str = Field(
        description="Agent that executed the task",
    )
    result_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Task result data",
    )
    execution_time_ms: int = Field(
        description="Task execution time in milliseconds",
    )


class TaskFailedEvent(OrchestrationEvent):
    """Event emitted when a task fails."""

    event_type: EventType = Field(default=EventType.TASK_FAILED, frozen=True)
    task_id: UUID = Field(
        description="Unique task identifier",
    )
    agent_id: str | None = Field(
        default=None,
        description="Agent that attempted the task",
    )
    error_message: str = Field(
        description="Error message describing the failure",
    )
    error_type: str = Field(
        description="Type/class of the error",
    )
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts",
    )


class AgentStartedEvent(OrchestrationEvent):
    """Event emitted when an agent starts."""

    event_type: EventType = Field(default=EventType.AGENT_STARTED, frozen=True)
    agent_id: str = Field(
        description="Agent identifier",
    )
    agent_type: str = Field(
        description="Type of agent",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Agent capabilities",
    )


class AgentStoppedEvent(OrchestrationEvent):
    """Event emitted when an agent stops."""

    event_type: EventType = Field(default=EventType.AGENT_STOPPED, frozen=True)
    agent_id: str = Field(
        description="Agent identifier",
    )
    reason: str = Field(
        description="Reason for stopping",
    )
    uptime_seconds: int = Field(
        description="Agent uptime in seconds",
    )


class AgentFailedEvent(OrchestrationEvent):
    """Event emitted when an agent fails."""

    event_type: EventType = Field(default=EventType.AGENT_FAILED, frozen=True)
    agent_id: str = Field(
        description="Agent identifier",
    )
    error_message: str = Field(
        description="Error message describing the failure",
    )
    error_type: str = Field(
        description="Type/class of the error",
    )


class WorkflowCreatedEvent(OrchestrationEvent):
    """Event emitted when a workflow is created."""

    event_type: EventType = Field(default=EventType.WORKFLOW_CREATED, frozen=True)
    workflow_name: str = Field(
        description="Workflow name",
    )
    workflow_version: str = Field(
        description="Workflow version",
    )
    orchestration_pattern: str = Field(
        description="Orchestration pattern type",
    )


class WorkflowStartedEvent(OrchestrationEvent):
    """Event emitted when a workflow starts execution."""

    event_type: EventType = Field(default=EventType.WORKFLOW_STARTED, frozen=True)
    execution_id: UUID = Field(
        description="Workflow execution identifier",
    )


class WorkflowCompletedEvent(OrchestrationEvent):
    """Event emitted when a workflow completes."""

    event_type: EventType = Field(default=EventType.WORKFLOW_COMPLETED, frozen=True)
    execution_id: UUID = Field(
        description="Workflow execution identifier",
    )
    total_execution_time_ms: int = Field(
        description="Total workflow execution time",
    )
    tasks_completed: int = Field(
        description="Number of tasks completed",
    )


class WorkflowFailedEvent(OrchestrationEvent):
    """Event emitted when a workflow fails."""

    event_type: EventType = Field(default=EventType.WORKFLOW_FAILED, frozen=True)
    execution_id: UUID = Field(
        description="Workflow execution identifier",
    )
    error_message: str = Field(
        description="Error message",
    )
    failed_task_id: UUID | None = Field(
        default=None,
        description="ID of the task that caused failure",
    )
