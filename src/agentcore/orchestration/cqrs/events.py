"""
CQRS Events Module

Event models for event sourcing and audit trails.
All state changes are recorded as immutable events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of domain events."""

    # Workflow events
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"

    # Agent events
    AGENT_ASSIGNED = "agent_assigned"
    AGENT_UNASSIGNED = "agent_unassigned"
    AGENT_REPLACED = "agent_replaced"

    # Task events
    TASK_SCHEDULED = "task_scheduled"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRIED = "task_retried"

    # State events
    STATE_SNAPSHOT_CREATED = "state_snapshot_created"


class DomainEvent(BaseModel):
    """
    Base domain event for event sourcing.

    All events are immutable records of state changes.
    """

    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    event_type: EventType = Field(description="Type of event")
    aggregate_id: UUID = Field(description="ID of aggregate this event belongs to")
    aggregate_type: str = Field(description="Type of aggregate (workflow, task, etc)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Event timestamp"
    )
    version: int = Field(
        default=1, ge=1, description="Event version for aggregate versioning"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional event metadata"
    )

    model_config = {"frozen": True}


class WorkflowCreatedEvent(DomainEvent):
    """Event published when a workflow is created."""

    event_type: EventType = Field(default=EventType.WORKFLOW_CREATED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    workflow_name: str = Field(description="Workflow name")
    orchestration_pattern: str = Field(description="Orchestration pattern type")
    agent_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Required agent capabilities"
    )
    task_definitions: list[dict[str, Any]] = Field(
        default_factory=list, description="Task definitions"
    )
    created_by: str | None = Field(default=None, description="User who created workflow")


class WorkflowUpdatedEvent(DomainEvent):
    """Event published when a workflow is updated."""

    event_type: EventType = Field(default=EventType.WORKFLOW_UPDATED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    changes: dict[str, Any] = Field(description="Changed fields and new values")
    updated_by: str | None = Field(default=None, description="User who updated workflow")


class WorkflowStartedEvent(DomainEvent):
    """Event published when workflow execution starts."""

    event_type: EventType = Field(default=EventType.WORKFLOW_STARTED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Workflow input data"
    )
    started_by: str | None = Field(
        default=None, description="User who started execution"
    )


class WorkflowPausedEvent(DomainEvent):
    """Event published when workflow execution is paused."""

    event_type: EventType = Field(default=EventType.WORKFLOW_PAUSED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    reason: str | None = Field(default=None, description="Reason for pausing")
    paused_by: str | None = Field(default=None, description="User who paused execution")


class WorkflowResumedEvent(DomainEvent):
    """Event published when workflow execution is resumed."""

    event_type: EventType = Field(default=EventType.WORKFLOW_RESUMED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    resumed_by: str | None = Field(
        default=None, description="User who resumed execution"
    )


class WorkflowCompletedEvent(DomainEvent):
    """Event published when workflow execution completes successfully."""

    event_type: EventType = Field(default=EventType.WORKFLOW_COMPLETED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Workflow output data"
    )
    execution_time_ms: int = Field(description="Total execution time in milliseconds")


class WorkflowFailedEvent(DomainEvent):
    """Event published when workflow execution fails."""

    event_type: EventType = Field(default=EventType.WORKFLOW_FAILED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    error_message: str = Field(description="Error message")
    error_type: str = Field(description="Error type/class")
    failed_task_id: UUID | None = Field(
        default=None, description="Task that caused failure"
    )


class WorkflowCancelledEvent(DomainEvent):
    """Event published when workflow execution is cancelled."""

    event_type: EventType = Field(default=EventType.WORKFLOW_CANCELLED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    execution_id: UUID = Field(description="Execution instance identifier")
    reason: str | None = Field(default=None, description="Cancellation reason")
    cancelled_by: str | None = Field(
        default=None, description="User who cancelled execution"
    )


class AgentAssignedEvent(DomainEvent):
    """Event published when an agent is assigned to a workflow."""

    event_type: EventType = Field(default=EventType.AGENT_ASSIGNED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    agent_id: str = Field(description="Agent identifier")
    agent_role: str = Field(description="Role assigned to agent in workflow")
    capabilities: list[str] = Field(
        default_factory=list, description="Agent capabilities"
    )


class AgentUnassignedEvent(DomainEvent):
    """Event published when an agent is unassigned from a workflow."""

    event_type: EventType = Field(default=EventType.AGENT_UNASSIGNED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    agent_id: str = Field(description="Agent identifier")
    agent_role: str = Field(description="Role of unassigned agent")
    reason: str | None = Field(default=None, description="Unassignment reason")


class AgentReplacedEvent(DomainEvent):
    """Event published when an agent is replaced in a workflow."""

    event_type: EventType = Field(default=EventType.AGENT_REPLACED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    old_agent_id: str = Field(description="Replaced agent identifier")
    new_agent_id: str = Field(description="New agent identifier")
    agent_role: str = Field(description="Role of replaced agent")
    reason: str = Field(description="Replacement reason")


class TaskScheduledEvent(DomainEvent):
    """Event published when a task is scheduled for execution."""

    event_type: EventType = Field(default=EventType.TASK_SCHEDULED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_id: UUID = Field(description="Task identifier")
    agent_id: str = Field(description="Assigned agent identifier")
    task_type: str = Field(description="Type of task")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Task input data"
    )
    dependencies: list[UUID] = Field(
        default_factory=list, description="Task dependencies"
    )


class TaskStartedEvent(DomainEvent):
    """Event published when a task starts execution."""

    event_type: EventType = Field(default=EventType.TASK_STARTED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_id: UUID = Field(description="Task identifier")
    agent_id: str = Field(description="Executing agent identifier")


class TaskCompletedEvent(DomainEvent):
    """Event published when a task completes successfully."""

    event_type: EventType = Field(default=EventType.TASK_COMPLETED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_id: UUID = Field(description="Task identifier")
    agent_id: str = Field(description="Executing agent identifier")
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Task output data"
    )
    execution_time_ms: int = Field(description="Task execution time in milliseconds")


class TaskFailedEvent(DomainEvent):
    """Event published when a task fails."""

    event_type: EventType = Field(default=EventType.TASK_FAILED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_id: UUID = Field(description="Task identifier")
    agent_id: str = Field(description="Executing agent identifier")
    error_message: str = Field(description="Error message")
    error_type: str = Field(description="Error type/class")
    retry_count: int = Field(default=0, description="Number of retry attempts")


class TaskRetriedEvent(DomainEvent):
    """Event published when a task is retried after failure."""

    event_type: EventType = Field(default=EventType.TASK_RETRIED, frozen=True)
    workflow_id: UUID = Field(description="Workflow identifier")
    task_id: UUID = Field(description="Task identifier")
    agent_id: str = Field(description="Executing agent identifier")
    retry_count: int = Field(description="Current retry attempt number")
    reason: str = Field(description="Reason for retry")


class StateSnapshotCreatedEvent(DomainEvent):
    """Event published when a state snapshot is created."""

    event_type: EventType = Field(
        default=EventType.STATE_SNAPSHOT_CREATED, frozen=True
    )
    aggregate_id: UUID = Field(description="Aggregate identifier")
    aggregate_type: str = Field(description="Type of aggregate")
    snapshot_version: int = Field(description="Version number at snapshot")
    snapshot_data: dict[str, Any] = Field(description="Snapshot state data")


def deserialize_event(event_data: dict[str, Any]) -> DomainEvent:
    """
    Deserialize event data to appropriate event class.

    Args:
        event_data: Event data dictionary

    Returns:
        Deserialized domain event

    Raises:
        ValueError: If event type is unknown
    """
    event_type_str = event_data.get("event_type")
    if not event_type_str:
        raise ValueError("Event data missing event_type field")

    event_type = EventType(event_type_str)

    # Map event types to classes
    event_class_map: dict[EventType, type[DomainEvent]] = {
        EventType.WORKFLOW_CREATED: WorkflowCreatedEvent,
        EventType.WORKFLOW_UPDATED: WorkflowUpdatedEvent,
        EventType.WORKFLOW_STARTED: WorkflowStartedEvent,
        EventType.WORKFLOW_PAUSED: WorkflowPausedEvent,
        EventType.WORKFLOW_RESUMED: WorkflowResumedEvent,
        EventType.WORKFLOW_COMPLETED: WorkflowCompletedEvent,
        EventType.WORKFLOW_FAILED: WorkflowFailedEvent,
        EventType.WORKFLOW_CANCELLED: WorkflowCancelledEvent,
        EventType.AGENT_ASSIGNED: AgentAssignedEvent,
        EventType.AGENT_UNASSIGNED: AgentUnassignedEvent,
        EventType.AGENT_REPLACED: AgentReplacedEvent,
        EventType.TASK_SCHEDULED: TaskScheduledEvent,
        EventType.TASK_STARTED: TaskStartedEvent,
        EventType.TASK_COMPLETED: TaskCompletedEvent,
        EventType.TASK_FAILED: TaskFailedEvent,
        EventType.TASK_RETRIED: TaskRetriedEvent,
        EventType.STATE_SNAPSHOT_CREATED: StateSnapshotCreatedEvent,
    }

    event_class = event_class_map.get(event_type)
    if not event_class:
        raise ValueError(f"Unknown event type: {event_type}")

    return event_class.model_validate(event_data)
