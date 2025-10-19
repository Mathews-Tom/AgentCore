"""
Projections Module

Event handlers that update read models from event store.
Projections handle eventual consistency between write and read sides.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.orchestration.cqrs.events import (
    AgentAssignedEvent,
    AgentReplacedEvent,
    AgentUnassignedEvent,
    DomainEvent,
    EventType,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskRetriedEvent,
    TaskScheduledEvent,
    TaskStartedEvent,
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowCreatedEvent,
    WorkflowFailedEvent,
    WorkflowPausedEvent,
    WorkflowResumedEvent,
    WorkflowStartedEvent,
    WorkflowUpdatedEvent,
)
from agentcore.orchestration.cqrs.read_models import (
    AgentAssignmentReadModel,
    ExecutionReadModel,
    TaskReadModel,
    WorkflowMetricsReadModel,
    WorkflowReadModel,
)


class Projection(ABC):
    """
    Abstract base class for projections.

    Projections update read models based on events.
    """

    @abstractmethod
    async def project(self, event: DomainEvent, session: AsyncSession) -> None:
        """
        Project event to update read model.

        Args:
            event: Domain event to project
            session: Database session for read model updates
        """
        pass

    @abstractmethod
    def handles_event_type(self, event_type: EventType) -> bool:
        """
        Check if this projection handles the event type.

        Args:
            event_type: Event type to check

        Returns:
            True if projection handles this event type
        """
        pass


class WorkflowProjection(Projection):
    """
    Projection for workflow read model.

    Updates workflow read model from workflow events.
    """

    def handles_event_type(self, event_type: EventType) -> bool:
        """Check if handles workflow events."""
        return event_type in {
            EventType.WORKFLOW_CREATED,
            EventType.WORKFLOW_UPDATED,
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
            EventType.WORKFLOW_FAILED,
        }

    async def project(self, event: DomainEvent, session: AsyncSession) -> None:
        """Project workflow event to read model."""
        if isinstance(event, WorkflowCreatedEvent):
            await self._handle_workflow_created(event, session)
        elif isinstance(event, WorkflowUpdatedEvent):
            await self._handle_workflow_updated(event, session)
        elif isinstance(event, WorkflowCompletedEvent):
            await self._handle_workflow_completed(event, session)
        elif isinstance(event, WorkflowFailedEvent):
            await self._handle_workflow_failed(event, session)

    async def _handle_workflow_created(
        self, event: WorkflowCreatedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow created event."""
        workflow = WorkflowReadModel(
            workflow_id=str(event.workflow_id),
            workflow_name=event.workflow_name,
            orchestration_pattern=event.orchestration_pattern,
            status="created",
            agent_requirements=event.agent_requirements,
            task_definitions=event.task_definitions,
            created_by=event.created_by,
            created_at=event.timestamp,
            updated_at=event.timestamp,
        )
        session.add(workflow)

    async def _handle_workflow_updated(
        self, event: WorkflowUpdatedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow updated event."""
        stmt = (
            update(WorkflowReadModel)
            .where(WorkflowReadModel.workflow_id == str(event.workflow_id))
            .values(updated_at=event.timestamp)
        )
        await session.execute(stmt)

    async def _handle_workflow_completed(
        self, event: WorkflowCompletedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow completed event."""
        stmt = (
            update(WorkflowReadModel)
            .where(WorkflowReadModel.workflow_id == str(event.workflow_id))
            .values(
                total_executions=WorkflowReadModel.total_executions + 1,
                successful_executions=WorkflowReadModel.successful_executions + 1,
            )
        )
        await session.execute(stmt)

    async def _handle_workflow_failed(
        self, event: WorkflowFailedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow failed event."""
        stmt = (
            update(WorkflowReadModel)
            .where(WorkflowReadModel.workflow_id == str(event.workflow_id))
            .values(
                total_executions=WorkflowReadModel.total_executions + 1,
                failed_executions=WorkflowReadModel.failed_executions + 1,
            )
        )
        await session.execute(stmt)


class ExecutionProjection(Projection):
    """
    Projection for execution read model.

    Updates execution read model from workflow execution events.
    """

    def handles_event_type(self, event_type: EventType) -> bool:
        """Check if handles execution events."""
        return event_type in {
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_PAUSED,
            EventType.WORKFLOW_RESUMED,
            EventType.WORKFLOW_COMPLETED,
            EventType.WORKFLOW_FAILED,
            EventType.WORKFLOW_CANCELLED,
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED,
        }

    async def project(self, event: DomainEvent, session: AsyncSession) -> None:
        """Project execution event to read model."""
        if isinstance(event, WorkflowStartedEvent):
            await self._handle_workflow_started(event, session)
        elif isinstance(event, WorkflowPausedEvent):
            await self._handle_workflow_paused(event, session)
        elif isinstance(event, WorkflowResumedEvent):
            await self._handle_workflow_resumed(event, session)
        elif isinstance(event, WorkflowCompletedEvent):
            await self._handle_workflow_completed(event, session)
        elif isinstance(event, WorkflowFailedEvent):
            await self._handle_workflow_failed(event, session)
        elif isinstance(event, WorkflowCancelledEvent):
            await self._handle_workflow_cancelled(event, session)
        elif isinstance(event, TaskCompletedEvent):
            await self._handle_task_completed(event, session)
        elif isinstance(event, TaskFailedEvent):
            await self._handle_task_failed(event, session)

    async def _handle_workflow_started(
        self, event: WorkflowStartedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow started event."""
        execution = ExecutionReadModel(
            execution_id=str(event.execution_id),
            workflow_id=str(event.workflow_id),
            status="running",
            input_data=event.input_data,
            started_by=event.started_by,
            started_at=event.timestamp,
        )
        session.add(execution)

    async def _handle_workflow_paused(
        self, event: WorkflowPausedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow paused event."""
        stmt = (
            update(ExecutionReadModel)
            .where(ExecutionReadModel.execution_id == str(event.execution_id))
            .values(status="paused", paused_at=event.timestamp)
        )
        await session.execute(stmt)

    async def _handle_workflow_resumed(
        self, event: WorkflowResumedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow resumed event."""
        stmt = (
            update(ExecutionReadModel)
            .where(ExecutionReadModel.execution_id == str(event.execution_id))
            .values(status="running", paused_at=None)
        )
        await session.execute(stmt)

    async def _handle_workflow_completed(
        self, event: WorkflowCompletedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow completed event."""
        stmt = (
            update(ExecutionReadModel)
            .where(ExecutionReadModel.execution_id == str(event.execution_id))
            .values(
                status="completed",
                completed_at=event.timestamp,
                output_data=event.output_data,
                execution_time_ms=event.execution_time_ms,
            )
        )
        await session.execute(stmt)

    async def _handle_workflow_failed(
        self, event: WorkflowFailedEvent, session: AsyncSession
    ) -> None:
        """Handle workflow failed event."""
        stmt = (
            update(ExecutionReadModel)
            .where(ExecutionReadModel.execution_id == str(event.execution_id))
            .values(
                status="failed",
                completed_at=event.timestamp,
                error_message=event.error_message,
                error_type=event.error_type,
                failed_task_id=str(event.failed_task_id) if event.failed_task_id else None,
            )
        )
        await session.execute(stmt)

    async def _handle_workflow_cancelled(
        self, event: WorkflowCancelledEvent, session: AsyncSession
    ) -> None:
        """Handle workflow cancelled event."""
        stmt = (
            update(ExecutionReadModel)
            .where(ExecutionReadModel.execution_id == str(event.execution_id))
            .values(status="cancelled", cancelled_at=event.timestamp)
        )
        await session.execute(stmt)

    async def _handle_task_completed(
        self, event: TaskCompletedEvent, session: AsyncSession
    ) -> None:
        """Handle task completed event - update execution task counts."""
        stmt = (
            update(ExecutionReadModel)
            .where(ExecutionReadModel.workflow_id == str(event.workflow_id))
            .values(
                tasks_completed=ExecutionReadModel.tasks_completed + 1,
                tasks_pending=ExecutionReadModel.tasks_pending - 1,
            )
        )
        await session.execute(stmt)

    async def _handle_task_failed(
        self, event: TaskFailedEvent, session: AsyncSession
    ) -> None:
        """Handle task failed event - update execution task counts."""
        stmt = (
            update(ExecutionReadModel)
            .where(ExecutionReadModel.workflow_id == str(event.workflow_id))
            .values(
                tasks_failed=ExecutionReadModel.tasks_failed + 1,
                tasks_pending=ExecutionReadModel.tasks_pending - 1,
            )
        )
        await session.execute(stmt)


class AgentAssignmentProjection(Projection):
    """
    Projection for agent assignment read model.

    Updates agent assignments from agent events.
    """

    def handles_event_type(self, event_type: EventType) -> bool:
        """Check if handles agent assignment events."""
        return event_type in {
            EventType.AGENT_ASSIGNED,
            EventType.AGENT_UNASSIGNED,
            EventType.AGENT_REPLACED,
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED,
        }

    async def project(self, event: DomainEvent, session: AsyncSession) -> None:
        """Project agent event to read model."""
        if isinstance(event, AgentAssignedEvent):
            await self._handle_agent_assigned(event, session)
        elif isinstance(event, AgentUnassignedEvent):
            await self._handle_agent_unassigned(event, session)
        elif isinstance(event, AgentReplacedEvent):
            await self._handle_agent_replaced(event, session)
        elif isinstance(event, TaskCompletedEvent):
            await self._handle_task_completed(event, session)
        elif isinstance(event, TaskFailedEvent):
            await self._handle_task_failed(event, session)

    async def _handle_agent_assigned(
        self, event: AgentAssignedEvent, session: AsyncSession
    ) -> None:
        """Handle agent assigned event."""
        assignment = AgentAssignmentReadModel(
            workflow_id=str(event.workflow_id),
            agent_id=event.agent_id,
            agent_role=event.agent_role,
            capabilities=event.capabilities,
            assigned_at=event.timestamp,
            is_active=1,
        )
        session.add(assignment)

    async def _handle_agent_unassigned(
        self, event: AgentUnassignedEvent, session: AsyncSession
    ) -> None:
        """Handle agent unassigned event."""
        stmt = (
            update(AgentAssignmentReadModel)
            .where(
                AgentAssignmentReadModel.workflow_id == str(event.workflow_id),
                AgentAssignmentReadModel.agent_id == event.agent_id,
                AgentAssignmentReadModel.agent_role == event.agent_role,
            )
            .values(is_active=0, unassigned_at=event.timestamp)
        )
        await session.execute(stmt)

    async def _handle_agent_replaced(
        self, event: AgentReplacedEvent, session: AsyncSession
    ) -> None:
        """Handle agent replaced event."""
        # Unassign old agent
        stmt = (
            update(AgentAssignmentReadModel)
            .where(
                AgentAssignmentReadModel.workflow_id == str(event.workflow_id),
                AgentAssignmentReadModel.agent_id == event.old_agent_id,
                AgentAssignmentReadModel.agent_role == event.agent_role,
            )
            .values(is_active=0, unassigned_at=event.timestamp)
        )
        await session.execute(stmt)

    async def _handle_task_completed(
        self, event: TaskCompletedEvent, session: AsyncSession
    ) -> None:
        """Handle task completed - update agent stats."""
        stmt = (
            update(AgentAssignmentReadModel)
            .where(
                AgentAssignmentReadModel.workflow_id == str(event.workflow_id),
                AgentAssignmentReadModel.agent_id == event.agent_id,
            )
            .values(tasks_completed=AgentAssignmentReadModel.tasks_completed + 1)
        )
        await session.execute(stmt)

    async def _handle_task_failed(
        self, event: TaskFailedEvent, session: AsyncSession
    ) -> None:
        """Handle task failed - update agent stats."""
        stmt = (
            update(AgentAssignmentReadModel)
            .where(
                AgentAssignmentReadModel.workflow_id == str(event.workflow_id),
                AgentAssignmentReadModel.agent_id == event.agent_id,
            )
            .values(tasks_failed=AgentAssignmentReadModel.tasks_failed + 1)
        )
        await session.execute(stmt)


class TaskProjection(Projection):
    """
    Projection for task read model.

    Updates task read model from task events.
    """

    def handles_event_type(self, event_type: EventType) -> bool:
        """Check if handles task events."""
        return event_type in {
            EventType.TASK_SCHEDULED,
            EventType.TASK_STARTED,
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED,
            EventType.TASK_RETRIED,
        }

    async def project(self, event: DomainEvent, session: AsyncSession) -> None:
        """Project task event to read model."""
        if isinstance(event, TaskScheduledEvent):
            await self._handle_task_scheduled(event, session)
        elif isinstance(event, TaskStartedEvent):
            await self._handle_task_started(event, session)
        elif isinstance(event, TaskCompletedEvent):
            await self._handle_task_completed(event, session)
        elif isinstance(event, TaskFailedEvent):
            await self._handle_task_failed(event, session)
        elif isinstance(event, TaskRetriedEvent):
            await self._handle_task_retried(event, session)

    async def _handle_task_scheduled(
        self, event: TaskScheduledEvent, session: AsyncSession
    ) -> None:
        """Handle task scheduled event."""
        task = TaskReadModel(
            task_id=str(event.task_id),
            workflow_id=str(event.workflow_id),
            task_type=event.task_type,
            status="scheduled",
            agent_id=event.agent_id,
            input_data=event.input_data,
            dependencies=event.dependencies,
            scheduled_at=event.timestamp,
        )
        session.add(task)

    async def _handle_task_started(
        self, event: TaskStartedEvent, session: AsyncSession
    ) -> None:
        """Handle task started event."""
        stmt = (
            update(TaskReadModel)
            .where(TaskReadModel.task_id == str(event.task_id))
            .values(status="running", started_at=event.timestamp)
        )
        await session.execute(stmt)

    async def _handle_task_completed(
        self, event: TaskCompletedEvent, session: AsyncSession
    ) -> None:
        """Handle task completed event."""
        stmt = (
            update(TaskReadModel)
            .where(TaskReadModel.task_id == str(event.task_id))
            .values(
                status="completed",
                completed_at=event.timestamp,
                output_data=event.output_data,
                execution_time_ms=event.execution_time_ms,
            )
        )
        await session.execute(stmt)

    async def _handle_task_failed(
        self, event: TaskFailedEvent, session: AsyncSession
    ) -> None:
        """Handle task failed event."""
        stmt = (
            update(TaskReadModel)
            .where(TaskReadModel.task_id == str(event.task_id))
            .values(
                status="failed",
                failed_at=event.timestamp,
                error_message=event.error_message,
                error_type=event.error_type,
                retry_count=event.retry_count,
            )
        )
        await session.execute(stmt)

    async def _handle_task_retried(
        self, event: TaskRetriedEvent, session: AsyncSession
    ) -> None:
        """Handle task retried event."""
        stmt = (
            update(TaskReadModel)
            .where(TaskReadModel.task_id == str(event.task_id))
            .values(status="retrying", retry_count=event.retry_count)
        )
        await session.execute(stmt)


class ProjectionManager:
    """
    Manager for coordinating multiple projections.

    Routes events to appropriate projections.
    """

    def __init__(self) -> None:
        """Initialize projection manager."""
        self._projections: list[Projection] = []

    def register(self, projection: Projection) -> None:
        """
        Register a projection.

        Args:
            projection: Projection instance to register
        """
        self._projections.append(projection)

    async def project_event(
        self, event: DomainEvent, session: AsyncSession
    ) -> None:
        """
        Project event through all applicable projections.

        Args:
            event: Event to project
            session: Database session for updates
        """
        for projection in self._projections:
            if projection.handles_event_type(event.event_type):
                await projection.project(event, session)

    async def project_events(
        self, events: list[DomainEvent], session: AsyncSession
    ) -> None:
        """
        Project multiple events.

        Args:
            events: Events to project
            session: Database session for updates
        """
        for event in events:
            await self.project_event(event, session)
