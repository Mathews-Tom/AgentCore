"""
State Management Integration

Integrates PostgreSQL state persistence with orchestration patterns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.orchestration.patterns.saga import (
    SagaDefinition,
    SagaExecution,
    SagaStatus,
)
from agentcore.orchestration.state.models import WorkflowStatus
from agentcore.orchestration.state.repository import WorkflowStateRepository

logger = structlog.get_logger()


class PersistentSagaOrchestrator:
    """
    Saga orchestrator with PostgreSQL state persistence.

    Extends the in-memory saga orchestrator with durable state management.
    """

    def __init__(
        self,
        orchestrator_id: str,
        session_factory: Any,  # Callable that returns AsyncSession context manager
    ) -> None:
        """
        Initialize persistent saga orchestrator.

        Args:
            orchestrator_id: Unique orchestrator identifier
            session_factory: Factory for creating database sessions
        """
        self.orchestrator_id = orchestrator_id
        self.session_factory = session_factory

        # In-memory cache for active sagas
        self._saga_definitions: dict[UUID, SagaDefinition] = {}

    async def register_saga(self, saga: SagaDefinition) -> None:
        """
        Register a saga definition.

        Args:
            saga: Saga definition to register
        """
        self._saga_definitions[saga.saga_id] = saga

        logger.info(
            "Registered saga definition",
            saga_id=str(saga.saga_id),
            saga_name=saga.name,
            steps=len(saga.steps),
        )

    async def create_execution(
        self,
        saga_id: UUID,
        input_data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """
        Create saga execution with persistent state.

        Args:
            saga_id: Saga definition ID
            input_data: Input data
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            Execution ID

        Raises:
            ValueError: If saga not found
        """
        if saga_id not in self._saga_definitions:
            raise ValueError(f"Saga not found: {saga_id}")

        saga = self._saga_definitions[saga_id]

        # Create execution ID
        from uuid import uuid4

        execution_id = uuid4()

        # Build workflow definition from saga
        workflow_definition = {
            "saga_id": str(saga_id),
            "saga_name": saga.name,
            "description": saga.description,
            "steps": [
                {
                    "step_id": str(step.step_id),
                    "name": step.name,
                    "order": step.order,
                    "max_retries": step.max_retries,
                }
                for step in saga.steps
            ],
            "compensation_strategy": saga.compensation_strategy,
            "step_timeout_seconds": saga.step_timeout_seconds,
            "saga_timeout_seconds": saga.saga_timeout_seconds,
            "enable_state_persistence": saga.enable_state_persistence,
            "checkpoint_interval": saga.checkpoint_interval,
        }

        # Initialize execution state
        execution_state = {
            "status": SagaStatus.PENDING,
            "current_step": 0,
            "completed_steps": [],
            "failed_steps": [],
            "compensated_steps": [],
        }

        # Initialize task states from steps
        task_states = {
            str(step.step_id): {
                "status": "pending",
                "name": step.name,
                "order": step.order,
                "retry_count": 0,
            }
            for step in saga.steps
        }

        # Persist to database
        async with self.session_factory() as session:
            await WorkflowStateRepository.create_execution(
                session=session,
                execution_id=str(execution_id),
                workflow_id=str(saga_id),
                workflow_name=saga.name,
                orchestration_pattern="saga",
                workflow_definition=workflow_definition,
                workflow_version="1.0",
                input_data=input_data,
                tags=tags,
                metadata=metadata,
            )

            # Update initial state
            await WorkflowStateRepository.update_execution_state(
                session=session,
                execution_id=str(execution_id),
                execution_state=execution_state,
                task_states=task_states,
                create_snapshot=True,
            )

            await session.commit()

        logger.info(
            "Created saga execution with persistent state",
            execution_id=str(execution_id),
            saga_id=str(saga_id),
            saga_name=saga.name,
        )

        return execution_id

    async def update_execution_state(
        self,
        execution_id: UUID,
        status: SagaStatus,
        current_step: int,
        completed_steps: list[UUID],
        failed_steps: list[UUID],
        compensated_steps: list[UUID],
        error_message: str | None = None,
    ) -> None:
        """
        Update saga execution state in database.

        Args:
            execution_id: Execution identifier
            status: Saga status
            current_step: Current step index
            completed_steps: Completed step IDs
            failed_steps: Failed step IDs
            compensated_steps: Compensated step IDs
            error_message: Error message if any
        """
        # Map saga status to workflow status
        workflow_status_map = {
            SagaStatus.PENDING: WorkflowStatus.PENDING,
            SagaStatus.RUNNING: WorkflowStatus.EXECUTING,
            SagaStatus.COMPLETED: WorkflowStatus.COMPLETED,
            SagaStatus.FAILED: WorkflowStatus.FAILED,
            SagaStatus.COMPENSATING: WorkflowStatus.COMPENSATING,
            SagaStatus.COMPENSATED: WorkflowStatus.COMPENSATED,
            SagaStatus.COMPENSATION_FAILED: WorkflowStatus.COMPENSATION_FAILED,
        }

        workflow_status = workflow_status_map.get(status, WorkflowStatus.EXECUTING)

        # Build execution state
        execution_state = {
            "status": status,
            "current_step": current_step,
            "completed_steps": [str(s) for s in completed_steps],
            "failed_steps": [str(s) for s in failed_steps],
            "compensated_steps": [str(s) for s in compensated_steps],
            "last_updated": datetime.now(UTC).isoformat(),
        }

        # Update in database
        async with self.session_factory() as session:
            await WorkflowStateRepository.update_execution_status(
                session=session,
                execution_id=str(execution_id),
                status=workflow_status,
                error_message=error_message,
            )

            await WorkflowStateRepository.update_execution_state(
                session=session,
                execution_id=str(execution_id),
                execution_state=execution_state,
                create_snapshot=True,
            )

            await session.commit()

        logger.debug(
            "Updated saga execution state",
            execution_id=str(execution_id),
            status=status,
            current_step=current_step,
        )

    async def update_step_state(
        self,
        execution_id: UUID,
        step_id: UUID,
        status: str,
        retry_count: int = 0,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """
        Update individual step state.

        Args:
            execution_id: Execution identifier
            step_id: Step identifier
            status: Step status
            retry_count: Retry count
            result: Step result
            error: Error message
        """
        async with self.session_factory() as session:
            # Get current execution
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )
            if not execution:
                raise ValueError(f"Execution not found: {execution_id}")

            # Update task state
            task_states = execution.task_states.copy()
            task_states[str(step_id)] = {
                "status": status,
                "retry_count": retry_count,
                "result": result,
                "error": error,
                "updated_at": datetime.now(UTC).isoformat(),
            }

            # Update execution
            await WorkflowStateRepository.update_execution_state(
                session=session,
                execution_id=str(execution_id),
                execution_state=execution.execution_state,
                task_states=task_states,
                create_snapshot=False,  # Don't create snapshot for every step update
            )

            await session.commit()

        logger.debug(
            "Updated step state",
            execution_id=str(execution_id),
            step_id=str(step_id),
            status=status,
        )

    async def create_checkpoint(
        self, execution_id: UUID, checkpoint_data: dict[str, Any]
    ) -> None:
        """
        Create workflow checkpoint.

        Args:
            execution_id: Execution identifier
            checkpoint_data: Checkpoint data
        """
        async with self.session_factory() as session:
            await WorkflowStateRepository.create_checkpoint(
                session=session,
                execution_id=str(execution_id),
                checkpoint_data=checkpoint_data,
            )

            await session.commit()

        logger.info(
            "Created workflow checkpoint",
            execution_id=str(execution_id),
        )

    async def get_execution_status(self, execution_id: UUID) -> SagaExecution | None:
        """
        Get saga execution status from database.

        Args:
            execution_id: Execution identifier

        Returns:
            Saga execution or None
        """
        async with self.session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            if not execution:
                return None

            # Reconstruct saga execution from database state
            execution_state = execution.execution_state

            saga_execution = SagaExecution(
                execution_id=UUID(execution.execution_id),
                saga_id=UUID(execution.workflow_id),
                saga_name=execution.workflow_name,
                status=SagaStatus(execution_state.get("status", "pending")),
                current_step=execution_state.get("current_step", 0),
                completed_steps=[
                    UUID(s) for s in execution_state.get("completed_steps", [])
                ],
                failed_steps=[UUID(s) for s in execution_state.get("failed_steps", [])],
                compensated_steps=[
                    UUID(s) for s in execution_state.get("compensated_steps", [])
                ],
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                error_message=execution.error_message,
            )

            return saga_execution

    async def recover_from_checkpoint(
        self, execution_id: UUID, checkpoint_version: int | None = None
    ) -> dict[str, Any]:
        """
        Recover saga from checkpoint.

        Args:
            execution_id: Execution identifier
            checkpoint_version: Specific checkpoint version (None = latest)

        Returns:
            Checkpoint data

        Raises:
            ValueError: If execution or checkpoint not found
        """
        async with self.session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            if not execution:
                raise ValueError(f"Execution not found: {execution_id}")

            # Get checkpoint data
            if checkpoint_version is not None:
                # Get specific checkpoint version from history
                state = await WorkflowStateRepository.get_state_at_version(
                    session, str(execution_id), checkpoint_version
                )
                if not state or state.state_type != "checkpoint":
                    raise ValueError(
                        f"Checkpoint not found: {execution_id}@v{checkpoint_version}"
                    )
                checkpoint_data = state.state_snapshot
            else:
                # Get latest checkpoint
                if not execution.checkpoint_data:
                    raise ValueError(f"No checkpoints available for: {execution_id}")
                checkpoint_data = execution.checkpoint_data

            logger.info(
                "Recovered saga from checkpoint",
                execution_id=str(execution_id),
                checkpoint_version=checkpoint_version,
            )

            return checkpoint_data

    async def get_execution_statistics(
        self, workflow_id: UUID | None = None
    ) -> dict[str, Any]:
        """
        Get execution statistics.

        Args:
            workflow_id: Filter by workflow ID

        Returns:
            Statistics dictionary
        """
        async with self.session_factory() as session:
            stats = await WorkflowStateRepository.get_execution_stats(
                session=session,
                workflow_id=str(workflow_id) if workflow_id else None,
                orchestration_pattern="saga",
            )

            return stats
