"""
Saga Pattern Implementation

Long-running transaction management with compensation actions for distributed workflows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentcore.orchestration.streams.producer import StreamProducer


class SagaStepStatus(str, Enum):
    """Status of a saga step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


class SagaStatus(str, Enum):
    """Status of a saga execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


class CompensationStrategy(str, Enum):
    """Strategy for compensation execution."""

    BACKWARD = "backward"  # Compensate in reverse order
    FORWARD = "forward"  # Compensate in forward order
    PARALLEL = "parallel"  # Compensate all steps in parallel


class SagaStep(BaseModel):
    """
    Individual step in a saga.

    Each step has a forward action and optional compensation action.
    """

    step_id: UUID = Field(default_factory=uuid4, description="Unique step identifier")
    name: str = Field(description="Step name")
    order: int = Field(description="Execution order")

    # Forward action
    action_data: dict[str, Any] = Field(
        default_factory=dict, description="Data for forward action"
    )
    action_result: dict[str, Any] | None = Field(
        default=None, description="Result of forward action"
    )

    # Compensation action
    compensation_data: dict[str, Any] | None = Field(
        default=None, description="Data for compensation action"
    )
    compensation_result: dict[str, Any] | None = Field(
        default=None, description="Result of compensation"
    )

    # Status tracking
    status: SagaStepStatus = Field(default=SagaStepStatus.PENDING)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    compensated_at: datetime | None = None

    error_message: str | None = None
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)

    # Idempotency
    idempotency_key: str | None = Field(
        default=None, description="Key for idempotent operations"
    )

    model_config = {"frozen": False}


class SagaDefinition(BaseModel):
    """Definition of a saga workflow."""

    saga_id: UUID = Field(default_factory=uuid4, description="Unique saga identifier")
    name: str = Field(description="Saga name")
    description: str | None = None

    steps: list[SagaStep] = Field(description="Ordered list of saga steps")

    compensation_strategy: CompensationStrategy = Field(
        default=CompensationStrategy.BACKWARD
    )

    # Timeouts
    step_timeout_seconds: int = Field(default=300, description="Timeout per step")
    saga_timeout_seconds: int = Field(default=3600, description="Total saga timeout")

    # State persistence
    enable_state_persistence: bool = Field(
        default=True, description="Enable state checkpointing"
    )
    checkpoint_interval: int = Field(default=1, description="Checkpoint every N steps")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class SagaExecution(BaseModel):
    """Runtime state of a saga execution."""

    execution_id: UUID = Field(
        default_factory=uuid4, description="Unique execution identifier"
    )
    saga_id: UUID = Field(description="Reference to saga definition")
    saga_name: str = Field(description="Saga name")

    status: SagaStatus = Field(default=SagaStatus.PENDING)

    current_step: int = Field(default=0, description="Current step index")
    completed_steps: list[UUID] = Field(
        default_factory=list, description="Completed step IDs"
    )
    failed_steps: list[UUID] = Field(
        default_factory=list, description="Failed step IDs"
    )
    compensated_steps: list[UUID] = Field(
        default_factory=list, description="Compensated step IDs"
    )

    # Timeline
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None

    # State snapshots for recovery
    checkpoints: list[dict[str, Any]] = Field(
        default_factory=list, description="State checkpoints"
    )
    last_checkpoint_at: datetime | None = None

    error_message: str | None = None
    compensation_errors: list[str] = Field(
        default_factory=list, description="Errors during compensation"
    )

    model_config = {"frozen": False}


class SagaConfig(BaseModel):
    """Configuration for saga orchestrator."""

    enable_retry: bool = Field(default=True, description="Enable automatic retry")
    max_retries: int = Field(default=3, description="Max retries per step")
    retry_delay_seconds: int = Field(default=5, description="Delay between retries")

    enable_checkpointing: bool = Field(
        default=True, description="Enable state checkpointing"
    )
    checkpoint_storage_path: str | None = Field(
        default=None, description="Path for checkpoint storage"
    )

    enable_idempotency: bool = Field(
        default=True, description="Enforce idempotent operations"
    )

    compensation_timeout_seconds: int = Field(
        default=300, description="Timeout for compensation actions"
    )


class SagaOrchestrator:
    """
    Saga pattern orchestrator.

    Manages long-running transactions with:
    - Sequential step execution
    - Automatic compensation on failure
    - State recovery and rollback
    - Consistency guarantees
    """

    def __init__(
        self,
        orchestrator_id: str,
        config: SagaConfig | None = None,
        event_producer: StreamProducer | None = None,
    ) -> None:
        """
        Initialize saga orchestrator.

        Args:
            orchestrator_id: Unique orchestrator identifier
            config: Saga configuration
            event_producer: Event stream producer
        """
        self.orchestrator_id = orchestrator_id
        self.config = config or SagaConfig()
        self.event_producer = event_producer

        # Saga tracking
        self._saga_definitions: dict[UUID, SagaDefinition] = {}
        self._active_executions: dict[UUID, SagaExecution] = {}
        self._completed_executions: dict[UUID, SagaExecution] = {}

        # Action handlers (registered externally)
        self._action_handlers: dict[str, Callable] = {}
        self._compensation_handlers: dict[str, Callable] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

    def register_saga(self, saga: SagaDefinition) -> None:
        """
        Register a saga definition.

        Args:
            saga: Saga definition to register
        """
        self._saga_definitions[saga.saga_id] = saga

    def register_action_handler(
        self,
        step_name: str,
        handler: Callable,
        compensation_handler: Callable | None = None,
    ) -> None:
        """
        Register action and compensation handlers for a step.

        Args:
            step_name: Name of the step
            handler: Forward action handler
            compensation_handler: Compensation action handler
        """
        self._action_handlers[step_name] = handler
        if compensation_handler:
            self._compensation_handlers[step_name] = compensation_handler

    async def execute_saga(
        self, saga_id: UUID, input_data: dict[str, Any] | None = None
    ) -> UUID:
        """
        Execute a saga.

        Args:
            saga_id: Saga definition ID
            input_data: Input data for the saga

        Returns:
            Execution ID

        Raises:
            ValueError: If saga not found
        """
        async with self._lock:
            if saga_id not in self._saga_definitions:
                raise ValueError(f"Saga not found: {saga_id}")

            saga = self._saga_definitions[saga_id]

            # Create execution
            execution = SagaExecution(
                saga_id=saga_id,
                saga_name=saga.name,
                status=SagaStatus.PENDING,
            )

            self._active_executions[execution.execution_id] = execution

        # Publish saga started event
        if self.event_producer:
            await self._publish_saga_event(
                "saga_started",
                execution.execution_id,
                {"saga_id": str(saga_id), "saga_name": saga.name},
            )

        # Start execution in background
        asyncio.create_task(
            self._execute_saga_steps(execution.execution_id, input_data or {})
        )

        return execution.execution_id

    async def _execute_saga_steps(
        self, execution_id: UUID, input_data: dict[str, Any]
    ) -> None:
        """
        Execute saga steps sequentially.

        Args:
            execution_id: Execution identifier
            input_data: Input data
        """
        async with self._lock:
            if execution_id not in self._active_executions:
                return

            execution = self._active_executions[execution_id]
            saga = self._saga_definitions[execution.saga_id]

        execution.status = SagaStatus.RUNNING
        execution.started_at = datetime.now(UTC)

        context_data = input_data.copy()

        try:
            # Execute steps in order
            for step in saga.steps:
                await self._execute_step(execution, step, context_data)

                # Check if step failed
                if step.status == SagaStepStatus.FAILED:
                    # Trigger compensation
                    await self._compensate_saga(execution, saga)
                    return

                # Checkpoint state if enabled
                if saga.enable_state_persistence and (
                    step.order % saga.checkpoint_interval == 0
                ):
                    await self._create_checkpoint(execution, context_data)

            # All steps completed successfully
            execution.status = SagaStatus.COMPLETED
            execution.completed_at = datetime.now(UTC)

            # Move to completed
            async with self._lock:
                self._completed_executions[execution_id] = execution
                del self._active_executions[execution_id]

            # Publish completion event
            if self.event_producer:
                await self._publish_saga_event(
                    "saga_completed",
                    execution_id,
                    {
                        "saga_id": str(execution.saga_id),
                        "steps_completed": len(execution.completed_steps),
                    },
                )

        except Exception as e:
            execution.status = SagaStatus.FAILED
            execution.failed_at = datetime.now(UTC)
            execution.error_message = str(e)

            # Trigger compensation
            await self._compensate_saga(execution, saga)

    async def _execute_step(
        self,
        execution: SagaExecution,
        step: SagaStep,
        context_data: dict[str, Any],
    ) -> None:
        """
        Execute a single saga step.

        Args:
            execution: Saga execution
            step: Step to execute
            context_data: Execution context
        """
        step.status = SagaStepStatus.RUNNING
        step.started_at = datetime.now(UTC)

        # Get action handler
        if step.name not in self._action_handlers:
            step.status = SagaStepStatus.FAILED
            step.error_message = f"No handler registered for step: {step.name}"
            execution.failed_steps.append(step.step_id)
            return

        handler = self._action_handlers[step.name]

        # Retry loop
        retry_count = 0
        while retry_count <= step.max_retries:
            try:
                # Execute action
                result = await handler(step.action_data, context_data)

                # Store result
                step.action_result = result
                step.status = SagaStepStatus.COMPLETED
                step.completed_at = datetime.now(UTC)

                # Add to completed steps
                execution.completed_steps.append(step.step_id)
                execution.current_step += 1

                # Publish step completed event
                if self.event_producer:
                    await self._publish_saga_event(
                        "step_completed",
                        execution.execution_id,
                        {
                            "step_id": str(step.step_id),
                            "step_name": step.name,
                            "order": step.order,
                        },
                    )

                return

            except Exception as e:
                retry_count += 1
                step.retry_count = retry_count

                if retry_count > step.max_retries:
                    # Max retries reached - fail step
                    step.status = SagaStepStatus.FAILED
                    step.failed_at = datetime.now(UTC)
                    step.error_message = str(e)
                    execution.failed_steps.append(step.step_id)

                    # Publish step failed event
                    if self.event_producer:
                        await self._publish_saga_event(
                            "step_failed",
                            execution.execution_id,
                            {
                                "step_id": str(step.step_id),
                                "step_name": step.name,
                                "error": str(e),
                                "retry_count": retry_count,
                            },
                        )

                    return
                else:
                    # Retry with delay
                    await asyncio.sleep(self.config.retry_delay_seconds)

    async def _compensate_saga(
        self, execution: SagaExecution, saga: SagaDefinition
    ) -> None:
        """
        Compensate a failed saga.

        Args:
            execution: Saga execution
            saga: Saga definition
        """
        execution.status = SagaStatus.COMPENSATING

        # Publish compensation started event
        if self.event_producer:
            await self._publish_saga_event(
                "compensation_started",
                execution.execution_id,
                {
                    "saga_id": str(saga.saga_id),
                    "completed_steps": len(execution.completed_steps),
                },
            )

        # Get completed steps in compensation order
        completed_step_ids = set(execution.completed_steps)
        completed_steps = [s for s in saga.steps if s.step_id in completed_step_ids]

        if saga.compensation_strategy == CompensationStrategy.BACKWARD:
            # Reverse order
            steps_to_compensate = list(reversed(completed_steps))
        elif saga.compensation_strategy == CompensationStrategy.FORWARD:
            # Forward order
            steps_to_compensate = completed_steps
        else:
            # Parallel - all at once
            steps_to_compensate = completed_steps

        # Execute compensation
        if saga.compensation_strategy == CompensationStrategy.PARALLEL:
            # Parallel compensation
            tasks = [
                self._compensate_step(execution, step) for step in steps_to_compensate
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential compensation
            for step in steps_to_compensate:
                await self._compensate_step(execution, step)

        # Check compensation result
        if len(execution.compensation_errors) == 0:
            execution.status = SagaStatus.COMPENSATED
        else:
            execution.status = SagaStatus.COMPENSATION_FAILED

        execution.completed_at = datetime.now(UTC)

        # Move to completed
        async with self._lock:
            self._completed_executions[execution.execution_id] = execution
            if execution.execution_id in self._active_executions:
                del self._active_executions[execution.execution_id]

        # Publish compensation completed event
        if self.event_producer:
            await self._publish_saga_event(
                "compensation_completed",
                execution.execution_id,
                {
                    "saga_id": str(saga.saga_id),
                    "status": execution.status,
                    "compensated_steps": len(execution.compensated_steps),
                    "errors": execution.compensation_errors,
                },
            )

    async def _compensate_step(self, execution: SagaExecution, step: SagaStep) -> None:
        """
        Compensate a single step.

        Args:
            execution: Saga execution
            step: Step to compensate
        """
        step.status = SagaStepStatus.COMPENSATING

        # Get compensation handler
        if step.name not in self._compensation_handlers:
            error = f"No compensation handler for step: {step.name}"
            execution.compensation_errors.append(error)
            step.status = SagaStepStatus.COMPENSATION_FAILED
            step.error_message = error
            return

        handler = self._compensation_handlers[step.name]

        try:
            # Execute compensation
            result = await asyncio.wait_for(
                handler(step.compensation_data or step.action_result or {}),
                timeout=self.config.compensation_timeout_seconds,
            )

            step.compensation_result = result
            step.status = SagaStepStatus.COMPENSATED
            step.compensated_at = datetime.now(UTC)

            execution.compensated_steps.append(step.step_id)

        except asyncio.TimeoutError:
            error = f"Compensation timeout for step: {step.name}"
            execution.compensation_errors.append(error)
            step.status = SagaStepStatus.COMPENSATION_FAILED
            step.error_message = error

        except Exception as e:
            error = f"Compensation failed for step {step.name}: {e!s}"
            execution.compensation_errors.append(error)
            step.status = SagaStepStatus.COMPENSATION_FAILED
            step.error_message = str(e)

    async def _create_checkpoint(
        self, execution: SagaExecution, context_data: dict[str, Any]
    ) -> None:
        """
        Create state checkpoint.

        Args:
            execution: Saga execution
            context_data: Current context
        """
        checkpoint = {
            "timestamp": datetime.now(UTC).isoformat(),
            "current_step": execution.current_step,
            "completed_steps": [str(s) for s in execution.completed_steps],
            "context_data": context_data.copy(),
        }

        execution.checkpoints.append(checkpoint)
        execution.last_checkpoint_at = datetime.now(UTC)

    async def _publish_saga_event(
        self, event_type: str, execution_id: UUID, data: dict[str, Any]
    ) -> None:
        """
        Publish saga event.

        Args:
            event_type: Event type
            execution_id: Execution ID
            data: Event data
        """
        if not self.event_producer:
            return

        event_data = {
            "event_type": event_type,
            "execution_id": str(execution_id),
            "orchestrator_id": self.orchestrator_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **data,
        }

        await self.event_producer.publish(event_data)

    async def get_execution_status(self, execution_id: UUID) -> SagaExecution:
        """
        Get execution status.

        Args:
            execution_id: Execution identifier

        Returns:
            Saga execution

        Raises:
            ValueError: If execution not found
        """
        async with self._lock:
            if execution_id in self._active_executions:
                return self._active_executions[execution_id]
            elif execution_id in self._completed_executions:
                return self._completed_executions[execution_id]
            else:
                raise ValueError(f"Execution not found: {execution_id}")

    async def recover_from_checkpoint(
        self, execution_id: UUID, checkpoint_index: int = -1
    ) -> None:
        """
        Recover saga from checkpoint.

        Args:
            execution_id: Execution identifier
            checkpoint_index: Checkpoint index to recover from (-1 = latest)

        Raises:
            ValueError: If execution or checkpoint not found
        """
        execution = await self.get_execution_status(execution_id)

        if not execution.checkpoints:
            raise ValueError("No checkpoints available for recovery")

        checkpoint = execution.checkpoints[checkpoint_index]

        # Restore state
        execution.current_step = checkpoint["current_step"]
        execution.completed_steps = [UUID(s) for s in checkpoint["completed_steps"]]

        # Resume execution from checkpoint
        # (Implementation would continue from current_step)

    async def get_orchestrator_status(self) -> dict[str, Any]:
        """
        Get orchestrator status.

        Returns:
            Status dictionary
        """
        async with self._lock:
            return {
                "orchestrator_id": self.orchestrator_id,
                "sagas_registered": len(self._saga_definitions),
                "active_executions": len(self._active_executions),
                "completed_executions": len(self._completed_executions),
                "registered_handlers": len(self._action_handlers),
                "config": self.config.model_dump(),
            }
