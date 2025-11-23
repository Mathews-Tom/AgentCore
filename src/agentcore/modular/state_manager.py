"""
Module State Management for Modular Agent Core

Provides state persistence and recovery for module executions:
- Checkpoint creation and restoration
- State serialization using Pydantic models
- Database persistence for execution state
- Crash recovery mechanisms
- State cleanup for completed executions

This enables resumption of interrupted executions and crash recovery.
"""

from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.modular.models import (
    EnhancedExecutionPlan,
    EnhancedPlanStep,
    PlanStatus,
    StepStatus,
    ModuleType,
)

logger = structlog.get_logger()


# ============================================================================
# State Models
# ============================================================================


class ExecutionCheckpoint(BaseModel):
    """Checkpoint for execution state."""

    checkpoint_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique checkpoint ID"
    )
    execution_id: str = Field(..., description="Execution identifier")
    plan_id: str = Field(..., description="Plan identifier")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Checkpoint creation time",
    )
    plan_state: dict[str, Any] = Field(
        ..., description="Serialized execution plan state"
    )
    current_iteration: int = Field(default=0, description="Current iteration number")
    current_module: ModuleType | None = Field(
        None, description="Currently active module"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional checkpoint metadata"
    )


class RecoveryInfo(BaseModel):
    """Information for recovering from crash."""

    execution_id: str = Field(..., description="Execution identifier")
    plan_id: str = Field(..., description="Plan identifier")
    last_checkpoint_id: str | None = Field(
        None, description="ID of last successful checkpoint"
    )
    crash_detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When crash was detected",
    )
    recoverable: bool = Field(
        default=True, description="Whether execution is recoverable"
    )
    recovery_point: dict[str, Any] = Field(
        default_factory=dict, description="State to recover from"
    )
    failed_step_id: str | None = Field(
        None, description="ID of step that was in progress when crashed"
    )


# ============================================================================
# State Manager
# ============================================================================


class StateManager:
    """
    Manages execution state persistence and recovery.

    Provides:
    - Checkpoint creation during execution
    - State serialization to database
    - Crash detection and recovery
    - State cleanup after completion
    - Execution resumption from checkpoints
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize state manager.

        Args:
            session: Database session for persistence
        """
        self.session = session
        logger.info("StateManager initialized")

    # ========================================================================
    # Checkpoint Management
    # ========================================================================

    async def init_execution(
        self,
        execution_id: str,
        query: str,
        trace_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize a new execution record in the database.

        Args:
            execution_id: Execution identifier
            query: User query
            trace_id: Trace ID for distributed tracing
            metadata: Additional execution metadata
        """
        from agentcore.a2a_protocol.database.models import ModularExecutionDB

        # Check if execution already exists
        result = await self.session.execute(
            select(ModularExecutionDB).where(
                ModularExecutionDB.id == execution_id
            )
        )
        existing = result.scalar_one_or_none()

        if not existing:
            # Create new execution record
            execution_db = ModularExecutionDB(
                id=execution_id,
                query=query,
                trace_id=trace_id,
                status="IN_PROGRESS",
                iterations=0,
                execution_metadata=metadata or {},
            )
            self.session.add(execution_db)
            await self.session.flush()

            logger.info(
                "Execution initialized",
                execution_id=execution_id,
                trace_id=trace_id,
            )
        else:
            logger.debug(
                "Execution already exists",
                execution_id=execution_id,
                trace_id=trace_id,
            )

    async def create_checkpoint(
        self,
        execution_id: str,
        plan: EnhancedExecutionPlan,
        current_iteration: int = 0,
        current_module: ModuleType | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionCheckpoint:
        """
        Create a checkpoint for current execution state.

        Args:
            execution_id: Execution identifier
            plan: Current execution plan state
            current_iteration: Current iteration number
            current_module: Currently active module
            metadata: Additional checkpoint metadata

        Returns:
            Created checkpoint
        """
        checkpoint = ExecutionCheckpoint(
            execution_id=execution_id,
            plan_id=plan.plan_id,
            plan_state=plan.model_dump(mode="json"),
            current_iteration=current_iteration,
            current_module=current_module,
            metadata=metadata or {},
        )

        logger.info(
            "Checkpoint created",
            checkpoint_id=checkpoint.checkpoint_id,
            execution_id=execution_id,
            plan_id=plan.plan_id,
            iteration=current_iteration,
        )

        # Persist to database
        await self._persist_checkpoint(checkpoint)

        return checkpoint

    async def _persist_checkpoint(self, checkpoint: ExecutionCheckpoint) -> None:
        """
        Persist checkpoint to database.

        Args:
            checkpoint: Checkpoint to persist
        """
        # Import here to avoid circular dependency
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        # Check if execution exists
        result = await self.session.execute(
            select(ModularExecutionDB).where(
                ModularExecutionDB.id == checkpoint.execution_id
            )
        )
        execution_db = result.scalar_one_or_none()

        if not execution_db:
            logger.warning(
                "Execution not found for checkpoint",
                execution_id=checkpoint.execution_id,
            )
            return

        # Check if plan exists
        result = await self.session.execute(
            select(ExecutionPlanDB).where(
                ExecutionPlanDB.plan_id == checkpoint.plan_id
            )
        )
        plan_db = result.scalar_one_or_none()

        if plan_db:
            # Update plan with checkpoint data
            plan_db.plan_data = checkpoint.plan_state
            plan_db.current_iteration = checkpoint.current_iteration
            plan_db.plan_metadata = {
                **(plan_db.plan_metadata or {}),
                "last_checkpoint_id": checkpoint.checkpoint_id,
                "last_checkpoint_at": checkpoint.created_at,
            }
            await self.session.flush()

            logger.debug(
                "Checkpoint persisted",
                checkpoint_id=checkpoint.checkpoint_id,
                plan_id=checkpoint.plan_id,
            )

    async def restore_from_checkpoint(
        self, checkpoint: ExecutionCheckpoint
    ) -> EnhancedExecutionPlan:
        """
        Restore execution plan from checkpoint.

        Args:
            checkpoint: Checkpoint to restore from

        Returns:
            Restored execution plan
        """
        plan = EnhancedExecutionPlan.model_validate(checkpoint.plan_state)

        logger.info(
            "Execution restored from checkpoint",
            checkpoint_id=checkpoint.checkpoint_id,
            plan_id=plan.plan_id,
            iteration=checkpoint.current_iteration,
        )

        return plan

    async def get_latest_checkpoint(
        self, execution_id: str
    ) -> ExecutionCheckpoint | None:
        """
        Get most recent checkpoint for execution.

        Args:
            execution_id: Execution identifier

        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        # Import here to avoid circular dependency
        from agentcore.a2a_protocol.database.models import ExecutionPlanDB

        # Get plan with latest checkpoint
        result = await self.session.execute(
            select(ExecutionPlanDB)
            .join(
                ExecutionPlanDB.execution,
                isouter=True,
            )
            .where(ExecutionPlanDB.execution_id == execution_id)
            .order_by(ExecutionPlanDB.created_at.desc())
        )
        plan_db = result.scalar_one_or_none()

        if not plan_db or not plan_db.plan_data:
            return None

        # Build checkpoint from stored data
        metadata = plan_db.plan_metadata or {}
        checkpoint = ExecutionCheckpoint(
            checkpoint_id=metadata.get("last_checkpoint_id", str(uuid4())),
            execution_id=execution_id,
            plan_id=plan_db.plan_id,
            created_at=metadata.get(
                "last_checkpoint_at", plan_db.created_at.isoformat()
            ),
            plan_state=plan_db.plan_data,
            current_iteration=plan_db.current_iteration,
            metadata=metadata,
        )

        return checkpoint

    # ========================================================================
    # Crash Recovery
    # ========================================================================

    async def detect_crashed_executions(self) -> list[RecoveryInfo]:
        """
        Detect executions that crashed without proper completion.

        Returns:
            List of recovery information for crashed executions
        """
        # Import here to avoid circular dependency
        from agentcore.a2a_protocol.database.models import ExecutionPlanDB

        # Find plans IN_PROGRESS for more than timeout period
        # (indicating likely crash)
        result = await self.session.execute(
            select(ExecutionPlanDB).where(
                ExecutionPlanDB.status == PlanStatus.IN_PROGRESS.value
            )
        )
        stale_plans = result.scalars().all()

        recovery_list: list[RecoveryInfo] = []

        for plan_db in stale_plans:
            # Check if plan has been running too long
            if plan_db.started_at:
                running_time = (
                    datetime.now(timezone.utc) - plan_db.started_at
                ).total_seconds()

                # Consider crashed if running > 1 hour without updates
                if running_time > 3600:
                    metadata = plan_db.plan_metadata or {}
                    recovery_info = RecoveryInfo(
                        execution_id=plan_db.execution_id,
                        plan_id=plan_db.plan_id,
                        last_checkpoint_id=metadata.get("last_checkpoint_id"),
                        recoverable=True,
                        recovery_point=plan_db.plan_data or {},
                    )
                    recovery_list.append(recovery_info)

                    logger.warning(
                        "Crashed execution detected",
                        execution_id=plan_db.execution_id,
                        plan_id=plan_db.plan_id,
                        running_time_seconds=running_time,
                    )

        return recovery_list

    async def recover_execution(
        self, recovery_info: RecoveryInfo
    ) -> EnhancedExecutionPlan | None:
        """
        Recover execution from crash.

        Args:
            recovery_info: Recovery information

        Returns:
            Recovered execution plan or None if not recoverable
        """
        if not recovery_info.recoverable:
            logger.error(
                "Execution not recoverable",
                execution_id=recovery_info.execution_id,
            )
            return None

        # Get latest checkpoint
        checkpoint = await self.get_latest_checkpoint(recovery_info.execution_id)

        if not checkpoint:
            logger.error(
                "No checkpoint found for recovery",
                execution_id=recovery_info.execution_id,
            )
            return None

        # Restore plan from checkpoint
        plan = await self.restore_from_checkpoint(checkpoint)

        # Mark failed step (if any) as FAILED
        if recovery_info.failed_step_id:
            for step in plan.steps:
                if step.step_id == recovery_info.failed_step_id:
                    if step.status == StepStatus.IN_PROGRESS:
                        step.status = StepStatus.FAILED
                        step.completed_at = datetime.now(timezone.utc).isoformat()
                        step.error = "Execution crashed during step"
                        logger.info(
                            "Marked crashed step as failed",
                            step_id=step.step_id,
                            action=step.action,
                        )

        # Reset IN_PROGRESS steps to PENDING (for retry)
        for step in plan.steps:
            if step.status == StepStatus.IN_PROGRESS:
                step.status = StepStatus.PENDING
                step.started_at = None
                logger.info(
                    "Reset interrupted step to pending",
                    step_id=step.step_id,
                    action=step.action,
                )

        # Update plan status
        if plan.status == PlanStatus.IN_PROGRESS:
            plan.status = PlanStatus.PENDING

        logger.info(
            "Execution recovered",
            execution_id=recovery_info.execution_id,
            plan_id=plan.plan_id,
            checkpoint_id=checkpoint.checkpoint_id,
        )

        # Create recovery checkpoint
        await self.create_checkpoint(
            execution_id=recovery_info.execution_id,
            plan=plan,
            current_iteration=checkpoint.current_iteration,
            metadata={
                "recovered_from_crash": True,
                "recovery_checkpoint_id": checkpoint.checkpoint_id,
                "recovery_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        return plan

    # ========================================================================
    # State Cleanup
    # ========================================================================

    async def cleanup_completed_state(
        self, execution_id: str, keep_final_state: bool = True
    ) -> None:
        """
        Clean up state for completed execution.

        Args:
            execution_id: Execution identifier
            keep_final_state: Whether to keep final state for audit
        """
        # Import here to avoid circular dependency
        from agentcore.a2a_protocol.database.models import ExecutionPlanDB

        if keep_final_state:
            # Keep final plan state but mark as archived
            result = await self.session.execute(
                select(ExecutionPlanDB).where(
                    ExecutionPlanDB.execution_id == execution_id
                )
            )
            plan_db = result.scalar_one_or_none()

            if plan_db:
                if plan_db.plan_metadata is None:
                    plan_db.plan_metadata = {}
                plan_db.plan_metadata["archived"] = True
                plan_db.plan_metadata["archived_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                await self.session.flush()

                logger.info(
                    "Execution state archived",
                    execution_id=execution_id,
                    plan_id=plan_db.plan_id,
                )
        else:
            # Delete intermediate checkpoints (keep final plan)
            # No intermediate checkpoint table yet, so just log
            logger.info(
                "Execution state cleanup completed",
                execution_id=execution_id,
                keep_final_state=keep_final_state,
            )

    async def cleanup_failed_state(self, execution_id: str) -> None:
        """
        Clean up state for failed execution.

        Args:
            execution_id: Execution identifier
        """
        # Import here to avoid circular dependency
        from agentcore.a2a_protocol.database.models import ExecutionPlanDB

        # Mark as failed and archived
        result = await self.session.execute(
            select(ExecutionPlanDB).where(ExecutionPlanDB.execution_id == execution_id)
        )
        plan_db = result.scalar_one_or_none()

        if plan_db:
            plan_db.status = PlanStatus.FAILED.value
            if plan_db.plan_metadata is None:
                plan_db.plan_metadata = {}
            plan_db.plan_metadata["archived"] = True
            plan_db.plan_metadata["archived_at"] = datetime.now(timezone.utc).isoformat()
            await self.session.flush()

            logger.info(
                "Failed execution state archived",
                execution_id=execution_id,
                plan_id=plan_db.plan_id,
            )

    # ========================================================================
    # Status & Monitoring
    # ========================================================================

    async def get_execution_state(self, execution_id: str) -> dict[str, Any]:
        """
        Get current state of execution.

        Args:
            execution_id: Execution identifier

        Returns:
            State information dictionary
        """
        # Import here to avoid circular dependency
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        # Get execution
        result = await self.session.execute(
            select(ModularExecutionDB).where(ModularExecutionDB.id == execution_id)
        )
        execution_db = result.scalar_one_or_none()

        if not execution_db:
            return {"error": "Execution not found"}

        # Get plan
        result = await self.session.execute(
            select(ExecutionPlanDB).where(ExecutionPlanDB.execution_id == execution_id)
        )
        plan_db = result.scalar_one_or_none()

        state: dict[str, Any] = {
            "execution_id": execution_id,
            "status": execution_db.status,
            "iterations": execution_db.iterations,
            "created_at": execution_db.created_at.isoformat(),
        }

        if plan_db:
            state["plan_id"] = plan_db.plan_id
            state["plan_status"] = plan_db.status
            state["current_iteration"] = plan_db.current_iteration
            state["max_iterations"] = plan_db.max_iterations

            if plan_db.started_at:
                state["started_at"] = plan_db.started_at.isoformat()
            if plan_db.completed_at:
                state["completed_at"] = plan_db.completed_at.isoformat()

            # Add checkpoint info
            if plan_db.plan_metadata:
                state["has_checkpoint"] = "last_checkpoint_id" in plan_db.plan_metadata
                state["last_checkpoint_at"] = plan_db.plan_metadata.get(
                    "last_checkpoint_at"
                )

        return state
