"""
Tests for Module State Management

Validates checkpoint creation, state serialization, crash recovery,
and state cleanup in the module state manager.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from agentcore.modular.state_manager import (
    StateManager,
    ExecutionCheckpoint,
    RecoveryInfo,
)
from agentcore.modular.models import (
    EnhancedExecutionPlan,
    EnhancedPlanStep,
    PlanStatus,
    StepStatus,
    ModuleType,
    SuccessCriteria,
    SuccessCriterion,
)


class TestExecutionCheckpoint:
    """Test ExecutionCheckpoint model."""

    def test_checkpoint_creation(self) -> None:
        """Test creating execution checkpoint."""
        checkpoint = ExecutionCheckpoint(
            execution_id="exec-123",
            plan_id="plan-456",
            plan_state={"status": "in_progress"},
            current_iteration=2,
            current_module=ModuleType.EXECUTOR,
        )

        assert checkpoint.execution_id == "exec-123"
        assert checkpoint.plan_id == "plan-456"
        assert checkpoint.current_iteration == 2
        assert checkpoint.current_module == ModuleType.EXECUTOR
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.created_at is not None

    def test_checkpoint_with_metadata(self) -> None:
        """Test checkpoint with additional metadata."""
        checkpoint = ExecutionCheckpoint(
            execution_id="exec-123",
            plan_id="plan-456",
            plan_state={},
            metadata={"notes": "test checkpoint", "version": "1.0"},
        )

        assert checkpoint.metadata["notes"] == "test checkpoint"
        assert checkpoint.metadata["version"] == "1.0"


class TestRecoveryInfo:
    """Test RecoveryInfo model."""

    def test_recovery_info_creation(self) -> None:
        """Test creating recovery info."""
        recovery = RecoveryInfo(
            execution_id="exec-123",
            plan_id="plan-456",
            last_checkpoint_id="chk-789",
            recoverable=True,
            failed_step_id="step-001",
        )

        assert recovery.execution_id == "exec-123"
        assert recovery.plan_id == "plan-456"
        assert recovery.last_checkpoint_id == "chk-789"
        assert recovery.recoverable is True
        assert recovery.failed_step_id == "step-001"
        assert recovery.crash_detected_at is not None

    def test_recovery_info_defaults(self) -> None:
        """Test recovery info default values."""
        recovery = RecoveryInfo(
            execution_id="exec-123",
            plan_id="plan-456",
        )

        assert recovery.recoverable is True
        assert recovery.recovery_point == {}
        assert recovery.last_checkpoint_id is None
        assert recovery.failed_step_id is None


@pytest.mark.asyncio
class TestStateManager:
    """Test StateManager class."""

    @pytest.fixture
    def sample_plan(self) -> EnhancedExecutionPlan:
        """Create a sample execution plan."""
        return EnhancedExecutionPlan(
            plan_id="plan-123",
            steps=[
                EnhancedPlanStep(
                    step_id="step-1",
                    action="test_action_1",
                    parameters={"param": "value1"},
                    status=StepStatus.COMPLETED,
                ),
                EnhancedPlanStep(
                    step_id="step-2",
                    action="test_action_2",
                    parameters={"param": "value2"},
                    status=StepStatus.IN_PROGRESS,
                ),
                EnhancedPlanStep(
                    step_id="step-3",
                    action="test_action_3",
                    parameters={"param": "value3"},
                    status=StepStatus.PENDING,
                ),
            ],
            status=PlanStatus.IN_PROGRESS,
            max_iterations=5,
            current_iteration=2,
        )

    async def test_create_checkpoint(
        self, async_session, sample_plan: EnhancedExecutionPlan
    ) -> None:
        """Test creating checkpoint."""
        state_manager = StateManager(async_session)

        # Create execution and plan in database first
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        execution_db = ModularExecutionDB(
            id="exec-123",
            query="test query",
            status="in_progress",
            created_at=datetime.now(timezone.utc),
        )
        async_session.add(execution_db)
        await async_session.flush()

        plan_db = ExecutionPlanDB(
            plan_id="plan-123",
            execution_id="exec-123",
            plan_data={},
            status="in_progress",
            created_at=datetime.now(timezone.utc),
        )
        async_session.add(plan_db)
        await async_session.flush()

        # Create checkpoint
        checkpoint = await state_manager.create_checkpoint(
            execution_id="exec-123",
            plan=sample_plan,
            current_iteration=2,
            current_module=ModuleType.EXECUTOR,
            metadata={"test": "data"},
        )

        assert checkpoint.execution_id == "exec-123"
        assert checkpoint.plan_id == "plan-123"
        assert checkpoint.current_iteration == 2
        assert checkpoint.current_module == ModuleType.EXECUTOR
        assert checkpoint.metadata["test"] == "data"
        assert "steps" in checkpoint.plan_state

    async def test_restore_from_checkpoint(
        self, async_session, sample_plan: EnhancedExecutionPlan
    ) -> None:
        """Test restoring execution from checkpoint."""
        state_manager = StateManager(async_session)

        # Create checkpoint
        checkpoint = ExecutionCheckpoint(
            execution_id="exec-123",
            plan_id="plan-123",
            plan_state=sample_plan.model_dump(mode="json"),
            current_iteration=2,
        )

        # Restore from checkpoint
        restored_plan = await state_manager.restore_from_checkpoint(checkpoint)

        assert restored_plan.plan_id == sample_plan.plan_id
        assert len(restored_plan.steps) == len(sample_plan.steps)
        assert restored_plan.status == sample_plan.status
        assert restored_plan.current_iteration == sample_plan.current_iteration

    async def test_get_latest_checkpoint_none(self, async_session) -> None:
        """Test getting latest checkpoint when none exist."""
        state_manager = StateManager(async_session)

        checkpoint = await state_manager.get_latest_checkpoint("nonexistent-exec")

        assert checkpoint is None

    async def test_get_latest_checkpoint(
        self, async_session, sample_plan: EnhancedExecutionPlan
    ) -> None:
        """Test getting latest checkpoint."""
        state_manager = StateManager(async_session)

        # Create execution and plan
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        execution_db = ModularExecutionDB(
            id="exec-123",
            query="test query",
            status="in_progress",
            created_at=datetime.now(timezone.utc),
        )
        async_session.add(execution_db)
        await async_session.flush()

        plan_db = ExecutionPlanDB(
            plan_id="plan-123",
            execution_id="exec-123",
            plan_data=sample_plan.model_dump(mode="json"),
            status="in_progress",
            current_iteration=2,
            created_at=datetime.now(timezone.utc),
            plan_metadata={
                "last_checkpoint_id": "chk-123",
                "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        async_session.add(plan_db)
        await async_session.flush()

        # Get latest checkpoint
        checkpoint = await state_manager.get_latest_checkpoint("exec-123")

        assert checkpoint is not None
        assert checkpoint.execution_id == "exec-123"
        assert checkpoint.plan_id == "plan-123"
        assert checkpoint.current_iteration == 2

    async def test_detect_crashed_executions_none(self, async_session) -> None:
        """Test detecting crashed executions when none exist."""
        state_manager = StateManager(async_session)

        crashed = await state_manager.detect_crashed_executions()

        assert len(crashed) == 0

    async def test_detect_crashed_executions(self, async_session) -> None:
        """Test detecting crashed executions."""
        state_manager = StateManager(async_session)

        # Create stale execution (running > 1 hour)
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        execution_db = ModularExecutionDB(
            id="exec-crashed",
            query="test query",
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        async_session.add(execution_db)
        await async_session.flush()

        plan_db = ExecutionPlanDB(
            plan_id="plan-crashed",
            execution_id="exec-crashed",
            plan_data={"status": "in_progress"},
            status="in_progress",
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            started_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        async_session.add(plan_db)
        await async_session.flush()

        # Detect crashed executions
        crashed = await state_manager.detect_crashed_executions()

        assert len(crashed) == 1
        assert crashed[0].execution_id == "exec-crashed"
        assert crashed[0].plan_id == "plan-crashed"
        assert crashed[0].recoverable is True

    async def test_recover_execution_no_checkpoint(self, async_session) -> None:
        """Test recovering execution without checkpoint."""
        state_manager = StateManager(async_session)

        recovery_info = RecoveryInfo(
            execution_id="exec-123",
            plan_id="plan-123",
            recoverable=True,
        )

        recovered_plan = await state_manager.recover_execution(recovery_info)

        assert recovered_plan is None

    async def test_recover_execution_not_recoverable(self, async_session) -> None:
        """Test recovering non-recoverable execution."""
        state_manager = StateManager(async_session)

        recovery_info = RecoveryInfo(
            execution_id="exec-123",
            plan_id="plan-123",
            recoverable=False,
        )

        recovered_plan = await state_manager.recover_execution(recovery_info)

        assert recovered_plan is None

    async def test_recover_execution(
        self, async_session, sample_plan: EnhancedExecutionPlan
    ) -> None:
        """Test recovering execution from crash."""
        state_manager = StateManager(async_session)

        # Create crashed execution
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        execution_db = ModularExecutionDB(
            id="exec-crashed",
            query="test query",
            status="in_progress",
            created_at=datetime.now(timezone.utc),
        )
        async_session.add(execution_db)
        await async_session.flush()

        # Create a copy of sample_plan with correct plan_id
        crashed_plan = sample_plan.model_copy(update={"plan_id": "plan-crashed"})

        plan_db = ExecutionPlanDB(
            plan_id="plan-crashed",
            execution_id="exec-crashed",
            plan_data=crashed_plan.model_dump(mode="json"),
            status="in_progress",
            current_iteration=2,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            plan_metadata={
                "last_checkpoint_id": "chk-123",
                "last_checkpoint_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        async_session.add(plan_db)
        await async_session.flush()

        # Create recovery info
        recovery_info = RecoveryInfo(
            execution_id="exec-crashed",
            plan_id="plan-crashed",
            recoverable=True,
            failed_step_id="step-2",  # The IN_PROGRESS step
        )

        # Recover execution
        recovered_plan = await state_manager.recover_execution(recovery_info)

        assert recovered_plan is not None
        assert recovered_plan.plan_id == "plan-crashed"

        # Check that IN_PROGRESS step was marked as FAILED
        failed_step = next(
            (s for s in recovered_plan.steps if s.step_id == "step-2"), None
        )
        assert failed_step is not None
        assert failed_step.status == StepStatus.FAILED
        assert failed_step.error is not None

    async def test_cleanup_completed_state_with_keep(self, async_session) -> None:
        """Test cleaning up completed execution state (keeping final state)."""
        state_manager = StateManager(async_session)

        # Create completed execution
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        execution_db = ModularExecutionDB(
            id="exec-completed",
            query="test query",
            status="completed",
            created_at=datetime.now(timezone.utc),
        )
        async_session.add(execution_db)
        await async_session.flush()

        plan_db = ExecutionPlanDB(
            plan_id="plan-completed",
            execution_id="exec-completed",
            plan_data={"status": "completed"},
            status="completed",
            created_at=datetime.now(timezone.utc),
            plan_metadata={},
        )
        async_session.add(plan_db)
        await async_session.flush()

        # Cleanup
        await state_manager.cleanup_completed_state("exec-completed", keep_final_state=True)
        await async_session.commit()

        # Check that plan was archived
        from sqlalchemy import select

        result = await async_session.execute(
            select(ExecutionPlanDB).where(
                ExecutionPlanDB.plan_id == "plan-completed"
            )
        )
        plan_db = result.scalar_one_or_none()

        assert plan_db is not None
        assert plan_db.plan_metadata is not None
        assert plan_db.plan_metadata.get("archived") is True

    async def test_cleanup_failed_state(self, async_session) -> None:
        """Test cleaning up failed execution state."""
        state_manager = StateManager(async_session)

        # Create failed execution
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        execution_db = ModularExecutionDB(
            id="exec-failed",
            query="test query",
            status="failed",
            created_at=datetime.now(timezone.utc),
        )
        async_session.add(execution_db)
        await async_session.flush()

        plan_db = ExecutionPlanDB(
            plan_id="plan-failed",
            execution_id="exec-failed",
            plan_data={"status": "failed"},
            status="in_progress",  # Should be updated to failed
            created_at=datetime.now(timezone.utc),
            plan_metadata={},
        )
        async_session.add(plan_db)
        await async_session.flush()

        # Cleanup
        await state_manager.cleanup_failed_state("exec-failed")
        await async_session.commit()

        # Check that plan was marked as failed and archived
        from sqlalchemy import select

        result = await async_session.execute(
            select(ExecutionPlanDB).where(ExecutionPlanDB.plan_id == "plan-failed")
        )
        plan_db = result.scalar_one_or_none()

        assert plan_db is not None
        assert plan_db.status == "failed"
        assert plan_db.plan_metadata is not None
        assert plan_db.plan_metadata.get("archived") is True

    async def test_get_execution_state_not_found(self, async_session) -> None:
        """Test getting state for non-existent execution."""
        state_manager = StateManager(async_session)

        state = await state_manager.get_execution_state("nonexistent")

        assert "error" in state
        assert state["error"] == "Execution not found"

    async def test_get_execution_state(self, async_session) -> None:
        """Test getting execution state."""
        state_manager = StateManager(async_session)

        # Create execution
        from agentcore.a2a_protocol.database.models import (
            ModularExecutionDB,
            ExecutionPlanDB,
        )

        execution_db = ModularExecutionDB(
            id="exec-123",
            query="test query",
            status="in_progress",
            iterations=2,
            created_at=datetime.now(timezone.utc),
        )
        async_session.add(execution_db)
        await async_session.flush()

        plan_db = ExecutionPlanDB(
            plan_id="plan-123",
            execution_id="exec-123",
            plan_data={},
            status="in_progress",
            current_iteration=2,
            max_iterations=5,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            plan_metadata={"last_checkpoint_id": "chk-123"},
        )
        async_session.add(plan_db)
        await async_session.flush()

        # Get state
        state = await state_manager.get_execution_state("exec-123")

        assert state["execution_id"] == "exec-123"
        assert state["status"] == "in_progress"
        assert state["iterations"] == 2
        assert state["plan_id"] == "plan-123"
        assert state["plan_status"] == "in_progress"
        assert state["current_iteration"] == 2
        assert state["max_iterations"] == 5
        assert state["has_checkpoint"] is True
