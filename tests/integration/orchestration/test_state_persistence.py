"""
Integration Tests for State Persistence

End-to-end tests for PostgreSQL state management with saga orchestrator.
"""

import pytest
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.orchestration.patterns.saga import (
    SagaDefinition,
    SagaStep,
    SagaStatus,
    CompensationStrategy)
from agentcore.orchestration.state.integration import PersistentSagaOrchestrator
from agentcore.orchestration.state.repository import WorkflowStateRepository
from agentcore.orchestration.state.models import WorkflowStatus


class TestStatePersistence:
    """Integration tests for state persistence with saga orchestrator."""

    @pytest.mark.asyncio
    async def test_saga_execution_persistence(
        self, db_session_factory
    ) -> None:
        """Test full saga execution with state persistence."""
        # Create saga definition
        saga = SagaDefinition(
            name="test_saga",
            description="Test saga for persistence",
            steps=[
                SagaStep(name="step1", order=1, action_data={"action": "data1"}),
                SagaStep(name="step2", order=2, action_data={"action": "data2"}),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True,
            checkpoint_interval=1)

        # Create orchestrator with persistent storage
        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator",
            session_factory=db_session_factory)

        # Register saga
        await orchestrator.register_saga(saga)

        # Create execution
        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"input": "test_data"},
            tags=["integration", "test"],
            metadata={"test": True})

        assert execution_id is not None

        # Verify execution was persisted
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.workflow_name == "test_saga"
            assert execution.orchestration_pattern == "saga"
            assert execution.status == WorkflowStatus.PENDING
            assert execution.input_data == {"input": "test_data"}
            assert "integration" in execution.tags
            assert execution.workflow_metadata["test"] is True

            # Verify task states initialized
            assert len(execution.task_states) == 2
            assert execution.total_tasks == 2

    @pytest.mark.asyncio
    async def test_saga_state_updates(self, db_session_factory) -> None:
        """Test saga state updates during execution."""
        saga = SagaDefinition(
            name="update_test_saga",
            steps=[
                SagaStep(name="step1", order=1),
                SagaStep(name="step2", order=2),
            ])

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)
        execution_id = await orchestrator.create_execution(saga_id=saga.saga_id)

        # Update execution to running
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[],
            compensated_steps=[])

        # Verify state was updated
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.EXECUTING
            assert execution.started_at is not None
            assert execution.execution_state["current_step"] == 1
            assert len(execution.execution_state["completed_steps"]) == 1

            # Verify state history was created
            history = await WorkflowStateRepository.get_state_history(
                session, str(execution_id)
            )

            # Should have initial state + update
            assert len(history) >= 1

    @pytest.mark.asyncio
    async def test_saga_checkpoint_creation(
        self, db_session_factory
    ) -> None:
        """Test checkpoint creation and recovery."""
        saga = SagaDefinition(
            name="checkpoint_saga",
            steps=[SagaStep(name="step1", order=1)],
            enable_state_persistence=True)

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)
        execution_id = await orchestrator.create_execution(saga_id=saga.saga_id)

        # Create checkpoint
        checkpoint_data = {
            "current_step": 1,
            "state": "checkpoint_test",
            "context": {"data": "value"},
        }

        await orchestrator.create_checkpoint(
            execution_id=execution_id, checkpoint_data=checkpoint_data
        )

        # Verify checkpoint was created
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.checkpoint_data == checkpoint_data
            assert execution.checkpoint_count == 1
            assert execution.last_checkpoint_at is not None

            # Verify checkpoint in history
            checkpoints = await WorkflowStateRepository.get_state_history(
                session, str(execution_id), state_type="checkpoint"
            )

            assert len(checkpoints) == 1
            assert checkpoints[0].state_snapshot == checkpoint_data

        # Test recovery from checkpoint
        recovered_data = await orchestrator.recover_from_checkpoint(execution_id)

        assert recovered_data == checkpoint_data

    @pytest.mark.asyncio
    async def test_step_state_tracking(self, db_session_factory) -> None:
        """Test individual step state tracking."""
        saga = SagaDefinition(
            name="step_tracking_saga",
            steps=[
                SagaStep(name="step1", order=1),
                SagaStep(name="step2", order=2),
            ])

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)
        execution_id = await orchestrator.create_execution(saga_id=saga.saga_id)

        step1_id = saga.steps[0].step_id
        step2_id = saga.steps[1].step_id

        # Update step1 to completed
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=step1_id,
            status="completed",
            retry_count=0,
            result={"output": "step1_result"})

        # Update step2 to running
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=step2_id,
            status="running",
            retry_count=1)

        # Verify step states
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None

            step1_state = execution.task_states[str(step1_id)]
            assert step1_state["status"] == "completed"
            assert step1_state["result"] == {"output": "step1_result"}

            step2_state = execution.task_states[str(step2_id)]
            assert step2_state["status"] == "running"
            assert step2_state["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_saga_completion_flow(
        self, db_session_factory
    ) -> None:
        """Test complete saga execution flow with state persistence."""
        saga = SagaDefinition(
            name="completion_saga",
            steps=[SagaStep(name="step1", order=1)])

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)
        execution_id = await orchestrator.create_execution(saga_id=saga.saga_id)

        # Simulate execution flow: pending -> running -> completed
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[])

        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=1,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[],
            compensated_steps=[])

        # Verify final state
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED
            assert execution.started_at is not None
            assert execution.completed_at is not None
            assert execution.duration_seconds is not None
            assert execution.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_execution_statistics(self, db_session_factory) -> None:
        """Test execution statistics aggregation."""
        saga = SagaDefinition(
            name="stats_saga",
            steps=[SagaStep(name="step1", order=1)])

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)

        # Create multiple executions
        exec1_id = await orchestrator.create_execution(saga_id=saga.saga_id)
        exec2_id = await orchestrator.create_execution(saga_id=saga.saga_id)
        exec3_id = await orchestrator.create_execution(saga_id=saga.saga_id)

        # Complete first execution
        await orchestrator.update_execution_state(
            execution_id=exec1_id,
            status=SagaStatus.RUNNING,
            current_step=0,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[])

        await orchestrator.update_execution_state(
            execution_id=exec1_id,
            status=SagaStatus.COMPLETED,
            current_step=1,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[],
            compensated_steps=[])

        # Fail second execution
        await orchestrator.update_execution_state(
            execution_id=exec2_id,
            status=SagaStatus.FAILED,
            current_step=0,
            completed_steps=[],
            failed_steps=[saga.steps[0].step_id],
            compensated_steps=[],
            error_message="Test failure")

        # Get statistics
        stats = await orchestrator.get_execution_statistics(workflow_id=saga.saga_id)

        assert stats["total_executions"] == 3
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["failed"] == 1
        assert stats["by_status"]["pending"] == 1
        assert stats["by_pattern"]["saga"] == 3
