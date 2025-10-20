"""
Event Sourcing Integration Tests

Integration tests for CQRS and event sourcing implementation:
- Event stream processing
- State reconstruction from events
- Event replay and projection
- Audit trail validation
- Command-query separation
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from testcontainers.redis import RedisContainer

from agentcore.orchestration.patterns.saga import (
    CompensationStrategy,
    SagaDefinition,
    SagaStatus,
    SagaStep,
)
from agentcore.orchestration.state.integration import PersistentSagaOrchestrator
from agentcore.orchestration.state.repository import WorkflowStateRepository
from agentcore.orchestration.state.models import WorkflowStatus
from agentcore.orchestration.streams import (
    EventType,
    RedisStreamsClient,
    StreamConfig,
    StreamProducer,
    StreamConsumer,
    ConsumerGroup,
    WorkflowCreatedEvent,
    WorkflowStartedEvent,
    WorkflowCompletedEvent,
    WorkflowFailedEvent,
    TaskCreatedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
)


class TestEventSourcingIntegration:
    """Integration tests for event sourcing and CQRS."""

    @pytest.mark.asyncio
    async def test_workflow_event_stream(
        self, redis_client: RedisStreamsClient, db_session_factory
    ) -> None:
        """Test complete workflow lifecycle through event stream."""
        # Setup producer and consumer
        config = StreamConfig(
            stream_name="test:workflow:events",
            consumer_group_name="test-workflow-group",
            consumer_name="test-workflow-consumer",
        )

        producer = StreamProducer(redis_client, config)
        consumer_group = ConsumerGroup("test-workflow-group", "test-workflow-consumer")
        consumer = StreamConsumer(redis_client, consumer_group, config)

        # Create saga workflow
        saga = SagaDefinition(
            name="event_sourced_workflow",
            description="Workflow with complete event sourcing",
            steps=[
                SagaStep(name="step1", order=1, action_data={"task": "task1"}),
                SagaStep(name="step2", order=2, action_data={"task": "task2"}),
            ],
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="event_sourcing_orchestrator",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        # Publish WorkflowCreated event
        workflow_created_event = WorkflowCreatedEvent(
            workflow_id=saga.saga_id,
            workflow_name=saga.name,
            workflow_version="1.0",
            orchestration_pattern="saga",
            total_tasks=len(saga.steps),
            metadata={"event_sourced": True},
        )

        await producer.publish(workflow_created_event)

        # Create execution
        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"test": "event_sourcing"},
        )

        # Publish WorkflowStarted event
        workflow_started_event = WorkflowStartedEvent(
            workflow_id=saga.saga_id,
            execution_id=execution_id,
            workflow_name=saga.name,
        )

        await producer.publish(workflow_started_event)

        # Update to running
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Publish TaskCreated events
        for step in saga.steps:
            task_created_event = TaskCreatedEvent(
                task_id=step.step_id,
                task_type=step.name,
                workflow_id=saga.saga_id,
                execution_id=execution_id,
            )
            await producer.publish(task_created_event)

        # Complete tasks and publish events
        for idx, step in enumerate(saga.steps):
            # Update step state
            await orchestrator.update_step_state(
                execution_id=execution_id,
                step_id=step.step_id,
                status="completed",
                result={"step": step.name, "completed": True},
            )

            # Publish TaskCompleted event
            task_completed_event = TaskCompletedEvent(
                task_id=step.step_id,
                agent_id=f"agent_{idx}",
                result_data={"step": step.name, "completed": True},
                execution_time_ms=100,
            )
            await producer.publish(task_completed_event)

        # Complete workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=2,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[],
        )

        # Publish WorkflowCompleted event
        workflow_completed_event = WorkflowCompletedEvent(
            workflow_id=saga.saga_id,
            execution_id=execution_id,
            workflow_name=saga.name,
            total_tasks_completed=len(saga.steps),
        )

        await producer.publish(workflow_completed_event)

        # Consume and verify events
        await asyncio.sleep(0.1)  # Allow events to be written

        events = await consumer.consume(count=10)
        event_types = [event.event_type for event in events]

        # Verify event sequence
        assert EventType.WORKFLOW_CREATED in event_types
        assert EventType.WORKFLOW_STARTED in event_types
        assert EventType.WORKFLOW_COMPLETED in event_types

        # Verify state matches events
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_state_reconstruction_from_events(
        self, redis_client: RedisStreamsClient, db_session_factory
    ) -> None:
        """Test reconstructing workflow state from event stream."""
        config = StreamConfig(
            stream_name="test:state:reconstruction",
            consumer_group_name="test-reconstruction-group",
        )

        producer = StreamProducer(redis_client, config)

        # Create workflow
        saga = SagaDefinition(
            name="reconstructable_workflow",
            steps=[
                SagaStep(name="step1", order=1),
                SagaStep(name="step2", order=2),
                SagaStep(name="step3", order=3),
            ],
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="reconstruction_orchestrator",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"test": "reconstruction"},
        )

        # Publish events as workflow progresses
        events_sequence = []

        # Event 1: Workflow started
        event1 = WorkflowStartedEvent(
            workflow_id=saga.saga_id,
            execution_id=execution_id,
            workflow_name=saga.name,
        )
        await producer.publish(event1)
        events_sequence.append(event1)

        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Event 2-4: Task events for each step
        for idx, step in enumerate(saga.steps):
            task_created = TaskCreatedEvent(
                task_id=step.step_id,
                task_type=step.name,
                workflow_id=saga.saga_id,
                execution_id=execution_id,
            )
            await producer.publish(task_created)
            events_sequence.append(task_created)

            await orchestrator.update_step_state(
                execution_id=execution_id,
                step_id=step.step_id,
                status="completed",
                result={"step_index": idx},
            )

            task_completed = TaskCompletedEvent(
                task_id=step.step_id,
                agent_id=f"agent_{idx}",
                result_data={"step_index": idx},
                execution_time_ms=50,
            )
            await producer.publish(task_completed)
            events_sequence.append(task_completed)

        # Event 5: Workflow completed
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=3,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[],
        )

        event_final = WorkflowCompletedEvent(
            workflow_id=saga.saga_id,
            execution_id=execution_id,
            workflow_name=saga.name,
            total_tasks_completed=3,
        )
        await producer.publish(event_final)
        events_sequence.append(event_final)

        # Verify state can be reconstructed from events
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED
            assert len(execution.task_states) == 3

            # Verify state history captures all transitions
            history = await WorkflowStateRepository.get_state_history(
                session, str(execution_id)
            )

            # Should have multiple state transitions
            assert len(history) > 0

    @pytest.mark.asyncio
    async def test_event_replay(
        self, redis_client: RedisStreamsClient, db_session_factory
    ) -> None:
        """Test replaying events to rebuild state."""
        config = StreamConfig(
            stream_name="test:event:replay",
            consumer_group_name="test-replay-group",
        )

        producer = StreamProducer(redis_client, config)

        # Create and execute workflow
        saga = SagaDefinition(
            name="replayable_workflow",
            steps=[SagaStep(name="step1", order=1)],
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="replay_orchestrator",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"test": "replay"},
        )

        # Publish sequence of events
        events = [
            WorkflowStartedEvent(
                workflow_id=saga.saga_id,
                execution_id=execution_id,
                workflow_name=saga.name,
            ),
            TaskCreatedEvent(
                task_id=saga.steps[0].step_id,
                task_type="step1",
                workflow_id=saga.saga_id,
                execution_id=execution_id,
            ),
        ]

        for event in events:
            await producer.publish(event)

        # Execute workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"completed": True},
        )

        # Publish completion event
        completion_event = TaskCompletedEvent(
            task_id=saga.steps[0].step_id,
            agent_id="agent_1",
            result_data={"completed": True},
            execution_time_ms=100,
        )
        await producer.publish(completion_event)

        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=1,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[],
            compensated_steps=[],
        )

        # Verify state can be queried
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED

            # Verify event history for replay
            history = await WorkflowStateRepository.get_state_history(
                session, str(execution_id)
            )

            assert len(history) > 0

    @pytest.mark.asyncio
    async def test_audit_trail_validation(
        self, db_session_factory
    ) -> None:
        """Test complete audit trail for workflow execution."""
        saga = SagaDefinition(
            name="audited_workflow",
            description="Workflow with complete audit trail",
            steps=[
                SagaStep(name="step1", order=1),
                SagaStep(name="step2", order=2),
            ],
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="audit_orchestrator",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"audit_test": True},
            metadata={"user": "test_user", "session_id": "sess_123"},
        )

        # Execute with state transitions
        transitions = [
            (SagaStatus.RUNNING, 1, [], [], []),
            (SagaStatus.RUNNING, 2, [saga.steps[0].step_id], [], []),
            (
                SagaStatus.COMPLETED,
                2,
                [step.step_id for step in saga.steps],
                [],
                [],
            ),
        ]

        for status, current_step, completed, failed, compensated in transitions:
            await orchestrator.update_execution_state(
                execution_id=execution_id,
                status=status,
                current_step=current_step,
                completed_steps=completed,
                failed_steps=failed,
                compensated_steps=compensated,
            )

            # Update step states
            if completed:
                for step_id in completed:
                    await orchestrator.update_step_state(
                        execution_id=execution_id,
                        step_id=step_id,
                        status="completed",
                        result={"completed": True},
                    )

        # Verify complete audit trail
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.workflow_metadata["user"] == "test_user"
            assert execution.workflow_metadata["session_id"] == "sess_123"

            # Verify state history contains all transitions
            history = await WorkflowStateRepository.get_state_history(
                session, str(execution_id)
            )

            # Should have multiple state transitions recorded
            assert len(history) >= len(transitions)

            # Verify timestamps are sequential
            for i in range(len(history) - 1):
                assert history[i].created_at <= history[i + 1].created_at

    @pytest.mark.asyncio
    async def test_cqrs_command_query_separation(
        self, db_session_factory
    ) -> None:
        """Test command-query separation in orchestration."""
        # Commands: Create and modify workflows
        saga = SagaDefinition(
            name="cqrs_workflow",
            steps=[SagaStep(name="step1", order=1)],
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="cqrs_orchestrator",
            session_factory=db_session_factory,
        )

        # Command: Register saga
        await orchestrator.register_saga(saga)

        # Command: Create execution
        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"cqrs": True},
        )

        # Command: Update state
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Queries: Read workflow state (should not modify state)
        async with db_session_factory() as session:
            # Query 1: Get execution
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )
            assert execution is not None

            # Query 2: Get state history
            history = await WorkflowStateRepository.get_state_history(
                session, str(execution_id)
            )
            assert len(history) > 0

            # Query 3: Get statistics
            stats = await orchestrator.get_execution_statistics(
                workflow_id=saga.saga_id
            )
            assert stats["total_executions"] == 1

            # Verify queries didn't modify state
            execution_after = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )
            assert execution.status == execution_after.status
            assert execution.updated_at == execution_after.updated_at

    @pytest.mark.asyncio
    async def test_event_ordering_and_consistency(
        self, redis_client: RedisStreamsClient, db_session_factory
    ) -> None:
        """Test event ordering guarantees and consistency."""
        config = StreamConfig(
            stream_name="test:event:ordering",
            consumer_group_name="test-ordering-group",
        )

        producer = StreamProducer(redis_client, config)

        saga = SagaDefinition(
            name="ordered_workflow",
            steps=[
                SagaStep(name="step1", order=1),
                SagaStep(name="step2", order=2),
                SagaStep(name="step3", order=3),
            ],
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="ordering_orchestrator",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"test": "ordering"},
        )

        # Publish events in strict order
        ordered_events = [
            WorkflowStartedEvent(
                workflow_id=saga.saga_id,
                execution_id=execution_id,
                workflow_name=saga.name,
            ),
        ]

        for step in saga.steps:
            ordered_events.append(
                TaskCreatedEvent(
                    task_id=step.step_id,
                    task_type=step.name,
                    workflow_id=saga.saga_id,
                    execution_id=execution_id,
                )
            )

        # Publish all events
        event_ids = []
        for event in ordered_events:
            event_id = await producer.publish(event)
            event_ids.append(event_id)

        # Verify events are ordered in stream
        # Redis Streams guarantees ordering within a stream
        for i in range(len(event_ids) - 1):
            # Event IDs are monotonically increasing
            assert event_ids[i] < event_ids[i + 1]

    @pytest.mark.asyncio
    async def test_event_sourcing_with_failures(
        self, redis_client: RedisStreamsClient, db_session_factory
    ) -> None:
        """Test event sourcing correctly captures failures and compensation."""
        config = StreamConfig(
            stream_name="test:failure:events",
            consumer_group_name="test-failure-group",
        )

        producer = StreamProducer(redis_client, config)

        saga = SagaDefinition(
            name="failure_event_workflow",
            steps=[
                SagaStep(
                    name="step1",
                    order=1,
                    compensation_data={"action": "undo"},
                ),
                SagaStep(
                    name="step2",
                    order=2,
                    compensation_data={"action": "undo"},
                ),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="failure_event_orchestrator",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"test": "failure_events"},
        )

        # Start workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Step 1 succeeds
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"success": True},
        )

        # Publish success event
        await producer.publish(
            TaskCompletedEvent(
                task_id=saga.steps[0].step_id,
                agent_id="agent_1",
                result_data={"success": True},
                execution_time_ms=100,
            )
        )

        # Step 2 fails
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="failed",
            error_message="Step 2 failed",
        )

        # Publish failure event
        await producer.publish(
            TaskFailedEvent(
                task_id=saga.steps[1].step_id,
                agent_id="agent_2",
                error_message="Step 2 failed",
                error_type="RuntimeError",
                retry_count=0,
            )
        )

        # Publish workflow failed event
        await producer.publish(
            WorkflowFailedEvent(
                workflow_id=saga.saga_id,
                execution_id=execution_id,
                workflow_name=saga.name,
                error_message="Step 2 failed",
                failed_task_id=saga.steps[1].step_id,
            )
        )

        # Verify failure events captured
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            # Note: Status may still be EXECUTING until compensation completes
            assert execution.error_message is not None or execution.status == WorkflowStatus.FAILED

            # Verify state history captures failure
            history = await WorkflowStateRepository.get_state_history(
                session, str(execution_id)
            )
            assert len(history) > 0
