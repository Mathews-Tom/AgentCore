"""
End-to-End Workflow Integration Tests

Comprehensive integration tests for complete workflow execution including:
- Simple sequential workflows
- Complex parallel workflows with dependencies
- Multi-agent coordination
- Error handling and recovery
- State persistence and checkpointing
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from agentcore.orchestration.patterns.saga import (
    CompensationStrategy,
    SagaDefinition,
    SagaStatus,
    SagaStep,
)
from agentcore.orchestration.state.integration import PersistentSagaOrchestrator
from agentcore.orchestration.state.repository import WorkflowStateRepository
from agentcore.orchestration.state.models import WorkflowStatus


class TestEndToEndWorkflows:
    """End-to-end workflow execution tests."""

    @pytest.mark.asyncio
    async def test_simple_sequential_workflow(
        self, db_session_factory
    ) -> None:
        """Test simple sequential workflow execution from start to finish."""
        # Define a simple 3-step sequential workflow
        saga = SagaDefinition(
            name="simple_sequential_workflow",
            description="Simple sequential workflow with 3 steps",
            steps=[
                SagaStep(
                    name="initialize",
                    order=1,
                    action_data={"task": "initialize_system"},
                ),
                SagaStep(
                    name="process",
                    order=2,
                    action_data={"task": "process_data"},
                ),
                SagaStep(
                    name="finalize",
                    order=3,
                    action_data={"task": "finalize_results"},
                ),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True,
        )

        # Create orchestrator
        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator_001",
            session_factory=db_session_factory,
        )

        # Register saga
        await orchestrator.register_saga(saga)

        # Create and execute workflow
        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"workflow_type": "sequential", "priority": "high"},
            tags=["test", "sequential"],
        )

        # Simulate workflow execution by marking steps as completed
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Complete step 1
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"initialized": True, "timestamp": "2025-10-20T00:00:00Z"},
        )

        # Complete step 2
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="completed",
            result={"processed_items": 100, "status": "success"},
        )

        # Complete step 3 and workflow
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[2].step_id,
            status="completed",
            result={"finalized": True, "output_file": "results.json"},
        )

        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=3,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[],
        )

        # Verify final state
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED
            assert execution.total_tasks == 3
            assert len(execution.task_states) == 3
            assert all(
                state["status"] == "completed"
                for state in execution.task_states.values()
            )

    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(
        self, db_session_factory
    ) -> None:
        """Test workflow with parallel task execution."""
        # Define workflow with parallel steps
        saga = SagaDefinition(
            name="parallel_workflow",
            description="Workflow with parallel processing steps",
            steps=[
                SagaStep(name="init", order=1, action_data={"phase": "setup"}),
                # These steps can run in parallel (same order)
                SagaStep(
                    name="analyze_text",
                    order=2,
                    action_data={"task": "text_analysis"},
                ),
                SagaStep(
                    name="analyze_images",
                    order=2,
                    action_data={"task": "image_analysis"},
                ),
                SagaStep(
                    name="analyze_audio",
                    order=2,
                    action_data={"task": "audio_analysis"},
                ),
                SagaStep(
                    name="merge_results",
                    order=3,
                    action_data={"task": "merge_all"},
                ),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator_002",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"content_types": ["text", "image", "audio"]},
        )

        # Start execution
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Complete init step
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"setup_complete": True},
        )

        # Simulate parallel execution of analysis steps
        parallel_steps = saga.steps[1:4]
        await asyncio.gather(
            *[
                orchestrator.update_step_state(
                    execution_id=execution_id,
                    step_id=step.step_id,
                    status="completed",
                    result={
                        "analysis_type": step.action_data["task"],
                        "items_processed": 50,
                    },
                )
                for step in parallel_steps
            ]
        )

        # Complete merge step
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[4].step_id,
            status="completed",
            result={"merged_results": 150, "status": "success"},
        )

        # Complete workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=3,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[],
        )

        # Verify all parallel steps completed
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED
            assert len(execution.task_states) == 5

            # Verify parallel steps all completed
            for step in parallel_steps:
                step_state = execution.task_states[str(step.step_id)]
                assert step_state["status"] == "completed"

    @pytest.mark.asyncio
    async def test_workflow_with_failure_and_compensation(
        self, db_session_factory
    ) -> None:
        """Test workflow failure handling with saga compensation."""
        saga = SagaDefinition(
            name="compensating_workflow",
            description="Workflow with failure and compensation",
            steps=[
                SagaStep(
                    name="create_order",
                    order=1,
                    action_data={"action": "create"},
                    compensation_data={"action": "delete_order"},
                ),
                SagaStep(
                    name="charge_payment",
                    order=2,
                    action_data={"action": "charge"},
                    compensation_data={"action": "refund_payment"},
                ),
                SagaStep(
                    name="send_confirmation",
                    order=3,
                    action_data={"action": "send_email"},
                    compensation_data={"action": "send_cancellation"},
                ),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator_003",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"order_id": "ORD-12345", "amount": 99.99},
        )

        # Start execution
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Complete step 1 successfully
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"order_created": True, "order_id": "ORD-12345"},
        )

        # Complete step 2 successfully
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="completed",
            result={"payment_charged": True, "transaction_id": "TXN-67890"},
        )

        # Step 3 fails
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[2].step_id,
            status="failed",
            error_message="Email service unavailable",
        )

        # Mark execution as failed
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.FAILED,
            current_step=3,
            completed_steps=[saga.steps[0].step_id, saga.steps[1].step_id],
            failed_steps=[saga.steps[2].step_id],
            compensated_steps=[],
            error_message="Email service unavailable",
        )

        # Start compensation
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPENSATING,
            current_step=3,
            completed_steps=[saga.steps[0].step_id, saga.steps[1].step_id],
            failed_steps=[saga.steps[2].step_id],
            compensated_steps=[],
        )

        # Compensate in reverse order (step 2, then step 1)
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="compensated",
            result={"refunded": True, "refund_id": "REF-11111"},
        )

        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="compensated",
            result={"order_deleted": True},
        )

        # Mark as compensated
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPENSATED,
            current_step=3,
            completed_steps=[saga.steps[0].step_id, saga.steps[1].step_id],
            failed_steps=[saga.steps[2].step_id],
            compensated_steps=[saga.steps[0].step_id, saga.steps[1].step_id],
        )

        # Verify compensation completed
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPENSATED
            assert execution.error_message == "Email service unavailable"

            # Verify step states
            step1_state = execution.task_states[str(saga.steps[0].step_id)]
            step2_state = execution.task_states[str(saga.steps[1].step_id)]
            step3_state = execution.task_states[str(saga.steps[2].step_id)]

            assert step1_state["status"] == "compensated"
            assert step2_state["status"] == "compensated"
            assert step3_state["status"] == "failed"

    @pytest.mark.asyncio
    async def test_workflow_checkpointing(self, db_session_factory) -> None:
        """Test workflow execution with checkpointing for recovery."""
        saga = SagaDefinition(
            name="checkpointed_workflow",
            description="Long-running workflow with checkpoints",
            steps=[
                SagaStep(name="step1", order=1, action_data={"task": "task1"}),
                SagaStep(name="step2", order=2, action_data={"task": "task2"}),
                SagaStep(name="step3", order=3, action_data={"task": "task3"}),
                SagaStep(name="step4", order=4, action_data={"task": "task4"}),
            ],
            enable_state_persistence=True,
            checkpoint_interval=2,  # Checkpoint every 2 steps
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator_004",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"batch_size": 1000},
        )

        # Start execution
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Complete step 1
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"processed": 250},
        )

        # Complete step 2 and create checkpoint
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="completed",
            result={"processed": 500},
        )

        checkpoint_data_1 = {
            "completed_steps": 2,
            "total_processed": 500,
            "current_phase": "phase_1_complete",
        }
        await orchestrator.create_checkpoint(
            execution_id=execution_id,
            checkpoint_data=checkpoint_data_1,
        )

        # Complete step 3
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[2].step_id,
            status="completed",
            result={"processed": 750},
        )

        # Complete step 4 and create final checkpoint
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[3].step_id,
            status="completed",
            result={"processed": 1000},
        )

        checkpoint_data_2 = {
            "completed_steps": 4,
            "total_processed": 1000,
            "current_phase": "all_complete",
        }
        await orchestrator.create_checkpoint(
            execution_id=execution_id,
            checkpoint_data=checkpoint_data_2,
        )

        # Complete workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=4,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[],
        )

        # Verify checkpoints were created
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.checkpoint_count == 2
            assert execution.checkpoint_data == checkpoint_data_2

            # Verify checkpoint history
            checkpoints = await WorkflowStateRepository.get_state_history(
                session, str(execution_id), state_type="checkpoint"
            )

            assert len(checkpoints) == 2

        # Test recovery from checkpoint
        recovered_data = await orchestrator.recover_from_checkpoint(execution_id)
        assert recovered_data == checkpoint_data_2

    @pytest.mark.asyncio
    async def test_complex_multi_stage_workflow(
        self, db_session_factory
    ) -> None:
        """Test complex workflow with multiple stages and dependencies."""
        saga = SagaDefinition(
            name="complex_pipeline",
            description="Complex data processing pipeline",
            steps=[
                # Stage 1: Data ingestion
                SagaStep(
                    name="ingest_sources",
                    order=1,
                    action_data={"sources": ["api", "database", "files"]},
                ),
                # Stage 2: Parallel validation
                SagaStep(
                    name="validate_api_data",
                    order=2,
                    action_data={"source": "api"},
                ),
                SagaStep(
                    name="validate_db_data",
                    order=2,
                    action_data={"source": "database"},
                ),
                SagaStep(
                    name="validate_file_data",
                    order=2,
                    action_data={"source": "files"},
                ),
                # Stage 3: Transformation
                SagaStep(
                    name="transform_data",
                    order=3,
                    action_data={"transformations": ["normalize", "enrich"]},
                ),
                # Stage 4: Parallel analysis
                SagaStep(
                    name="statistical_analysis",
                    order=4,
                    action_data={"analysis_type": "stats"},
                ),
                SagaStep(
                    name="ml_analysis",
                    order=4,
                    action_data={"analysis_type": "ml"},
                ),
                # Stage 5: Output
                SagaStep(
                    name="generate_reports",
                    order=5,
                    action_data={"formats": ["pdf", "json"]},
                ),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True,
            checkpoint_interval=2,
        )

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="test_orchestrator_005",
            session_factory=db_session_factory,
        )

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={
                "pipeline_id": "PIPE-001",
                "priority": "high",
                "deadline": "2025-10-21T00:00:00Z",
            },
            tags=["complex", "pipeline", "production"],
        )

        # Execute workflow stages
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[],
        )

        # Stage 1: Ingest
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"total_records": 10000},
        )

        # Stage 2: Parallel validation
        for step in saga.steps[1:4]:
            await orchestrator.update_step_state(
                execution_id=execution_id,
                step_id=step.step_id,
                status="completed",
                result={"validated_records": 3333, "errors": 0},
            )

        # Checkpoint after validation
        await orchestrator.create_checkpoint(
            execution_id=execution_id,
            checkpoint_data={"stage": "validation_complete", "records": 10000},
        )

        # Stage 3: Transform
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[4].step_id,
            status="completed",
            result={"transformed_records": 10000},
        )

        # Stage 4: Parallel analysis
        for step in saga.steps[5:7]:
            await orchestrator.update_step_state(
                execution_id=execution_id,
                step_id=step.step_id,
                status="completed",
                result={"analysis_complete": True, "insights": 50},
            )

        # Checkpoint after analysis
        await orchestrator.create_checkpoint(
            execution_id=execution_id,
            checkpoint_data={"stage": "analysis_complete", "insights": 100},
        )

        # Stage 5: Output
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[7].step_id,
            status="completed",
            result={"reports_generated": 2, "output_path": "/reports/"},
        )

        # Complete workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=5,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[],
        )

        # Verify complete workflow execution
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED
            assert execution.total_tasks == 8
            assert len(execution.task_states) == 8
            assert execution.checkpoint_count == 2

            # Verify all steps completed
            assert all(
                state["status"] == "completed"
                for state in execution.task_states.values()
            )

            # Verify tags
            assert "complex" in execution.tags
            assert "pipeline" in execution.tags
