"""
Multi-Pattern Integration Tests

Integration tests for multiple orchestration patterns working together:
- Saga pattern with circuit breaker
- Swarm coordination with fault tolerance
- Custom patterns with hooks
- Pattern composition and interoperability
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from agentcore.orchestration.patterns.saga import (
    CompensationStrategy,
    SagaDefinition,
    SagaStatus,
    SagaStep)
from agentcore.orchestration.patterns.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    FaultToleranceCoordinator)
from agentcore.orchestration.patterns.swarm import (
    AgentRole,
    AgentState,
    ConsensusStrategy,
    SwarmConfig,
    SwarmCoordinator,
    SwarmTask)
from agentcore.orchestration.patterns.custom import (
    AgentRequirement,
    CoordinationConfig,
    CoordinationModel,
    PatternDefinition,
    PatternType,
    TaskNode,
    pattern_registry)
from agentcore.orchestration.state.integration import PersistentSagaOrchestrator
from agentcore.orchestration.state.repository import WorkflowStateRepository
from agentcore.orchestration.state.models import WorkflowStatus


class TestMultiPatternIntegration:
    """Multi-pattern orchestration integration tests."""

    @pytest.mark.asyncio
    async def test_saga_with_circuit_breaker(
        self, db_session_factory
    ) -> None:
        """Test saga pattern with circuit breaker for fault tolerance."""
        # Create circuit breaker for external service calls
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=30,
            recovery_timeout_seconds=60)
        circuit_breaker = CircuitBreaker(
            service_name="payment_service_breaker",
            config=circuit_config)

        # Define saga with circuit breaker protection
        saga = SagaDefinition(
            name="payment_saga_with_breaker",
            description="Payment workflow with circuit breaker protection",
            steps=[
                SagaStep(
                    name="validate_payment",
                    order=1,
                    action_data={"service": "validation", "breaker": True}),
                SagaStep(
                    name="process_payment",
                    order=2,
                    action_data={"service": "processing", "breaker": True},
                    compensation_data={"action": "refund"}),
                SagaStep(
                    name="update_ledger",
                    order=3,
                    action_data={"service": "ledger"}),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True)

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="saga_breaker_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"amount": 150.00, "currency": "USD"})

        # Start execution
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[])

        # Simulate circuit breaker state transitions
        assert circuit_breaker.state == CircuitState.CLOSED

        # Step 1: Validation succeeds
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"validated": True})

        # Step 2: Payment processing fails (circuit breaker opens after threshold)
        for retry in range(3):
            await circuit_breaker.record_failure(
                error=Exception("Payment service timeout")
            )

        assert circuit_breaker.state == CircuitState.OPEN

        # Mark step as failed due to circuit open
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="failed",
            error_message="Circuit breaker open - payment service unavailable",
            retry_count=3)

        # Trigger compensation
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPENSATING,
            current_step=2,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[saga.steps[1].step_id],
            compensated_steps=[])

        # Compensate step 1 (validation doesn't need compensation)
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPENSATED,
            current_step=2,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[saga.steps[1].step_id],
            compensated_steps=[],
            error_message="Circuit breaker open - payment service unavailable")

        # Verify workflow compensated correctly
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPENSATED
            assert "Circuit breaker open" in execution.error_message

    @pytest.mark.asyncio
    async def test_swarm_coordination(self) -> None:
        """Test swarm pattern for distributed agent coordination."""
        swarm_config = SwarmConfig(
            min_agents=3,
            max_agents=10,
            consensus_strategy=ConsensusStrategy.MAJORITY_VOTE,
            quorum_threshold=0.51,
            task_distribution_strategy="round_robin")

        coordinator = SwarmCoordinator(
            swarm_id=str(uuid4()),
            config=swarm_config)

        # Register agents in swarm
        agents = [
            AgentState(
                agent_id=f"agent_{i}",
                role=AgentRole.MEMBER,
                capabilities=["task_processing"],
                status="active")
            for i in range(5)
        ]

        for agent in agents:
            coordinator.register_agent(agent)

        assert len(coordinator.agents) == 5

        # Create swarm task
        task = SwarmTask(
            task_id=uuid4(),
            task_type="distributed_computation",
            input_data={"data": list(range(100))},
            required_agents=3)

        # Assign task to swarm
        assigned_agents = coordinator.assign_task(task)
        assert len(assigned_agents) >= 3

        # Simulate voting/consensus
        votes = [
            coordinator.submit_vote(
                agent_id=agent_id,
                proposal_id=str(task.task_id),
                vote=True)
            for agent_id in assigned_agents[:3]
        ]

        # Check consensus reached
        consensus_result = coordinator.check_consensus(str(task.task_id))
        assert consensus_result is True

    @pytest.mark.asyncio
    async def test_custom_pattern_with_hooks(
        self, db_session_factory
    ) -> None:
        """Test custom orchestration pattern with hooks integration."""
        # Define custom pattern
        from agentcore.orchestration.patterns.custom import PatternMetadata

        custom_pattern = PatternDefinition(
            metadata=PatternMetadata(
                name="research_pipeline",
                description="Custom research and analysis pipeline",
                version="1.0.0"),
            pattern_type=PatternType.CUSTOM,
            agents={
                "researcher": AgentRequirement(
                    role="researcher",
                    capabilities=["web_search", "data_collection"],
                    min_count=1,
                    max_count=3),
                "analyzer": AgentRequirement(
                    role="analyzer",
                    capabilities=["data_analysis", "ml_inference"],
                    min_count=1,
                    max_count=2),
                "synthesizer": AgentRequirement(
                    role="synthesizer",
                    capabilities=["report_generation"],
                    min_count=1,
                    max_count=1),
            },
            tasks=[
                TaskNode(
                    task_id="research",
                    task_name="Research Phase",
                    agent_role="researcher",
                    dependencies=[],
                    parallel=True),
                TaskNode(
                    task_id="analyze",
                    task_name="Analysis Phase",
                    agent_role="analyzer",
                    dependencies=["research"],
                    parallel=False),
                TaskNode(
                    task_id="synthesize",
                    task_name="Synthesis Phase",
                    agent_role="synthesizer",
                    dependencies=["analyze"],
                    parallel=False),
            ],
            coordination=CoordinationConfig(
                model=CoordinationModel.HYBRID,
                event_driven_triggers=["agent_status", "task_completion"],
                graph_based_tasks=["task_dependencies"],
                max_concurrent_tasks=10))

        # Register custom pattern
        pattern_registry.register(custom_pattern)
        assert pattern_registry.get(custom_pattern.pattern_id) is not None

        # Convert to saga for execution
        saga_steps = [
            SagaStep(
                name=task.task_name,
                order=idx + 1,
                action_data={
                    "task_id": task.task_id,
                    "agent_role": task.agent_role,
                    "parallel": task.parallel,
                })
            for idx, task in enumerate(custom_pattern.tasks)
        ]

        saga = SagaDefinition(
            name=custom_pattern.metadata.name,
            description=custom_pattern.metadata.description,
            steps=saga_steps,
            enable_state_persistence=True)

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="custom_pattern_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"research_topic": "AI orchestration patterns"},
            metadata={"pattern_type": "custom", "pattern_id": str(custom_pattern.pattern_id)})

        # Execute custom pattern workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[])

        # Execute research phase (parallel)
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"sources_found": 50, "data_collected": True})

        # Execute analysis phase
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="completed",
            result={"insights_generated": 20, "patterns_identified": 5})

        # Execute synthesis phase
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[2].step_id,
            status="completed",
            result={"report_generated": True, "pages": 15})

        # Complete workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=3,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[])

        # Verify custom pattern execution
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED
            assert execution.workflow_metadata["pattern_type"] == "custom"

    @pytest.mark.asyncio
    async def test_fault_tolerance_coordinator(self) -> None:
        """Test fault tolerance coordinator with multiple patterns."""
        coordinator = FaultToleranceCoordinator(
            coordinator_id=uuid4()
        )

        # Register multiple circuit breakers for different services
        breakers = {
            "database": CircuitBreaker(
                service_name="database_breaker",
                config=CircuitBreakerConfig(
                    failure_threshold=5,
                    timeout_seconds=60)),
            "api": CircuitBreaker(
                service_name="api_breaker",
                config=CircuitBreakerConfig(
                    failure_threshold=3,
                    timeout_seconds=30)),
            "cache": CircuitBreaker(
                service_name="cache_breaker",
                config=CircuitBreakerConfig(
                    failure_threshold=10,
                    timeout_seconds=15)),
        }

        for name, breaker in breakers.items():
            coordinator.register_circuit_breaker(name, breaker)

        # Simulate failures in different services
        for _ in range(3):
            await breakers["api"].record_failure(Exception("API timeout"))

        assert breakers["api"].state == CircuitState.OPEN
        assert breakers["database"].state == CircuitState.CLOSED
        assert breakers["cache"].state == CircuitState.CLOSED

        # Test coordinator's ability to route around failures
        available_services = coordinator.get_available_services()
        assert "database" in available_services
        assert "cache" in available_services
        assert "api" not in available_services

    @pytest.mark.asyncio
    async def test_hybrid_pattern_composition(
        self, db_session_factory
    ) -> None:
        """Test composition of multiple patterns in a single workflow."""
        # Create a hybrid workflow combining saga, swarm, and circuit breaker
        saga = SagaDefinition(
            name="hybrid_workflow",
            description="Hybrid workflow combining multiple patterns",
            steps=[
                # Stage 1: Single-agent initialization (saga)
                SagaStep(
                    name="initialize",
                    order=1,
                    action_data={"pattern": "saga", "task": "init"}),
                # Stage 2: Distributed processing (swarm)
                SagaStep(
                    name="swarm_processing",
                    order=2,
                    action_data={
                        "pattern": "swarm",
                        "agents": 5,
                        "task": "distributed_compute",
                    }),
                # Stage 3: External API call (circuit breaker protected)
                SagaStep(
                    name="api_integration",
                    order=3,
                    action_data={
                        "pattern": "circuit_breaker",
                        "service": "external_api",
                    },
                    compensation_data={"action": "rollback_integration"}),
                # Stage 4: Finalization (saga)
                SagaStep(
                    name="finalize",
                    order=4,
                    action_data={"pattern": "saga", "task": "finalize"}),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True)

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="hybrid_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"workflow_type": "hybrid", "complexity": "high"},
            tags=["hybrid", "multi-pattern"])

        # Execute hybrid workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[])

        # Stage 1: Initialize (saga pattern)
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"initialized": True, "pattern": "saga"})

        # Stage 2: Swarm processing
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="completed",
            result={
                "agents_used": 5,
                "tasks_completed": 100,
                "pattern": "swarm",
            })

        # Stage 3: Circuit breaker protected API call
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[2].step_id,
            status="completed",
            result={
                "api_response": {"status": "success"},
                "pattern": "circuit_breaker",
            })

        # Stage 4: Finalize
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[3].step_id,
            status="completed",
            result={"finalized": True, "pattern": "saga"})

        # Complete workflow
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPLETED,
            current_step=4,
            completed_steps=[step.step_id for step in saga.steps],
            failed_steps=[],
            compensated_steps=[])

        # Verify hybrid workflow execution
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPLETED
            assert "hybrid" in execution.tags
            assert "multi-pattern" in execution.tags

            # Verify all pattern stages completed
            for step in saga.steps:
                step_state = execution.task_states[str(step.step_id)]
                assert step_state["status"] == "completed"
                assert "pattern" in step_state.get("result", {})

    @pytest.mark.asyncio
    async def test_pattern_failure_propagation(
        self, db_session_factory
    ) -> None:
        """Test failure propagation across multiple patterns."""
        saga = SagaDefinition(
            name="failure_propagation_test",
            description="Test failure handling across patterns",
            steps=[
                SagaStep(
                    name="step1",
                    order=1,
                    action_data={"pattern": "saga"},
                    compensation_data={"action": "undo_step1"}),
                SagaStep(
                    name="step2_swarm",
                    order=2,
                    action_data={"pattern": "swarm"},
                    compensation_data={"action": "undo_step2"}),
                SagaStep(
                    name="step3_breaker",
                    order=3,
                    action_data={"pattern": "circuit_breaker"},
                    compensation_data={"action": "undo_step3"}),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD,
            enable_state_persistence=True)

        orchestrator = PersistentSagaOrchestrator(
            orchestrator_id="failure_prop_orchestrator",
            session_factory=db_session_factory)

        await orchestrator.register_saga(saga)

        execution_id = await orchestrator.create_execution(
            saga_id=saga.saga_id,
            input_data={"test": "failure_propagation"})

        # Start execution
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.RUNNING,
            current_step=1,
            completed_steps=[],
            failed_steps=[],
            compensated_steps=[])

        # Step 1 succeeds
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="completed",
            result={"success": True})

        # Step 2 fails (swarm consensus failure)
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[1].step_id,
            status="failed",
            error_message="Swarm consensus not reached")

        # Trigger compensation
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPENSATING,
            current_step=2,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[saga.steps[1].step_id],
            compensated_steps=[])

        # Compensate step 1
        await orchestrator.update_step_state(
            execution_id=execution_id,
            step_id=saga.steps[0].step_id,
            status="compensated",
            result={"undone": True})

        # Mark as compensated
        await orchestrator.update_execution_state(
            execution_id=execution_id,
            status=SagaStatus.COMPENSATED,
            current_step=2,
            completed_steps=[saga.steps[0].step_id],
            failed_steps=[saga.steps[1].step_id],
            compensated_steps=[saga.steps[0].step_id],
            error_message="Swarm consensus not reached")

        # Verify failure propagation and compensation
        async with db_session_factory() as session:
            execution = await WorkflowStateRepository.get_execution(
                session, str(execution_id)
            )

            assert execution is not None
            assert execution.status == WorkflowStatus.COMPENSATED
            assert "Swarm consensus not reached" in execution.error_message
