"""
Tests for Saga Pattern Implementation
"""

from uuid import uuid4

import pytest

from agentcore.orchestration.patterns.saga import (
    CompensationStrategy,
    SagaConfig,
    SagaDefinition,
    SagaExecution,
    SagaOrchestrator,
    SagaStatus,
    SagaStep,
    SagaStepStatus)


@pytest.fixture
def saga_config() -> SagaConfig:
    """Create test saga configuration."""
    return SagaConfig(
        enable_retry=True,
        max_retries=2,
        retry_delay_seconds=0,  # No delay for tests
        enable_checkpointing=True)


@pytest.fixture
def saga_orchestrator(saga_config: SagaConfig) -> SagaOrchestrator:
    """Create test saga orchestrator."""
    return SagaOrchestrator(
        orchestrator_id="test-orchestrator",
        config=saga_config)


class TestSagaOrchestrator:
    """Test SagaOrchestrator class."""

    @pytest.mark.asyncio
    async def test_register_saga(self, saga_orchestrator: SagaOrchestrator) -> None:
        """Test registering a saga definition."""
        saga = SagaDefinition(
            name="test-saga",
            description="Test saga",
            steps=[
                SagaStep(name="step-1", order=0),
                SagaStep(name="step-2", order=1),
            ])

        saga_orchestrator.register_saga(saga)

        assert saga.saga_id in saga_orchestrator._saga_definitions

    @pytest.mark.asyncio
    async def test_register_action_handlers(
        self, saga_orchestrator: SagaOrchestrator
    ) -> None:
        """Test registering action and compensation handlers."""

        async def action_handler(action_data, context):
            return {"result": "success"}

        async def compensation_handler(compensation_data):
            return {"compensated": True}

        saga_orchestrator.register_action_handler(
            "test-step", action_handler, compensation_handler
        )

        assert "test-step" in saga_orchestrator._action_handlers
        assert "test-step" in saga_orchestrator._compensation_handlers

    @pytest.mark.asyncio
    async def test_execute_saga_success(
        self, saga_orchestrator: SagaOrchestrator
    ) -> None:
        """Test successful saga execution."""
        # Define saga
        saga = SagaDefinition(
            name="payment-saga",
            steps=[
                SagaStep(name="reserve-inventory", order=0),
                SagaStep(name="charge-payment", order=1),
                SagaStep(name="ship-order", order=2),
            ])

        saga_orchestrator.register_saga(saga)

        # Register handlers
        async def reserve_handler(data, context):
            context["inventory_reserved"] = True
            return {"reserved": True}

        async def charge_handler(data, context):
            context["payment_charged"] = True
            return {"charged": True}

        async def ship_handler(data, context):
            context["order_shipped"] = True
            return {"shipped": True}

        saga_orchestrator.register_action_handler("reserve-inventory", reserve_handler)
        saga_orchestrator.register_action_handler("charge-payment", charge_handler)
        saga_orchestrator.register_action_handler("ship-order", ship_handler)

        # Execute saga
        execution_id = await saga_orchestrator.execute_saga(saga.saga_id)

        # Wait for completion
        await asyncio.sleep(0.2)

        # Check execution status
        execution = await saga_orchestrator.get_execution_status(execution_id)
        assert execution.status == SagaStatus.COMPLETED
        assert len(execution.completed_steps) == 3

    @pytest.mark.asyncio
    async def test_execute_saga_with_failure_and_compensation(
        self, saga_orchestrator: SagaOrchestrator
    ) -> None:
        """Test saga failure and compensation."""
        # Define saga
        saga = SagaDefinition(
            name="failing-saga",
            steps=[
                SagaStep(name="step-1", order=0),
                SagaStep(name="step-2-fail", order=1),
                SagaStep(name="step-3", order=2),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD)

        saga_orchestrator.register_saga(saga)

        # Register handlers
        async def step1_handler(data, context):
            context["step1_done"] = True
            return {"result": "step1"}

        async def step1_compensation(data):
            return {"compensated": "step1"}

        async def step2_handler(data, context):
            raise ValueError("Intentional failure")

        async def step2_compensation(data):
            return {"compensated": "step2"}

        saga_orchestrator.register_action_handler(
            "step-1", step1_handler, step1_compensation
        )
        saga_orchestrator.register_action_handler(
            "step-2-fail", step2_handler, step2_compensation
        )

        # Execute saga
        execution_id = await saga_orchestrator.execute_saga(saga.saga_id)

        # Wait for compensation
        await asyncio.sleep(0.3)

        # Check execution status
        execution = await saga_orchestrator.get_execution_status(execution_id)
        assert execution.status == SagaStatus.COMPENSATED
        assert len(execution.completed_steps) == 1  # Only step-1 completed
        assert len(execution.compensated_steps) == 1  # step-1 compensated

    @pytest.mark.asyncio
    async def test_saga_retry_logic(self, saga_orchestrator: SagaOrchestrator) -> None:
        """Test step retry on transient failure."""
        retry_count = {"count": 0}

        # Define saga
        saga = SagaDefinition(
            name="retry-saga",
            steps=[
                SagaStep(name="retry-step", order=0, max_retries=2),
            ])

        saga_orchestrator.register_saga(saga)

        # Handler that fails once then succeeds
        async def retry_handler(data, context):
            retry_count["count"] += 1
            if retry_count["count"] < 2:
                raise ValueError("Transient failure")
            return {"result": "success after retry"}

        saga_orchestrator.register_action_handler("retry-step", retry_handler)

        # Execute saga
        execution_id = await saga_orchestrator.execute_saga(saga.saga_id)

        # Wait for completion
        await asyncio.sleep(0.3)

        # Check execution status
        execution = await saga_orchestrator.get_execution_status(execution_id)
        assert execution.status == SagaStatus.COMPLETED
        assert retry_count["count"] == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_compensation_strategy_backward(
        self, saga_orchestrator: SagaOrchestrator
    ) -> None:
        """Test backward compensation strategy."""
        compensation_order = []

        saga = SagaDefinition(
            name="backward-compensation-saga",
            steps=[
                SagaStep(name="step-1", order=0),
                SagaStep(name="step-2", order=1),
                SagaStep(name="step-3-fail", order=2),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD)

        saga_orchestrator.register_saga(saga)

        # Handlers
        async def step1_handler(data, context):
            return {"result": "step1"}

        async def step1_compensation(data):
            compensation_order.append("step1")
            return {"compensated": "step1"}

        async def step2_handler(data, context):
            return {"result": "step2"}

        async def step2_compensation(data):
            compensation_order.append("step2")
            return {"compensated": "step2"}

        async def step3_handler(data, context):
            raise ValueError("Step 3 fails")

        saga_orchestrator.register_action_handler(
            "step-1", step1_handler, step1_compensation
        )
        saga_orchestrator.register_action_handler(
            "step-2", step2_handler, step2_compensation
        )
        saga_orchestrator.register_action_handler("step-3-fail", step3_handler)

        # Execute saga
        execution_id = await saga_orchestrator.execute_saga(saga.saga_id)

        # Wait for compensation
        await asyncio.sleep(0.3)

        # Check compensation order (should be reverse: step2, step1)
        assert compensation_order == ["step2", "step1"]

    @pytest.mark.asyncio
    async def test_get_orchestrator_status(
        self, saga_orchestrator: SagaOrchestrator
    ) -> None:
        """Test getting orchestrator status."""
        status = await saga_orchestrator.get_orchestrator_status()

        assert status["orchestrator_id"] == "test-orchestrator"
        assert "sagas_registered" in status
        assert "active_executions" in status
        assert "completed_executions" in status


class TestSagaModels:
    """Test Saga data models."""

    def test_saga_step_creation(self) -> None:
        """Test creating a saga step."""
        step = SagaStep(
            name="test-step",
            order=0,
            action_data={"input": "data"})

        assert step.name == "test-step"
        assert step.status == SagaStepStatus.PENDING
        assert step.retry_count == 0

    def test_saga_definition_creation(self) -> None:
        """Test creating a saga definition."""
        saga = SagaDefinition(
            name="test-saga",
            steps=[
                SagaStep(name="step-1", order=0),
                SagaStep(name="step-2", order=1),
            ],
            compensation_strategy=CompensationStrategy.BACKWARD)

        assert saga.name == "test-saga"
        assert len(saga.steps) == 2
        assert saga.compensation_strategy == CompensationStrategy.BACKWARD

    def test_saga_execution_creation(self) -> None:
        """Test creating a saga execution."""
        saga_id = uuid4()
        execution = SagaExecution(
            saga_id=saga_id,
            saga_name="test-saga")

        assert execution.saga_id == saga_id
        assert execution.status == SagaStatus.PENDING
        assert len(execution.completed_steps) == 0
        assert len(execution.compensated_steps) == 0


# Import asyncio for sleep
import asyncio
