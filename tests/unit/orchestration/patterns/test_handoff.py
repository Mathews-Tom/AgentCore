"""
Unit tests for Handoff Pattern Implementation.

Tests sequential task handoff, context preservation, quality gates, and rollback.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from agentcore.orchestration.patterns.handoff import (
    CapabilityGate,
    HandoffConfig,
    HandoffContext,
    HandoffCoordinator,
    HandoffGate,
    HandoffStatus,
    InputValidationGate,
    OutputValidationGate,
    ValidationResult,
)


class TestHandoffContext:
    """Test suite for HandoffContext model."""

    def test_create_context(self) -> None:
        """Test creating handoff context."""
        task_id = uuid4()
        context = HandoffContext(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"key": "value"},
        )

        assert context.task_id == task_id
        assert context.source_agent_id == "agent-1"
        assert context.target_agent_id == "agent-2"
        assert context.task_data["key"] == "value"
        assert len(context.handoff_chain) == 0
        assert context.completed_at is None

    def test_context_with_history(self) -> None:
        """Test context with handoff chain history."""
        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-2",
            target_agent_id="agent-3",
            task_data={},
            handoff_chain=["agent-1", "agent-2"],
        )

        assert len(context.handoff_chain) == 2
        assert context.handoff_chain[0] == "agent-1"
        assert context.handoff_chain[1] == "agent-2"

    def test_context_with_previous_state(self) -> None:
        """Test context preserves previous state for rollback."""
        task_data = {"status": "in_progress", "result": None}
        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data=task_data.copy(),
            previous_state=task_data.copy(),
        )

        assert context.previous_state is not None
        assert context.previous_state["status"] == "in_progress"


class TestInputValidationGate:
    """Test suite for InputValidationGate."""

    @pytest.mark.asyncio
    async def test_valid_input(self) -> None:
        """Test validation passes with valid input."""
        gate = InputValidationGate(required_fields=["name", "age"])

        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"name": "Alice", "age": 30},
        )

        result = await gate.validate(context)

        assert result.valid is True
        assert result.gate_name == "input_validation"

    @pytest.mark.asyncio
    async def test_missing_required_field(self) -> None:
        """Test validation fails with missing required field."""
        gate = InputValidationGate(required_fields=["name", "age"])

        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"name": "Alice"},
        )

        result = await gate.validate(context)

        assert result.valid is False
        assert "age" in result.message

    @pytest.mark.asyncio
    async def test_field_validator(self) -> None:
        """Test custom field validators."""
        gate = InputValidationGate(
            required_fields=["age"],
            field_validators={"age": lambda x: x >= 18},
        )

        # Valid age
        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"age": 25},
        )

        result = await gate.validate(context)
        assert result.valid is True

        # Invalid age
        context.task_data["age"] = 15
        result = await gate.validate(context)
        assert result.valid is False


class TestOutputValidationGate:
    """Test suite for OutputValidationGate."""

    @pytest.mark.asyncio
    async def test_valid_output(self) -> None:
        """Test validation passes with valid output."""
        gate = OutputValidationGate(output_key="result", min_completeness=0.8)

        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={
                "result": {
                    "field1": "value1",
                    "field2": "value2",
                    "field3": "value3",
                }
            },
        )

        result = await gate.validate(context)

        assert result.valid is True
        assert result.metadata["completeness"] >= 0.8

    @pytest.mark.asyncio
    async def test_missing_output(self) -> None:
        """Test validation fails with missing output."""
        gate = OutputValidationGate(output_key="result")

        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
        )

        result = await gate.validate(context)

        assert result.valid is False
        assert "Missing output field" in result.message

    @pytest.mark.asyncio
    async def test_incomplete_output(self) -> None:
        """Test validation fails with incomplete output."""
        gate = OutputValidationGate(output_key="result", min_completeness=0.8)

        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={
                "result": {
                    "field1": "value1",
                    "field2": None,
                    "field3": "",
                    "field4": None,
                }
            },
        )

        result = await gate.validate(context)

        assert result.valid is False
        assert "completeness" in result.message.lower()


class TestCapabilityGate:
    """Test suite for CapabilityGate."""

    @pytest.mark.asyncio
    async def test_valid_capabilities(self) -> None:
        """Test validation passes with required capabilities."""
        gate = CapabilityGate(required_capabilities=["capability_a", "capability_b"])

        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
            metadata={
                "target_capabilities": ["capability_a", "capability_b", "capability_c"]
            },
        )

        result = await gate.validate(context)

        assert result.valid is True

    @pytest.mark.asyncio
    async def test_missing_capabilities(self) -> None:
        """Test validation fails with missing capabilities."""
        gate = CapabilityGate(required_capabilities=["capability_a", "capability_b"])

        context = HandoffContext(
            task_id=uuid4(),
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
            metadata={"target_capabilities": ["capability_a"]},
        )

        result = await gate.validate(context)

        assert result.valid is False
        assert "capability_b" in str(result.metadata.get("missing", []))


class TestHandoffCoordinator:
    """Test suite for HandoffCoordinator."""

    @pytest.fixture
    def config(self) -> HandoffConfig:
        """Create test configuration."""
        return HandoffConfig(
            enable_quality_gates=True,
            enable_rollback=True,
            validation_timeout_seconds=30,
            handoff_timeout_seconds=60,
            preserve_history=True,
        )

    @pytest.fixture
    def coordinator(self, config: HandoffConfig) -> HandoffCoordinator:
        """Create handoff coordinator instance."""
        return HandoffCoordinator(
            coordinator_id="test-coordinator",
            config=config,
        )

    @pytest.mark.asyncio
    async def test_initiate_handoff(self, coordinator: HandoffCoordinator) -> None:
        """Test initiating a handoff."""
        task_id = uuid4()

        handoff_id = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"key": "value"},
        )

        assert isinstance(handoff_id, UUID)

        # Verify handoff status
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.PENDING

        # Verify context
        context = await coordinator.get_handoff_context(handoff_id)
        assert context.task_id == task_id
        assert context.source_agent_id == "agent-1"
        assert context.target_agent_id == "agent-2"

    @pytest.mark.asyncio
    async def test_execute_handoff_without_gates(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test executing handoff without quality gates."""
        # Disable quality gates
        coordinator.config.enable_quality_gates = False

        task_id = uuid4()
        handoff_id = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"key": "value"},
        )

        # Execute handoff
        success = await coordinator.execute_handoff(handoff_id)

        assert success is True

        # Verify status
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.COMPLETED

        # Verify handoff chain
        context = await coordinator.get_handoff_context(handoff_id)
        assert len(context.handoff_chain) == 2
        assert context.handoff_chain[0] == "agent-1"
        assert context.handoff_chain[1] == "agent-2"

    @pytest.mark.asyncio
    async def test_execute_handoff_with_gates(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test executing handoff with quality gates."""
        # Register quality gates
        coordinator.register_quality_gate(
            InputValidationGate(required_fields=["name"])
        )

        task_id = uuid4()
        handoff_id = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"name": "Alice"},
        )

        # Execute handoff
        success = await coordinator.execute_handoff(handoff_id)

        assert success is True
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_handoff_validation_failure(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test handoff fails with quality gate validation failure."""
        # Register quality gate that will fail
        coordinator.register_quality_gate(
            InputValidationGate(required_fields=["missing_field"])
        )

        task_id = uuid4()
        handoff_id = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"name": "Alice"},
        )

        # Execute handoff
        success = await coordinator.execute_handoff(handoff_id)

        assert success is False
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.FAILED

    @pytest.mark.asyncio
    async def test_rollback_handoff(self, coordinator: HandoffCoordinator) -> None:
        """Test rolling back a handoff."""
        task_id = uuid4()
        original_data = {"status": "in_progress", "result": None}

        handoff_id = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data=original_data.copy(),
        )

        # Execute handoff
        coordinator.config.enable_quality_gates = False
        await coordinator.execute_handoff(handoff_id)

        # Modify task data
        context = await coordinator.get_handoff_context(handoff_id)
        context.task_data["status"] = "completed"
        context.task_data["result"] = "success"

        # Rollback
        success = await coordinator.rollback_handoff(
            handoff_id, reason="Testing rollback"
        )

        assert success is True

        # Verify rollback
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.ROLLED_BACK

        context = await coordinator.get_handoff_context(handoff_id)
        assert context.task_data["status"] == "in_progress"
        assert context.task_data["result"] is None

    @pytest.mark.asyncio
    async def test_rollback_disabled(self, coordinator: HandoffCoordinator) -> None:
        """Test rollback fails when disabled."""
        coordinator.config.enable_rollback = False

        task_id = uuid4()
        handoff_id = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"key": "value"},
        )

        # Attempt rollback
        with pytest.raises(ValueError, match="Rollback is disabled"):
            await coordinator.rollback_handoff(handoff_id, reason="Test")

    @pytest.mark.asyncio
    async def test_get_handoff_chain(self, coordinator: HandoffCoordinator) -> None:
        """Test getting handoff chain for a task."""
        coordinator.config.enable_quality_gates = False

        task_id = uuid4()

        # First handoff
        handoff_id_1 = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
        )
        await coordinator.execute_handoff(handoff_id_1)

        # Get chain
        chain = await coordinator.get_handoff_chain(task_id)

        assert len(chain) == 2
        assert chain[0] == "agent-1"
        assert chain[1] == "agent-2"

    @pytest.mark.asyncio
    async def test_register_unregister_gate(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test registering and unregistering quality gates."""
        gate = InputValidationGate(required_fields=["test"])

        # Register
        coordinator.register_quality_gate(gate)
        status = await coordinator.get_coordinator_status()
        assert status["registered_gates"] == 1
        assert "input_validation" in status["quality_gates"]

        # Unregister
        coordinator.unregister_quality_gate("input_validation")
        status = await coordinator.get_coordinator_status()
        assert status["registered_gates"] == 0

    @pytest.mark.asyncio
    async def test_coordinator_status(self, coordinator: HandoffCoordinator) -> None:
        """Test getting coordinator status."""
        status = await coordinator.get_coordinator_status()

        assert status["coordinator_id"] == "test-coordinator"
        assert status["active_handoffs"] == 0
        assert status["completed_handoffs"] == 0
        assert status["registered_gates"] == 0
        assert "config" in status

    @pytest.mark.asyncio
    async def test_handoff_not_found(self, coordinator: HandoffCoordinator) -> None:
        """Test error handling for non-existent handoff."""
        fake_id = uuid4()

        with pytest.raises(ValueError, match="Handoff not found"):
            await coordinator.get_handoff_status(fake_id)

        with pytest.raises(ValueError, match="Handoff not found"):
            await coordinator.get_handoff_context(fake_id)

        with pytest.raises(ValueError, match="Handoff not found"):
            await coordinator.execute_handoff(fake_id)

    @pytest.mark.asyncio
    async def test_multiple_handoffs(self, coordinator: HandoffCoordinator) -> None:
        """Test handling multiple concurrent handoffs."""
        coordinator.config.enable_quality_gates = False

        task_ids = [uuid4() for _ in range(3)]
        handoff_ids = []

        # Initiate multiple handoffs
        for i, task_id in enumerate(task_ids):
            handoff_id = await coordinator.initiate_handoff(
                task_id=task_id,
                task_type="test_task",
                source_agent_id=f"agent-{i}",
                target_agent_id=f"agent-{i+1}",
                task_data={},
            )
            handoff_ids.append(handoff_id)

        # Execute all handoffs
        for handoff_id in handoff_ids:
            success = await coordinator.execute_handoff(handoff_id)
            assert success is True

        # Verify all completed
        status = await coordinator.get_coordinator_status()
        assert status["active_handoffs"] == 0
        assert status["completed_handoffs"] == 3

    @pytest.mark.asyncio
    async def test_handoff_history_disabled(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test handoff without history preservation."""
        coordinator.config.preserve_history = False
        coordinator.config.enable_quality_gates = False

        task_id = uuid4()
        handoff_id = await coordinator.initiate_handoff(
            task_id=task_id,
            task_type="test_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
        )

        await coordinator.execute_handoff(handoff_id)

        # Chain should be empty
        chain = await coordinator.get_handoff_chain(task_id)
        assert len(chain) == 0
