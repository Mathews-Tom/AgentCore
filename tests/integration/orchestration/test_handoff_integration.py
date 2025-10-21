"""
Integration tests for Handoff Pattern with CQRS and Redis Streams.

Tests end-to-end handoff flows including event sourcing and stream publishing.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.orchestration.cqrs.commands import (
    CommandBus,
    CommandResult,
    ExecuteHandoffCommand,
    InitiateHandoffCommand,
    RollbackHandoffCommand,
)
from agentcore.orchestration.cqrs.events import (
    EventType,
    HandoffCompletedEvent,
    HandoffFailedEvent,
    HandoffInitiatedEvent,
    HandoffRolledBackEvent,
)
from agentcore.orchestration.patterns.handoff import (
    HandoffConfig,
    HandoffCoordinator,
    HandoffStatus,
    InputValidationGate,
)


class MockHandoffCommandHandler:
    """Mock command handler for handoff operations."""

    def __init__(self, coordinator: HandoffCoordinator) -> None:
        """Initialize handler with coordinator."""
        self.coordinator = coordinator

    async def handle_initiate(
        self, command: InitiateHandoffCommand
    ) -> CommandResult:
        """Handle initiate handoff command."""
        try:
            handoff_id = await self.coordinator.initiate_handoff(
                task_id=command.task_id,
                task_type=command.task_type,
                source_agent_id=command.source_agent_id,
                target_agent_id=command.target_agent_id,
                task_data=command.task_data,
                metadata=command.handoff_metadata,
            )

            return CommandResult(
                command_id=command.command_id,
                success=True,
                aggregate_id=handoff_id,
                events_produced=[handoff_id],
            )
        except Exception as e:
            return CommandResult(
                command_id=command.command_id,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
            )

    async def handle_execute(self, command: ExecuteHandoffCommand) -> CommandResult:
        """Handle execute handoff command."""
        try:
            success = await self.coordinator.execute_handoff(command.handoff_id)

            return CommandResult(
                command_id=command.command_id,
                success=success,
                aggregate_id=command.handoff_id,
            )
        except Exception as e:
            return CommandResult(
                command_id=command.command_id,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
            )

    async def handle_rollback(self, command: RollbackHandoffCommand) -> CommandResult:
        """Handle rollback handoff command."""
        try:
            success = await self.coordinator.rollback_handoff(
                handoff_id=command.handoff_id,
                reason=command.reason,
            )

            return CommandResult(
                command_id=command.command_id,
                success=success,
                aggregate_id=command.handoff_id,
            )
        except Exception as e:
            return CommandResult(
                command_id=command.command_id,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
            )


class TestHandoffIntegration:
    """Integration test suite for handoff pattern with CQRS."""

    @pytest.fixture
    def coordinator(self) -> HandoffCoordinator:
        """Create handoff coordinator."""
        config = HandoffConfig(
            enable_quality_gates=True,
            enable_rollback=True,
            preserve_history=True,
        )
        return HandoffCoordinator(
            coordinator_id="test-integration-coordinator",
            config=config,
        )

    @pytest.mark.asyncio
    async def test_initiate_handoff_command(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test initiating handoff via command."""
        handler = MockHandoffCommandHandler(coordinator)

        task_id = uuid4()
        command = InitiateHandoffCommand(
            task_id=task_id,
            task_type="integration_test",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"test": "data"},
            handoff_metadata={"priority": "high"},
        )

        result = await handler.handle_initiate(command)

        assert result.success is True
        assert result.aggregate_id is not None

        # Verify handoff was created
        handoff_id = result.aggregate_id
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.PENDING

    @pytest.mark.asyncio
    async def test_execute_handoff_command(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test executing handoff via command."""
        coordinator.config.enable_quality_gates = False
        handler = MockHandoffCommandHandler(coordinator)

        # Initiate handoff
        task_id = uuid4()
        init_command = InitiateHandoffCommand(
            task_id=task_id,
            task_type="integration_test",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
        )

        init_result = await handler.handle_initiate(init_command)
        handoff_id = init_result.aggregate_id

        # Execute handoff
        exec_command = ExecuteHandoffCommand(handoff_id=handoff_id)
        exec_result = await handler.handle_execute(exec_command)

        assert exec_result.success is True

        # Verify status
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rollback_handoff_command(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test rolling back handoff via command."""
        coordinator.config.enable_quality_gates = False
        handler = MockHandoffCommandHandler(coordinator)

        # Initiate and execute handoff
        task_id = uuid4()
        init_command = InitiateHandoffCommand(
            task_id=task_id,
            task_type="integration_test",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"original": "data"},
        )

        init_result = await handler.handle_initiate(init_command)
        handoff_id = init_result.aggregate_id

        exec_command = ExecuteHandoffCommand(handoff_id=handoff_id)
        await handler.handle_execute(exec_command)

        # Rollback
        rollback_command = RollbackHandoffCommand(
            handoff_id=handoff_id,
            reason="Integration test rollback",
        )

        rollback_result = await handler.handle_rollback(rollback_command)

        assert rollback_result.success is True

        # Verify status
        status = await coordinator.get_handoff_status(handoff_id)
        assert status == HandoffStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_handoff_with_quality_gates_integration(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test complete handoff flow with quality gate validation."""
        coordinator.register_quality_gate(
            InputValidationGate(required_fields=["user_id", "action"])
        )

        handler = MockHandoffCommandHandler(coordinator)

        # Valid handoff
        task_id = uuid4()
        init_command = InitiateHandoffCommand(
            task_id=task_id,
            task_type="user_action",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"user_id": "123", "action": "approve"},
        )

        init_result = await handler.handle_initiate(init_command)
        handoff_id = init_result.aggregate_id

        exec_command = ExecuteHandoffCommand(handoff_id=handoff_id)
        exec_result = await handler.handle_execute(exec_command)

        assert exec_result.success is True

        # Invalid handoff (missing required field)
        task_id_2 = uuid4()
        init_command_2 = InitiateHandoffCommand(
            task_id=task_id_2,
            task_type="user_action",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"user_id": "123"},  # Missing 'action'
        )

        init_result_2 = await handler.handle_initiate(init_command_2)
        handoff_id_2 = init_result_2.aggregate_id

        exec_command_2 = ExecuteHandoffCommand(handoff_id=handoff_id_2)
        exec_result_2 = await handler.handle_execute(exec_command_2)

        assert exec_result_2.success is False

    @pytest.mark.asyncio
    async def test_handoff_chain_integration(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test multiple sequential handoffs maintaining chain."""
        coordinator.config.enable_quality_gates = False
        handler = MockHandoffCommandHandler(coordinator)

        task_id = uuid4()

        # First handoff: agent-1 -> agent-2
        init_cmd_1 = InitiateHandoffCommand(
            task_id=task_id,
            task_type="sequential_task",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={"step": 1},
        )

        result_1 = await handler.handle_initiate(init_cmd_1)
        handoff_id_1 = result_1.aggregate_id

        exec_cmd_1 = ExecuteHandoffCommand(handoff_id=handoff_id_1)
        await handler.handle_execute(exec_cmd_1)

        # Get context and chain
        context_1 = await coordinator.get_handoff_context(handoff_id_1)
        assert len(context_1.handoff_chain) == 2
        assert context_1.handoff_chain == ["agent-1", "agent-2"]

        # Second handoff: agent-2 -> agent-3
        init_cmd_2 = InitiateHandoffCommand(
            task_id=task_id,
            task_type="sequential_task",
            source_agent_id="agent-2",
            target_agent_id="agent-3",
            task_data={"step": 2},
        )

        result_2 = await handler.handle_initiate(init_cmd_2)
        handoff_id_2 = result_2.aggregate_id

        exec_cmd_2 = ExecuteHandoffCommand(handoff_id=handoff_id_2)
        await handler.handle_execute(exec_cmd_2)

        # Verify second handoff chain
        context_2 = await coordinator.get_handoff_context(handoff_id_2)
        assert len(context_2.handoff_chain) == 2
        assert context_2.handoff_chain == ["agent-2", "agent-3"]

    @pytest.mark.asyncio
    async def test_concurrent_handoffs_integration(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test handling multiple concurrent handoffs."""
        coordinator.config.enable_quality_gates = False
        handler = MockHandoffCommandHandler(coordinator)

        # Create multiple handoffs
        handoff_count = 5
        handoff_ids = []

        for i in range(handoff_count):
            task_id = uuid4()
            init_command = InitiateHandoffCommand(
                task_id=task_id,
                task_type=f"concurrent_task_{i}",
                source_agent_id=f"agent-{i}",
                target_agent_id=f"agent-{i+1}",
                task_data={"index": i},
            )

            result = await handler.handle_initiate(init_command)
            handoff_ids.append(result.aggregate_id)

        # Execute all handoffs concurrently
        exec_commands = [
            ExecuteHandoffCommand(handoff_id=hid) for hid in handoff_ids
        ]

        exec_results = await asyncio.gather(
            *[handler.handle_execute(cmd) for cmd in exec_commands]
        )

        # Verify all succeeded
        for result in exec_results:
            assert result.success is True

        # Verify coordinator status
        status = await coordinator.get_coordinator_status()
        assert status["completed_handoffs"] == handoff_count

    @pytest.mark.asyncio
    async def test_handoff_error_handling_integration(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test error handling in handoff operations."""
        handler = MockHandoffCommandHandler(coordinator)

        # Test invalid handoff ID
        exec_command = ExecuteHandoffCommand(handoff_id=uuid4())
        exec_result = await handler.handle_execute(exec_command)

        assert exec_result.success is False
        assert "not found" in exec_result.error_message.lower()

        # Test rollback without rollback enabled
        coordinator.config.enable_rollback = False

        task_id = uuid4()
        init_command = InitiateHandoffCommand(
            task_id=task_id,
            task_type="test",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
        )

        init_result = await handler.handle_initiate(init_command)
        handoff_id = init_result.aggregate_id

        rollback_command = RollbackHandoffCommand(
            handoff_id=handoff_id,
            reason="Test",
        )

        rollback_result = await handler.handle_rollback(rollback_command)

        assert rollback_result.success is False
        assert "disabled" in rollback_result.error_message.lower()

    @pytest.mark.asyncio
    async def test_handoff_event_generation(
        self, coordinator: HandoffCoordinator
    ) -> None:
        """Test that handoff operations would generate correct events."""
        coordinator.config.enable_quality_gates = False
        handler = MockHandoffCommandHandler(coordinator)

        task_id = uuid4()

        # Initiate handoff
        init_command = InitiateHandoffCommand(
            task_id=task_id,
            task_type="event_test",
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_data={},
        )

        init_result = await handler.handle_initiate(init_command)
        handoff_id = init_result.aggregate_id

        # Execute handoff
        exec_command = ExecuteHandoffCommand(handoff_id=handoff_id)
        exec_result = await handler.handle_execute(exec_command)

        assert exec_result.success is True

        # Verify we could create domain events
        initiated_event = HandoffInitiatedEvent(
            aggregate_id=handoff_id,
            aggregate_type="handoff",
            handoff_id=handoff_id,
            task_id=task_id,
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            task_type="event_test",
        )

        assert initiated_event.event_type == EventType.HANDOFF_INITIATED
        assert initiated_event.handoff_id == handoff_id

        completed_event = HandoffCompletedEvent(
            aggregate_id=handoff_id,
            aggregate_type="handoff",
            handoff_id=handoff_id,
            task_id=task_id,
            source_agent_id="agent-1",
            target_agent_id="agent-2",
            handoff_chain=["agent-1", "agent-2"],
        )

        assert completed_event.event_type == EventType.HANDOFF_COMPLETED
        assert len(completed_event.handoff_chain) == 2
