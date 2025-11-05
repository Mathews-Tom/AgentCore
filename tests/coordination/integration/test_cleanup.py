"""Integration tests for signal cleanup service.

Tests the background cleanup task and remove_expired_signals functionality
with realistic scenarios including periodic cleanup and background task lifecycle.

Coverage targets:
- Periodic cleanup execution
- Expired signal removal
- Stale agent state removal
- Score recomputation after cleanup
- Background task start/stop
- Cleanup statistics logging
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import CoordinationService


class TestSignalCleanup:
    """Integration tests for signal cleanup functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def test_cleanup_removes_expired_signals(self) -> None:
        """Test that cleanup removes signals that have exceeded TTL."""
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        agent_id = "cleanup-test-agent"
        base_time = datetime.now(timezone.utc)

        # Create expired signal (directly, bypass registration validation)
        expired_signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.8,
            ttl_seconds=1,  # 1 second TTL
        )
        # Override timestamp to make it expired
        expired_signal.timestamp = base_time - timedelta(seconds=5)

        # Create fresh signal
        fresh_signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.CAPACITY,
            value=0.5,
            ttl_seconds=300,
        )

        # Directly insert signals into coordination state (bypass registration)
        state = AgentCoordinationState(agent_id=agent_id)
        state.signals[SignalType.LOAD] = expired_signal
        state.signals[SignalType.CAPACITY] = fresh_signal
        self.service.coordination_states[agent_id] = state

        # Verify both signals exist before cleanup
        state_before = self.service.get_coordination_state(agent_id)
        assert state_before is not None
        assert len(state_before.signals) == 2

        # Run cleanup
        stats = self.service.remove_expired_signals()

        # Verify expired signal removed, fresh signal remains
        state_after = self.service.get_coordination_state(agent_id)
        assert state_after is not None
        assert len(state_after.signals) == 1
        assert SignalType.CAPACITY in state_after.signals
        assert SignalType.LOAD not in state_after.signals

        # Verify stats
        assert stats["signals_removed"] == 1
        assert stats["agents_removed"] == 0
        assert stats["agents_remaining"] == 1

    def test_cleanup_removes_agent_with_no_active_signals(self) -> None:
        """Test that cleanup removes agent state when all signals expired."""
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        agent_id = "cleanup-agent-removal"
        base_time = datetime.now(timezone.utc)

        # Create expired signal
        expired_signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.7,
            ttl_seconds=1,
        )
        expired_signal.timestamp = base_time - timedelta(seconds=10)

        # Directly insert into coordination state
        state = AgentCoordinationState(agent_id=agent_id)
        state.signals[SignalType.LOAD] = expired_signal
        self.service.coordination_states[agent_id] = state

        # Verify agent exists before cleanup
        assert self.service.get_coordination_state(agent_id) is not None

        # Run cleanup
        stats = self.service.remove_expired_signals()

        # Verify agent removed
        assert self.service.get_coordination_state(agent_id) is None

        # Verify stats
        assert stats["signals_removed"] == 1
        assert stats["agents_removed"] == 1
        assert stats["agents_remaining"] == 0

    def test_cleanup_recomputes_scores(self) -> None:
        """Test that cleanup recomputes routing scores for affected agents."""
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        agent_id = "score-recompute-agent"
        base_time = datetime.now(timezone.utc)

        # Create expired load signal (high load)
        expired_load = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.9,
            ttl_seconds=1,
        )
        expired_load.timestamp = base_time - timedelta(seconds=5)

        # Create fresh capacity signal
        fresh_capacity = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.CAPACITY,
            value=0.8,
            ttl_seconds=300,
        )

        # Directly insert signals into coordination state
        state = AgentCoordinationState(agent_id=agent_id)
        state.signals[SignalType.LOAD] = expired_load
        state.signals[SignalType.CAPACITY] = fresh_capacity
        self.service.coordination_states[agent_id] = state

        # Compute initial score (should be affected by high load)
        initial_score = self.service.compute_routing_score(agent_id)

        # Run cleanup (removes expired load signal)
        stats = self.service.remove_expired_signals()

        # Get state after cleanup
        state_after = self.service.get_coordination_state(agent_id)
        assert state_after is not None

        # Score should be different after cleanup (no longer penalized by high load)
        # Default load score (0.5) instead of inverted 0.9
        assert state_after.routing_score != initial_score

        # Verify cleanup stats
        assert stats["signals_removed"] == 1

    def test_cleanup_multiple_agents(self) -> None:
        """Test cleanup with multiple agents and mixed signal states."""
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        base_time = datetime.now(timezone.utc)

        # Agent 1: All signals expired (should be removed)
        agent1 = "agent-1-all-expired"
        expired_signal1 = SensitivitySignal(
            agent_id=agent1,
            signal_type=SignalType.LOAD,
            value=0.5,
            ttl_seconds=1,
        )
        expired_signal1.timestamp = base_time - timedelta(seconds=10)

        state1 = AgentCoordinationState(agent_id=agent1)
        state1.signals[SignalType.LOAD] = expired_signal1
        self.service.coordination_states[agent1] = state1

        # Agent 2: Mix of expired and fresh (should remain)
        agent2 = "agent-2-partial-expired"
        expired_signal2 = SensitivitySignal(
            agent_id=agent2,
            signal_type=SignalType.LOAD,
            value=0.6,
            ttl_seconds=1,
        )
        expired_signal2.timestamp = base_time - timedelta(seconds=10)

        fresh_signal2 = SensitivitySignal(
            agent_id=agent2,
            signal_type=SignalType.QUALITY,
            value=0.9,
            ttl_seconds=300,
        )

        state2 = AgentCoordinationState(agent_id=agent2)
        state2.signals[SignalType.LOAD] = expired_signal2
        state2.signals[SignalType.QUALITY] = fresh_signal2
        self.service.coordination_states[agent2] = state2

        # Agent 3: All signals fresh (should remain)
        agent3 = "agent-3-all-fresh"
        fresh_signal3 = SensitivitySignal(
            agent_id=agent3,
            signal_type=SignalType.CAPACITY,
            value=0.7,
            ttl_seconds=300,
        )

        state3 = AgentCoordinationState(agent_id=agent3)
        state3.signals[SignalType.CAPACITY] = fresh_signal3
        self.service.coordination_states[agent3] = state3

        # Verify initial state
        assert len(self.service.coordination_states) == 3

        # Run cleanup
        stats = self.service.remove_expired_signals()

        # Verify agent1 removed
        assert self.service.get_coordination_state(agent1) is None

        # Verify agent2 remains with only fresh signal
        state2 = self.service.get_coordination_state(agent2)
        assert state2 is not None
        assert len(state2.signals) == 1
        assert SignalType.QUALITY in state2.signals

        # Verify agent3 unchanged
        state3 = self.service.get_coordination_state(agent3)
        assert state3 is not None
        assert len(state3.signals) == 1

        # Verify stats
        assert stats["signals_removed"] == 2  # expired from agent1 and agent2
        assert stats["agents_removed"] == 1  # agent1 removed
        assert stats["agents_remaining"] == 2  # agent2 and agent3 remain

    def test_cleanup_no_expired_signals(self) -> None:
        """Test cleanup when no signals are expired."""
        agent_id = "fresh-agent"

        # Register fresh signal
        fresh_signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.5,
            ttl_seconds=300,
        )
        self.service.register_signal(fresh_signal)

        # Run cleanup
        stats = self.service.remove_expired_signals()

        # Verify nothing removed
        assert stats["signals_removed"] == 0
        assert stats["agents_removed"] == 0
        assert stats["agents_remaining"] == 1

        # Verify agent still exists
        state = self.service.get_coordination_state(agent_id)
        assert state is not None
        assert len(state.signals) == 1

    def test_cleanup_empty_state(self) -> None:
        """Test cleanup with no agents registered."""
        # Run cleanup on empty service
        stats = self.service.remove_expired_signals()

        # Verify zero activity
        assert stats["signals_removed"] == 0
        assert stats["agents_removed"] == 0
        assert stats["agents_remaining"] == 0

    @pytest.mark.asyncio
    async def test_background_task_starts_and_stops(self) -> None:
        """Test background cleanup task lifecycle."""
        # Start cleanup task
        self.service.start_cleanup_task()

        # Verify task is running
        assert self.service._cleanup_task is not None
        assert not self.service._cleanup_task.done()

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop cleanup task
        await self.service.stop_cleanup_task()

        # Verify task stopped
        assert self.service._cleanup_task.done()

    @pytest.mark.asyncio
    async def test_background_task_periodic_cleanup(self) -> None:
        """Test that background task performs periodic cleanup."""
        from agentcore.a2a_protocol.config import settings
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        original_interval = settings.COORDINATION_CLEANUP_INTERVAL
        settings.COORDINATION_CLEANUP_INTERVAL = 1  # 1 second for testing

        try:
            # Create an expired signal
            agent_id = "bg-cleanup-agent"
            base_time = datetime.now(timezone.utc)

            expired_signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.LOAD,
                value=0.8,
                ttl_seconds=1,
            )
            expired_signal.timestamp = base_time - timedelta(seconds=10)

            # Directly insert into coordination state
            state = AgentCoordinationState(agent_id=agent_id)
            state.signals[SignalType.LOAD] = expired_signal
            self.service.coordination_states[agent_id] = state

            # Verify signal exists
            assert self.service.get_coordination_state(agent_id) is not None

            # Start background task
            self.service.start_cleanup_task()

            # Wait for cleanup interval + buffer
            await asyncio.sleep(1.5)

            # Verify signal was cleaned up
            assert self.service.get_coordination_state(agent_id) is None

            # Stop task
            await self.service.stop_cleanup_task()
        finally:
            # Restore original interval
            settings.COORDINATION_CLEANUP_INTERVAL = original_interval

    @pytest.mark.asyncio
    async def test_background_task_double_start_prevented(self) -> None:
        """Test that starting cleanup task twice doesn't create duplicate tasks."""
        # Start task
        self.service.start_cleanup_task()
        first_task = self.service._cleanup_task

        # Try to start again
        self.service.start_cleanup_task()
        second_task = self.service._cleanup_task

        # Verify same task instance
        assert first_task is second_task

        # Cleanup
        await self.service.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_background_task_graceful_shutdown(self) -> None:
        """Test graceful shutdown of background cleanup task."""
        # Start task
        self.service.start_cleanup_task()

        # Verify running
        assert not self.service._cleanup_task.done()

        # Stop task
        await self.service.stop_cleanup_task()

        # Verify cleanly stopped
        assert self.service._cleanup_task.done()
        # Should not raise CancelledError
        assert not self.service._cleanup_task.cancelled()
