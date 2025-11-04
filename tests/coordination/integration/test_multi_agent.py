"""Multi-agent coordination integration tests.

End-to-end tests with multiple agents demonstrating the Ripple Effect Protocol
in realistic scenarios with concurrent operations and large-scale coordination.

Coverage:
- Multi-agent signal registration
- Concurrent signal updates
- Large-scale coordination (10+ agents)
- Signal expiry workflows
- Score computation under load
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import CoordinationService


class TestMultiAgentCoordination:
    """Integration tests for multi-agent coordination."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.service.clear_state()

    def test_multi_agent_signal_registration(self) -> None:
        """Test registering signals for multiple agents."""
        # Register signals for 5 agents
        for i in range(5):
            agent_id = f"agent-{i:03d}"

            load_signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.LOAD,
                value=0.1 * i,  # Increasing load
                ttl_seconds=60,
            )
            capacity_signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.CAPACITY,
                value=1.0 - (0.1 * i),  # Decreasing capacity
                ttl_seconds=60,
            )

            self.service.register_signal(load_signal)
            self.service.register_signal(capacity_signal)

        # Verify all agents tracked
        assert self.service.metrics.agents_tracked == 5
        assert self.service.metrics.total_signals == 10

        # Verify routing scores calculated for all agents
        for i in range(5):
            agent_id = f"agent-{i:03d}"
            state = self.service.get_coordination_state(agent_id)
            assert state is not None
            assert 0.0 <= state.routing_score <= 1.0

    def test_coordination_with_10_plus_agents(self) -> None:
        """Test coordination system with 10+ agents."""
        num_agents = 15

        # Register varied signals for 15 agents
        for i in range(num_agents):
            agent_id = f"agent-scale-{i:03d}"

            # Vary signal types and values for diversity
            if i % 3 == 0:
                # Low load, high capacity
                self.service.register_signal(
                    SensitivitySignal(
                        agent_id=agent_id,
                        signal_type=SignalType.LOAD,
                        value=0.2,
                        ttl_seconds=60,
                    )
                )
                self.service.register_signal(
                    SensitivitySignal(
                        agent_id=agent_id,
                        signal_type=SignalType.CAPACITY,
                        value=0.9,
                        ttl_seconds=60,
                    )
                )
            elif i % 3 == 1:
                # Medium load and capacity
                self.service.register_signal(
                    SensitivitySignal(
                        agent_id=agent_id,
                        signal_type=SignalType.LOAD,
                        value=0.5,
                        ttl_seconds=60,
                    )
                )
                self.service.register_signal(
                    SensitivitySignal(
                        agent_id=agent_id,
                        signal_type=SignalType.CAPACITY,
                        value=0.5,
                        ttl_seconds=60,
                    )
                )
            else:
                # High load, low capacity
                self.service.register_signal(
                    SensitivitySignal(
                        agent_id=agent_id,
                        signal_type=SignalType.LOAD,
                        value=0.9,
                        ttl_seconds=60,
                    )
                )
                self.service.register_signal(
                    SensitivitySignal(
                        agent_id=agent_id,
                        signal_type=SignalType.CAPACITY,
                        value=0.2,
                        ttl_seconds=60,
                    )
                )

        # Verify metrics
        assert self.service.metrics.agents_tracked == num_agents
        assert self.service.metrics.total_signals == num_agents * 2

        # Select optimal agent multiple times
        candidates = [f"agent-scale-{i:03d}" for i in range(num_agents)]

        for _ in range(5):
            selected = self.service.select_optimal_agent(candidates)
            assert selected in candidates

            # Selected agent should have good routing score
            state = self.service.get_coordination_state(selected)
            assert state is not None
            # Agents with low load % 3 == 0 should be preferred
            assert selected.endswith(("000", "003", "006", "009", "012"))

    @pytest.mark.asyncio
    async def test_concurrent_signal_registration(self) -> None:
        """Test concurrent signal registration from multiple agents."""
        num_agents = 20

        async def register_agent_signals(agent_idx: int) -> None:
            """Register signals for a single agent."""
            agent_id = f"agent-concurrent-{agent_idx:03d}"

            # Simulate concurrent registration
            await asyncio.sleep(0.01 * agent_idx)  # Slight stagger

            load_signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.LOAD,
                value=0.3 + (agent_idx * 0.02),
                ttl_seconds=60,
            )
            self.service.register_signal(load_signal)

            capacity_signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=SignalType.CAPACITY,
                value=0.8 - (agent_idx * 0.02),
                ttl_seconds=60,
            )
            self.service.register_signal(capacity_signal)

        # Register signals concurrently
        tasks = [register_agent_signals(i) for i in range(num_agents)]
        await asyncio.gather(*tasks)

        # Verify all signals registered correctly
        assert self.service.metrics.agents_tracked == num_agents
        assert self.service.metrics.total_signals == num_agents * 2

        # Verify state integrity for all agents
        for i in range(num_agents):
            agent_id = f"agent-concurrent-{i:03d}"
            state = self.service.get_coordination_state(agent_id)
            assert state is not None
            assert len(state.signals) == 2
            assert SignalType.LOAD in state.signals
            assert SignalType.CAPACITY in state.signals

    def test_signal_expiry_workflow(self) -> None:
        """Test signal expiry and cleanup workflow."""
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        base_time = datetime.now(timezone.utc)
        agent_id = "agent-expiry-workflow"

        # Create mix of expired and fresh signals
        expired_signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.9,
            ttl_seconds=1,
        )
        expired_signal.timestamp = base_time - timedelta(seconds=10)

        fresh_signal1 = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.CAPACITY,
            value=0.7,
            ttl_seconds=300,
        )

        fresh_signal2 = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.QUALITY,
            value=0.9,
            ttl_seconds=300,
        )

        # Directly insert signals (bypass registration for expired)
        state = AgentCoordinationState(agent_id=agent_id)
        state.signals[SignalType.LOAD] = expired_signal
        state.signals[SignalType.CAPACITY] = fresh_signal1
        state.signals[SignalType.QUALITY] = fresh_signal2
        self.service.coordination_states[agent_id] = state

        # Verify initial state
        assert len(state.signals) == 3

        # Get active signals (should filter expired)
        active = self.service.get_active_signals(agent_id)
        assert len(active) == 2
        assert SignalType.LOAD not in active
        assert SignalType.CAPACITY in active
        assert SignalType.QUALITY in active

        # Run cleanup
        stats = self.service.remove_expired_signals()
        assert stats["signals_removed"] == 1
        assert stats["agents_removed"] == 0

        # Verify state after cleanup
        state_after = self.service.get_coordination_state(agent_id)
        assert state_after is not None
        assert len(state_after.signals) == 2

    def test_signal_update_replaces_previous(self) -> None:
        """Test that updating a signal replaces the previous value."""
        agent_id = "agent-update"

        # Register initial signal
        signal1 = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.3,
            ttl_seconds=60,
        )
        self.service.register_signal(signal1)

        state1 = self.service.get_coordination_state(agent_id)
        assert state1 is not None
        assert state1.signals[SignalType.LOAD].value == 0.3

        # Update signal with new value
        signal2 = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=0.8,
            ttl_seconds=60,
        )
        self.service.register_signal(signal2)

        state2 = self.service.get_coordination_state(agent_id)
        assert state2 is not None
        assert state2.signals[SignalType.LOAD].value == 0.8
        assert len(state2.signals) == 1  # Only one LOAD signal

        # Metrics track total registrations (2 registrations occurred)
        assert self.service.metrics.total_signals == 2
        # But signals_by_type counts current signal instances
        assert self.service.metrics.signals_by_type[SignalType.LOAD] == 2

    def test_multi_signal_type_coordination(self) -> None:
        """Test coordination with all signal types."""
        agent_id = "agent-all-signals"

        # Register all signal types
        signal_types_values = [
            (SignalType.LOAD, 0.4),
            (SignalType.CAPACITY, 0.8),
            (SignalType.QUALITY, 0.95),
            (SignalType.COST, 0.7),
            (SignalType.LATENCY, 0.3),
            (SignalType.AVAILABILITY, 0.99),
        ]

        for signal_type, value in signal_types_values:
            signal = SensitivitySignal(
                agent_id=agent_id,
                signal_type=signal_type,
                value=value,
                ttl_seconds=60,
            )
            self.service.register_signal(signal)

        # Verify all signals registered
        state = self.service.get_coordination_state(agent_id)
        assert state is not None
        assert len(state.signals) == 6

        # Verify individual scores computed
        assert 0.0 <= state.load_score <= 1.0
        assert 0.0 <= state.capacity_score <= 1.0
        assert 0.0 <= state.quality_score <= 1.0
        assert 0.0 <= state.cost_score <= 1.0
        assert 0.0 <= state.availability_score <= 1.0

        # Verify routing score is composite
        assert 0.0 <= state.routing_score <= 1.0

    def test_agent_selection_consistency(self) -> None:
        """Test that agent selection is consistent with same signals."""
        # Register signals for 3 agents with clear ordering
        agents = [
            ("agent-best", 0.1, 0.95),  # Best: low load, high capacity
            ("agent-medium", 0.5, 0.6),  # Medium
            ("agent-worst", 0.9, 0.2),  # Worst: high load, low capacity
        ]

        for agent_id, load, capacity in agents:
            self.service.register_signal(
                SensitivitySignal(
                    agent_id=agent_id,
                    signal_type=SignalType.LOAD,
                    value=load,
                    ttl_seconds=60,
                )
            )
            self.service.register_signal(
                SensitivitySignal(
                    agent_id=agent_id,
                    signal_type=SignalType.CAPACITY,
                    value=capacity,
                    ttl_seconds=60,
                )
            )

        # Select agent multiple times
        candidates = [agent_id for agent_id, _, _ in agents]
        selections = []

        for _ in range(10):
            selected = self.service.select_optimal_agent(candidates)
            selections.append(selected)

        # All selections should be consistent (same agent)
        assert len(set(selections)) == 1
        # Should always select the best agent
        assert selections[0] == "agent-best"

    def test_metrics_accuracy_multi_agent(self) -> None:
        """Test metrics accuracy with multiple agents and operations."""
        # Register signals for multiple agents
        for i in range(5):
            agent_id = f"agent-metrics-{i}"
            self.service.register_signal(
                SensitivitySignal(
                    agent_id=agent_id,
                    signal_type=SignalType.LOAD,
                    value=0.5,
                    ttl_seconds=60,
                )
            )

        # Perform selections
        candidates = [f"agent-metrics-{i}" for i in range(5)]
        for _ in range(10):
            self.service.select_optimal_agent(candidates)

        # Verify metrics
        metrics = self.service.metrics
        assert metrics.agents_tracked == 5
        assert metrics.total_signals == 5
        assert metrics.total_selections == 10
        assert metrics.signals_by_type[SignalType.LOAD] == 5
