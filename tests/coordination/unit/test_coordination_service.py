"""Unit tests for coordination service.

Tests the CoordinationService class methods covering:
- Signal registration and validation
- Score computation (individual and composite)
- Agent selection optimization
- Signal history management
- Active signal filtering
- State management

Coverage targets:
- register_signal(): Validation, normalization, state creation
- compute_individual_scores(): Score calculation with decay
- compute_routing_score(): Weighted aggregation
- select_optimal_agent(): Multi-objective optimization
- get_active_signals(): Expiry filtering
- get_signal_history(): History retrieval with filters
- Signal TTL and temporal decay

This file complements test_overload_prediction.py (20 tests) and
test_cleanup.py (10 integration tests) for comprehensive coverage.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agentcore.a2a_protocol.models.coordination import (
    AgentCoordinationState,
    SensitivitySignal,
    SignalType,
)
from agentcore.a2a_protocol.services.coordination_service import CoordinationService


class TestSignalRegistration:
    """Test signal registration and validation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def test_register_valid_signal(self) -> None:
        """Test registering a valid signal creates coordination state."""
        signal = SensitivitySignal(
            agent_id="agent-001",
            signal_type=SignalType.LOAD,
            value=0.75,
            ttl_seconds=60,
        )

        self.service.register_signal(signal)

        # Verify state created
        state = self.service.get_coordination_state("agent-001")
        assert state is not None
        assert state.agent_id == "agent-001"
        assert SignalType.LOAD in state.signals
        assert state.signals[SignalType.LOAD] == signal

        # Verify metrics updated
        assert self.service.metrics.total_signals == 1
        assert self.service.metrics.signals_by_type[SignalType.LOAD] == 1
        assert self.service.metrics.agents_tracked == 1

    def test_register_multiple_signal_types(self) -> None:
        """Test registering multiple signal types for same agent."""
        agent_id = "agent-multi"

        load_signal = SensitivitySignal(
            agent_id=agent_id, signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        capacity_signal = SensitivitySignal(
            agent_id=agent_id, signal_type=SignalType.CAPACITY, value=0.8, ttl_seconds=60
        )
        quality_signal = SensitivitySignal(
            agent_id=agent_id, signal_type=SignalType.QUALITY, value=0.9, ttl_seconds=60
        )

        self.service.register_signal(load_signal)
        self.service.register_signal(capacity_signal)
        self.service.register_signal(quality_signal)

        state = self.service.get_coordination_state(agent_id)
        assert state is not None
        assert len(state.signals) == 3
        assert SignalType.LOAD in state.signals
        assert SignalType.CAPACITY in state.signals
        assert SignalType.QUALITY in state.signals

        assert self.service.metrics.total_signals == 3

    def test_register_signal_updates_existing(self) -> None:
        """Test registering signal of same type replaces previous."""
        agent_id = "agent-update"

        # Register first signal
        signal1 = SensitivitySignal(
            agent_id=agent_id, signal_type=SignalType.LOAD, value=0.3, ttl_seconds=60
        )
        self.service.register_signal(signal1)

        # Register second signal of same type
        signal2 = SensitivitySignal(
            agent_id=agent_id, signal_type=SignalType.LOAD, value=0.7, ttl_seconds=60
        )
        self.service.register_signal(signal2)

        state = self.service.get_coordination_state(agent_id)
        assert state is not None
        assert len(state.signals) == 1
        assert state.signals[SignalType.LOAD] == signal2
        assert state.signals[SignalType.LOAD].value == 0.7

    def test_register_expired_signal_raises_error(self) -> None:
        """Test registering expired signal raises ValueError."""
        base_time = datetime.now(timezone.utc)
        signal = SensitivitySignal(
            agent_id="agent-expired", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=1
        )
        signal.timestamp = base_time - timedelta(seconds=10)

        with pytest.raises(ValueError, match="Cannot register expired signal"):
            self.service.register_signal(signal)

    def test_register_signal_value_out_of_range_rejected(self) -> None:
        """Test signal with value outside [0.0, 1.0] is rejected."""
        # This is validated by Pydantic before reaching service
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SensitivitySignal(
                agent_id="agent", signal_type=SignalType.LOAD, value=-0.1, ttl_seconds=60
            )

        with pytest.raises(ValidationError):
            SensitivitySignal(
                agent_id="agent", signal_type=SignalType.LOAD, value=1.5, ttl_seconds=60
            )

    def test_register_signal_stores_in_history(self) -> None:
        """Test signal registration stores signal in history."""
        signal = SensitivitySignal(
            agent_id="agent-history", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )

        self.service.register_signal(signal)

        history = self.service.get_signal_history("agent-history")
        assert len(history) == 1
        assert history[0] == signal


class TestSignalHistory:
    """Test signal history management."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()
        self.agent_id = "agent-history"

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def test_get_signal_history_empty(self) -> None:
        """Test getting history for agent with no signals."""
        history = self.service.get_signal_history("unknown-agent")
        assert history == []

    def test_get_signal_history_ordered_by_timestamp(self) -> None:
        """Test signal history is ordered by timestamp (most recent first)."""
        base_time = datetime.now(timezone.utc)

        # Register signals in non-sequential order
        signal2 = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        signal2.timestamp = base_time + timedelta(seconds=20)
        self.service.register_signal(signal2)

        signal1 = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.CAPACITY, value=0.7, ttl_seconds=60
        )
        signal1.timestamp = base_time + timedelta(seconds=10)
        self.service.register_signal(signal1)

        signal3 = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.QUALITY, value=0.9, ttl_seconds=60
        )
        signal3.timestamp = base_time + timedelta(seconds=30)
        self.service.register_signal(signal3)

        history = self.service.get_signal_history(self.agent_id)
        assert len(history) == 3
        assert history[0] == signal3  # Most recent
        assert history[1] == signal2
        assert history[2] == signal1  # Oldest

    def test_get_signal_history_filter_by_type(self) -> None:
        """Test filtering signal history by signal type."""
        # Register multiple signal types
        load1 = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.LOAD, value=0.3, ttl_seconds=60
        )
        self.service.register_signal(load1)

        capacity1 = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.CAPACITY, value=0.5, ttl_seconds=60
        )
        self.service.register_signal(capacity1)

        load2 = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.LOAD, value=0.7, ttl_seconds=60
        )
        self.service.register_signal(load2)

        # Get only LOAD signals
        load_history = self.service.get_signal_history(self.agent_id, signal_type=SignalType.LOAD)
        assert len(load_history) == 2
        assert all(s.signal_type == SignalType.LOAD for s in load_history)

        # Get only CAPACITY signals
        capacity_history = self.service.get_signal_history(
            self.agent_id, signal_type=SignalType.CAPACITY
        )
        assert len(capacity_history) == 1
        assert capacity_history[0] == capacity1

    def test_get_signal_history_with_limit(self) -> None:
        """Test limiting number of signals returned from history."""
        # Register 5 signals
        for i in range(5):
            signal = SensitivitySignal(
                agent_id=self.agent_id, signal_type=SignalType.LOAD, value=0.1 * i, ttl_seconds=60
            )
            self.service.register_signal(signal)

        # Get only 3 most recent
        history = self.service.get_signal_history(self.agent_id, limit=3)
        assert len(history) == 3


class TestActiveSignals:
    """Test active signal filtering (non-expired)."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()
        self.agent_id = "agent-active"

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def test_get_active_signals_empty(self) -> None:
        """Test getting active signals for unknown agent."""
        active = self.service.get_active_signals("unknown-agent")
        assert active == {}

    def test_get_active_signals_all_fresh(self) -> None:
        """Test all fresh signals are returned as active."""
        load = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.LOAD, value=0.5, ttl_seconds=300
        )
        capacity = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.CAPACITY, value=0.7, ttl_seconds=300
        )

        self.service.register_signal(load)
        self.service.register_signal(capacity)

        active = self.service.get_active_signals(self.agent_id)
        assert len(active) == 2
        assert SignalType.LOAD in active
        assert SignalType.CAPACITY in active

    def test_get_active_signals_filters_expired(self) -> None:
        """Test expired signals are filtered out from active signals."""
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        base_time = datetime.now(timezone.utc)

        # Create expired and fresh signals
        expired_signal = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.LOAD, value=0.5, ttl_seconds=1
        )
        expired_signal.timestamp = base_time - timedelta(seconds=10)

        fresh_signal = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.CAPACITY, value=0.7, ttl_seconds=300
        )

        # Manually insert into state (bypass registration validation for expired signal)
        state = AgentCoordinationState(agent_id=self.agent_id)
        state.signals[SignalType.LOAD] = expired_signal
        state.signals[SignalType.CAPACITY] = fresh_signal
        self.service.coordination_states[self.agent_id] = state

        # Get active signals
        active = self.service.get_active_signals(self.agent_id)

        # Only fresh signal should be active
        assert len(active) == 1
        assert SignalType.CAPACITY in active
        assert SignalType.LOAD not in active


class TestScoreComputation:
    """Test score computation methods."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()
        self.agent_id = "agent-scores"

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def test_compute_individual_scores_no_signals(self) -> None:
        """Test computing scores for agent with no signals uses defaults."""
        self.service.compute_individual_scores(self.agent_id)

        state = self.service.get_coordination_state(self.agent_id)
        assert state is not None
        assert state.load_score == 0.5
        assert state.capacity_score == 0.5
        assert state.quality_score == 0.5
        assert state.cost_score == 0.5
        assert state.availability_score == 0.5

    def test_compute_individual_scores_with_load(self) -> None:
        """Test load score is inverted (1.0 - value)."""
        # High load (0.9) should result in low score (1.0 - 0.9 = 0.1)
        load = SensitivitySignal(
            agent_id=self.agent_id,
            signal_type=SignalType.LOAD,
            value=0.9,
            ttl_seconds=60,
            confidence=1.0,
        )
        self.service.register_signal(load)
        self.service.compute_individual_scores(self.agent_id)

        state = self.service.get_coordination_state(self.agent_id)
        assert state is not None
        # load_score = 1.0 - (0.9 * 1.0 * decay)
        # decay ≈ 1.0 for fresh signal
        assert 0.05 <= state.load_score <= 0.15  # Should be around 0.1

    def test_compute_individual_scores_with_capacity(self) -> None:
        """Test capacity score is direct value."""
        capacity = SensitivitySignal(
            agent_id=self.agent_id,
            signal_type=SignalType.CAPACITY,
            value=0.8,
            ttl_seconds=60,
            confidence=1.0,
        )
        self.service.register_signal(capacity)
        self.service.compute_individual_scores(self.agent_id)

        state = self.service.get_coordination_state(self.agent_id)
        assert state is not None
        # capacity_score = 0.8 * 1.0 * decay ≈ 0.8
        assert 0.75 <= state.capacity_score <= 0.85

    def test_compute_individual_scores_with_quality(self) -> None:
        """Test quality score is direct value."""
        quality = SensitivitySignal(
            agent_id=self.agent_id,
            signal_type=SignalType.QUALITY,
            value=0.95,
            ttl_seconds=60,
            confidence=1.0,
        )
        self.service.register_signal(quality)
        self.service.compute_individual_scores(self.agent_id)

        state = self.service.get_coordination_state(self.agent_id)
        assert state is not None
        assert 0.90 <= state.quality_score <= 1.0

    def test_compute_individual_scores_with_confidence(self) -> None:
        """Test scores are weighted by signal confidence."""
        # Signal with low confidence
        signal = SensitivitySignal(
            agent_id=self.agent_id,
            signal_type=SignalType.CAPACITY,
            value=0.8,
            ttl_seconds=60,
            confidence=0.5,  # Low confidence
        )
        self.service.register_signal(signal)
        self.service.compute_individual_scores(self.agent_id)

        state = self.service.get_coordination_state(self.agent_id)
        assert state is not None
        # capacity_score = 0.8 * 0.5 * decay ≈ 0.4
        assert 0.35 <= state.capacity_score <= 0.45

    def test_compute_individual_scores_with_temporal_decay(self) -> None:
        """Test scores apply temporal decay for aging signals."""
        from agentcore.a2a_protocol.models.coordination import AgentCoordinationState

        base_time = datetime.now(timezone.utc)

        # Create signal with age = 30 seconds (half of TTL)
        signal = SensitivitySignal(
            agent_id=self.agent_id,
            signal_type=SignalType.CAPACITY,
            value=0.8,
            ttl_seconds=60,
            confidence=1.0,
        )
        signal.timestamp = base_time - timedelta(seconds=30)

        # Manually insert signal
        state = AgentCoordinationState(agent_id=self.agent_id)
        state.signals[SignalType.CAPACITY] = signal
        self.service.coordination_states[self.agent_id] = state

        self.service.compute_individual_scores(self.agent_id)

        state = self.service.get_coordination_state(self.agent_id)
        assert state is not None
        # Expected decay at half-life: e^(-0.5) ≈ 0.606
        # capacity_score = 0.8 * 1.0 * 0.606 ≈ 0.485
        assert 0.45 <= state.capacity_score <= 0.52

    def test_compute_routing_score_weighted_average(self) -> None:
        """Test routing score is weighted average of individual scores."""
        from agentcore.a2a_protocol.config import settings

        # Register signals
        load = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.LOAD, value=0.2, ttl_seconds=60
        )
        capacity = SensitivitySignal(
            agent_id=self.agent_id, signal_type=SignalType.CAPACITY, value=0.8, ttl_seconds=60
        )
        self.service.register_signal(load)
        self.service.register_signal(capacity)

        routing_score = self.service.compute_routing_score(self.agent_id)

        # Verify score is in valid range
        assert 0.0 <= routing_score <= 1.0

        # Verify state updated
        state = self.service.get_coordination_state(self.agent_id)
        assert state is not None
        assert state.routing_score == routing_score

        # Verify score uses weighted average (rough check)
        # load_score ≈ 1.0 - 0.2 = 0.8, capacity_score ≈ 0.8
        # Default weights sum to 1.0
        expected_min = 0.5  # Conservative estimate
        assert routing_score >= expected_min

    def test_compute_routing_score_clamped_to_range(self) -> None:
        """Test routing score is clamped to [0.0, 1.0] range."""
        # Even with extreme values, score should be clamped
        routing_score = self.service.compute_routing_score(self.agent_id)
        assert 0.0 <= routing_score <= 1.0


class TestAgentSelection:
    """Test optimal agent selection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def test_select_optimal_agent_empty_list(self) -> None:
        """Test selection with empty candidate list returns None."""
        selected = self.service.select_optimal_agent([])
        assert selected is None

    def test_select_optimal_agent_single_candidate(self) -> None:
        """Test selection with single candidate returns that agent."""
        selected = self.service.select_optimal_agent(["agent-001"])
        assert selected == "agent-001"

    def test_select_optimal_agent_highest_score(self) -> None:
        """Test selection chooses agent with highest routing score."""
        # Agent 1: Low load (high score)
        load1 = SensitivitySignal(
            agent_id="agent-001", signal_type=SignalType.LOAD, value=0.2, ttl_seconds=60
        )
        capacity1 = SensitivitySignal(
            agent_id="agent-001", signal_type=SignalType.CAPACITY, value=0.9, ttl_seconds=60
        )
        self.service.register_signal(load1)
        self.service.register_signal(capacity1)

        # Agent 2: High load (low score)
        load2 = SensitivitySignal(
            agent_id="agent-002", signal_type=SignalType.LOAD, value=0.9, ttl_seconds=60
        )
        capacity2 = SensitivitySignal(
            agent_id="agent-002", signal_type=SignalType.CAPACITY, value=0.3, ttl_seconds=60
        )
        self.service.register_signal(load2)
        self.service.register_signal(capacity2)

        # Agent 3: Medium (default scores)
        # No signals for agent-003

        candidates = ["agent-001", "agent-002", "agent-003"]
        selected = self.service.select_optimal_agent(candidates)

        # Agent 1 should be selected (low load + high capacity)
        assert selected == "agent-001"

    def test_select_optimal_agent_updates_metrics(self) -> None:
        """Test selection updates service metrics."""
        initial_selections = self.service.metrics.total_selections

        selected = self.service.select_optimal_agent(["agent-001", "agent-002"])

        assert selected is not None
        assert self.service.metrics.total_selections == initial_selections + 1
        assert self.service.metrics.coordination_score_avg >= 0.0

    def test_select_optimal_agent_with_no_signals(self) -> None:
        """Test selection works even when agents have no signals (use defaults)."""
        candidates = ["agent-no-signals-1", "agent-no-signals-2", "agent-no-signals-3"]
        selected = self.service.select_optimal_agent(candidates)

        # Should still select an agent (all have default 0.5 scores)
        assert selected in candidates

    def test_select_optimal_agent_multiple_candidates(self) -> None:
        """Test selection with multiple candidates chooses optimal."""
        # Create 5 agents with varying loads
        for i in range(5):
            load = SensitivitySignal(
                agent_id=f"agent-{i:03d}",
                signal_type=SignalType.LOAD,
                value=0.1 * i,  # 0.0, 0.1, 0.2, 0.3, 0.4
                ttl_seconds=60,
            )
            self.service.register_signal(load)

        candidates = [f"agent-{i:03d}" for i in range(5)]
        selected = self.service.select_optimal_agent(candidates)

        # Agent with lowest load (agent-000) should be selected
        assert selected == "agent-000"


class TestStateManagement:
    """Test coordination state management."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def test_get_coordination_state_unknown_agent(self) -> None:
        """Test getting state for unknown agent returns None."""
        state = self.service.get_coordination_state("unknown-agent")
        assert state is None

    def test_get_coordination_state_after_registration(self) -> None:
        """Test state exists after signal registration."""
        signal = SensitivitySignal(
            agent_id="agent-state", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        self.service.register_signal(signal)

        state = self.service.get_coordination_state("agent-state")
        assert state is not None
        assert state.agent_id == "agent-state"

    def test_clear_state_removes_all_data(self) -> None:
        """Test clear_state removes all coordination data."""
        # Register signals for multiple agents
        for i in range(3):
            signal = SensitivitySignal(
                agent_id=f"agent-{i}",
                signal_type=SignalType.LOAD,
                value=0.5,
                ttl_seconds=60,
            )
            self.service.register_signal(signal)

        assert len(self.service.coordination_states) == 3

        # Clear state
        self.service.clear_state()

        # Verify all data cleared
        assert len(self.service.coordination_states) == 0
        assert len(self.service.signal_history) == 0

    def test_state_last_updated_timestamp(self) -> None:
        """Test state.last_updated is updated on signal registration."""
        signal = SensitivitySignal(
            agent_id="agent-time", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )

        before = datetime.now(timezone.utc)
        self.service.register_signal(signal)
        after = datetime.now(timezone.utc)

        state = self.service.get_coordination_state("agent-time")
        assert state is not None
        assert before <= state.last_updated <= after
