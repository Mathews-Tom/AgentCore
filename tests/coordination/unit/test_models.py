"""Unit tests for coordination models.

Tests Pydantic validation, model methods, and business logic for:
- SensitivitySignal: Validation, expiry, decay factor
- AgentCoordinationState: State management
- CoordinationMetrics: Metrics tracking
- SignalType enum

Coverage targets:
- Field validation (value ranges, required fields)
- Signal expiry detection
- Temporal decay calculation
- Default value handling
- Edge cases and error conditions
"""

import math
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from agentcore.a2a_protocol.models.coordination import (
    AgentCoordinationState,
    CoordinationMetrics,
    SensitivitySignal,
    SignalType,
)


class TestSignalType:
    """Test SignalType enum."""

    def test_signal_types_defined(self) -> None:
        """Test all signal types are defined."""
        assert SignalType.LOAD == "LOAD"
        assert SignalType.CAPACITY == "CAPACITY"
        assert SignalType.QUALITY == "QUALITY"
        assert SignalType.COST == "COST"
        assert SignalType.LATENCY == "LATENCY"
        assert SignalType.AVAILABILITY == "AVAILABILITY"

    def test_signal_type_iteration(self) -> None:
        """Test iterating over signal types."""
        signal_types = list(SignalType)
        assert len(signal_types) == 6
        assert SignalType.LOAD in signal_types


class TestSensitivitySignal:
    """Test SensitivitySignal model."""

    def test_create_valid_signal(self) -> None:
        """Test creating a valid signal with required fields."""
        signal = SensitivitySignal(
            agent_id="agent-001",
            signal_type=SignalType.LOAD,
            value=0.75,
            ttl_seconds=60,
        )

        assert signal.agent_id == "agent-001"
        assert signal.signal_type == SignalType.LOAD
        assert signal.value == 0.75
        assert signal.ttl_seconds == 60
        assert signal.confidence == 1.0  # default
        assert signal.trace_id is None  # default
        assert isinstance(signal.signal_id, UUID)
        assert isinstance(signal.timestamp, datetime)

    def test_signal_with_optional_fields(self) -> None:
        """Test signal with all fields including optional."""
        signal = SensitivitySignal(
            agent_id="agent-002",
            signal_type=SignalType.CAPACITY,
            value=0.5,
            ttl_seconds=120,
            confidence=0.8,
            trace_id="trace-123",
        )

        assert signal.confidence == 0.8
        assert signal.trace_id == "trace-123"

    def test_value_validation_within_range(self) -> None:
        """Test value must be in [0.0, 1.0] range."""
        # Valid values
        SensitivitySignal(agent_id="a", signal_type=SignalType.LOAD, value=0.0, ttl_seconds=60)
        SensitivitySignal(agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60)
        SensitivitySignal(agent_id="a", signal_type=SignalType.LOAD, value=1.0, ttl_seconds=60)

    def test_value_validation_below_range(self) -> None:
        """Test value below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SensitivitySignal(
                agent_id="a", signal_type=SignalType.LOAD, value=-0.1, ttl_seconds=60
            )

    def test_value_validation_above_range(self) -> None:
        """Test value above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SensitivitySignal(
                agent_id="a", signal_type=SignalType.LOAD, value=1.5, ttl_seconds=60
            )

    def test_ttl_validation_positive(self) -> None:
        """Test TTL must be positive."""
        # Valid TTL
        SensitivitySignal(agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=1)
        SensitivitySignal(agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=3600)

    def test_ttl_validation_zero(self) -> None:
        """Test TTL of 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SensitivitySignal(agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=0)

    def test_ttl_validation_negative(self) -> None:
        """Test negative TTL raises ValidationError."""
        with pytest.raises(ValidationError):
            SensitivitySignal(
                agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=-10
            )

    def test_confidence_validation_within_range(self) -> None:
        """Test confidence must be in [0.0, 1.0] range."""
        SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60, confidence=0.0
        )
        SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60, confidence=1.0
        )

    def test_confidence_validation_below_range(self) -> None:
        """Test confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SensitivitySignal(
                agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60, confidence=-0.1
            )

    def test_confidence_validation_above_range(self) -> None:
        """Test confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            SensitivitySignal(
                agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60, confidence=1.5
            )

    def test_is_expired_with_current_time(self) -> None:
        """Test is_expired with explicit current_time."""
        base_time = datetime.now(timezone.utc)
        signal = SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        signal.timestamp = base_time

        # Not expired (within TTL)
        assert not signal.is_expired(base_time + timedelta(seconds=30))

        # At TTL boundary (still valid)
        assert not signal.is_expired(base_time + timedelta(seconds=60))

        # Expired (exceeded TTL)
        assert signal.is_expired(base_time + timedelta(seconds=61))

    def test_is_expired_default_current_time(self) -> None:
        """Test is_expired with default current time (now)."""
        # Create fresh signal
        signal = SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=3600
        )
        assert not signal.is_expired()

        # Create old signal
        old_signal = SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=1
        )
        old_signal.timestamp = datetime.now(timezone.utc) - timedelta(seconds=10)
        assert old_signal.is_expired()

    def test_decay_factor_fresh_signal(self) -> None:
        """Test decay factor for freshly created signal."""
        base_time = datetime.now(timezone.utc)
        signal = SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        signal.timestamp = base_time

        # Fresh signal (age = 0) should have decay = 1.0
        decay = signal.decay_factor(base_time)
        assert decay == 1.0

    def test_decay_factor_half_life(self) -> None:
        """Test decay factor at various ages."""
        base_time = datetime.now(timezone.utc)
        signal = SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        signal.timestamp = base_time

        # Age = 30 seconds (half of TTL)
        decay_30 = signal.decay_factor(base_time + timedelta(seconds=30))
        expected_30 = math.exp(-30 / 60)  # e^(-0.5) ≈ 0.606
        assert abs(decay_30 - expected_30) < 0.001

        # Age = TTL (at expiry boundary)
        decay_60 = signal.decay_factor(base_time + timedelta(seconds=60))
        expected_60 = math.exp(-1)  # e^(-1) ≈ 0.368
        assert abs(decay_60 - expected_60) < 0.001

    def test_decay_factor_expired_signal(self) -> None:
        """Test decay factor for expired signal returns 0.0."""
        base_time = datetime.now(timezone.utc)
        signal = SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=60
        )
        signal.timestamp = base_time

        # Expired signal should have decay = 0.0
        decay = signal.decay_factor(base_time + timedelta(seconds=61))
        assert decay == 0.0

    def test_decay_factor_default_current_time(self) -> None:
        """Test decay factor with default current time."""
        signal = SensitivitySignal(
            agent_id="a", signal_type=SignalType.LOAD, value=0.5, ttl_seconds=3600
        )
        decay = signal.decay_factor()
        assert 0.95 < decay <= 1.0  # Should be very close to 1.0 for fresh signal


class TestAgentCoordinationState:
    """Test AgentCoordinationState model."""

    def test_create_default_state(self) -> None:
        """Test creating state with default values."""
        state = AgentCoordinationState(agent_id="agent-001")

        assert state.agent_id == "agent-001"
        assert state.signals == {}
        assert state.load_score == 0.5
        assert state.capacity_score == 0.5
        assert state.quality_score == 0.5
        assert state.cost_score == 0.5
        assert state.availability_score == 0.5
        assert state.routing_score == 0.5
        assert isinstance(state.last_updated, datetime)

    def test_state_with_custom_scores(self) -> None:
        """Test creating state with custom score values."""
        state = AgentCoordinationState(
            agent_id="agent-002",
            load_score=0.8,
            capacity_score=0.7,
            quality_score=0.9,
            cost_score=0.6,
            availability_score=1.0,
            routing_score=0.85,
        )

        assert state.load_score == 0.8
        assert state.capacity_score == 0.7
        assert state.quality_score == 0.9
        assert state.cost_score == 0.6
        assert state.availability_score == 1.0
        assert state.routing_score == 0.85

    def test_state_with_signals(self) -> None:
        """Test state with signals dictionary."""
        load_signal = SensitivitySignal(
            agent_id="agent-003", signal_type=SignalType.LOAD, value=0.6, ttl_seconds=60
        )
        capacity_signal = SensitivitySignal(
            agent_id="agent-003", signal_type=SignalType.CAPACITY, value=0.8, ttl_seconds=60
        )

        state = AgentCoordinationState(
            agent_id="agent-003",
            signals={
                SignalType.LOAD: load_signal,
                SignalType.CAPACITY: capacity_signal,
            },
        )

        assert len(state.signals) == 2
        assert state.signals[SignalType.LOAD] == load_signal
        assert state.signals[SignalType.CAPACITY] == capacity_signal

    def test_score_validation_within_range(self) -> None:
        """Test scores must be in [0.0, 1.0] range."""
        AgentCoordinationState(agent_id="a", load_score=0.0)
        AgentCoordinationState(agent_id="a", load_score=1.0)
        AgentCoordinationState(agent_id="a", routing_score=0.5)

    def test_score_validation_below_range(self) -> None:
        """Test score below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentCoordinationState(agent_id="a", load_score=-0.1)

    def test_score_validation_above_range(self) -> None:
        """Test score above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentCoordinationState(agent_id="a", routing_score=1.5)


class TestCoordinationMetrics:
    """Test CoordinationMetrics model."""

    def test_create_default_metrics(self) -> None:
        """Test creating metrics with default values."""
        metrics = CoordinationMetrics()

        assert metrics.total_signals == 0
        assert metrics.signals_by_type == {}
        assert metrics.total_selections == 0
        assert metrics.average_selection_time_ms == 0.0
        assert metrics.average_signal_age_seconds == 0.0
        assert metrics.coordination_score_avg == 0.5
        assert metrics.expired_signals_cleaned == 0
        assert metrics.agents_tracked == 0

    def test_metrics_with_custom_values(self) -> None:
        """Test metrics with custom values."""
        metrics = CoordinationMetrics(
            total_signals=100,
            signals_by_type={SignalType.LOAD: 50, SignalType.CAPACITY: 50},
            total_selections=25,
            average_selection_time_ms=5.5,
            average_signal_age_seconds=30.0,
            coordination_score_avg=0.75,
            expired_signals_cleaned=10,
            agents_tracked=5,
        )

        assert metrics.total_signals == 100
        assert metrics.signals_by_type[SignalType.LOAD] == 50
        assert metrics.total_selections == 25
        assert metrics.average_selection_time_ms == 5.5
        assert metrics.average_signal_age_seconds == 30.0
        assert metrics.coordination_score_avg == 0.75
        assert metrics.expired_signals_cleaned == 10
        assert metrics.agents_tracked == 5

    def test_metrics_validation_non_negative(self) -> None:
        """Test metrics counters must be non-negative."""
        # Valid
        CoordinationMetrics(total_signals=0)
        CoordinationMetrics(total_selections=100)

        # Invalid
        with pytest.raises(ValidationError):
            CoordinationMetrics(total_signals=-1)

        with pytest.raises(ValidationError):
            CoordinationMetrics(agents_tracked=-5)

    def test_coordination_score_validation(self) -> None:
        """Test coordination_score_avg must be in [0.0, 1.0] range."""
        CoordinationMetrics(coordination_score_avg=0.0)
        CoordinationMetrics(coordination_score_avg=1.0)

        with pytest.raises(ValidationError):
            CoordinationMetrics(coordination_score_avg=-0.1)

        with pytest.raises(ValidationError):
            CoordinationMetrics(coordination_score_avg=1.5)
