"""Unit tests for overload prediction functionality.

Tests the predict_overload method which uses linear regression to forecast
agent load and predict overload conditions.

Coverage targets:
- Insufficient data handling
- Linear regression calculation
- Flat load (no trend) handling
- Positive trend detection
- Negative trend handling
- Threshold detection
- Warning logging for predicted overloads
"""

from datetime import datetime, timedelta, timezone

import pytest

from agentcore.a2a_protocol.models.coordination import SensitivitySignal, SignalType
from agentcore.a2a_protocol.services.coordination_service import CoordinationService


class TestOverloadPrediction:
    """Test suite for predict_overload method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.service = CoordinationService()
        self.agent_id = "test-agent-001"
        self.base_time = datetime.now(timezone.utc)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.service.clear_state()

    def _register_load_signal(
        self, value: float, time_offset_seconds: int = 0
    ) -> None:
        """Helper to register a load signal with time offset.

        Args:
            value: Load value (0.0-1.0)
            time_offset_seconds: Seconds to offset from base_time
        """
        signal = SensitivitySignal(
            agent_id=self.agent_id,
            signal_type=SignalType.LOAD,
            value=value,
            ttl_seconds=300,  # 5 minutes
        )
        # Override timestamp for testing
        signal.timestamp = self.base_time + timedelta(seconds=time_offset_seconds)
        self.service.register_signal(signal)

    def test_insufficient_data_returns_no_overload(self) -> None:
        """Test prediction with insufficient historical data (<3 signals)."""
        # Register only 2 signals
        self._register_load_signal(0.5, time_offset_seconds=0)
        self._register_load_signal(0.6, time_offset_seconds=10)

        will_overload, probability = self.service.predict_overload(self.agent_id)

        assert will_overload is False
        assert probability == 0.0

    def test_no_data_returns_no_overload(self) -> None:
        """Test prediction with no historical data."""
        will_overload, probability = self.service.predict_overload(self.agent_id)

        assert will_overload is False
        assert probability == 0.0

    def test_flat_load_below_threshold(self) -> None:
        """Test prediction with flat load (no trend) below threshold."""
        # Register signals with constant load
        for i in range(5):
            self._register_load_signal(0.5, time_offset_seconds=i * 10)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        assert will_overload is False
        assert probability == 0.5  # Current load (flat, no trend)

    def test_flat_load_above_threshold(self) -> None:
        """Test prediction with flat load (no trend) above threshold."""
        # Register signals with high constant load
        for i in range(5):
            self._register_load_signal(0.9, time_offset_seconds=i * 10)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        assert will_overload is True
        assert probability == 0.9  # Current load is above threshold

    def test_increasing_load_predicts_overload(self) -> None:
        """Test prediction with increasing load trend (predicts overload)."""
        # Register increasing load signals: 0.3 → 0.5 → 0.7 → 0.75
        loads = [0.3, 0.5, 0.7, 0.75]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 15)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        # With increasing trend, should predict overload
        assert will_overload is True
        assert 0.8 < probability <= 1.0  # Predicted load exceeds threshold

    def test_decreasing_load_no_overload(self) -> None:
        """Test prediction with decreasing load trend (no overload)."""
        # Register decreasing load signals: 0.9 → 0.7 → 0.5 → 0.4
        loads = [0.9, 0.7, 0.5, 0.4]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 15)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        # With decreasing trend, should not predict overload
        assert will_overload is False
        # Probability is the predicted load (which should be low with decreasing trend)
        assert probability < 0.5  # Predicted load should be below current

    def test_slow_increasing_load_no_overload(self) -> None:
        """Test prediction with slow increasing load (stays below threshold)."""
        # Register slowly increasing load: 0.1 → 0.2 → 0.3 → 0.35
        loads = [0.1, 0.2, 0.3, 0.35]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 20)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        # Slow trend should not reach threshold in 60 seconds
        assert will_overload is False

    def test_steep_increasing_load_high_probability(self) -> None:
        """Test prediction with steep increasing load (high overload probability)."""
        # Register steeply increasing load: 0.2 → 0.4 → 0.6 → 0.75
        loads = [0.2, 0.4, 0.6, 0.75]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 10)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=30, threshold=0.8
        )

        # Steep trend should predict high probability overload
        assert will_overload is True
        assert probability >= 0.85  # High predicted load

    def test_custom_threshold(self) -> None:
        """Test prediction with custom overload threshold."""
        # Register slow increasing load that won't reach very high levels
        loads = [0.2, 0.25, 0.3, 0.32]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 20)

        # Lower threshold should predict overload
        will_overload_low, prob_low = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.4
        )

        # Higher threshold should not predict overload (slow trend won't reach 0.9)
        will_overload_high, prob_high = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.9
        )

        assert will_overload_low is True  # Predicted load exceeds 0.4
        assert will_overload_high is False  # Predicted load well below 0.9
        # With a slow trend, the predicted load should be relatively low
        assert prob_high < 0.7  # Sanity check on predicted load

    def test_custom_forecast_window(self) -> None:
        """Test prediction with different forecast windows."""
        # Register increasing load
        loads = [0.3, 0.4, 0.5, 0.6]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 20)

        # Short forecast window
        will_overload_short, prob_short = self.service.predict_overload(
            self.agent_id, forecast_seconds=30, threshold=0.8
        )

        # Long forecast window
        will_overload_long, prob_long = self.service.predict_overload(
            self.agent_id, forecast_seconds=120, threshold=0.8
        )

        # Longer forecast should have higher predicted load
        assert prob_long > prob_short

    def test_clamping_to_valid_range(self) -> None:
        """Test that predicted load is clamped to 0.0-1.0 range."""
        # Register extremely steep increasing load
        loads = [0.5, 0.7, 0.9, 0.95]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 5)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        # Probability should be clamped to 1.0 even if extrapolation exceeds it
        assert 0.0 <= probability <= 1.0

    def test_uses_last_10_signals_only(self) -> None:
        """Test that prediction uses only the 10 most recent signals."""
        # Register 15 signals (only last 10 should be used)
        for i in range(15):
            # First 5 signals are high load, last 10 are increasing from low
            load = 0.9 if i < 5 else 0.1 + (i - 5) * 0.08
            self._register_load_signal(load, time_offset_seconds=i * 10)

        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        # Prediction should be based on last 10 signals (increasing trend)
        # not the first 5 high signals
        # With the increasing trend from the last 10, should predict overload
        assert will_overload is True

    def test_prediction_with_mixed_signal_types(self) -> None:
        """Test that prediction only considers load signals, not other types."""
        # Register load signals
        loads = [0.3, 0.5, 0.7]
        for i, load in enumerate(loads):
            self._register_load_signal(load, time_offset_seconds=i * 15)

        # Register non-load signals (should be ignored)
        capacity_signal = SensitivitySignal(
            agent_id=self.agent_id,
            signal_type=SignalType.CAPACITY,
            value=0.5,
            ttl_seconds=300,
        )
        self.service.register_signal(capacity_signal)

        # Should still work with only 3 load signals
        will_overload, probability = self.service.predict_overload(
            self.agent_id, forecast_seconds=60, threshold=0.8
        )

        # Should make prediction based on 3 load signals
        assert isinstance(will_overload, bool)
        assert isinstance(probability, float)

    def test_unknown_agent_returns_no_overload(self) -> None:
        """Test prediction for agent with no signals."""
        will_overload, probability = self.service.predict_overload(
            "unknown-agent-999", forecast_seconds=60, threshold=0.8
        )

        assert will_overload is False
        assert probability == 0.0


@pytest.mark.parametrize(
    "loads,forecast_seconds,threshold,expected_overload",
    [
        # Slow increase, low threshold
        ([0.1, 0.2, 0.3, 0.4], 60, 0.5, True),
        # Slow increase, high threshold
        ([0.1, 0.2, 0.3, 0.4], 60, 0.9, False),
        # Steep increase, moderate threshold
        ([0.2, 0.5, 0.75, 0.85], 30, 0.8, True),
        # Decrease, any threshold
        ([0.9, 0.7, 0.5, 0.3], 60, 0.5, False),
        # Constant below threshold
        ([0.5, 0.5, 0.5, 0.5], 60, 0.8, False),
        # Constant above threshold
        ([0.9, 0.9, 0.9, 0.9], 60, 0.8, True),
    ],
)
def test_prediction_scenarios(
    loads: list[float],
    forecast_seconds: int,
    threshold: float,
    expected_overload: bool,
) -> None:
    """Parameterized test for various prediction scenarios.

    Args:
        loads: List of load values to register
        forecast_seconds: Forecast window
        threshold: Overload threshold
        expected_overload: Expected overload prediction
    """
    service = CoordinationService()
    agent_id = "parametrized-agent"
    base_time = datetime.now(timezone.utc)

    # Register signals
    for i, load in enumerate(loads):
        signal = SensitivitySignal(
            agent_id=agent_id,
            signal_type=SignalType.LOAD,
            value=load,
            ttl_seconds=300,
        )
        signal.timestamp = base_time + timedelta(seconds=i * 15)
        service.register_signal(signal)

    # Make prediction
    will_overload, probability = service.predict_overload(
        agent_id, forecast_seconds=forecast_seconds, threshold=threshold
    )

    # Verify expectation
    assert will_overload == expected_overload

    # Cleanup
    service.clear_state()
