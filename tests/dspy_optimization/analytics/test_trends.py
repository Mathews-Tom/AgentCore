"""
Tests for trend analysis and forecasting
"""

from datetime import datetime, timedelta

import pytest

from agentcore.dspy_optimization.analytics.trends import (
    ForecastConfig,
    TimeSeriesPoint,
    TrendAnalyzer,
    TrendDirection,
    TrendStrength,
)
from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    OptimizationTargetType,
)


@pytest.fixture
def analyzer() -> TrendAnalyzer:
    """Create trend analyzer"""
    return TrendAnalyzer()


@pytest.fixture
def target() -> OptimizationTarget:
    """Create optimization target"""
    return OptimizationTarget(type=OptimizationTargetType.AGENT, id="test-agent")


@pytest.fixture
def improving_series() -> list[TimeSeriesPoint]:
    """Create improving time series"""
    base_time = datetime.utcnow()
    return [
        TimeSeriesPoint(
            timestamp=base_time + timedelta(hours=i),
            value=0.70 + i * 0.02,  # Steady improvement
        )
        for i in range(15)
    ]


@pytest.fixture
def degrading_series() -> list[TimeSeriesPoint]:
    """Create degrading time series"""
    base_time = datetime.utcnow()
    return [
        TimeSeriesPoint(
            timestamp=base_time + timedelta(hours=i),
            value=0.90 - i * 0.02,  # Steady degradation
        )
        for i in range(15)
    ]


@pytest.fixture
def stable_series() -> list[TimeSeriesPoint]:
    """Create stable time series"""
    base_time = datetime.utcnow()
    return [
        TimeSeriesPoint(
            timestamp=base_time + timedelta(hours=i),
            value=0.80 + (i % 3 - 1) * 0.01,  # Minimal variation
        )
        for i in range(15)
    ]


@pytest.fixture
def volatile_series() -> list[TimeSeriesPoint]:
    """Create volatile time series"""
    base_time = datetime.utcnow()
    # Create highly volatile series with large swings
    # Mean ~0.7, std > 0.35 gives volatility > 0.5
    return [
        TimeSeriesPoint(
            timestamp=base_time + timedelta(hours=i),
            value=0.30 + (i % 2) * 0.80,  # Swing between 0.3 and 1.1 (very volatile)
        )
        for i in range(15)
    ]


@pytest.mark.asyncio
async def test_analyze_improving_trend(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test trend analysis for improving series"""
    result = await analyzer.analyze_trend(
        target,
        "success_rate",
        improving_series,
    )

    assert result.direction == TrendDirection.IMPROVING
    assert result.slope > 0
    assert result.strength in [TrendStrength.STRONG, TrendStrength.MODERATE]
    assert result.data_points == 15


@pytest.mark.asyncio
async def test_analyze_degrading_trend(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    degrading_series: list[TimeSeriesPoint],
) -> None:
    """Test trend analysis for degrading series"""
    result = await analyzer.analyze_trend(
        target,
        "success_rate",
        degrading_series,
    )

    assert result.direction == TrendDirection.DEGRADING
    assert result.slope < 0
    assert result.strength in [TrendStrength.STRONG, TrendStrength.MODERATE]


@pytest.mark.asyncio
async def test_analyze_stable_trend(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    stable_series: list[TimeSeriesPoint],
) -> None:
    """Test trend analysis for stable series"""
    result = await analyzer.analyze_trend(
        target,
        "success_rate",
        stable_series,
    )

    assert result.direction == TrendDirection.STABLE
    assert abs(result.correlation_coefficient) < 0.3


@pytest.mark.asyncio
async def test_analyze_volatile_trend(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    volatile_series: list[TimeSeriesPoint],
) -> None:
    """Test trend analysis for volatile series"""
    result = await analyzer.analyze_trend(
        target,
        "success_rate",
        volatile_series,
    )

    assert result.direction == TrendDirection.VOLATILE
    assert result.volatility > 0.3


@pytest.mark.asyncio
async def test_forecast_trend_exponential_smoothing(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test trend forecasting with exponential smoothing"""
    forecast = await analyzer.forecast_trend(
        target,
        "success_rate",
        improving_series,
    )

    assert forecast.forecast_periods == 7
    assert len(forecast.predicted_values) == 7
    assert len(forecast.lower_bounds) == 7
    assert len(forecast.upper_bounds) == 7
    assert forecast.method == "exponential_smoothing"

    # Check predictions are higher than baseline (improving trend)
    assert forecast.predicted_values[-1] > forecast.baseline_value


@pytest.mark.asyncio
async def test_forecast_trend_linear(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test trend forecasting with linear regression"""
    config = ForecastConfig(
        use_exponential_smoothing=False,
        forecast_periods=5,
    )

    forecast = await analyzer.forecast_trend(
        target,
        "success_rate",
        improving_series,
        config=config,
    )

    assert forecast.method == "linear_regression"
    assert len(forecast.predicted_values) == 5


@pytest.mark.asyncio
async def test_forecast_insufficient_data(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
) -> None:
    """Test forecast fails with insufficient data"""
    short_series = [
        TimeSeriesPoint(timestamp=datetime.utcnow(), value=0.8)
        for _ in range(5)
    ]

    with pytest.raises(ValueError, match="Insufficient historical data"):
        await analyzer.forecast_trend(target, "success_rate", short_series)


@pytest.mark.asyncio
async def test_forecast_timestamps(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test forecast generates correct timestamps"""
    forecast = await analyzer.forecast_trend(
        target,
        "success_rate",
        improving_series,
    )

    assert len(forecast.forecast_timestamps) == forecast.forecast_periods

    # Timestamps should be in future
    last_historical = improving_series[-1].timestamp
    assert all(ts > last_historical for ts in forecast.forecast_timestamps)

    # Timestamps should be sequential
    for i in range(len(forecast.forecast_timestamps) - 1):
        assert forecast.forecast_timestamps[i] < forecast.forecast_timestamps[i + 1]


@pytest.mark.asyncio
async def test_prediction_intervals(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test prediction interval calculation"""
    forecast = await analyzer.forecast_trend(
        target,
        "success_rate",
        improving_series,
    )

    # Lower bound should be less than predicted, upper bound should be greater
    for i in range(len(forecast.predicted_values)):
        assert forecast.lower_bounds[i] <= forecast.predicted_values[i]
        assert forecast.upper_bounds[i] >= forecast.predicted_values[i]

    # Uncertainty should increase with forecast horizon
    first_width = forecast.upper_bounds[0] - forecast.lower_bounds[0]
    last_width = forecast.upper_bounds[-1] - forecast.lower_bounds[-1]
    assert last_width >= first_width


@pytest.mark.asyncio
async def test_analyze_multiple_metrics(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
    stable_series: list[TimeSeriesPoint],
) -> None:
    """Test analysis of multiple metrics"""
    metrics_data = {
        "success_rate": improving_series,
        "cost": stable_series,
    }

    results = await analyzer.analyze_multiple_metrics(target, metrics_data)

    assert len(results) == 2
    assert "success_rate" in results
    assert "cost" in results
    assert results["success_rate"].direction == TrendDirection.IMPROVING
    assert results["cost"].direction == TrendDirection.STABLE


@pytest.mark.asyncio
async def test_forecast_multiple_metrics(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
    stable_series: list[TimeSeriesPoint],
) -> None:
    """Test forecasting multiple metrics"""
    metrics_data = {
        "success_rate": improving_series,
        "cost": stable_series,
    }

    forecasts = await analyzer.forecast_multiple_metrics(target, metrics_data)

    assert len(forecasts) == 2
    assert "success_rate" in forecasts
    assert "cost" in forecasts


@pytest.mark.asyncio
async def test_get_trend_summary(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
    degrading_series: list[TimeSeriesPoint],
    stable_series: list[TimeSeriesPoint],
) -> None:
    """Test trend summary generation"""
    metrics_data = {
        "metric1": improving_series,
        "metric2": degrading_series,
        "metric3": stable_series,
    }

    results = await analyzer.analyze_multiple_metrics(target, metrics_data)
    summary = await analyzer.get_trend_summary(results)

    assert summary["total_metrics"] == 3
    assert summary["improving_count"] >= 1
    assert summary["degrading_count"] >= 1
    assert summary["stable_count"] >= 0
    assert "avg_correlation" in summary
    assert "avg_volatility" in summary


@pytest.mark.asyncio
async def test_trend_statistical_significance(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test statistical significance of trend"""
    result = await analyzer.analyze_trend(
        target,
        "success_rate",
        improving_series,
    )

    assert result.p_value >= 0.0
    assert result.p_value <= 1.0

    # Strong improving trend should be significant
    if result.strength == TrendStrength.STRONG:
        assert result.is_significant is True


@pytest.mark.asyncio
async def test_expected_change_calculation(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test expected change calculation in forecast"""
    forecast = await analyzer.forecast_trend(
        target,
        "success_rate",
        improving_series,
    )

    # For improving series, expected change should be positive
    assert forecast.expected_change > 0

    # Should be percentage change from baseline
    calculated_change = (
        (forecast.predicted_values[-1] - forecast.baseline_value)
        / forecast.baseline_value
    )
    assert forecast.expected_change == pytest.approx(calculated_change, rel=0.01)


@pytest.mark.asyncio
async def test_custom_forecast_config(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
    improving_series: list[TimeSeriesPoint],
) -> None:
    """Test custom forecast configuration"""
    config = ForecastConfig(
        forecast_periods=14,
        confidence_level=0.99,
        smoothing_alpha=0.5,
    )

    forecast = await analyzer.forecast_trend(
        target,
        "success_rate",
        improving_series,
        config=config,
    )

    assert len(forecast.predicted_values) == 14
    assert forecast.confidence_level == 0.99


@pytest.mark.asyncio
async def test_analyze_insufficient_data(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
) -> None:
    """Test analysis fails with insufficient data"""
    single_point = [TimeSeriesPoint(timestamp=datetime.utcnow(), value=0.8)]

    with pytest.raises(ValueError, match="Insufficient data points"):
        await analyzer.analyze_trend(target, "success_rate", single_point)


@pytest.mark.asyncio
async def test_time_span_calculation(
    analyzer: TrendAnalyzer,
    target: OptimizationTarget,
) -> None:
    """Test time span calculation"""
    base_time = datetime.utcnow()
    series = [
        TimeSeriesPoint(timestamp=base_time + timedelta(days=i), value=0.7 + i * 0.01)
        for i in range(30)
    ]

    result = await analyzer.analyze_trend(target, "success_rate", series)

    # Should be approximately 29 days (30 points, 0-indexed)
    assert result.time_span_days >= 28
    assert result.time_span_days <= 30
