"""
Trend analysis and forecasting

Provides time-series analysis, trend detection, and forecasting
for optimization performance over time.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    PerformanceMetrics,
)


class TrendDirection(str, Enum):
    """Direction of trend"""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


class TrendStrength(str, Enum):
    """Strength of trend"""

    STRONG = "strong"  # |correlation| > 0.7
    MODERATE = "moderate"  # 0.4 < |correlation| <= 0.7
    WEAK = "weak"  # 0.2 < |correlation| <= 0.4
    NONE = "none"  # |correlation| <= 0.2


class ForecastConfig(BaseModel):
    """Configuration for forecasting"""

    forecast_periods: int = Field(
        default=7,
        description="Number of periods to forecast",
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for prediction intervals",
    )
    min_history_points: int = Field(
        default=10,
        description="Minimum historical points required",
    )
    use_exponential_smoothing: bool = Field(
        default=True,
        description="Use exponential smoothing for forecasts",
    )
    smoothing_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Exponential smoothing parameter",
    )


class TrendResult(BaseModel):
    """Result of trend analysis"""

    target: OptimizationTarget
    metric_name: str
    direction: TrendDirection
    strength: TrendStrength
    correlation_coefficient: float
    slope: float
    intercept: float
    p_value: float
    is_significant: bool
    volatility: float
    data_points: int
    time_span_days: float
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrendForecast(BaseModel):
    """Forecast result"""

    target: OptimizationTarget
    metric_name: str
    forecast_periods: int
    predicted_values: list[float]
    lower_bounds: list[float]
    upper_bounds: list[float]
    confidence_level: float
    forecast_timestamps: list[datetime]
    method: str
    baseline_value: float
    expected_change: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimeSeriesPoint(BaseModel):
    """Time-series data point"""

    timestamp: datetime
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrendAnalyzer:
    """
    Trend analysis and forecasting

    Provides comprehensive time-series analysis including trend detection,
    statistical validation, and future performance forecasting.

    Key features:
    - Linear regression trend analysis
    - Statistical significance testing
    - Volatility analysis
    - Exponential smoothing forecasts
    - Confidence intervals
    - Multi-metric analysis
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        """
        Initialize trend analyzer

        Args:
            config: Forecast configuration
        """
        self.config = config or ForecastConfig()

    async def analyze_trend(
        self,
        target: OptimizationTarget,
        metric_name: str,
        time_series: list[TimeSeriesPoint],
    ) -> TrendResult:
        """
        Analyze trend in time-series data

        Args:
            target: Optimization target
            metric_name: Name of metric being analyzed
            time_series: Time-series data points

        Returns:
            Trend analysis result

        Raises:
            ValueError: If insufficient data points
        """
        if len(time_series) < 2:
            raise ValueError(f"Insufficient data points: {len(time_series)}")

        # Sort by timestamp
        sorted_series = sorted(time_series, key=lambda p: p.timestamp)

        # Extract timestamps and values
        timestamps = [p.timestamp for p in sorted_series]
        values = [p.value for p in sorted_series]

        # Convert timestamps to numeric (days since first point)
        time_deltas = [(t - timestamps[0]).total_seconds() / 86400 for t in timestamps]
        x = np.array(time_deltas)
        y = np.array(values)

        # Calculate linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Determine direction
        direction = self._determine_direction(slope, r_value, values)

        # Determine strength
        strength = self._determine_strength(abs(r_value))

        # Calculate volatility (coefficient of variation)
        volatility = np.std(y) / np.mean(y) if np.mean(y) != 0 else 0.0

        # Check significance (p < 0.05)
        is_significant = p_value < 0.05

        # Calculate time span
        time_span = (timestamps[-1] - timestamps[0]).total_seconds() / 86400

        return TrendResult(
            target=target,
            metric_name=metric_name,
            direction=direction,
            strength=strength,
            correlation_coefficient=r_value,
            slope=slope,
            intercept=intercept,
            p_value=p_value,
            is_significant=is_significant,
            volatility=volatility,
            data_points=len(time_series),
            time_span_days=time_span,
        )

    async def forecast_trend(
        self,
        target: OptimizationTarget,
        metric_name: str,
        time_series: list[TimeSeriesPoint],
        config: ForecastConfig | None = None,
    ) -> TrendForecast:
        """
        Forecast future trend values

        Args:
            target: Optimization target
            metric_name: Name of metric to forecast
            time_series: Historical time-series data
            config: Optional forecast configuration override

        Returns:
            Forecast result

        Raises:
            ValueError: If insufficient historical data
        """
        forecast_config = config or self.config

        if len(time_series) < forecast_config.min_history_points:
            raise ValueError(
                f"Insufficient historical data: {len(time_series)} "
                f"< {forecast_config.min_history_points}"
            )

        # Sort by timestamp
        sorted_series = sorted(time_series, key=lambda p: p.timestamp)

        # Extract values
        values = [p.value for p in sorted_series]
        timestamps = [p.timestamp for p in sorted_series]

        # Determine time interval (average)
        intervals = [
            (timestamps[i + 1] - timestamps[i]).total_seconds()
            for i in range(len(timestamps) - 1)
        ]
        avg_interval_seconds = sum(intervals) / len(intervals)

        # Choose forecasting method
        if forecast_config.use_exponential_smoothing:
            predicted_values = self._exponential_smoothing_forecast(
                values,
                forecast_config.forecast_periods,
                forecast_config.smoothing_alpha,
            )
            method = "exponential_smoothing"
        else:
            predicted_values = self._linear_forecast(
                values,
                forecast_config.forecast_periods,
            )
            method = "linear_regression"

        # Calculate prediction intervals
        lower_bounds, upper_bounds = self._calculate_prediction_intervals(
            values,
            predicted_values,
            forecast_config.confidence_level,
        )

        # Generate forecast timestamps
        forecast_timestamps = []
        last_timestamp = timestamps[-1]
        for i in range(1, forecast_config.forecast_periods + 1):
            next_timestamp = last_timestamp + timedelta(seconds=avg_interval_seconds * i)
            forecast_timestamps.append(next_timestamp)

        # Calculate expected change
        baseline_value = values[-1]
        expected_change = (
            (predicted_values[-1] - baseline_value) / baseline_value
            if baseline_value != 0
            else 0.0
        )

        return TrendForecast(
            target=target,
            metric_name=metric_name,
            forecast_periods=forecast_config.forecast_periods,
            predicted_values=predicted_values,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            confidence_level=forecast_config.confidence_level,
            forecast_timestamps=forecast_timestamps,
            method=method,
            baseline_value=baseline_value,
            expected_change=expected_change,
            metadata={
                "historical_points": len(time_series),
                "avg_interval_seconds": avg_interval_seconds,
            },
        )

    async def analyze_multiple_metrics(
        self,
        target: OptimizationTarget,
        metrics_data: dict[str, list[TimeSeriesPoint]],
    ) -> dict[str, TrendResult]:
        """
        Analyze trends for multiple metrics

        Args:
            target: Optimization target
            metrics_data: Dictionary of metric names to time-series data

        Returns:
            Dictionary of metric names to trend results
        """
        results = {}

        for metric_name, time_series in metrics_data.items():
            try:
                result = await self.analyze_trend(target, metric_name, time_series)
                results[metric_name] = result
            except Exception:
                # Skip metrics with insufficient data
                continue

        return results

    async def forecast_multiple_metrics(
        self,
        target: OptimizationTarget,
        metrics_data: dict[str, list[TimeSeriesPoint]],
        config: ForecastConfig | None = None,
    ) -> dict[str, TrendForecast]:
        """
        Forecast trends for multiple metrics

        Args:
            target: Optimization target
            metrics_data: Dictionary of metric names to time-series data
            config: Optional forecast configuration

        Returns:
            Dictionary of metric names to forecast results
        """
        forecasts = {}

        for metric_name, time_series in metrics_data.items():
            try:
                forecast = await self.forecast_trend(
                    target, metric_name, time_series, config
                )
                forecasts[metric_name] = forecast
            except Exception:
                # Skip metrics with insufficient data
                continue

        return forecasts

    async def get_trend_summary(
        self,
        trends: dict[str, TrendResult],
    ) -> dict[str, Any]:
        """
        Get summary of trend analysis

        Args:
            trends: Dictionary of trend results

        Returns:
            Summary statistics
        """
        if not trends:
            return {
                "total_metrics": 0,
                "improving_count": 0,
                "degrading_count": 0,
                "stable_count": 0,
            }

        improving = sum(1 for t in trends.values() if t.direction == TrendDirection.IMPROVING)
        degrading = sum(1 for t in trends.values() if t.direction == TrendDirection.DEGRADING)
        stable = sum(1 for t in trends.values() if t.direction == TrendDirection.STABLE)
        volatile = sum(1 for t in trends.values() if t.direction == TrendDirection.VOLATILE)

        significant = sum(1 for t in trends.values() if t.is_significant)
        strong = sum(1 for t in trends.values() if t.strength == TrendStrength.STRONG)

        avg_correlation = sum(abs(t.correlation_coefficient) for t in trends.values()) / len(
            trends
        )
        avg_volatility = sum(t.volatility for t in trends.values()) / len(trends)

        return {
            "total_metrics": len(trends),
            "improving_count": improving,
            "degrading_count": degrading,
            "stable_count": stable,
            "volatile_count": volatile,
            "significant_count": significant,
            "strong_trend_count": strong,
            "avg_correlation": avg_correlation,
            "avg_volatility": avg_volatility,
        }

    def _determine_direction(
        self,
        slope: float,
        correlation: float,
        values: list[float],
    ) -> TrendDirection:
        """
        Determine trend direction

        Args:
            slope: Regression slope
            correlation: Correlation coefficient
            values: Data values

        Returns:
            Trend direction
        """
        # Check volatility first
        volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0.0

        if volatility > 0.5:
            return TrendDirection.VOLATILE

        # Weak correlation means stable
        if abs(correlation) < 0.2:
            return TrendDirection.STABLE

        # Strong correlation determines direction
        if slope > 0:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DEGRADING

    def _determine_strength(self, abs_correlation: float) -> TrendStrength:
        """
        Determine trend strength

        Args:
            abs_correlation: Absolute correlation coefficient

        Returns:
            Trend strength
        """
        if abs_correlation > 0.7:
            return TrendStrength.STRONG
        elif abs_correlation > 0.4:
            return TrendStrength.MODERATE
        elif abs_correlation > 0.2:
            return TrendStrength.WEAK
        else:
            return TrendStrength.NONE

    def _exponential_smoothing_forecast(
        self,
        values: list[float],
        periods: int,
        alpha: float,
    ) -> list[float]:
        """
        Exponential smoothing forecast

        Args:
            values: Historical values
            periods: Number of periods to forecast
            alpha: Smoothing parameter

        Returns:
            Forecasted values
        """
        # Initialize with first value
        smoothed = [values[0]]

        # Calculate smoothed values for historical data
        for i in range(1, len(values)):
            smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[-1]
            smoothed.append(smoothed_value)

        # Forecast future values (use last smoothed value with trend adjustment)
        forecasts = []
        last_value = smoothed[-1]

        # Calculate trend from last few points
        trend = 0.0
        if len(smoothed) >= 3:
            trend = (smoothed[-1] - smoothed[-3]) / 2

        for i in range(periods):
            forecast_value = last_value + trend * (i + 1)
            forecasts.append(forecast_value)

        return forecasts

    def _linear_forecast(
        self,
        values: list[float],
        periods: int,
    ) -> list[float]:
        """
        Linear regression forecast

        Args:
            values: Historical values
            periods: Number of periods to forecast

        Returns:
            Forecasted values
        """
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate linear regression
        slope, intercept, _, _, _ = stats.linregress(x, y)

        # Forecast future values
        forecasts = []
        for i in range(periods):
            future_x = len(values) + i
            forecast_value = slope * future_x + intercept
            forecasts.append(forecast_value)

        return forecasts

    def _calculate_prediction_intervals(
        self,
        historical_values: list[float],
        predicted_values: list[float],
        confidence_level: float,
    ) -> tuple[list[float], list[float]]:
        """
        Calculate prediction intervals

        Args:
            historical_values: Historical data
            predicted_values: Predicted values
            confidence_level: Confidence level

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        # Calculate residual standard error
        residuals = np.diff(historical_values)
        std_error = np.std(residuals)

        # Calculate margin based on confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Calculate bounds (uncertainty increases with forecast horizon)
        lower_bounds = []
        upper_bounds = []

        for i, pred_value in enumerate(predicted_values):
            # Increase uncertainty with time
            uncertainty = std_error * z_score * np.sqrt(i + 1)
            lower_bounds.append(max(0, pred_value - uncertainty))
            upper_bounds.append(pred_value + uncertainty)

        return lower_bounds, upper_bounds
