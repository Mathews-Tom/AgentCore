"""
Dashboard service for performance monitoring

Provides real-time performance dashboards, trend analysis, and
optimization history for monitoring optimization effectiveness.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    OptimizationStatus,
    OptimizationTarget,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.monitoring.baseline import BaselineService
from agentcore.dspy_optimization.monitoring.collector import (
    MetricsCollector,
    MetricSnapshot,
)


class PerformanceTrend(BaseModel):
    """Performance trend data point"""

    timestamp: datetime
    metrics: PerformanceMetrics
    optimization_version: str = "baseline"
    is_optimized: bool = False


class OptimizationHistory(BaseModel):
    """Optimization deployment history record"""

    deployed_at: datetime
    algorithm: str
    improvement_percentage: float
    status: OptimizationStatus
    version_id: str
    baseline_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics


class PerformanceRecommendation(BaseModel):
    """Performance improvement recommendation"""

    type: str
    suggestion: str
    confidence: float = Field(ge=0.0, le=1.0)
    expected_improvement: float | None = None
    priority: str = "medium"


class DashboardData(BaseModel):
    """Complete dashboard data"""

    target: OptimizationTarget
    current_metrics: PerformanceMetrics
    baseline_metrics: PerformanceMetrics | None = None
    trends: list[PerformanceTrend] = Field(default_factory=list)
    optimization_history: list[OptimizationHistory] = Field(default_factory=list)
    recommendations: list[PerformanceRecommendation] = Field(default_factory=list)
    time_range: tuple[datetime, datetime]
    last_updated: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DashboardService:
    """
    Dashboard service for performance monitoring

    Provides comprehensive dashboard data including trends, history,
    and recommendations for optimization targets.
    """

    def __init__(
        self,
        collector: MetricsCollector,
        baseline_service: BaselineService,
    ) -> None:
        """
        Initialize dashboard service

        Args:
            collector: Metrics collector instance
            baseline_service: Baseline service instance
        """
        self.collector = collector
        self.baseline_service = baseline_service
        self._optimization_history: dict[str, list[OptimizationHistory]] = {}

    async def get_dashboard_data(
        self,
        target: OptimizationTarget,
        hours: int = 24,
    ) -> DashboardData:
        """
        Get complete dashboard data for target

        Args:
            target: Optimization target
            hours: Time window in hours (default: 24)

        Returns:
            Dashboard data
        """
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=hours)

        # Get current metrics
        current_metrics = await self.collector.get_latest_metrics(target)
        if not current_metrics:
            current_metrics = PerformanceMetrics(
                success_rate=0.0,
                avg_cost_per_task=0.0,
                avg_latency_ms=0,
                quality_score=0.0,
            )

        # Get baseline metrics
        baseline = await self.baseline_service.get_baseline(target)
        baseline_metrics = baseline.metrics if baseline else None

        # Get performance trends
        trends = await self.get_performance_trends(target, start_time, end_time)

        # Get optimization history
        history = await self.get_optimization_history(target)

        # Generate recommendations
        recommendations = await self.generate_recommendations(
            target, current_metrics, baseline_metrics
        )

        return DashboardData(
            target=target,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            trends=trends,
            optimization_history=history,
            recommendations=recommendations,
            time_range=(start_time, end_time),
        )

    async def get_performance_trends(
        self,
        target: OptimizationTarget,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 60,
    ) -> list[PerformanceTrend]:
        """
        Get performance trends over time

        Args:
            target: Optimization target
            start_time: Start of time window
            end_time: End of time window
            interval_minutes: Sampling interval in minutes

        Returns:
            List of performance trend data points
        """
        trends = []

        # Get snapshots
        snapshots = await self.collector.get_snapshots(target, start_time, end_time)

        if not snapshots:
            return trends

        # Group snapshots by interval
        interval_delta = timedelta(minutes=interval_minutes)
        current_time = start_time

        while current_time <= end_time:
            interval_end = current_time + interval_delta

            # Get snapshots in interval
            interval_snapshots = [
                s for s in snapshots if current_time <= s.timestamp < interval_end
            ]

            if interval_snapshots:
                # Average metrics in interval
                avg_metrics = self._average_snapshots(interval_snapshots)

                # Determine if optimized version
                is_optimized = any(
                    s.metadata.get("is_optimized", False) for s in interval_snapshots
                )
                version = (
                    interval_snapshots[0].metadata.get("version", "baseline")
                    if interval_snapshots
                    else "baseline"
                )

                trends.append(
                    PerformanceTrend(
                        timestamp=current_time,
                        metrics=avg_metrics,
                        optimization_version=version,
                        is_optimized=is_optimized,
                    )
                )

            current_time = interval_end

        return trends

    async def get_optimization_history(
        self,
        target: OptimizationTarget,
        limit: int = 10,
    ) -> list[OptimizationHistory]:
        """
        Get optimization deployment history

        Args:
            target: Optimization target
            limit: Maximum number of history records

        Returns:
            List of optimization history records
        """
        target_key = self._get_target_key(target)
        history = self._optimization_history.get(target_key, [])

        # Sort by deployment date (newest first)
        history.sort(key=lambda h: h.deployed_at, reverse=True)

        return history[:limit]

    async def record_optimization(
        self,
        target: OptimizationTarget,
        algorithm: str,
        baseline_metrics: PerformanceMetrics,
        optimized_metrics: PerformanceMetrics,
        improvement_percentage: float,
        version_id: str,
    ) -> OptimizationHistory:
        """
        Record optimization deployment

        Args:
            target: Optimization target
            algorithm: Algorithm used
            baseline_metrics: Baseline metrics
            optimized_metrics: Optimized metrics
            improvement_percentage: Improvement percentage
            version_id: Version identifier

        Returns:
            Created history record
        """
        history_record = OptimizationHistory(
            deployed_at=datetime.now(UTC),
            algorithm=algorithm,
            improvement_percentage=improvement_percentage,
            status=OptimizationStatus.COMPLETED,
            version_id=version_id,
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
        )

        # Store history
        target_key = self._get_target_key(target)
        if target_key not in self._optimization_history:
            self._optimization_history[target_key] = []

        self._optimization_history[target_key].append(history_record)

        return history_record

    async def generate_recommendations(
        self,
        target: OptimizationTarget,
        current_metrics: PerformanceMetrics,
        baseline_metrics: PerformanceMetrics | None,
    ) -> list[PerformanceRecommendation]:
        """
        Generate performance recommendations

        Args:
            target: Optimization target
            current_metrics: Current performance metrics
            baseline_metrics: Baseline metrics (if available)

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check if baseline exists
        if not baseline_metrics:
            recommendations.append(
                PerformanceRecommendation(
                    type="baseline_missing",
                    suggestion="Establish baseline metrics for accurate improvement tracking",
                    confidence=1.0,
                    priority="high",
                )
            )
            return recommendations

        # Check success rate
        if current_metrics.success_rate < 0.7:
            recommendations.append(
                PerformanceRecommendation(
                    type="low_success_rate",
                    suggestion="Success rate is below 70%. Consider using GEPA algorithm for reflective optimization",
                    confidence=0.8,
                    expected_improvement=0.2,
                    priority="high",
                )
            )

        # Check cost efficiency
        if current_metrics.avg_cost_per_task > baseline_metrics.avg_cost_per_task * 1.2:
            recommendations.append(
                PerformanceRecommendation(
                    type="high_cost",
                    suggestion="Cost has increased 20%+ above baseline. Review optimization trade-offs",
                    confidence=0.9,
                    priority="medium",
                )
            )

        # Check latency
        if current_metrics.avg_latency_ms > baseline_metrics.avg_latency_ms * 1.5:
            recommendations.append(
                PerformanceRecommendation(
                    type="high_latency",
                    suggestion="Latency has increased 50%+ above baseline. Consider latency-focused optimization",
                    confidence=0.85,
                    priority="medium",
                )
            )

        # Check for optimization opportunities
        if (
            baseline_metrics.success_rate > 0
            and current_metrics.success_rate == baseline_metrics.success_rate
        ):
            # Get history
            history = await self.get_optimization_history(target, limit=1)
            if not history:
                recommendations.append(
                    PerformanceRecommendation(
                        type="optimization_opportunity",
                        suggestion="No optimizations detected. Try MIPROv2 for instruction-based improvements",
                        confidence=0.7,
                        expected_improvement=0.25,
                        priority="medium",
                    )
                )

        # Quality score check
        if current_metrics.quality_score < 0.6:
            recommendations.append(
                PerformanceRecommendation(
                    type="low_quality",
                    suggestion="Quality score is below 60%. Focus on quality-oriented optimization objectives",
                    confidence=0.75,
                    priority="high",
                )
            )

        return recommendations

    async def get_realtime_stats(
        self,
        target: OptimizationTarget,
    ) -> dict[str, Any]:
        """
        Get real-time statistics for target

        Args:
            target: Optimization target

        Returns:
            Real-time statistics
        """
        # Get latest metrics
        current = await self.collector.get_latest_metrics(target)

        # Get 1-hour average
        hour_avg = await self.collector.get_average_metrics(target, window_hours=1)

        # Get 24-hour average
        day_avg = await self.collector.get_average_metrics(target, window_hours=24)

        # Get baseline
        baseline = await self.baseline_service.get_baseline(target)

        return {
            "current": current.model_dump() if current else None,
            "1h_average": hour_avg.model_dump() if hour_avg else None,
            "24h_average": day_avg.model_dump() if day_avg else None,
            "baseline": baseline.metrics.model_dump() if baseline else None,
            "baseline_age_hours": (
                (datetime.now(UTC) - baseline.created_at).total_seconds() / 3600
                if baseline
                else None
            ),
        }

    def _average_snapshots(
        self,
        snapshots: list[MetricSnapshot],
    ) -> PerformanceMetrics:
        """
        Calculate average metrics from snapshots

        Args:
            snapshots: Metric snapshots

        Returns:
            Average performance metrics
        """
        if not snapshots:
            return PerformanceMetrics(
                success_rate=0.0,
                avg_cost_per_task=0.0,
                avg_latency_ms=0,
                quality_score=0.0,
            )

        success_rates = [s.metrics.success_rate for s in snapshots]
        costs = [s.metrics.avg_cost_per_task for s in snapshots]
        latencies = [s.metrics.avg_latency_ms for s in snapshots]
        quality_scores = [s.metrics.quality_score for s in snapshots]

        return PerformanceMetrics(
            success_rate=sum(success_rates) / len(success_rates),
            avg_cost_per_task=sum(costs) / len(costs),
            avg_latency_ms=int(sum(latencies) / len(latencies)),
            quality_score=sum(quality_scores) / len(quality_scores),
        )

    def _get_target_key(self, target: OptimizationTarget) -> str:
        """
        Get target storage key

        Args:
            target: Optimization target

        Returns:
            Target key
        """
        return f"{target.type.value}:{target.id}:{target.scope.value}"
