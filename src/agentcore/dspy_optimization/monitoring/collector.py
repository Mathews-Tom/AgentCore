"""
Metrics collection framework

Collects, aggregates, and stores performance metrics for optimization
monitoring and analysis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    PerformanceMetrics,
    OptimizationTarget,
)


class AggregationMethod(str, Enum):
    """Methods for metric aggregation"""

    AVERAGE = "average"
    MEDIAN = "median"
    PERCENTILE_95 = "percentile_95"
    PERCENTILE_99 = "percentile_99"
    MIN = "min"
    MAX = "max"


class CollectorConfig(BaseModel):
    """Configuration for metrics collector"""

    aggregation_interval_seconds: int = Field(
        default=300,
        description="Interval for aggregating metrics (5 minutes default)",
    )
    retention_days: int = Field(
        default=90,
        description="Days to retain metrics",
    )
    max_snapshots_per_target: int = Field(
        default=10000,
        description="Maximum snapshots to store per target",
    )
    aggregation_method: AggregationMethod = Field(
        default=AggregationMethod.AVERAGE,
        description="Default aggregation method",
    )


class MetricSnapshot(BaseModel):
    """Point-in-time metrics snapshot"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    metrics: PerformanceMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sample_count: int = Field(default=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricAggregation(BaseModel):
    """Aggregated metrics over time window"""

    target: OptimizationTarget
    start_time: datetime
    end_time: datetime
    aggregated_metrics: PerformanceMetrics
    snapshot_count: int
    aggregation_method: AggregationMethod


class MetricsCollector:
    """
    Collects and aggregates performance metrics

    Manages real-time collection, aggregation, and storage of
    performance metrics for optimization targets.
    """

    def __init__(self, config: CollectorConfig | None = None) -> None:
        """
        Initialize metrics collector

        Args:
            config: Collector configuration
        """
        self.config = config or CollectorConfig()
        self._snapshots: dict[str, list[MetricSnapshot]] = {}
        self._aggregations: dict[str, list[MetricAggregation]] = {}

    async def collect(
        self,
        target: OptimizationTarget,
        metrics: PerformanceMetrics,
        metadata: dict[str, Any] | None = None,
    ) -> MetricSnapshot:
        """
        Collect metrics snapshot

        Args:
            target: Optimization target
            metrics: Performance metrics
            metadata: Optional metadata

        Returns:
            Created metric snapshot
        """
        snapshot = MetricSnapshot(
            target=target,
            metrics=metrics,
            metadata=metadata or {},
        )

        # Store snapshot
        target_key = self._get_target_key(target)
        if target_key not in self._snapshots:
            self._snapshots[target_key] = []

        self._snapshots[target_key].append(snapshot)

        # Enforce retention limits
        await self._enforce_retention(target_key)

        return snapshot

    async def collect_batch(
        self,
        target: OptimizationTarget,
        metrics_list: list[PerformanceMetrics],
    ) -> list[MetricSnapshot]:
        """
        Collect batch of metrics

        Args:
            target: Optimization target
            metrics_list: List of performance metrics

        Returns:
            List of created snapshots
        """
        snapshots = []
        for metrics in metrics_list:
            snapshot = await self.collect(target, metrics)
            snapshots.append(snapshot)

        return snapshots

    async def get_snapshots(
        self,
        target: OptimizationTarget,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
    ) -> list[MetricSnapshot]:
        """
        Get metrics snapshots for target

        Args:
            target: Optimization target
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Optional result limit

        Returns:
            List of metric snapshots
        """
        target_key = self._get_target_key(target)
        snapshots = self._snapshots.get(target_key, [])

        # Filter by time range
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]

        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)

        # Apply limit
        if limit:
            snapshots = snapshots[:limit]

        return snapshots

    async def aggregate(
        self,
        target: OptimizationTarget,
        start_time: datetime,
        end_time: datetime,
        method: AggregationMethod | None = None,
    ) -> MetricAggregation:
        """
        Aggregate metrics over time window

        Args:
            target: Optimization target
            start_time: Aggregation start time
            end_time: Aggregation end time
            method: Aggregation method (uses config default if not specified)

        Returns:
            Metric aggregation

        Raises:
            ValueError: If no metrics found in time window
        """
        # Get snapshots in time window
        snapshots = await self.get_snapshots(target, start_time, end_time)

        if not snapshots:
            raise ValueError(
                f"No metrics found for {target.type}:{target.id} "
                f"between {start_time} and {end_time}"
            )

        # Use provided method or config default
        agg_method = method or self.config.aggregation_method

        # Aggregate metrics
        aggregated = self._aggregate_metrics(snapshots, agg_method)

        aggregation = MetricAggregation(
            target=target,
            start_time=start_time,
            end_time=end_time,
            aggregated_metrics=aggregated,
            snapshot_count=len(snapshots),
            aggregation_method=agg_method,
        )

        # Store aggregation
        target_key = self._get_target_key(target)
        if target_key not in self._aggregations:
            self._aggregations[target_key] = []

        self._aggregations[target_key].append(aggregation)

        return aggregation

    async def get_latest_metrics(
        self,
        target: OptimizationTarget,
    ) -> PerformanceMetrics | None:
        """
        Get latest metrics for target

        Args:
            target: Optimization target

        Returns:
            Latest performance metrics or None
        """
        snapshots = await self.get_snapshots(target, limit=1)
        return snapshots[0].metrics if snapshots else None

    async def get_average_metrics(
        self,
        target: OptimizationTarget,
        window_hours: int = 24,
    ) -> PerformanceMetrics | None:
        """
        Get average metrics over time window

        Args:
            target: Optimization target
            window_hours: Time window in hours

        Returns:
            Average metrics or None if no data
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=window_hours)

        try:
            aggregation = await self.aggregate(
                target,
                start_time,
                end_time,
                AggregationMethod.AVERAGE,
            )
            return aggregation.aggregated_metrics
        except ValueError:
            return None

    def _aggregate_metrics(
        self,
        snapshots: list[MetricSnapshot],
        method: AggregationMethod,
    ) -> PerformanceMetrics:
        """
        Aggregate metrics from snapshots

        Args:
            snapshots: Metric snapshots
            method: Aggregation method

        Returns:
            Aggregated performance metrics
        """
        # Extract metric values
        success_rates = [s.metrics.success_rate for s in snapshots]
        costs = [s.metrics.avg_cost_per_task for s in snapshots]
        latencies = [s.metrics.avg_latency_ms for s in snapshots]
        quality_scores = [s.metrics.quality_score for s in snapshots]

        # Apply aggregation method
        if method == AggregationMethod.AVERAGE:
            return PerformanceMetrics(
                success_rate=sum(success_rates) / len(success_rates),
                avg_cost_per_task=sum(costs) / len(costs),
                avg_latency_ms=int(sum(latencies) / len(latencies)),
                quality_score=sum(quality_scores) / len(quality_scores),
            )
        elif method == AggregationMethod.MEDIAN:
            success_rates.sort()
            costs.sort()
            latencies.sort()
            quality_scores.sort()
            mid = len(snapshots) // 2
            return PerformanceMetrics(
                success_rate=success_rates[mid],
                avg_cost_per_task=costs[mid],
                avg_latency_ms=latencies[mid],
                quality_score=quality_scores[mid],
            )
        elif method == AggregationMethod.PERCENTILE_95:
            return self._percentile_aggregation(snapshots, 0.95)
        elif method == AggregationMethod.PERCENTILE_99:
            return self._percentile_aggregation(snapshots, 0.99)
        elif method == AggregationMethod.MIN:
            return PerformanceMetrics(
                success_rate=min(success_rates),
                avg_cost_per_task=min(costs),
                avg_latency_ms=min(latencies),
                quality_score=min(quality_scores),
            )
        elif method == AggregationMethod.MAX:
            return PerformanceMetrics(
                success_rate=max(success_rates),
                avg_cost_per_task=max(costs),
                avg_latency_ms=max(latencies),
                quality_score=max(quality_scores),
            )

        # Default to average
        return PerformanceMetrics(
            success_rate=sum(success_rates) / len(success_rates),
            avg_cost_per_task=sum(costs) / len(costs),
            avg_latency_ms=int(sum(latencies) / len(latencies)),
            quality_score=sum(quality_scores) / len(quality_scores),
        )

    def _percentile_aggregation(
        self,
        snapshots: list[MetricSnapshot],
        percentile: float,
    ) -> PerformanceMetrics:
        """
        Calculate percentile aggregation

        Args:
            snapshots: Metric snapshots
            percentile: Percentile value (0.0-1.0)

        Returns:
            Percentile metrics
        """
        success_rates = sorted([s.metrics.success_rate for s in snapshots])
        costs = sorted([s.metrics.avg_cost_per_task for s in snapshots])
        latencies = sorted([s.metrics.avg_latency_ms for s in snapshots])
        quality_scores = sorted([s.metrics.quality_score for s in snapshots])

        idx = int(len(snapshots) * percentile)

        return PerformanceMetrics(
            success_rate=success_rates[idx],
            avg_cost_per_task=costs[idx],
            avg_latency_ms=latencies[idx],
            quality_score=quality_scores[idx],
        )

    async def _enforce_retention(self, target_key: str) -> None:
        """
        Enforce retention policies

        Args:
            target_key: Target key
        """
        snapshots = self._snapshots.get(target_key, [])

        # Remove old snapshots beyond retention period
        cutoff_time = datetime.utcnow() - timedelta(days=self.config.retention_days)
        snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]

        # Limit total snapshots
        if len(snapshots) > self.config.max_snapshots_per_target:
            # Keep newest snapshots
            snapshots.sort(key=lambda s: s.timestamp, reverse=True)
            snapshots = snapshots[: self.config.max_snapshots_per_target]

        self._snapshots[target_key] = snapshots

    def _get_target_key(self, target: OptimizationTarget) -> str:
        """
        Get target storage key

        Args:
            target: Optimization target

        Returns:
            Target key
        """
        return f"{target.type.value}:{target.id}:{target.scope.value}"
