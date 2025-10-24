"""Performance monitoring and SLA tracking.

Real-time performance analysis, SLA monitoring and alerting, performance
optimization insights, and dashboard data aggregation with Prometheus export.
"""

from __future__ import annotations

import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import structlog

from agentcore.integration.portkey.metrics_collector import get_metrics_collector
from agentcore.integration.portkey.metrics_models import (
    AlertSeverity,
    DashboardData,
    MetricType,
    PerformanceAlert,
    PerformanceInsight,
    PerformanceLevel,
    PerformanceMetrics,
    PrometheusMetrics,
    ProviderPerformanceMetrics,
    RequestMetrics,
    SLAMetrics,
    SLAStatus,
)

logger = structlog.get_logger(__name__)


class PerformanceMonitor:
    """Performance monitoring with SLA tracking and alerts.

    Provides:
    - Real-time performance analysis
    - SLA monitoring and violation detection
    - Performance alerts and notifications
    - Optimization insights and recommendations
    - Dashboard data aggregation
    - Prometheus metrics export
    """

    def __init__(
        self,
        availability_target: float = 99.9,
        response_time_target_ms: int = 2000,
        success_rate_target: float = 99.5,
        alert_debounce_seconds: int = 300,
    ) -> None:
        """Initialize the performance monitor.

        Args:
            availability_target: Target availability percentage (default: 99.9%)
            response_time_target_ms: Target response time in ms (default: 2000ms)
            success_rate_target: Target success rate percentage (default: 99.5%)
            alert_debounce_seconds: Seconds between duplicate alerts (default: 300)
        """
        self._metrics_collector = get_metrics_collector()
        self._alerts: list[PerformanceAlert] = []
        self._insights: list[PerformanceInsight] = []
        self._last_alert_time: dict[str, datetime] = {}

        # SLA Targets
        self.availability_target = availability_target
        self.response_time_target_ms = response_time_target_ms
        self.success_rate_target = success_rate_target
        self.alert_debounce = timedelta(seconds=alert_debounce_seconds)

        logger.info(
            "performance_monitor_initialized",
            availability_target=availability_target,
            response_time_target_ms=response_time_target_ms,
            success_rate_target=success_rate_target,
        )

    def calculate_performance_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        provider_id: str | None = None,
        tenant_id: str | None = None,
    ) -> PerformanceMetrics:
        """Calculate aggregated performance metrics for a time period.

        Args:
            start_time: Period start time
            end_time: Period end time
            provider_id: Optional provider filter
            tenant_id: Optional tenant filter

        Returns:
            Aggregated performance metrics
        """
        # Get metrics from collector
        metrics = self._metrics_collector.get_metrics_history(
            provider_id=provider_id,
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
        )

        if not metrics:
            # Return empty metrics
            return PerformanceMetrics(
                period_start=start_time,
                period_end=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0.0,
                avg_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                max_latency_ms=0,
                min_latency_ms=0,
                requests_per_second=0.0,
                tokens_per_second=0.0,
                avg_tokens_per_request=0.0,
                total_cost=0.0,
                avg_cost_per_request=0.0,
                avg_cost_per_1k_tokens=0.0,
                error_rate=0.0,
                timeout_count=0,
                rate_limit_count=0,
                provider_error_count=0,
                provider_breakdown={},
                model_breakdown={},
            )

        # Calculate request statistics
        total_requests = len(metrics)
        successful_requests = len([m for m in metrics if m.success])
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests) * 100

        # Calculate latency statistics
        latencies = [m.total_latency_ms for m in metrics]
        avg_latency_ms = statistics.mean(latencies)
        p50_latency_ms = statistics.median(latencies)
        p95_latency_ms = self._calculate_percentile(latencies, 95)
        p99_latency_ms = self._calculate_percentile(latencies, 99)
        max_latency_ms = max(latencies)
        min_latency_ms = min(latencies)

        # Calculate throughput
        duration_seconds = (end_time - start_time).total_seconds()
        requests_per_second = total_requests / duration_seconds if duration_seconds > 0 else 0.0

        total_tokens = sum(m.total_tokens for m in metrics)
        tokens_per_second = total_tokens / duration_seconds if duration_seconds > 0 else 0.0
        avg_tokens_per_request = total_tokens / total_requests if total_requests > 0 else 0.0

        # Calculate cost statistics
        total_cost = sum(m.total_cost for m in metrics)
        avg_cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0
        avg_cost_per_1k_tokens = (
            (total_cost / total_tokens) * 1000 if total_tokens > 0 else 0.0
        )

        # Calculate error statistics
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0.0
        timeout_count = len(
            [m for m in metrics if m.error_type and "timeout" in m.error_type.lower()]
        )
        rate_limit_count = len(
            [m for m in metrics if m.error_type and "rate" in m.error_type.lower()]
        )
        provider_error_count = len(
            [m for m in metrics if m.error_type and "provider" in m.error_type.lower()]
        )

        # Provider and model breakdown
        provider_breakdown: dict[str, int] = defaultdict(int)
        for m in metrics:
            provider_breakdown[m.provider_id] += 1

        model_breakdown: dict[str, int] = defaultdict(int)
        for m in metrics:
            model_breakdown[m.model] += 1

        return PerformanceMetrics(
            period_start=start_time,
            period_end=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            avg_latency_ms=avg_latency_ms,
            p50_latency_ms=p50_latency_ms,
            p95_latency_ms=p95_latency_ms,
            p99_latency_ms=p99_latency_ms,
            max_latency_ms=max_latency_ms,
            min_latency_ms=min_latency_ms,
            requests_per_second=requests_per_second,
            tokens_per_second=tokens_per_second,
            avg_tokens_per_request=avg_tokens_per_request,
            total_cost=total_cost,
            avg_cost_per_request=avg_cost_per_request,
            avg_cost_per_1k_tokens=avg_cost_per_1k_tokens,
            error_rate=error_rate,
            timeout_count=timeout_count,
            rate_limit_count=rate_limit_count,
            provider_error_count=provider_error_count,
            provider_breakdown=dict(provider_breakdown),
            model_breakdown=dict(model_breakdown),
        )

    def calculate_sla_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        measurement_window_hours: int = 24,
    ) -> SLAMetrics:
        """Calculate SLA compliance metrics.

        Args:
            start_time: Period start time
            end_time: Period end time
            measurement_window_hours: SLA measurement window

        Returns:
            SLA compliance metrics
        """
        # Get performance metrics
        perf_metrics = self.calculate_performance_metrics(start_time, end_time)

        # Calculate actual metrics
        actual_availability = perf_metrics.success_rate
        actual_response_time_ms = perf_metrics.avg_latency_ms
        actual_success_rate = perf_metrics.success_rate

        # Determine compliance status
        availability_status = self._determine_sla_status(
            actual_availability,
            self.availability_target,
            5.0,  # warning threshold: 5% below target
        )

        response_time_status = self._determine_sla_status_inverted(
            actual_response_time_ms,
            self.response_time_target_ms,
            0.2,  # warning threshold: 20% above target
        )

        success_rate_status = self._determine_sla_status(
            actual_success_rate,
            self.success_rate_target,
            5.0,  # warning threshold: 5% below target
        )

        # Overall status (worst of all)
        status_priority = {
            SLAStatus.VIOLATED: 3,
            SLAStatus.WARNING: 2,
            SLAStatus.COMPLIANT: 1,
            SLAStatus.UNKNOWN: 0,
        }

        overall_status = max(
            [availability_status, response_time_status, success_rate_status],
            key=lambda s: status_priority.get(s, 0),
        )

        # Count violations
        availability_violations = (
            1 if availability_status == SLAStatus.VIOLATED else 0
        )
        response_time_violations = (
            1 if response_time_status == SLAStatus.VIOLATED else 0
        )
        success_rate_violations = (
            1 if success_rate_status == SLAStatus.VIOLATED else 0
        )

        sla_metrics = SLAMetrics(
            availability_target=self.availability_target,
            response_time_target_ms=self.response_time_target_ms,
            success_rate_target=self.success_rate_target,
            actual_availability=actual_availability,
            actual_response_time_ms=actual_response_time_ms,
            actual_success_rate=actual_success_rate,
            availability_status=availability_status,
            response_time_status=response_time_status,
            success_rate_status=success_rate_status,
            overall_status=overall_status,
            period_start=start_time,
            period_end=end_time,
            measurement_window_hours=measurement_window_hours,
            availability_violations=availability_violations,
            response_time_violations=response_time_violations,
            success_rate_violations=success_rate_violations,
        )

        # Generate alerts if violations detected
        if overall_status in (SLAStatus.WARNING, SLAStatus.VIOLATED):
            self._generate_sla_alerts(sla_metrics)

        return sla_metrics

    def calculate_provider_performance(
        self,
        provider_id: str,
        provider_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> ProviderPerformanceMetrics:
        """Calculate performance metrics for a specific provider.

        Args:
            provider_id: Provider identifier
            provider_name: Provider display name
            start_time: Period start time
            end_time: Period end time

        Returns:
            Provider-specific performance metrics
        """
        # Get performance metrics for provider
        perf_metrics = self.calculate_performance_metrics(
            start_time=start_time,
            end_time=end_time,
            provider_id=provider_id,
        )

        # Calculate provider scores
        availability_score = min(perf_metrics.success_rate / 100, 1.0)

        # Reliability score based on error rate and retries
        reliability_score = max(0.0, 1.0 - (perf_metrics.error_rate / 100))

        # Cost efficiency score (lower cost per 1k tokens = higher score)
        # Normalized against a baseline of $0.01 per 1k tokens
        baseline_cost = 0.01
        if perf_metrics.avg_cost_per_1k_tokens > 0:
            cost_efficiency_score = min(
                baseline_cost / perf_metrics.avg_cost_per_1k_tokens, 1.0
            )
        else:
            cost_efficiency_score = 1.0

        # Overall score (weighted average)
        overall_score = (
            availability_score * 0.4 +
            reliability_score * 0.4 +
            cost_efficiency_score * 0.2
        )

        # Determine health status
        if perf_metrics.error_rate > 20:
            health_status = "unhealthy"
        elif perf_metrics.error_rate > 5:
            health_status = "degraded"
        else:
            health_status = "healthy"

        # Determine performance level
        if overall_score >= 0.9:
            performance_level = PerformanceLevel.EXCELLENT
        elif overall_score >= 0.75:
            performance_level = PerformanceLevel.GOOD
        elif overall_score >= 0.5:
            performance_level = PerformanceLevel.DEGRADED
        else:
            performance_level = PerformanceLevel.POOR

        return ProviderPerformanceMetrics(
            provider_id=provider_id,
            provider_name=provider_name,
            performance_metrics=perf_metrics,
            availability_score=availability_score,
            reliability_score=reliability_score,
            cost_efficiency_score=cost_efficiency_score,
            overall_score=overall_score,
            health_status=health_status,
            performance_level=performance_level,
            last_updated=datetime.now(),
        )

    def generate_performance_insights(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[PerformanceInsight]:
        """Generate performance optimization insights.

        Analyzes historical data to identify optimization opportunities.

        Args:
            start_time: Analysis period start
            end_time: Analysis period end

        Returns:
            List of performance insights
        """
        insights: list[PerformanceInsight] = []

        # Get metrics
        metrics = self._metrics_collector.get_metrics_history(
            start_time=start_time,
            end_time=end_time,
        )

        if not metrics:
            return insights

        # Analyze latency spikes
        latency_insight = self._analyze_latency_patterns(metrics, start_time, end_time)
        if latency_insight:
            insights.append(latency_insight)

        # Analyze error patterns
        error_insight = self._analyze_error_patterns(metrics, start_time, end_time)
        if error_insight:
            insights.append(error_insight)

        # Analyze cost optimization opportunities
        cost_insight = self._analyze_cost_patterns(metrics, start_time, end_time)
        if cost_insight:
            insights.append(cost_insight)

        # Store insights
        self._insights.extend(insights)

        return insights

    def export_prometheus_metrics(self) -> PrometheusMetrics:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics
        """
        # Get last 5 minutes of metrics
        now = datetime.now()
        start_time = now - timedelta(minutes=5)

        metrics = self._metrics_collector.get_metrics_history(
            start_time=start_time,
            end_time=now,
        )

        if not metrics:
            return PrometheusMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                current_latency_ms=0.0,
                current_throughput=0.0,
                current_error_rate=0.0,
                latency_histogram={},
                token_count_histogram={},
                latency_summary={},
                labels={},
                timestamp=now,
            )

        # Calculate metrics
        total_requests = len(metrics)
        successful_requests = len([m for m in metrics if m.success])
        failed_requests = total_requests - successful_requests

        latencies = [m.total_latency_ms for m in metrics]
        current_latency_ms = statistics.mean(latencies) if latencies else 0.0

        duration = (now - start_time).total_seconds()
        current_throughput = total_requests / duration if duration > 0 else 0.0

        current_error_rate = (
            (failed_requests / total_requests) * 100 if total_requests > 0 else 0.0
        )

        # Build latency histogram (buckets: 0-100, 100-500, 500-1000, 1000-2000, 2000+)
        latency_histogram = {
            "0-100": len([l for l in latencies if l < 100]),
            "100-500": len([l for l in latencies if 100 <= l < 500]),
            "500-1000": len([l for l in latencies if 500 <= l < 1000]),
            "1000-2000": len([l for l in latencies if 1000 <= l < 2000]),
            "2000+": len([l for l in latencies if l >= 2000]),
        }

        # Build token count histogram
        token_counts = [m.total_tokens for m in metrics]
        token_count_histogram = {
            "0-100": len([t for t in token_counts if t < 100]),
            "100-500": len([t for t in token_counts if 100 <= t < 500]),
            "500-1000": len([t for t in token_counts if 500 <= t < 1000]),
            "1000-5000": len([t for t in token_counts if 1000 <= t < 5000]),
            "5000+": len([t for t in token_counts if t >= 5000]),
        }

        # Calculate latency percentiles
        latency_summary = {
            "p50": self._calculate_percentile(latencies, 50),
            "p95": self._calculate_percentile(latencies, 95),
            "p99": self._calculate_percentile(latencies, 99),
        }

        return PrometheusMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            current_latency_ms=current_latency_ms,
            current_throughput=current_throughput,
            current_error_rate=current_error_rate,
            latency_histogram=latency_histogram,
            token_count_histogram=token_count_histogram,
            latency_summary=latency_summary,
            labels={},
            timestamp=now,
        )

    def get_dashboard_data(self) -> DashboardData:
        """Get real-time dashboard data.

        Returns:
            Dashboard data with overview, SLA, and provider metrics
        """
        now = datetime.now()
        last_24h_start = now - timedelta(hours=24)

        # Calculate 24h metrics
        perf_24h = self.calculate_performance_metrics(last_24h_start, now)

        # Calculate current metrics (last 5 minutes)
        last_5m_start = now - timedelta(minutes=5)
        perf_5m = self.calculate_performance_metrics(last_5m_start, now)

        # Calculate SLA compliance
        sla_compliance = self.calculate_sla_metrics(last_24h_start, now, 24)

        # Get top providers
        top_providers = self._get_top_providers(last_24h_start, now)

        # Get active alerts
        active_alerts = [a for a in self._alerts if not a.acknowledged]

        # Get recent insights
        recent_insights = self._insights[-10:]  # Last 10 insights

        return DashboardData(
            total_requests_24h=perf_24h.total_requests,
            success_rate_24h=perf_24h.success_rate,
            avg_latency_24h=perf_24h.avg_latency_ms,
            total_cost_24h=perf_24h.total_cost,
            current_throughput=perf_5m.requests_per_second,
            current_latency_ms=perf_5m.avg_latency_ms,
            current_error_rate=perf_5m.error_rate,
            sla_compliance=sla_compliance,
            top_providers=top_providers,
            active_alerts=active_alerts,
            recent_insights=recent_insights,
            latency_timeseries=[],  # Would need time-series storage
            throughput_timeseries=[],
            error_rate_timeseries=[],
            last_updated=now,
        )

    def get_alerts(
        self,
        acknowledged: bool | None = None,
        resolved: bool | None = None,
    ) -> list[PerformanceAlert]:
        """Get performance alerts.

        Args:
            acknowledged: Filter by acknowledged status
            resolved: Filter by resolved status

        Returns:
            List of alerts matching filters
        """
        alerts = self._alerts

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if alert found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info("alert_acknowledged", alert_id=alert_id)
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if alert found and resolved
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info("alert_resolved", alert_id=alert_id)
                return True
        return False

    def _calculate_percentile(self, values: list[float | int], percentile: float) -> float:
        """Calculate percentile for a list of values.

        Args:
            values: List of numeric values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)

        return float(sorted_values[index])

    def _determine_sla_status(
        self,
        actual: float,
        target: float,
        warning_threshold_percent: float,
    ) -> SLAStatus:
        """Determine SLA compliance status (higher is better).

        Args:
            actual: Actual value
            target: Target value
            warning_threshold_percent: Warning threshold percentage

        Returns:
            SLA compliance status
        """
        if actual >= target:
            return SLAStatus.COMPLIANT

        violation_percent = ((target - actual) / target) * 100

        if violation_percent >= warning_threshold_percent:
            return SLAStatus.VIOLATED

        return SLAStatus.WARNING

    def _determine_sla_status_inverted(
        self,
        actual: float,
        target: float,
        warning_threshold_percent: float,
    ) -> SLAStatus:
        """Determine SLA compliance status (lower is better).

        Args:
            actual: Actual value
            target: Target value
            warning_threshold_percent: Warning threshold percentage

        Returns:
            SLA compliance status
        """
        if actual <= target:
            return SLAStatus.COMPLIANT

        violation_percent = ((actual - target) / target) * 100

        if violation_percent >= (warning_threshold_percent * 100):
            return SLAStatus.VIOLATED

        return SLAStatus.WARNING

    def _generate_sla_alerts(self, sla_metrics: SLAMetrics) -> None:
        """Generate alerts for SLA violations.

        Args:
            sla_metrics: SLA metrics with violations
        """
        # Availability alert
        if sla_metrics.availability_status != SLAStatus.COMPLIANT:
            self._create_alert(
                metric_type=MetricType.QUALITY,
                severity=(
                    AlertSeverity.CRITICAL
                    if sla_metrics.availability_status == SLAStatus.VIOLATED
                    else AlertSeverity.WARNING
                ),
                threshold_name="availability",
                threshold_value=sla_metrics.availability_target,
                actual_value=sla_metrics.actual_availability,
                title="Availability SLA Violation",
                message=f"Availability {sla_metrics.actual_availability:.2f}% below target {sla_metrics.availability_target}%",
            )

        # Response time alert
        if sla_metrics.response_time_status != SLAStatus.COMPLIANT:
            self._create_alert(
                metric_type=MetricType.LATENCY,
                severity=(
                    AlertSeverity.CRITICAL
                    if sla_metrics.response_time_status == SLAStatus.VIOLATED
                    else AlertSeverity.WARNING
                ),
                threshold_name="response_time",
                threshold_value=float(sla_metrics.response_time_target_ms),
                actual_value=sla_metrics.actual_response_time_ms,
                title="Response Time SLA Violation",
                message=f"Response time {sla_metrics.actual_response_time_ms:.0f}ms exceeds target {sla_metrics.response_time_target_ms}ms",
            )

        # Success rate alert
        if sla_metrics.success_rate_status != SLAStatus.COMPLIANT:
            self._create_alert(
                metric_type=MetricType.QUALITY,
                severity=(
                    AlertSeverity.CRITICAL
                    if sla_metrics.success_rate_status == SLAStatus.VIOLATED
                    else AlertSeverity.WARNING
                ),
                threshold_name="success_rate",
                threshold_value=sla_metrics.success_rate_target,
                actual_value=sla_metrics.actual_success_rate,
                title="Success Rate SLA Violation",
                message=f"Success rate {sla_metrics.actual_success_rate:.2f}% below target {sla_metrics.success_rate_target}%",
            )

    def _create_alert(
        self,
        metric_type: MetricType,
        severity: AlertSeverity,
        threshold_name: str,
        threshold_value: float,
        actual_value: float,
        title: str,
        message: str,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> PerformanceAlert:
        """Create a performance alert.

        Args:
            metric_type: Type of metric violated
            severity: Alert severity
            threshold_name: Name of threshold
            threshold_value: Threshold value
            actual_value: Actual measured value
            title: Alert title
            message: Alert message
            provider_id: Optional provider ID
            model: Optional model name

        Returns:
            Created alert
        """
        # Check debounce
        alert_key = f"{metric_type}_{threshold_name}_{provider_id or 'global'}"
        last_alert = self._last_alert_time.get(alert_key)

        if last_alert and (datetime.now() - last_alert) < self.alert_debounce:
            # Skip duplicate alert
            logger.debug(
                "alert_debounced",
                alert_key=alert_key,
                threshold_name=threshold_name,
            )
            return self._alerts[-1]  # Return last alert

        # Calculate violation percentage
        if threshold_value > 0:
            violation_percent = abs((actual_value - threshold_value) / threshold_value) * 100
        else:
            violation_percent = 0.0

        # Create alert
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            metric_type=metric_type,
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            actual_value=actual_value,
            violation_percent=violation_percent,
            provider_id=provider_id,
            model=model,
            timestamp=datetime.now(),
            title=title,
            message=message,
            acknowledged=False,
            resolved=False,
        )

        self._alerts.append(alert)
        self._last_alert_time[alert_key] = datetime.now()

        logger.warning(
            "performance_alert_created",
            alert_id=alert.alert_id,
            severity=severity,
            metric_type=metric_type,
            threshold_name=threshold_name,
        )

        return alert

    def _analyze_latency_patterns(
        self,
        metrics: list[RequestMetrics],
        start_time: datetime,
        end_time: datetime,
    ) -> PerformanceInsight | None:
        """Analyze latency patterns for optimization insights.

        Args:
            metrics: Request metrics to analyze
            start_time: Analysis period start
            end_time: Analysis period end

        Returns:
            Latency optimization insight or None
        """
        if not metrics:
            return None

        # Find requests with high latency (> 2 standard deviations)
        latencies = [m.total_latency_ms for m in metrics]
        mean_latency = statistics.mean(latencies)
        stdev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0

        threshold = mean_latency + (2 * stdev_latency)
        high_latency_metrics = [m for m in metrics if m.total_latency_ms > threshold]

        if not high_latency_metrics or len(high_latency_metrics) < (len(metrics) * 0.05):
            return None  # Less than 5% high latency requests

        # Analyze patterns
        affected_providers = list(set(m.provider_id for m in high_latency_metrics))
        affected_models = list(set(m.model for m in high_latency_metrics))

        # Calculate potential improvement
        avg_high_latency = statistics.mean(
            [m.total_latency_ms for m in high_latency_metrics]
        )
        potential_improvement = {
            "latency_reduction_ms": avg_high_latency - mean_latency,
            "affected_requests_percent": (len(high_latency_metrics) / len(metrics)) * 100,
        }

        return PerformanceInsight(
            insight_id=str(uuid.uuid4()),
            insight_type="latency_spike",
            title="High Latency Detected",
            description=f"{len(high_latency_metrics)} requests ({potential_improvement['affected_requests_percent']:.1f}%) experienced high latency (avg: {avg_high_latency:.0f}ms vs baseline: {mean_latency:.0f}ms)",
            impact_level="high",
            confidence=0.85,
            affected_requests=len(high_latency_metrics),
            potential_improvement=potential_improvement,
            recommendations=[
                "Consider switching to lower-latency providers",
                "Implement request timeout optimization",
                "Enable caching for similar requests",
            ],
            action_items=[
                f"Investigate {affected_providers[0] if affected_providers else 'provider'} performance",
                "Review network connectivity and routing",
                "Enable L1/L2 caching if not already active",
            ],
            affected_providers=affected_providers,
            affected_models=affected_models,
            time_period={"start": start_time, "end": end_time},
            generated_at=datetime.now(),
            metadata={},
        )

    def _analyze_error_patterns(
        self,
        metrics: list[RequestMetrics],
        start_time: datetime,
        end_time: datetime,
    ) -> PerformanceInsight | None:
        """Analyze error patterns for insights.

        Args:
            metrics: Request metrics to analyze
            start_time: Analysis period start
            end_time: Analysis period end

        Returns:
            Error pattern insight or None
        """
        if not metrics:
            return None

        error_metrics = [m for m in metrics if m.error_occurred]
        error_rate = (len(error_metrics) / len(metrics)) * 100

        if error_rate < 5.0:  # Less than 5% error rate
            return None

        # Analyze error types
        error_types: dict[str, int] = defaultdict(int)
        for m in error_metrics:
            if m.error_type:
                error_types[m.error_type] += 1

        most_common_error = max(error_types.items(), key=lambda x: x[1]) if error_types else ("unknown", 0)

        affected_providers = list(set(m.provider_id for m in error_metrics))

        return PerformanceInsight(
            insight_id=str(uuid.uuid4()),
            insight_type="error_pattern",
            title="Elevated Error Rate Detected",
            description=f"Error rate at {error_rate:.1f}% ({len(error_metrics)} errors). Most common: {most_common_error[0]} ({most_common_error[1]} occurrences)",
            impact_level="high" if error_rate > 10 else "medium",
            confidence=0.9,
            affected_requests=len(error_metrics),
            potential_improvement={
                "error_rate_reduction_percent": error_rate,
            },
            recommendations=[
                "Review provider health and status",
                "Implement automatic failover to backup providers",
                "Check rate limits and quotas",
            ],
            action_items=[
                f"Investigate {most_common_error[0]} errors",
                "Enable provider health monitoring",
                "Configure fallback provider chains",
            ],
            affected_providers=affected_providers,
            affected_models=[],
            time_period={"start": start_time, "end": end_time},
            generated_at=datetime.now(),
            metadata={"error_types": dict(error_types)},
        )

    def _analyze_cost_patterns(
        self,
        metrics: list[RequestMetrics],
        start_time: datetime,
        end_time: datetime,
    ) -> PerformanceInsight | None:
        """Analyze cost patterns for optimization.

        Args:
            metrics: Request metrics to analyze
            start_time: Analysis period start
            end_time: Analysis period end

        Returns:
            Cost optimization insight or None
        """
        if not metrics:
            return None

        # Calculate average cost per request
        total_cost = sum(m.total_cost for m in metrics)
        avg_cost = total_cost / len(metrics)

        # Find high-cost requests
        high_cost_metrics = [m for m in metrics if m.total_cost > (avg_cost * 2)]

        if not high_cost_metrics:
            return None

        # Calculate potential savings from caching
        cacheable_requests = len([m for m in metrics if not m.cache_hit])
        potential_savings = total_cost * 0.5  # Estimate 50% savings

        return PerformanceInsight(
            insight_id=str(uuid.uuid4()),
            insight_type="cost_optimization",
            title="Cost Optimization Opportunity",
            description=f"${total_cost:.2f} spent on {len(metrics)} requests. {cacheable_requests} cacheable requests could save up to ${potential_savings:.2f}",
            impact_level="medium",
            confidence=0.75,
            affected_requests=len(metrics),
            potential_improvement={
                "cost_savings_usd": potential_savings,
                "cost_reduction_percent": 50.0,
            },
            recommendations=[
                "Enable semantic caching for similar requests",
                "Route requests to lower-cost providers",
                "Optimize token usage and prompts",
            ],
            action_items=[
                "Enable L1/L2 caching",
                "Review provider pricing and routing rules",
                "Implement request batching where possible",
            ],
            affected_providers=[],
            affected_models=[],
            time_period={"start": start_time, "end": end_time},
            generated_at=datetime.now(),
            metadata={},
        )

    def _get_top_providers(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 5,
    ) -> list[ProviderPerformanceMetrics]:
        """Get top performing providers.

        Args:
            start_time: Period start time
            end_time: Period end time
            limit: Maximum number of providers to return

        Returns:
            List of top providers by performance
        """
        # Get all metrics
        metrics = self._metrics_collector.get_metrics_history(
            start_time=start_time,
            end_time=end_time,
        )

        if not metrics:
            return []

        # Group by provider
        provider_metrics: dict[str, list[RequestMetrics]] = defaultdict(list)
        for m in metrics:
            provider_metrics[m.provider_id].append(m)

        # Calculate performance for each provider
        provider_performance = []
        for provider_id, provider_requests in provider_metrics.items():
            if not provider_requests:
                continue

            provider_name = provider_requests[0].provider_name

            perf = self.calculate_provider_performance(
                provider_id=provider_id,
                provider_name=provider_name,
                start_time=start_time,
                end_time=end_time,
            )
            provider_performance.append(perf)

        # Sort by overall score
        provider_performance.sort(key=lambda p: p.overall_score, reverse=True)

        return provider_performance[:limit]


# Global performance monitor instance
_performance_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance.

    Returns:
        Global PerformanceMonitor instance
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
