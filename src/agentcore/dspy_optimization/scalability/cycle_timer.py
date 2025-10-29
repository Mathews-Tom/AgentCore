"""
Optimization cycle timing and performance monitoring

Tracks optimization execution time, validates against constraints,
and generates performance alerts for degradation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CycleMetrics:
    """Metrics for an optimization cycle"""

    optimization_id: str
    start_time: datetime
    end_time: datetime | None
    duration_seconds: float
    target_duration_seconds: float
    status: str
    iterations: int
    throughput: float  # iterations per second
    exceeded_target: bool


@dataclass
class PerformanceAlert:
    """Alert for performance degradation"""

    severity: AlertSeverity
    message: str
    optimization_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime


class OptimizationTimer:
    """
    Tracks optimization cycle timing and validates performance

    Key features:
    - Real-time cycle timing
    - <2h optimization validation
    - Performance degradation detection
    - Alert generation
    - Historical tracking
    """

    def __init__(
        self,
        target_duration_seconds: float = 7200,  # 2 hours
        warning_threshold: float = 0.8,  # Warn at 80% of target
        enable_alerts: bool = True,
    ) -> None:
        """
        Initialize optimization timer

        Args:
            target_duration_seconds: Target cycle duration (default: 2h)
            warning_threshold: Warning threshold percentage (default: 0.8)
            enable_alerts: Enable performance alerts
        """
        self.target_duration_seconds = target_duration_seconds
        self.warning_threshold = warning_threshold
        self.enable_alerts = enable_alerts
        self._active_cycles: dict[str, dict[str, Any]] = {}
        self._completed_cycles: list[CycleMetrics] = []
        self._alerts: list[PerformanceAlert] = []

    def start_cycle(self, optimization_id: str) -> None:
        """
        Start timing optimization cycle

        Args:
            optimization_id: Unique optimization identifier
        """
        if optimization_id in self._active_cycles:
            logger.warning(f"Cycle {optimization_id} already started")
            return

        self._active_cycles[optimization_id] = {
            "start_time": datetime.utcnow(),
            "start_timestamp": time.perf_counter(),
            "iterations": 0,
        }

        logger.info(f"Started timing cycle {optimization_id}")

    def update_progress(self, optimization_id: str, iterations: int) -> None:
        """
        Update cycle progress

        Args:
            optimization_id: Optimization identifier
            iterations: Number of iterations completed
        """
        if optimization_id not in self._active_cycles:
            logger.warning(f"Cycle {optimization_id} not started")
            return

        self._active_cycles[optimization_id]["iterations"] = iterations

        # Check elapsed time and warn if approaching limit
        elapsed = self.get_elapsed_time(optimization_id)
        if elapsed and self.enable_alerts:
            warning_time = self.target_duration_seconds * self.warning_threshold

            if elapsed >= warning_time and elapsed < self.target_duration_seconds:
                # Generate warning alert
                alert = PerformanceAlert(
                    severity=AlertSeverity.WARNING,
                    message=f"Optimization approaching time limit: {elapsed:.1f}s / {self.target_duration_seconds}s",
                    optimization_id=optimization_id,
                    metric_name="duration",
                    current_value=elapsed,
                    threshold_value=warning_time,
                    timestamp=datetime.utcnow(),
                )
                self._add_alert(alert)

            elif elapsed >= self.target_duration_seconds:
                # Generate critical alert
                alert = PerformanceAlert(
                    severity=AlertSeverity.CRITICAL,
                    message=f"Optimization exceeded time limit: {elapsed:.1f}s > {self.target_duration_seconds}s",
                    optimization_id=optimization_id,
                    metric_name="duration",
                    current_value=elapsed,
                    threshold_value=self.target_duration_seconds,
                    timestamp=datetime.utcnow(),
                )
                self._add_alert(alert)

    def end_cycle(self, optimization_id: str, status: str = "completed") -> CycleMetrics:
        """
        End timing cycle and compute metrics

        Args:
            optimization_id: Optimization identifier
            status: Final cycle status

        Returns:
            CycleMetrics with cycle statistics
        """
        if optimization_id not in self._active_cycles:
            raise ValueError(f"Cycle {optimization_id} not started")

        cycle_data = self._active_cycles.pop(optimization_id)
        start_time = cycle_data["start_time"]
        start_timestamp = cycle_data["start_timestamp"]
        iterations = cycle_data["iterations"]

        end_time = datetime.utcnow()
        duration = time.perf_counter() - start_timestamp

        # Calculate throughput
        throughput = iterations / duration if duration > 0 else 0.0

        # Check if exceeded target
        exceeded = duration > self.target_duration_seconds

        metrics = CycleMetrics(
            optimization_id=optimization_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            target_duration_seconds=self.target_duration_seconds,
            status=status,
            iterations=iterations,
            throughput=throughput,
            exceeded_target=exceeded,
        )

        self._completed_cycles.append(metrics)

        # Log result
        if exceeded:
            logger.warning(
                f"Cycle {optimization_id} exceeded target: {duration:.1f}s > {self.target_duration_seconds}s"
            )
        else:
            logger.info(
                f"Cycle {optimization_id} completed in {duration:.1f}s "
                f"({duration / self.target_duration_seconds * 100:.1f}% of target)"
            )

        return metrics

    def get_elapsed_time(self, optimization_id: str) -> float | None:
        """
        Get elapsed time for active cycle

        Args:
            optimization_id: Optimization identifier

        Returns:
            Elapsed time in seconds, or None if cycle not active
        """
        if optimization_id not in self._active_cycles:
            return None

        start_timestamp = self._active_cycles[optimization_id]["start_timestamp"]
        return time.perf_counter() - start_timestamp

    def check_time_remaining(self, optimization_id: str) -> float | None:
        """
        Get remaining time before target

        Args:
            optimization_id: Optimization identifier

        Returns:
            Remaining time in seconds, or None if cycle not active
        """
        elapsed = self.get_elapsed_time(optimization_id)
        if elapsed is None:
            return None

        return max(0.0, self.target_duration_seconds - elapsed)

    def is_approaching_limit(self, optimization_id: str) -> bool:
        """
        Check if cycle is approaching time limit

        Args:
            optimization_id: Optimization identifier

        Returns:
            True if elapsed time > warning threshold
        """
        elapsed = self.get_elapsed_time(optimization_id)
        if elapsed is None:
            return False

        warning_time = self.target_duration_seconds * self.warning_threshold
        return elapsed >= warning_time

    def get_active_cycles(self) -> list[str]:
        """
        Get list of active cycle IDs

        Returns:
            List of optimization IDs
        """
        return list(self._active_cycles.keys())

    def get_cycle_statistics(self) -> dict[str, Any]:
        """
        Get statistics across all completed cycles

        Returns:
            Dictionary with cycle statistics
        """
        if not self._completed_cycles:
            return {
                "total_cycles": 0,
                "avg_duration": 0.0,
                "avg_throughput": 0.0,
                "exceeded_count": 0,
                "success_rate": 0.0,
            }

        durations = [c.duration_seconds for c in self._completed_cycles]
        throughputs = [c.throughput for c in self._completed_cycles]
        exceeded = sum(1 for c in self._completed_cycles if c.exceeded_target)
        successful = sum(
            1 for c in self._completed_cycles if c.status == "completed"
        )

        return {
            "total_cycles": len(self._completed_cycles),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "avg_throughput": sum(throughputs) / len(throughputs),
            "exceeded_count": exceeded,
            "exceeded_rate": exceeded / len(self._completed_cycles),
            "success_rate": successful / len(self._completed_cycles),
        }

    def get_recent_alerts(
        self,
        severity: AlertSeverity | None = None,
        limit: int = 10,
    ) -> list[PerformanceAlert]:
        """
        Get recent performance alerts

        Args:
            severity: Filter by severity (optional)
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        alerts = self._alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts[:limit]

    def clear_alerts(self) -> None:
        """Clear all performance alerts"""
        self._alerts.clear()
        logger.info("Cleared all performance alerts")

    def _add_alert(self, alert: PerformanceAlert) -> None:
        """
        Add performance alert

        Args:
            alert: Alert to add
        """
        # Check if similar alert already exists
        similar = any(
            a.optimization_id == alert.optimization_id
            and a.metric_name == alert.metric_name
            and a.severity == alert.severity
            for a in self._alerts[-10:]  # Check last 10 alerts
        )

        if not similar:
            self._alerts.append(alert)
            logger.log(
                logging.WARNING if alert.severity == AlertSeverity.WARNING else logging.ERROR,
                f"Performance alert ({alert.severity.value}): {alert.message}",
            )

    def reset(self) -> None:
        """Reset all timing data"""
        self._active_cycles.clear()
        self._completed_cycles.clear()
        self._alerts.clear()
        logger.info("Reset optimization timer")
