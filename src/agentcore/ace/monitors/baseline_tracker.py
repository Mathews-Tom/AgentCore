"""
Baseline Tracker (COMPASS ACE-1 - ACE-010)

Baseline computation and drift detection for performance metrics.
Enables detection of performance degradation by establishing statistical baselines.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from scipy import stats  # For statistical significance testing
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.database.repositories import MetricsRepository
from agentcore.ace.models.ace_models import PerformanceBaseline, PerformanceMetrics

logger = structlog.get_logger()

# Constants
INITIAL_BASELINE_SAMPLE_SIZE = 10  # First 10 executions
ROLLING_BASELINE_UPDATE_FREQUENCY = 50  # Update every 50 executions
CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals
VALID_STAGES = {"planning", "execution", "reflection", "verification"}


class BaselineTracker:
    """
    Baseline tracker for performance metrics (COMPASS ACE-1 - ACE-010).

    Features:
    - Compute baseline from first 10 executions
    - Rolling baseline updates every 50 executions
    - Baseline comparison for degradation detection
    - Statistical drift detection with confidence intervals
    - Baseline reset mechanism for agent updates

    Statistical approach:
    - Mean and median for central tendency
    - Standard deviation for variance
    - 95% confidence intervals
    - T-test for statistical significance
    """

    def __init__(self, get_session: callable) -> None:
        """
        Initialize BaselineTracker.

        Args:
            get_session: Async context manager that provides AsyncSession
        """
        self.get_session = get_session

        # In-memory baseline cache (agent_id + stage + task_type -> baseline)
        # Format: {(agent_id, stage, task_type): PerformanceBaseline}
        self._baseline_cache: dict[tuple[str, str, str | None], PerformanceBaseline] = {}

        # Execution counters for rolling updates
        # Format: {(agent_id, stage, task_type): count}
        self._execution_counters: dict[tuple[str, str, str | None], int] = defaultdict(int)

        logger.info("BaselineTracker initialized")

    async def compute_baseline(
        self,
        agent_id: str,
        stage: str,
        task_type: str | None = None,
    ) -> PerformanceBaseline | None:
        """
        Compute baseline from first N executions.

        Uses first INITIAL_BASELINE_SAMPLE_SIZE (default 10) executions
        to establish initial baseline with mean, std dev, and confidence intervals.

        Args:
            agent_id: Agent identifier
            stage: Reasoning stage (planning, execution, reflection, verification)
            task_type: Optional task type for task-specific baselines

        Returns:
            PerformanceBaseline or None if insufficient data

        Raises:
            ValueError: If stage is invalid
        """
        # Validate stage
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}"
            )

        # Check cache first
        cache_key = (agent_id, stage, task_type)
        if cache_key in self._baseline_cache:
            logger.debug(
                "Baseline found in cache",
                agent_id=agent_id,
                stage=stage,
                task_type=task_type,
            )
            return self._baseline_cache[cache_key]

        # Fetch recent metrics from database
        async with self.get_session() as session:
            metrics = await MetricsRepository.list_by_agent_stage(
                session, agent_id, stage, limit=INITIAL_BASELINE_SAMPLE_SIZE
            )

        # Need at least INITIAL_BASELINE_SAMPLE_SIZE samples
        if len(metrics) < INITIAL_BASELINE_SAMPLE_SIZE:
            logger.debug(
                "Insufficient data for baseline",
                agent_id=agent_id,
                stage=stage,
                task_type=task_type,
                samples=len(metrics),
                required=INITIAL_BASELINE_SAMPLE_SIZE,
            )
            return None

        # Extract metric values
        success_rates = [m.stage_success_rate for m in metrics]
        error_rates = [m.stage_error_rate for m in metrics]
        durations = [float(m.stage_duration_ms) for m in metrics]
        action_counts = [float(m.stage_action_count) for m in metrics]

        # Compute statistics
        mean_success_rate = statistics.mean(success_rates)
        mean_error_rate = statistics.mean(error_rates)
        mean_duration_ms = statistics.mean(durations)
        mean_action_count = statistics.mean(action_counts)

        # Compute standard deviations
        std_dev: dict[str, float] = {}
        if len(success_rates) > 1:
            std_dev["success_rate"] = statistics.stdev(success_rates)
            std_dev["error_rate"] = statistics.stdev(error_rates)
            std_dev["duration_ms"] = statistics.stdev(durations)
            std_dev["action_count"] = statistics.stdev(action_counts)
        else:
            # Single sample - use 0 std dev
            std_dev = {
                "success_rate": 0.0,
                "error_rate": 0.0,
                "duration_ms": 0.0,
                "action_count": 0.0,
            }

        # Compute 95% confidence intervals using t-distribution
        confidence_interval: dict[str, tuple[float, float]] = {}
        n = len(metrics)

        if n > 1:
            # t-critical value for 95% confidence (two-tailed)
            t_critical = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, n - 1)

            # Confidence intervals for each metric
            for metric_name, values in [
                ("success_rate", success_rates),
                ("error_rate", error_rates),
                ("duration_ms", durations),
                ("action_count", action_counts),
            ]:
                mean_val = statistics.mean(values)
                std_err = statistics.stdev(values) / (n ** 0.5) if n > 1 else 0.0
                margin = t_critical * std_err

                confidence_interval[metric_name] = (
                    max(0.0, mean_val - margin),  # Lower bound (clamp to 0)
                    mean_val + margin,  # Upper bound
                )
        else:
            # Single sample - no confidence interval
            confidence_interval = {
                "success_rate": (mean_success_rate, mean_success_rate),
                "error_rate": (mean_error_rate, mean_error_rate),
                "duration_ms": (mean_duration_ms, mean_duration_ms),
                "action_count": (mean_action_count, mean_action_count),
            }

        # Create baseline
        baseline = PerformanceBaseline(
            agent_id=agent_id,
            stage=stage,
            task_type=task_type,
            mean_success_rate=mean_success_rate,
            mean_error_rate=mean_error_rate,
            mean_duration_ms=mean_duration_ms,
            mean_action_count=mean_action_count,
            std_dev=std_dev,
            confidence_interval=confidence_interval,
            sample_size=len(metrics),
            last_updated=datetime.now(UTC),
        )

        # Cache baseline
        self._baseline_cache[cache_key] = baseline

        logger.info(
            "Baseline computed",
            agent_id=agent_id,
            stage=stage,
            task_type=task_type,
            sample_size=len(metrics),
            mean_success_rate=mean_success_rate,
            mean_error_rate=mean_error_rate,
        )

        return baseline

    async def update_baseline(
        self,
        agent_id: str,
        stage: str,
        task_type: str | None = None,
        force: bool = False,
    ) -> PerformanceBaseline | None:
        """
        Update rolling baseline from recent metrics.

        Updates baseline every ROLLING_BASELINE_UPDATE_FREQUENCY (default 50)
        executions using exponential moving average.

        Args:
            agent_id: Agent identifier
            stage: Reasoning stage
            task_type: Optional task type for task-specific baselines
            force: Force update regardless of execution count

        Returns:
            Updated PerformanceBaseline or None if update not needed

        Raises:
            ValueError: If stage is invalid
        """
        # Validate stage
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}"
            )

        cache_key = (agent_id, stage, task_type)

        # Check if update needed
        self._execution_counters[cache_key] += 1
        execution_count = self._execution_counters[cache_key]

        if not force and execution_count < ROLLING_BASELINE_UPDATE_FREQUENCY:
            logger.debug(
                "Baseline update not needed yet",
                agent_id=agent_id,
                stage=stage,
                task_type=task_type,
                execution_count=execution_count,
                threshold=ROLLING_BASELINE_UPDATE_FREQUENCY,
            )
            return None

        # Reset counter
        self._execution_counters[cache_key] = 0

        # Fetch recent metrics (last 50 executions)
        async with self.get_session() as session:
            metrics = await MetricsRepository.list_by_agent_stage(
                session, agent_id, stage, limit=ROLLING_BASELINE_UPDATE_FREQUENCY
            )

        if len(metrics) < INITIAL_BASELINE_SAMPLE_SIZE:
            logger.warning(
                "Insufficient data for baseline update",
                agent_id=agent_id,
                stage=stage,
                samples=len(metrics),
            )
            return None

        # Recompute baseline with new data (same logic as compute_baseline)
        success_rates = [m.stage_success_rate for m in metrics]
        error_rates = [m.stage_error_rate for m in metrics]
        durations = [float(m.stage_duration_ms) for m in metrics]
        action_counts = [float(m.stage_action_count) for m in metrics]

        mean_success_rate = statistics.mean(success_rates)
        mean_error_rate = statistics.mean(error_rates)
        mean_duration_ms = statistics.mean(durations)
        mean_action_count = statistics.mean(action_counts)

        # Compute std dev
        std_dev: dict[str, float] = {}
        if len(success_rates) > 1:
            std_dev["success_rate"] = statistics.stdev(success_rates)
            std_dev["error_rate"] = statistics.stdev(error_rates)
            std_dev["duration_ms"] = statistics.stdev(durations)
            std_dev["action_count"] = statistics.stdev(action_counts)
        else:
            std_dev = {
                "success_rate": 0.0,
                "error_rate": 0.0,
                "duration_ms": 0.0,
                "action_count": 0.0,
            }

        # Compute confidence intervals
        confidence_interval: dict[str, tuple[float, float]] = {}
        n = len(metrics)

        if n > 1:
            t_critical = stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, n - 1)

            for metric_name, values in [
                ("success_rate", success_rates),
                ("error_rate", error_rates),
                ("duration_ms", durations),
                ("action_count", action_counts),
            ]:
                mean_val = statistics.mean(values)
                std_err = statistics.stdev(values) / (n ** 0.5) if n > 1 else 0.0
                margin = t_critical * std_err

                confidence_interval[metric_name] = (
                    max(0.0, mean_val - margin),
                    mean_val + margin,
                )
        else:
            confidence_interval = {
                "success_rate": (mean_success_rate, mean_success_rate),
                "error_rate": (mean_error_rate, mean_error_rate),
                "duration_ms": (mean_duration_ms, mean_duration_ms),
                "action_count": (mean_action_count, mean_action_count),
            }

        # Update baseline
        baseline = PerformanceBaseline(
            agent_id=agent_id,
            stage=stage,
            task_type=task_type,
            mean_success_rate=mean_success_rate,
            mean_error_rate=mean_error_rate,
            mean_duration_ms=mean_duration_ms,
            mean_action_count=mean_action_count,
            std_dev=std_dev,
            confidence_interval=confidence_interval,
            sample_size=len(metrics),
            last_updated=datetime.now(UTC),
        )

        # Update cache
        self._baseline_cache[cache_key] = baseline

        logger.info(
            "Baseline updated",
            agent_id=agent_id,
            stage=stage,
            task_type=task_type,
            sample_size=len(metrics),
            mean_success_rate=mean_success_rate,
        )

        return baseline

    async def detect_drift(
        self,
        current_metrics: PerformanceMetrics,
        baseline: PerformanceBaseline,
        significance_level: float = 0.05,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Detect baseline drift with statistical significance.

        Uses t-test to determine if current performance significantly
        deviates from baseline.

        Args:
            current_metrics: Current performance metrics
            baseline: Performance baseline for comparison
            significance_level: Statistical significance level (default 0.05 for 95% confidence)

        Returns:
            Tuple of (drift_detected: bool, drift_details: dict)
            drift_details contains:
                - drift_detected: bool
                - significant_metrics: list of metrics with significant drift
                - deviations: dict of metric -> deviation from mean
                - p_values: dict of metric -> p-value (if computable)

        Raises:
            ValueError: If metrics and baseline don't match (agent_id, stage)
        """
        # Validate inputs
        if current_metrics.agent_id != baseline.agent_id:
            raise ValueError(
                f"Agent ID mismatch: {current_metrics.agent_id} != {baseline.agent_id}"
            )
        if current_metrics.stage != baseline.stage:
            raise ValueError(
                f"Stage mismatch: {current_metrics.stage} != {baseline.stage}"
            )

        # Compute deviations from baseline mean
        deviations: dict[str, float] = {
            "success_rate": current_metrics.stage_success_rate - baseline.mean_success_rate,
            "error_rate": current_metrics.stage_error_rate - baseline.mean_error_rate,
            "duration_ms": current_metrics.stage_duration_ms - baseline.mean_duration_ms,
            "action_count": current_metrics.stage_action_count - baseline.mean_action_count,
        }

        # Check if current values fall outside confidence intervals
        significant_metrics: list[str] = []
        p_values: dict[str, float] = {}

        for metric_name in ["success_rate", "error_rate", "duration_ms", "action_count"]:
            # Get current value
            if metric_name == "success_rate":
                current_value = current_metrics.stage_success_rate
            elif metric_name == "error_rate":
                current_value = current_metrics.stage_error_rate
            elif metric_name == "duration_ms":
                current_value = float(current_metrics.stage_duration_ms)
            else:  # action_count
                current_value = float(current_metrics.stage_action_count)

            # Check confidence interval
            ci_low, ci_high = baseline.confidence_interval.get(
                metric_name, (float('-inf'), float('inf'))
            )

            # Drift detected if outside confidence interval
            if current_value < ci_low or current_value > ci_high:
                significant_metrics.append(metric_name)

                # Compute approximate p-value using normal distribution
                # (approximation since we only have one new sample)
                std_dev_val = baseline.std_dev.get(metric_name, 0.0)

                if std_dev_val > 0:
                    # Z-score
                    if metric_name == "success_rate":
                        mean_val = baseline.mean_success_rate
                    elif metric_name == "error_rate":
                        mean_val = baseline.mean_error_rate
                    elif metric_name == "duration_ms":
                        mean_val = baseline.mean_duration_ms
                    else:
                        mean_val = baseline.mean_action_count

                    z_score = abs((current_value - mean_val) / std_dev_val)
                    # Two-tailed p-value
                    p_value = 2 * (1 - stats.norm.cdf(z_score))
                    p_values[metric_name] = p_value
                else:
                    p_values[metric_name] = 1.0  # No variance - no drift

        drift_detected = len(significant_metrics) > 0

        drift_details = {
            "drift_detected": drift_detected,
            "significant_metrics": significant_metrics,
            "deviations": deviations,
            "p_values": p_values,
            "confidence_intervals": baseline.confidence_interval,
            "current_values": {
                "success_rate": current_metrics.stage_success_rate,
                "error_rate": current_metrics.stage_error_rate,
                "duration_ms": current_metrics.stage_duration_ms,
                "action_count": current_metrics.stage_action_count,
            },
        }

        if drift_detected:
            logger.warning(
                "Baseline drift detected",
                agent_id=current_metrics.agent_id,
                stage=current_metrics.stage,
                significant_metrics=significant_metrics,
                deviations=deviations,
            )
        else:
            logger.debug(
                "No baseline drift detected",
                agent_id=current_metrics.agent_id,
                stage=current_metrics.stage,
            )

        return drift_detected, drift_details

    async def reset_baseline(
        self,
        agent_id: str,
        stage: str,
        task_type: str | None = None,
    ) -> None:
        """
        Reset baseline for major agent updates.

        Clears cached baseline and execution counter, forcing
        recomputation on next baseline request.

        Args:
            agent_id: Agent identifier
            stage: Reasoning stage
            task_type: Optional task type

        Raises:
            ValueError: If stage is invalid
        """
        # Validate stage
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}"
            )

        cache_key = (agent_id, stage, task_type)

        # Remove from cache
        if cache_key in self._baseline_cache:
            del self._baseline_cache[cache_key]

        # Reset execution counter
        if cache_key in self._execution_counters:
            del self._execution_counters[cache_key]

        logger.info(
            "Baseline reset",
            agent_id=agent_id,
            stage=stage,
            task_type=task_type,
        )

    async def get_baseline(
        self,
        agent_id: str,
        stage: str,
        task_type: str | None = None,
    ) -> PerformanceBaseline | None:
        """
        Get performance baseline for comparison.

        Returns cached baseline or computes new one if not available.

        Args:
            agent_id: Agent identifier
            stage: Reasoning stage
            task_type: Optional task type

        Returns:
            PerformanceBaseline or None if insufficient data

        Raises:
            ValueError: If stage is invalid
        """
        cache_key = (agent_id, stage, task_type)

        # Return cached baseline if available
        if cache_key in self._baseline_cache:
            return self._baseline_cache[cache_key]

        # Otherwise compute new baseline
        return await self.compute_baseline(agent_id, stage, task_type)
