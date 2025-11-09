"""
Performance Monitor (COMPASS ACE-1)

Stage-aware performance monitoring with batching and baseline comparison.
Implements CRITICAL <50ms latency requirement (p95).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.database.repositories import MetricsRepository
from agentcore.ace.models.ace_models import PerformanceBaseline, PerformanceMetrics

logger = structlog.get_logger()

# Stage validation
VALID_STAGES = {"planning", "execution", "reflection", "verification"}


class PerformanceMonitor:
    """
    Performance monitor with stage-aware metrics tracking (COMPASS ACE-1).

    Features:
    - Stage validation (planning, execution, reflection, verification)
    - Metrics batching (buffer 100 updates or 1 second)
    - <50ms latency target (p95) for record_metrics
    - Baseline comparison (stub implementation - ACE-010 will expand)
    - Async-first design

    Performance targets:
    - Metric computation: <100ms (p95)
    - Metrics update latency: <50ms (p95) - CRITICAL
    - Throughput: 10K+ metrics per hour
    """

    def __init__(
        self,
        get_session: callable,
        batch_size: int = 100,
        batch_timeout: float = 1.0,
    ) -> None:
        """
        Initialize PerformanceMonitor.

        Args:
            get_session: Async context manager that provides AsyncSession
            batch_size: Number of metrics to buffer before flush (default: 100)
            batch_timeout: Time in seconds before auto-flush (default: 1.0)

        Raises:
            ValueError: If batch_size < 1 or batch_timeout <= 0
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if batch_timeout <= 0:
            raise ValueError(f"batch_timeout must be > 0, got {batch_timeout}")

        self.get_session = get_session
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        # Batching state
        self._buffer: list[dict] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None
        self._last_flush_time = datetime.now(UTC)

        logger.info(
            "PerformanceMonitor initialized",
            batch_size=batch_size,
            batch_timeout=batch_timeout,
        )

    async def record_metrics(
        self,
        task_id: UUID,
        agent_id: str,
        stage: str,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Record performance metrics for a stage.

        Validates stage and buffers metrics for batching.
        Flushes when buffer reaches batch_size or batch_timeout.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            stage: Reasoning stage (planning, execution, reflection, verification)
            metrics: PerformanceMetrics instance with all metric values

        Raises:
            ValueError: If stage is invalid
        """
        # Validate stage (fail-fast)
        if stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of: {VALID_STAGES}"
            )

        # Prepare metric dict for buffering
        metric_dict = {
            "task_id": task_id,
            "agent_id": agent_id,
            "stage": stage,
            "stage_success_rate": metrics.stage_success_rate,
            "stage_error_rate": metrics.stage_error_rate,
            "stage_duration_ms": metrics.stage_duration_ms,
            "stage_action_count": metrics.stage_action_count,
            "overall_progress_velocity": metrics.overall_progress_velocity,
            "error_accumulation_rate": metrics.error_accumulation_rate,
            "context_staleness_score": metrics.context_staleness_score,
            "intervention_effectiveness": metrics.intervention_effectiveness,
            "baseline_delta": metrics.baseline_delta,
        }

        # Add to buffer (thread-safe)
        async with self._buffer_lock:
            self._buffer.append(metric_dict)
            buffer_length = len(self._buffer)

            logger.debug(
                "Metric buffered",
                task_id=str(task_id),
                agent_id=agent_id,
                stage=stage,
                buffer_length=buffer_length,
            )

            # Check if we need to flush
            should_flush = buffer_length >= self.batch_size

            # Start timeout task if not running
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._auto_flush())

        # Flush immediately if batch size reached
        if should_flush:
            await self._flush_buffer()

    async def get_current_metrics(
        self,
        task_id: UUID,
        agent_id: str,
    ) -> PerformanceMetrics | None:
        """
        Get latest metrics for task and agent.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier

        Returns:
            Latest PerformanceMetrics or None if not found
        """
        async with self.get_session() as session:
            db_metric = await MetricsRepository.get_latest_by_task(
                session, task_id, agent_id
            )

            if not db_metric:
                logger.debug(
                    "No metrics found",
                    task_id=str(task_id),
                    agent_id=agent_id,
                )
                return None

            # Convert DB model to Pydantic model
            return PerformanceMetrics(
                metric_id=db_metric.metric_id,
                task_id=db_metric.task_id,
                agent_id=db_metric.agent_id,
                stage=db_metric.stage,
                stage_success_rate=db_metric.stage_success_rate,
                stage_error_rate=db_metric.stage_error_rate,
                stage_duration_ms=db_metric.stage_duration_ms,
                stage_action_count=db_metric.stage_action_count,
                overall_progress_velocity=db_metric.overall_progress_velocity,
                error_accumulation_rate=db_metric.error_accumulation_rate,
                context_staleness_score=db_metric.context_staleness_score,
                intervention_effectiveness=db_metric.intervention_effectiveness,
                baseline_delta=db_metric.baseline_delta,
                recorded_at=db_metric.recorded_at,
            )

    async def get_baseline(
        self,
        agent_id: str,
        task_type: str | None,
        stage: str,
    ) -> PerformanceBaseline | None:
        """
        Get performance baseline for comparison.

        STUB IMPLEMENTATION: Returns None for now.
        ACE-010 (Baseline Tracker) will implement full baseline logic.

        Args:
            agent_id: Agent identifier
            task_type: Optional task type for task-specific baselines
            stage: Reasoning stage

        Returns:
            PerformanceBaseline or None (always None in stub)
        """
        logger.debug(
            "Baseline requested (stub - not implemented)",
            agent_id=agent_id,
            task_type=task_type,
            stage=stage,
        )
        # Stub: Will be implemented in ACE-010
        return None

    async def compute_baseline_delta(
        self,
        current_metrics: PerformanceMetrics,
        baseline: PerformanceBaseline | None,
    ) -> dict[str, float]:
        """
        Compute deviation from baseline.

        STUB IMPLEMENTATION: Returns empty dict for now.
        ACE-010 (Baseline Tracker) will implement full delta computation.

        Args:
            current_metrics: Current performance metrics
            baseline: Performance baseline (can be None)

        Returns:
            Dict of metric deltas (always empty dict in stub)
        """
        if baseline is None:
            logger.debug("No baseline available for delta computation")
            return {}

        # Stub: Will be implemented in ACE-010
        logger.debug(
            "Baseline delta computation (stub - not implemented)",
            agent_id=current_metrics.agent_id,
            stage=current_metrics.stage,
        )
        return {}

    async def update_baseline(
        self,
        agent_id: str,
        stage: str,
        metrics_history: list[PerformanceMetrics],
    ) -> PerformanceBaseline | None:
        """
        Update rolling baseline from recent metrics.

        STUB IMPLEMENTATION: Returns None for now.
        ACE-010 (Baseline Tracker) will implement full baseline updates.

        Args:
            agent_id: Agent identifier
            stage: Reasoning stage
            metrics_history: List of recent PerformanceMetrics

        Returns:
            Updated PerformanceBaseline or None (always None in stub)
        """
        logger.debug(
            "Baseline update requested (stub - not implemented)",
            agent_id=agent_id,
            stage=stage,
            sample_size=len(metrics_history),
        )
        # Stub: Will be implemented in ACE-010
        return None

    async def _auto_flush(self) -> None:
        """
        Auto-flush buffer after timeout.

        Runs continuously, checking buffer every batch_timeout seconds.
        """
        while True:
            await asyncio.sleep(self.batch_timeout)

            # Check if buffer has data
            async with self._buffer_lock:
                if len(self._buffer) > 0:
                    logger.debug(
                        "Auto-flush triggered by timeout",
                        buffer_length=len(self._buffer),
                        timeout=self.batch_timeout,
                    )

            # Flush buffer
            await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """
        Flush buffered metrics to database.

        Uses bulk insert for performance.
        """
        async with self._buffer_lock:
            if not self._buffer:
                return

            # Take snapshot of buffer and clear it
            metrics_to_flush = self._buffer.copy()
            self._buffer.clear()
            self._last_flush_time = datetime.now(UTC)

        # Bulk insert to database
        try:
            async with self.get_session() as session:
                count = await MetricsRepository.bulk_create(session, metrics_to_flush)
                await session.commit()

                logger.info(
                    "Metrics flushed to database",
                    count=count,
                    buffer_size=len(metrics_to_flush),
                )

        except Exception as e:
            logger.error(
                "Failed to flush metrics",
                error=str(e),
                buffer_size=len(metrics_to_flush),
            )
            # Re-add to buffer for retry (fail-fast disabled for buffering)
            async with self._buffer_lock:
                self._buffer.extend(metrics_to_flush)
            raise

    async def flush_and_shutdown(self) -> None:
        """
        Flush remaining metrics and shutdown gracefully.

        Call this during application shutdown to ensure no metrics are lost.
        """
        # Cancel auto-flush task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush any remaining metrics
        await self._flush_buffer()

        logger.info("PerformanceMonitor shutdown complete")
