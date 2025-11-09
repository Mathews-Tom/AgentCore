"""
InterventionEngine Core (COMPASS ACE-2)

Strategic intervention orchestration with priority-based queue management.
Implements COMPASS intervention precision target (85%+).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.database.repositories import InterventionRepository, MetricsRepository
from agentcore.ace.models.ace_models import (
    ExecutionStatus,
    InterventionRecord,
    InterventionType,
    PerformanceMetrics,
    TriggerType,
)

logger = structlog.get_logger()


@dataclass(order=True)
class QueuedIntervention:
    """Intervention queued for execution with priority ordering.

    Priority levels:
    - 0: Critical (performance_degradation, error_accumulation)
    - 1: Warning (context_staleness)
    - 2: Info (capability_mismatch)
    """

    priority: int
    task_id: UUID
    agent_id: str
    trigger_type: TriggerType
    trigger_signals: list[str]
    intervention_type: InterventionType
    intervention_rationale: str
    decision_confidence: float
    trigger_metric_id: UUID | None = None
    pre_metric_id: UUID | None = None
    queued_at: datetime = None  # type: ignore

    def __post_init__(self) -> None:
        """Set queued_at timestamp."""
        if self.queued_at is None:
            self.queued_at = datetime.now(UTC)


class InterventionEngine:
    """
    Intervention orchestration engine (COMPASS ACE-2).

    Features:
    - Priority-based queue (critical > warning > info)
    - Deduplication (avoid duplicate interventions for same task)
    - Cooldown period (configurable, default 60s between interventions)
    - Intervention execution tracking
    - History persistence
    - <100ms decision latency target (p95)

    Intervention precision target: 85%+ (COMPASS validated)
    """

    # Priority mapping
    PRIORITY_MAP = {
        TriggerType.PERFORMANCE_DEGRADATION: 0,  # Critical
        TriggerType.ERROR_ACCUMULATION: 0,  # Critical
        TriggerType.CONTEXT_STALENESS: 1,  # Warning
        TriggerType.CAPABILITY_MISMATCH: 2,  # Info
    }

    def __init__(
        self,
        get_session: Callable[[], AsyncSession],
        cooldown_seconds: int = 60,
        max_interventions_per_task: int = 5,
        queue_size: int = 50,
    ) -> None:
        """
        Initialize InterventionEngine.

        Args:
            get_session: Async context manager that provides AsyncSession
            cooldown_seconds: Minimum time between interventions for same task (default: 60)
            max_interventions_per_task: Maximum interventions per task (default: 5)
            queue_size: Maximum queue size (default: 50)

        Raises:
            ValueError: If parameters are invalid
        """
        if cooldown_seconds < 0:
            raise ValueError(f"cooldown_seconds must be >= 0, got {cooldown_seconds}")
        if max_interventions_per_task < 1:
            raise ValueError(
                f"max_interventions_per_task must be >= 1, got {max_interventions_per_task}"
            )
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")

        self.get_session = get_session
        self.cooldown_seconds = cooldown_seconds
        self.max_interventions_per_task = max_interventions_per_task

        # Priority queue (heapq-based via asyncio.PriorityQueue)
        self._queue: asyncio.PriorityQueue[QueuedIntervention] = asyncio.PriorityQueue(
            maxsize=queue_size
        )
        self._queue_lock = asyncio.Lock()

        # Deduplication tracking (task_id -> last intervention time)
        self._last_intervention: dict[UUID, datetime] = {}
        self._dedup_lock = asyncio.Lock()

        logger.info(
            "InterventionEngine initialized",
            cooldown_seconds=cooldown_seconds,
            max_interventions_per_task=max_interventions_per_task,
            queue_size=queue_size,
        )

    async def process_trigger(
        self,
        task_id: UUID,
        agent_id: str,
        trigger_type: TriggerType,
        trigger_signals: list[str],
        intervention_type: InterventionType,
        intervention_rationale: str,
        decision_confidence: float,
        trigger_metric_id: UUID | None = None,
        pre_metric_id: UUID | None = None,
    ) -> InterventionRecord:
        """
        Process intervention trigger and orchestrate execution.

        This is the main entry point for intervention processing.
        Validates cooldown, checks limits, queues, and executes intervention.

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            trigger_type: Type of trigger (performance_degradation, error_accumulation, etc.)
            trigger_signals: Specific signals that triggered intervention
            intervention_type: Type of intervention (replan, reflect, etc.)
            intervention_rationale: Reasoning for intervention
            decision_confidence: Confidence in decision (0-1)
            trigger_metric_id: Optional metric that triggered intervention
            pre_metric_id: Optional pre-intervention metric for comparison

        Returns:
            InterventionRecord with execution results

        Raises:
            ValueError: If intervention is in cooldown or limits exceeded
        """
        logger.info(
            "Processing intervention trigger",
            task_id=str(task_id),
            agent_id=agent_id,
            trigger_type=trigger_type.value,
            intervention_type=intervention_type.value,
        )

        # Check cooldown
        if not await self._check_cooldown(task_id):
            cooldown_remaining = await self._get_cooldown_remaining(task_id)
            raise ValueError(
                f"Intervention cooldown active for task {task_id}. "
                f"Remaining: {cooldown_remaining}s"
            )

        # Check intervention limit
        async with self.get_session() as session:
            count = await InterventionRepository.count_by_agent(session, agent_id)
            if count >= self.max_interventions_per_task:
                raise ValueError(
                    f"Maximum interventions per task exceeded ({self.max_interventions_per_task})"
                )

        # Queue intervention
        priority = self.PRIORITY_MAP.get(trigger_type, 2)
        queued = QueuedIntervention(
            priority=priority,
            task_id=task_id,
            agent_id=agent_id,
            trigger_type=trigger_type,
            trigger_signals=trigger_signals,
            intervention_type=intervention_type,
            intervention_rationale=intervention_rationale,
            decision_confidence=decision_confidence,
            trigger_metric_id=trigger_metric_id,
            pre_metric_id=pre_metric_id,
        )

        # Check for duplicates before queuing
        if await self._is_duplicate(queued):
            logger.warning(
                "Duplicate intervention detected, skipping",
                task_id=str(task_id),
                trigger_type=trigger_type.value,
            )
            raise ValueError(f"Duplicate intervention for task {task_id}")

        try:
            self._queue.put_nowait(queued)
            logger.debug(
                "Intervention queued",
                task_id=str(task_id),
                priority=priority,
                queue_size=self._queue.qsize(),
            )
        except asyncio.QueueFull:
            raise ValueError("Intervention queue is full")

        # Execute intervention
        intervention_record = await self._execute_queued_intervention(queued)

        # Update last intervention time
        async with self._dedup_lock:
            self._last_intervention[task_id] = datetime.now(UTC)

        return intervention_record

    async def get_intervention_history(
        self,
        agent_id: str,
        limit: int = 10,
    ) -> list[InterventionRecord]:
        """
        Get recent intervention history for an agent.

        Args:
            agent_id: Agent identifier
            limit: Maximum number of interventions to return (default: 10)

        Returns:
            List of InterventionRecord instances, most recent first
        """
        async with self.get_session() as session:
            interventions_db = await InterventionRepository.list_by_agent(
                session, agent_id, limit
            )

            # Convert ORM models to Pydantic models
            return [
                InterventionRecord(
                    intervention_id=db.intervention_id,
                    task_id=db.task_id,
                    agent_id=db.agent_id,
                    trigger_type=TriggerType(db.trigger_type),
                    trigger_signals=db.trigger_signals,
                    trigger_metric_id=db.trigger_metric_id,
                    intervention_type=InterventionType(db.intervention_type),
                    intervention_rationale=db.intervention_rationale,
                    decision_confidence=db.decision_confidence,
                    executed_at=db.executed_at,
                    execution_duration_ms=db.execution_duration_ms,
                    execution_status=ExecutionStatus(db.execution_status),
                    execution_error=db.execution_error,
                    pre_metric_id=db.pre_metric_id,
                    post_metric_id=db.post_metric_id,
                    effectiveness_delta=db.effectiveness_delta,
                    created_at=db.created_at,
                    updated_at=db.updated_at,
                )
                for db in interventions_db
            ]

    async def track_intervention_outcome(
        self,
        intervention_id: UUID,
        post_metrics: PerformanceMetrics,
    ) -> float:
        """
        Track effectiveness of intervention.

        Computes effectiveness delta by comparing post-intervention metrics
        with pre-intervention metrics.

        Args:
            intervention_id: Intervention identifier
            post_metrics: Performance metrics after intervention

        Returns:
            Effectiveness delta (-1 to 1, higher is better)

        Raises:
            ValueError: If intervention not found or pre-metrics missing
        """
        async with self.get_session() as session:
            # Get intervention record
            intervention_db = await InterventionRepository.get_by_id(session, intervention_id)
            if not intervention_db:
                raise ValueError(f"Intervention {intervention_id} not found")

            if not intervention_db.pre_metric_id:
                raise ValueError(f"Intervention {intervention_id} has no pre-metrics")

            # Get pre-intervention metrics
            pre_metrics_db = await MetricsRepository.get_by_id(
                session, intervention_db.pre_metric_id
            )
            if not pre_metrics_db:
                raise ValueError(
                    f"Pre-intervention metrics {intervention_db.pre_metric_id} not found"
                )

            # Compute effectiveness delta
            # Simple implementation: compare success rate and error rate
            success_rate_delta = (
                post_metrics.stage_success_rate - pre_metrics_db.stage_success_rate
            )
            error_rate_delta = -(
                post_metrics.stage_error_rate - pre_metrics_db.stage_error_rate
            )  # Negative because lower error rate is better

            # Average the deltas
            effectiveness_delta = (success_rate_delta + error_rate_delta) / 2.0

            # Clamp to [-1, 1]
            effectiveness_delta = max(-1.0, min(1.0, effectiveness_delta))

            # Update intervention record
            await InterventionRepository.update_outcome(
                session,
                intervention_id,
                post_metrics.metric_id,
                effectiveness_delta,
            )
            await session.commit()

            logger.info(
                "Intervention outcome tracked",
                intervention_id=str(intervention_id),
                effectiveness_delta=effectiveness_delta,
            )

            return effectiveness_delta

    async def get_queue_size(self) -> int:
        """Get current intervention queue size."""
        return self._queue.qsize()

    async def _check_cooldown(self, task_id: UUID) -> bool:
        """Check if intervention is allowed (not in cooldown)."""
        async with self._dedup_lock:
            last_time = self._last_intervention.get(task_id)
            if not last_time:
                return True

            elapsed = (datetime.now(UTC) - last_time).total_seconds()
            return elapsed >= self.cooldown_seconds

    async def _get_cooldown_remaining(self, task_id: UUID) -> int:
        """Get remaining cooldown time in seconds."""
        async with self._dedup_lock:
            last_time = self._last_intervention.get(task_id)
            if not last_time:
                return 0

            elapsed = (datetime.now(UTC) - last_time).total_seconds()
            remaining = max(0, self.cooldown_seconds - elapsed)
            return int(remaining)

    async def _is_duplicate(self, queued: QueuedIntervention) -> bool:
        """Check if intervention is duplicate (same task + type in queue)."""
        # Simple implementation: check if task_id is in recent interventions
        # More sophisticated deduplication can be added in ACE-016/ACE-017
        async with self._dedup_lock:
            last_time = self._last_intervention.get(queued.task_id)
            if not last_time:
                return False

            # Consider duplicate if within last 10 seconds
            elapsed = (datetime.now(UTC) - last_time).total_seconds()
            return elapsed < 10

    async def _execute_queued_intervention(
        self, queued: QueuedIntervention
    ) -> InterventionRecord:
        """
        Execute queued intervention.

        This is a stub implementation. Full execution will be implemented in ACE-018
        when Agent Runtime integration is ready.

        For now, it creates the intervention record and marks it as SUCCESS.
        """
        start_time = datetime.now(UTC)

        async with self.get_session() as session:
            # Create intervention record
            intervention_db = await InterventionRepository.create(
                session=session,
                task_id=queued.task_id,
                agent_id=queued.agent_id,
                trigger_type=queued.trigger_type.value,
                trigger_signals=queued.trigger_signals,
                intervention_type=queued.intervention_type.value,
                intervention_rationale=queued.intervention_rationale,
                decision_confidence=queued.decision_confidence,
                trigger_metric_id=queued.trigger_metric_id,
                pre_metric_id=queued.pre_metric_id,
            )

            # STUB: Mark as success for now (ACE-018 will implement actual execution)
            execution_duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
            await InterventionRepository.update_execution_status(
                session,
                intervention_db.intervention_id,
                ExecutionStatus.SUCCESS.value,
                execution_duration_ms,
            )

            await session.commit()

            logger.info(
                "Intervention executed (stub)",
                intervention_id=str(intervention_db.intervention_id),
                task_id=str(queued.task_id),
                intervention_type=queued.intervention_type.value,
                duration_ms=execution_duration_ms,
            )

            # Convert to Pydantic model
            return InterventionRecord(
                intervention_id=intervention_db.intervention_id,
                task_id=intervention_db.task_id,
                agent_id=intervention_db.agent_id,
                trigger_type=TriggerType(intervention_db.trigger_type),
                trigger_signals=intervention_db.trigger_signals,
                trigger_metric_id=intervention_db.trigger_metric_id,
                intervention_type=InterventionType(intervention_db.intervention_type),
                intervention_rationale=intervention_db.intervention_rationale,
                decision_confidence=intervention_db.decision_confidence,
                executed_at=intervention_db.executed_at,
                execution_duration_ms=execution_duration_ms,
                execution_status=ExecutionStatus.SUCCESS,
                execution_error=None,
                pre_metric_id=intervention_db.pre_metric_id,
                post_metric_id=None,
                effectiveness_delta=None,
                created_at=intervention_db.created_at,
                updated_at=intervention_db.updated_at,
            )
