"""
ACE Database Repositories

Repository pattern for ACE database operations.
Provides CRUD operations and data access patterns for ACE entities.
"""

from datetime import UTC, datetime
from uuid import UUID

import structlog
from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.database.ace_orm import (
    ContextDeltaDB,
    ContextPlaybookDB,
    EvolutionStatusDB,
    ExecutionTraceDB,
    PerformanceMetricsDB,
)
from agentcore.ace.models.ace_models import EvolutionStatusType

logger = structlog.get_logger()


class PlaybookRepository:
    """Repository for context playbook database operations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        agent_id: str,
        context: dict,
        metadata: dict | None = None,
    ) -> ContextPlaybookDB:
        """Create a new context playbook."""
        playbook = ContextPlaybookDB(
            agent_id=agent_id,
            context=context,
            playbook_metadata=metadata or {},
            version=1,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(playbook)
        await session.flush()
        return playbook

    @staticmethod
    async def get_by_id(
        session: AsyncSession, playbook_id: UUID
    ) -> ContextPlaybookDB | None:
        """Get playbook by ID."""
        result = await session.execute(
            select(ContextPlaybookDB).where(ContextPlaybookDB.playbook_id == playbook_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_agent_id(
        session: AsyncSession, agent_id: str
    ) -> ContextPlaybookDB | None:
        """Get the latest playbook for an agent."""
        result = await session.execute(
            select(ContextPlaybookDB)
            .where(ContextPlaybookDB.agent_id == agent_id)
            .order_by(desc(ContextPlaybookDB.updated_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update_context(
        session: AsyncSession,
        playbook_id: UUID,
        context: dict,
    ) -> bool:
        """Update playbook context and increment version."""
        result = await session.execute(
            update(ContextPlaybookDB)
            .where(ContextPlaybookDB.playbook_id == playbook_id)
            .values(
                context=context,
                version=ContextPlaybookDB.version + 1,
                updated_at=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, playbook_id: UUID) -> bool:
        """Delete a playbook."""
        result = await session.execute(
            delete(ContextPlaybookDB).where(
                ContextPlaybookDB.playbook_id == playbook_id
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def list_by_agent(
        session: AsyncSession, agent_id: str, limit: int = 10
    ) -> list[ContextPlaybookDB]:
        """List playbooks for an agent, ordered by most recent."""
        result = await session.execute(
            select(ContextPlaybookDB)
            .where(ContextPlaybookDB.agent_id == agent_id)
            .order_by(desc(ContextPlaybookDB.updated_at))
            .limit(limit)
        )
        return list(result.scalars().all())


class DeltaRepository:
    """Repository for context delta database operations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        playbook_id: UUID,
        changes: dict,
        confidence: float,
        reasoning: str,
    ) -> ContextDeltaDB:
        """Create a new context delta."""
        delta = ContextDeltaDB(
            playbook_id=playbook_id,
            changes=changes,
            confidence=confidence,
            reasoning=reasoning,
            generated_at=datetime.now(UTC),
            applied=False,
        )
        session.add(delta)
        await session.flush()
        return delta

    @staticmethod
    async def get_by_id(
        session: AsyncSession, delta_id: UUID
    ) -> ContextDeltaDB | None:
        """Get delta by ID."""
        result = await session.execute(
            select(ContextDeltaDB).where(ContextDeltaDB.delta_id == delta_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_unapplied_by_playbook(
        session: AsyncSession,
        playbook_id: UUID,
        confidence_threshold: float = 0.7,
    ) -> list[ContextDeltaDB]:
        """Get unapplied deltas for a playbook above confidence threshold."""
        result = await session.execute(
            select(ContextDeltaDB)
            .where(
                ContextDeltaDB.playbook_id == playbook_id,
                ContextDeltaDB.applied == False,  # noqa: E712
                ContextDeltaDB.confidence >= confidence_threshold,
            )
            .order_by(desc(ContextDeltaDB.confidence))
        )
        return list(result.scalars().all())

    @staticmethod
    async def mark_applied(
        session: AsyncSession, delta_id: UUID
    ) -> bool:
        """Mark a delta as applied."""
        result = await session.execute(
            update(ContextDeltaDB)
            .where(ContextDeltaDB.delta_id == delta_id)
            .values(applied=True, applied_at=datetime.now(UTC))
        )
        return result.rowcount > 0

    @staticmethod
    async def list_by_playbook(
        session: AsyncSession,
        playbook_id: UUID,
        limit: int = 50,
    ) -> list[ContextDeltaDB]:
        """List all deltas for a playbook."""
        result = await session.execute(
            select(ContextDeltaDB)
            .where(ContextDeltaDB.playbook_id == playbook_id)
            .order_by(desc(ContextDeltaDB.generated_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def count_by_playbook(
        session: AsyncSession, playbook_id: UUID, applied_only: bool = False
    ) -> int:
        """Count deltas for a playbook."""
        query = select(func.count(ContextDeltaDB.delta_id)).where(
            ContextDeltaDB.playbook_id == playbook_id
        )
        if applied_only:
            query = query.where(ContextDeltaDB.applied == True)  # noqa: E712
        result = await session.execute(query)
        return result.scalar_one()


class TraceRepository:
    """Repository for execution trace database operations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        agent_id: str,
        execution_time: float,
        success: bool,
        task_id: str | None = None,
        output_quality: float | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> ExecutionTraceDB:
        """Create a new execution trace."""
        trace = ExecutionTraceDB(
            agent_id=agent_id,
            task_id=task_id,
            execution_time=execution_time,
            success=success,
            output_quality=output_quality,
            error_message=error_message,
            trace_metadata=metadata or {},
            captured_at=datetime.now(UTC),
        )
        session.add(trace)
        await session.flush()
        return trace

    @staticmethod
    async def get_by_id(
        session: AsyncSession, trace_id: UUID
    ) -> ExecutionTraceDB | None:
        """Get trace by ID."""
        result = await session.execute(
            select(ExecutionTraceDB).where(ExecutionTraceDB.trace_id == trace_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_agent(
        session: AsyncSession,
        agent_id: str,
        limit: int = 100,
        success_only: bool = False,
    ) -> list[ExecutionTraceDB]:
        """List traces for an agent."""
        query = (
            select(ExecutionTraceDB)
            .where(ExecutionTraceDB.agent_id == agent_id)
            .order_by(desc(ExecutionTraceDB.captured_at))
            .limit(limit)
        )
        if success_only:
            query = query.where(ExecutionTraceDB.success == True)  # noqa: E712
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_recent_for_evolution(
        session: AsyncSession,
        agent_id: str,
        min_traces: int = 10,
    ) -> list[ExecutionTraceDB]:
        """Get recent traces for context evolution analysis."""
        result = await session.execute(
            select(ExecutionTraceDB)
            .where(ExecutionTraceDB.agent_id == agent_id)
            .order_by(desc(ExecutionTraceDB.captured_at))
            .limit(min_traces)
        )
        return list(result.scalars().all())

    @staticmethod
    async def count_by_agent(
        session: AsyncSession, agent_id: str, success_only: bool = False
    ) -> int:
        """Count traces for an agent."""
        query = select(func.count(ExecutionTraceDB.trace_id)).where(
            ExecutionTraceDB.agent_id == agent_id
        )
        if success_only:
            query = query.where(ExecutionTraceDB.success == True)  # noqa: E712
        result = await session.execute(query)
        return result.scalar_one()

    @staticmethod
    async def get_avg_execution_time(
        session: AsyncSession, agent_id: str
    ) -> float | None:
        """Get average execution time for successful traces."""
        result = await session.execute(
            select(func.avg(ExecutionTraceDB.execution_time)).where(
                ExecutionTraceDB.agent_id == agent_id,
                ExecutionTraceDB.success == True,  # noqa: E712
            )
        )
        return result.scalar_one()

    @staticmethod
    async def delete_old_traces(
        session: AsyncSession, agent_id: str, keep_count: int = 1000
    ) -> int:
        """Delete old traces, keeping only the most recent N."""
        # Get the cutoff trace_id
        cutoff_result = await session.execute(
            select(ExecutionTraceDB.trace_id)
            .where(ExecutionTraceDB.agent_id == agent_id)
            .order_by(desc(ExecutionTraceDB.captured_at))
            .offset(keep_count)
            .limit(1)
        )
        cutoff_trace = cutoff_result.scalar_one_or_none()

        if not cutoff_trace:
            return 0

        # Delete traces older than cutoff
        result = await session.execute(
            delete(ExecutionTraceDB).where(
                ExecutionTraceDB.agent_id == agent_id,
                ExecutionTraceDB.captured_at
                < select(ExecutionTraceDB.captured_at)
                .where(ExecutionTraceDB.trace_id == cutoff_trace)
                .scalar_subquery(),
            )
        )
        return result.rowcount


class EvolutionStatusRepository:
    """Repository for evolution status database operations."""

    @staticmethod
    async def create(
        session: AsyncSession,
        agent_id: str,
    ) -> EvolutionStatusDB:
        """Create a new evolution status record."""
        status = EvolutionStatusDB(
            agent_id=agent_id,
            pending_traces=0,
            deltas_generated=0,
            deltas_applied=0,
            total_cost=0.0,
            status=EvolutionStatusType.IDLE,
        )
        session.add(status)
        await session.flush()
        return status

    @staticmethod
    async def get_by_agent_id(
        session: AsyncSession, agent_id: str
    ) -> EvolutionStatusDB | None:
        """Get evolution status for an agent."""
        result = await session.execute(
            select(EvolutionStatusDB).where(EvolutionStatusDB.agent_id == agent_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_or_create(
        session: AsyncSession, agent_id: str
    ) -> EvolutionStatusDB:
        """Get evolution status or create if not exists."""
        status = await EvolutionStatusRepository.get_by_agent_id(session, agent_id)
        if not status:
            status = await EvolutionStatusRepository.create(session, agent_id)
        return status

    @staticmethod
    async def update_pending_traces(
        session: AsyncSession, agent_id: str, delta: int
    ) -> bool:
        """Increment or decrement pending traces count."""
        result = await session.execute(
            update(EvolutionStatusDB)
            .where(EvolutionStatusDB.agent_id == agent_id)
            .values(pending_traces=EvolutionStatusDB.pending_traces + delta)
        )
        return result.rowcount > 0

    @staticmethod
    async def record_evolution(
        session: AsyncSession,
        agent_id: str,
        deltas_generated: int,
        cost: float,
    ) -> bool:
        """Record an evolution event."""
        result = await session.execute(
            update(EvolutionStatusDB)
            .where(EvolutionStatusDB.agent_id == agent_id)
            .values(
                last_evolution=datetime.now(UTC),
                deltas_generated=EvolutionStatusDB.deltas_generated + deltas_generated,
                total_cost=EvolutionStatusDB.total_cost + cost,
                pending_traces=0,  # Reset pending traces after evolution
                status=EvolutionStatusType.IDLE,
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def increment_deltas_applied(
        session: AsyncSession, agent_id: str
    ) -> bool:
        """Increment deltas applied count."""
        result = await session.execute(
            update(EvolutionStatusDB)
            .where(EvolutionStatusDB.agent_id == agent_id)
            .values(deltas_applied=EvolutionStatusDB.deltas_applied + 1)
        )
        return result.rowcount > 0

    @staticmethod
    async def update_status(
        session: AsyncSession,
        agent_id: str,
        status: EvolutionStatusType,
    ) -> bool:
        """Update evolution status."""
        result = await session.execute(
            update(EvolutionStatusDB)
            .where(EvolutionStatusDB.agent_id == agent_id)
            .values(status=status)
        )
        return result.rowcount > 0

    @staticmethod
    async def list_all(
        session: AsyncSession, status: EvolutionStatusType | None = None
    ) -> list[EvolutionStatusDB]:
        """List all evolution status records."""
        query = select(EvolutionStatusDB).order_by(
            desc(EvolutionStatusDB.last_evolution)
        )
        if status:
            query = query.where(EvolutionStatusDB.status == status)
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_total_cost(session: AsyncSession) -> float:
        """Get total cost across all agents."""
        result = await session.execute(
            select(func.sum(EvolutionStatusDB.total_cost))
        )
        return result.scalar_one() or 0.0


class MetricsRepository:
    """Repository for performance metrics database operations (COMPASS ACE-1)."""

    @staticmethod
    async def create(
        session: AsyncSession,
        task_id: UUID,
        agent_id: str,
        stage: str,
        stage_success_rate: float,
        stage_error_rate: float,
        stage_duration_ms: int,
        stage_action_count: int,
        overall_progress_velocity: float,
        error_accumulation_rate: float,
        context_staleness_score: float,
        intervention_effectiveness: float | None = None,
        baseline_delta: dict | None = None,
    ) -> PerformanceMetricsDB:
        """Create a new performance metric record."""
        metric = PerformanceMetricsDB(
            task_id=task_id,
            agent_id=agent_id,
            stage=stage,
            stage_success_rate=stage_success_rate,
            stage_error_rate=stage_error_rate,
            stage_duration_ms=stage_duration_ms,
            stage_action_count=stage_action_count,
            overall_progress_velocity=overall_progress_velocity,
            error_accumulation_rate=error_accumulation_rate,
            context_staleness_score=context_staleness_score,
            intervention_effectiveness=intervention_effectiveness,
            baseline_delta=baseline_delta or {},
            recorded_at=datetime.now(UTC),
        )
        session.add(metric)
        await session.flush()
        return metric

    @staticmethod
    async def bulk_create(
        session: AsyncSession,
        metrics: list[dict],
    ) -> int:
        """Bulk insert performance metrics for batching.

        Args:
            session: Database session
            metrics: List of metric dictionaries with all required fields

        Returns:
            Number of metrics inserted
        """
        if not metrics:
            return 0

        # Add recorded_at timestamp if not present
        for metric in metrics:
            if "recorded_at" not in metric:
                metric["recorded_at"] = datetime.now(UTC)
            if "baseline_delta" not in metric:
                metric["baseline_delta"] = {}

        # Convert to ORM instances
        db_metrics = [PerformanceMetricsDB(**metric) for metric in metrics]
        session.add_all(db_metrics)
        await session.flush()

        return len(db_metrics)

    @staticmethod
    async def get_by_id(
        session: AsyncSession, metric_id: UUID
    ) -> PerformanceMetricsDB | None:
        """Get metric by ID."""
        result = await session.execute(
            select(PerformanceMetricsDB).where(
                PerformanceMetricsDB.metric_id == metric_id
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_latest_by_task(
        session: AsyncSession,
        task_id: UUID,
        agent_id: str,
    ) -> PerformanceMetricsDB | None:
        """Get latest metric for a task and agent."""
        result = await session.execute(
            select(PerformanceMetricsDB)
            .where(
                PerformanceMetricsDB.task_id == task_id,
                PerformanceMetricsDB.agent_id == agent_id,
            )
            .order_by(desc(PerformanceMetricsDB.recorded_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_task(
        session: AsyncSession,
        task_id: UUID,
        limit: int = 100,
    ) -> list[PerformanceMetricsDB]:
        """List metrics for a task."""
        result = await session.execute(
            select(PerformanceMetricsDB)
            .where(PerformanceMetricsDB.task_id == task_id)
            .order_by(desc(PerformanceMetricsDB.recorded_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_by_agent_stage(
        session: AsyncSession,
        agent_id: str,
        stage: str,
        limit: int = 50,
    ) -> list[PerformanceMetricsDB]:
        """List metrics for an agent and stage (for baseline computation)."""
        result = await session.execute(
            select(PerformanceMetricsDB)
            .where(
                PerformanceMetricsDB.agent_id == agent_id,
                PerformanceMetricsDB.stage == stage,
            )
            .order_by(desc(PerformanceMetricsDB.recorded_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def count_by_agent(
        session: AsyncSession, agent_id: str
    ) -> int:
        """Count metrics for an agent."""
        result = await session.execute(
            select(func.count(PerformanceMetricsDB.metric_id)).where(
                PerformanceMetricsDB.agent_id == agent_id
            )
        )
        return result.scalar_one()

    @staticmethod
    async def delete_old_metrics(
        session: AsyncSession,
        agent_id: str,
        keep_count: int = 1000,
    ) -> int:
        """Delete old metrics, keeping only the most recent N."""
        # Get the cutoff metric_id
        cutoff_result = await session.execute(
            select(PerformanceMetricsDB.metric_id)
            .where(PerformanceMetricsDB.agent_id == agent_id)
            .order_by(desc(PerformanceMetricsDB.recorded_at))
            .offset(keep_count)
            .limit(1)
        )
        cutoff_metric = cutoff_result.scalar_one_or_none()

        if not cutoff_metric:
            return 0

        # Delete metrics older than cutoff
        result = await session.execute(
            delete(PerformanceMetricsDB).where(
                PerformanceMetricsDB.agent_id == agent_id,
                PerformanceMetricsDB.recorded_at
                < select(PerformanceMetricsDB.recorded_at).where(
                    PerformanceMetricsDB.metric_id == cutoff_metric
                ),
            )
        )
        return result.rowcount


class InterventionRepository:
    """Repository for intervention record database operations (COMPASS ACE-2)."""

    @staticmethod
    async def create(
        session: AsyncSession,
        task_id: UUID,
        agent_id: str,
        trigger_type: str,
        trigger_signals: list[str],
        intervention_type: str,
        intervention_rationale: str,
        decision_confidence: float,
        trigger_metric_id: UUID | None = None,
        pre_metric_id: UUID | None = None,
    ) -> "InterventionRecordDB":
        """Create a new intervention record."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        intervention = InterventionRecordDB(
            task_id=task_id,
            agent_id=agent_id,
            trigger_type=trigger_type,
            trigger_signals=trigger_signals,
            trigger_metric_id=trigger_metric_id,
            intervention_type=intervention_type,
            intervention_rationale=intervention_rationale,
            decision_confidence=decision_confidence,
            pre_metric_id=pre_metric_id,
            executed_at=datetime.now(UTC),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(intervention)
        await session.flush()
        return intervention

    @staticmethod
    async def get_by_id(
        session: AsyncSession, intervention_id: UUID
    ) -> "InterventionRecordDB | None":
        """Get intervention by ID."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        result = await session.execute(
            select(InterventionRecordDB).where(
                InterventionRecordDB.intervention_id == intervention_id
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_agent(
        session: AsyncSession,
        agent_id: str,
        limit: int = 10,
    ) -> list["InterventionRecordDB"]:
        """List interventions for an agent, ordered by most recent."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        result = await session.execute(
            select(InterventionRecordDB)
            .where(InterventionRecordDB.agent_id == agent_id)
            .order_by(desc(InterventionRecordDB.executed_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_by_task(
        session: AsyncSession,
        task_id: UUID,
        limit: int = 10,
    ) -> list["InterventionRecordDB"]:
        """List interventions for a task, ordered by most recent."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        result = await session.execute(
            select(InterventionRecordDB)
            .where(InterventionRecordDB.task_id == task_id)
            .order_by(desc(InterventionRecordDB.executed_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_recent_for_task(
        session: AsyncSession,
        task_id: UUID,
        agent_id: str,
        minutes: int = 60,
    ) -> list["InterventionRecordDB"]:
        """Get recent interventions for a task within time window (for cooldown checks)."""
        from datetime import timedelta

        from agentcore.ace.database.ace_orm import InterventionRecordDB

        cutoff_time = datetime.now(UTC) - timedelta(minutes=minutes)
        result = await session.execute(
            select(InterventionRecordDB)
            .where(
                InterventionRecordDB.task_id == task_id,
                InterventionRecordDB.agent_id == agent_id,
                InterventionRecordDB.executed_at >= cutoff_time,
            )
            .order_by(desc(InterventionRecordDB.executed_at))
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_execution_status(
        session: AsyncSession,
        intervention_id: UUID,
        execution_status: str,
        execution_duration_ms: int,
        execution_error: str | None = None,
    ) -> bool:
        """Update intervention execution status."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        result = await session.execute(
            update(InterventionRecordDB)
            .where(InterventionRecordDB.intervention_id == intervention_id)
            .values(
                execution_status=execution_status,
                execution_duration_ms=execution_duration_ms,
                execution_error=execution_error,
                updated_at=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def update_outcome(
        session: AsyncSession,
        intervention_id: UUID,
        post_metric_id: UUID,
        effectiveness_delta: float,
    ) -> bool:
        """Update intervention outcome metrics."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        result = await session.execute(
            update(InterventionRecordDB)
            .where(InterventionRecordDB.intervention_id == intervention_id)
            .values(
                post_metric_id=post_metric_id,
                effectiveness_delta=effectiveness_delta,
                updated_at=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def count_by_agent(
        session: AsyncSession,
        agent_id: str,
        intervention_type: str | None = None,
    ) -> int:
        """Count interventions for an agent, optionally filtered by type."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        query = select(func.count(InterventionRecordDB.intervention_id)).where(
            InterventionRecordDB.agent_id == agent_id
        )
        if intervention_type:
            query = query.where(InterventionRecordDB.intervention_type == intervention_type)
        result = await session.execute(query)
        return result.scalar_one()

    @staticmethod
    async def get_effectiveness_stats(
        session: AsyncSession,
        agent_id: str,
        intervention_type: str | None = None,
    ) -> dict[str, float]:
        """Get intervention effectiveness statistics for an agent."""
        from agentcore.ace.database.ace_orm import InterventionRecordDB

        query = select(
            func.avg(InterventionRecordDB.effectiveness_delta).label("mean"),
            func.count(InterventionRecordDB.intervention_id).label("count"),
        ).where(
            InterventionRecordDB.agent_id == agent_id,
            InterventionRecordDB.effectiveness_delta.isnot(None),
        )

        if intervention_type:
            query = query.where(InterventionRecordDB.intervention_type == intervention_type)

        result = await session.execute(query)
        row = result.one_or_none()

        if not row or row.count == 0:
            return {"mean_effectiveness": 0.0, "total_interventions": 0}

        return {
            "mean_effectiveness": float(row.mean) if row.mean else 0.0,
            "total_interventions": int(row.count),
        }
