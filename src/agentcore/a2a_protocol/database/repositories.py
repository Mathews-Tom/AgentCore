"""
Database Repositories

Repository pattern for database operations on agents, tasks, and related entities.
"""

from datetime import datetime
from typing import List, Optional

import structlog
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.database.models import (
    AgentDB,
    TaskDB,
    AgentHealthMetricDB,
    MessageQueueDB,
    EventSubscriptionDB,
    SessionSnapshotDB,
)
from agentcore.a2a_protocol.models.agent import AgentCard, AgentStatus
from agentcore.a2a_protocol.models.task import TaskDefinition, TaskExecution, TaskStatus
from agentcore.a2a_protocol.models.session import SessionSnapshot, SessionState, SessionPriority

logger = structlog.get_logger()


class AgentRepository:
    """Repository for agent database operations."""

    @staticmethod
    async def create(session: AsyncSession, agent_card: AgentCard) -> AgentDB:
        """Create agent from AgentCard."""
        agent_db = AgentDB(
            id=agent_card.agent_id,
            name=agent_card.name,
            version=agent_card.version,
            status=agent_card.status,
            description=agent_card.description,
            capabilities=agent_card.capabilities,
            requirements=agent_card.requirements.model_dump(mode="json") if agent_card.requirements else None,
            agent_metadata=agent_card.metadata,
            endpoint=str(agent_card.endpoints[0].url) if agent_card.endpoints else None,
            current_load=0,
            max_load=10,
            created_at=agent_card.created_at,
            updated_at=agent_card.updated_at,
            last_seen=datetime.utcnow(),
        )
        session.add(agent_db)
        await session.flush()
        return agent_db

    @staticmethod
    async def get_by_id(session: AsyncSession, agent_id: str) -> Optional[AgentDB]:
        """Get agent by ID."""
        result = await session.execute(select(AgentDB).where(AgentDB.id == agent_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all(session: AsyncSession, status: Optional[AgentStatus] = None) -> List[AgentDB]:
        """Get all agents, optionally filtered by status."""
        query = select(AgentDB)
        if status:
            query = query.where(AgentDB.status == status)
        result = await session.execute(query.order_by(AgentDB.created_at.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def get_by_capabilities(
        session: AsyncSession,
        required_capabilities: List[str],
        status: Optional[AgentStatus] = AgentStatus.ACTIVE
    ) -> List[AgentDB]:
        """
        Get agents with required capabilities.

        Uses PostgreSQL JSON containment operator @>.
        """
        query = select(AgentDB).where(
            and_(
                AgentDB.status == status,
                AgentDB.capabilities.op('@>')(required_capabilities)  # JSON containment
            )
        )
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def update_status(
        session: AsyncSession,
        agent_id: str,
        status: AgentStatus
    ) -> bool:
        """Update agent status."""
        result = await session.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(status=status, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0

    @staticmethod
    async def update_load(
        session: AsyncSession,
        agent_id: str,
        load_delta: int
    ) -> bool:
        """Update agent load (increment or decrement)."""
        result = await session.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(
                current_load=AgentDB.current_load + load_delta,
                updated_at=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def update_last_seen(session: AsyncSession, agent_id: str) -> bool:
        """Update agent last_seen timestamp."""
        result = await session.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(last_seen=datetime.utcnow())
        )
        return result.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, agent_id: str) -> bool:
        """Delete agent."""
        result = await session.execute(delete(AgentDB).where(AgentDB.id == agent_id))
        return result.rowcount > 0

    @staticmethod
    async def count_by_status(session: AsyncSession) -> dict:
        """Count agents by status."""
        result = await session.execute(
            select(AgentDB.status, func.count(AgentDB.id))
            .group_by(AgentDB.status)
        )
        return {status.value: count for status, count in result.all()}


class TaskRepository:
    """Repository for task database operations."""

    @staticmethod
    async def create(session: AsyncSession, task_def: TaskDefinition) -> TaskDB:
        """Create task from TaskDefinition."""
        task_db = TaskDB(
            id=task_def.task_id,
            name=task_def.name,
            description=task_def.description,
            status=TaskStatus.PENDING,
            priority=task_def.priority,
            required_capabilities=task_def.required_capabilities,
            parameters=task_def.parameters,
            depends_on=task_def.depends_on,
            task_metadata=task_def.metadata,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(task_db)
        await session.flush()
        return task_db

    @staticmethod
    async def get_by_id(session: AsyncSession, task_id: str) -> Optional[TaskDB]:
        """Get task by ID."""
        result = await session.execute(select(TaskDB).where(TaskDB.id == task_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all(
        session: AsyncSession,
        status: Optional[TaskStatus] = None,
        agent_id: Optional[str] = None
    ) -> List[TaskDB]:
        """Get all tasks, optionally filtered by status and/or agent."""
        query = select(TaskDB)
        conditions = []
        if status:
            conditions.append(TaskDB.status == status)
        if agent_id:
            conditions.append(TaskDB.assigned_agent_id == agent_id)
        if conditions:
            query = query.where(and_(*conditions))
        result = await session.execute(query.order_by(TaskDB.priority.desc(), TaskDB.created_at))
        return list(result.scalars().all())

    @staticmethod
    async def assign_to_agent(
        session: AsyncSession,
        task_id: str,
        agent_id: str
    ) -> bool:
        """Assign task to agent."""
        result = await session.execute(
            update(TaskDB)
            .where(TaskDB.id == task_id)
            .values(
                assigned_agent_id=agent_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def update_status(
        session: AsyncSession,
        task_id: str,
        status: TaskStatus,
        result: Optional[dict] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update task status and optionally result/error."""
        values = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        if result is not None:
            values["result"] = result
        if error is not None:
            values["error"] = error
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            values["completed_at"] = datetime.utcnow()

        result_obj = await session.execute(
            update(TaskDB)
            .where(TaskDB.id == task_id)
            .values(**values)
        )
        return result_obj.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, task_id: str) -> bool:
        """Delete task."""
        result = await session.execute(delete(TaskDB).where(TaskDB.id == task_id))
        return result.rowcount > 0

    @staticmethod
    async def count_by_status(session: AsyncSession) -> dict:
        """Count tasks by status."""
        result = await session.execute(
            select(TaskDB.status, func.count(TaskDB.id))
            .group_by(TaskDB.status)
        )
        return {status.value: count for status, count in result.all()}

    @staticmethod
    async def get_pending_tasks(session: AsyncSession, limit: int = 100) -> List[TaskDB]:
        """Get pending tasks ordered by priority."""
        result = await session.execute(
            select(TaskDB)
            .where(TaskDB.status == TaskStatus.PENDING)
            .order_by(TaskDB.priority.desc(), TaskDB.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())


class HealthMetricRepository:
    """Repository for agent health metrics."""

    @staticmethod
    async def record_health_check(
        session: AsyncSession,
        agent_id: str,
        is_healthy: bool,
        response_time_ms: Optional[float] = None,
        status_code: Optional[int] = None,
        error_message: Optional[str] = None,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None
    ) -> AgentHealthMetricDB:
        """Record health check result."""
        metric = AgentHealthMetricDB(
            agent_id=agent_id,
            is_healthy=is_healthy,
            status_code=status_code,
            response_time_ms=response_time_ms,
            error_message=error_message,
            consecutive_failures=0,  # Will be updated by health service
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            checked_at=datetime.utcnow()
        )
        session.add(metric)
        await session.flush()
        return metric

    @staticmethod
    async def get_latest_metrics(
        session: AsyncSession,
        agent_id: str,
        limit: int = 10
    ) -> List[AgentHealthMetricDB]:
        """Get latest health metrics for agent."""
        result = await session.execute(
            select(AgentHealthMetricDB)
            .where(AgentHealthMetricDB.agent_id == agent_id)
            .order_by(AgentHealthMetricDB.checked_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_unhealthy_agents(session: AsyncSession) -> List[str]:
        """Get agent IDs with recent health failures."""
        # Get agents with failures in last check
        subquery = (
            select(
                AgentHealthMetricDB.agent_id,
                func.max(AgentHealthMetricDB.checked_at).label('last_check')
            )
            .group_by(AgentHealthMetricDB.agent_id)
            .subquery()
        )

        result = await session.execute(
            select(AgentHealthMetricDB.agent_id)
            .join(
                subquery,
                and_(
                    AgentHealthMetricDB.agent_id == subquery.c.agent_id,
                    AgentHealthMetricDB.checked_at == subquery.c.last_check
                )
            )
            .where(AgentHealthMetricDB.is_healthy == False)
        )
        return [row[0] for row in result.all()]

    @staticmethod
    async def cleanup_old_metrics(
        session: AsyncSession,
        days_to_keep: int = 7
    ) -> int:
        """Delete health metrics older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        result = await session.execute(
            delete(AgentHealthMetricDB)
            .where(AgentHealthMetricDB.checked_at < cutoff_date)
        )
        return result.rowcount


# Import timedelta for cleanup_old_metrics
from datetime import timedelta


class SessionRepository:
    """Repository for session snapshot database operations."""

    @staticmethod
    async def create(session: AsyncSession, snapshot: SessionSnapshot) -> SessionSnapshotDB:
        """Create session snapshot."""
        session_db = SessionSnapshotDB(
            session_id=snapshot.session_id,
            name=snapshot.name,
            description=snapshot.description,
            state=snapshot.state,
            priority=snapshot.priority,
            owner_agent=snapshot.owner_agent,
            participant_agents=snapshot.participant_agents,
            context=snapshot.context.model_dump(mode="json"),
            task_ids=snapshot.task_ids,
            artifact_ids=snapshot.artifact_ids,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
            expires_at=snapshot.expires_at,
            completed_at=snapshot.completed_at,
            timeout_seconds=snapshot.timeout_seconds,
            max_idle_seconds=snapshot.max_idle_seconds,
            tags=snapshot.tags,
            session_metadata=snapshot.metadata,
            checkpoint_interval_seconds=snapshot.checkpoint_interval_seconds,
            last_checkpoint_at=snapshot.last_checkpoint_at,
            checkpoint_count=snapshot.checkpoint_count,
        )
        session.add(session_db)
        await session.flush()
        return session_db

    @staticmethod
    async def get_by_id(session: AsyncSession, session_id: str) -> Optional[SessionSnapshotDB]:
        """Get session by ID."""
        result = await session.execute(
            select(SessionSnapshotDB).where(SessionSnapshotDB.session_id == session_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def update(session: AsyncSession, snapshot: SessionSnapshot) -> bool:
        """Update existing session snapshot."""
        result = await session.execute(
            update(SessionSnapshotDB)
            .where(SessionSnapshotDB.session_id == snapshot.session_id)
            .values(
                name=snapshot.name,
                description=snapshot.description,
                state=snapshot.state,
                priority=snapshot.priority,
                participant_agents=snapshot.participant_agents,
                context=snapshot.context.model_dump(mode="json"),
                task_ids=snapshot.task_ids,
                artifact_ids=snapshot.artifact_ids,
                updated_at=snapshot.updated_at,
                expires_at=snapshot.expires_at,
                completed_at=snapshot.completed_at,
                tags=snapshot.tags,
                session_metadata=snapshot.metadata,
                last_checkpoint_at=snapshot.last_checkpoint_at,
                checkpoint_count=snapshot.checkpoint_count,
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, session_id: str) -> bool:
        """Delete session snapshot."""
        result = await session.execute(
            delete(SessionSnapshotDB).where(SessionSnapshotDB.session_id == session_id)
        )
        return result.rowcount > 0

    @staticmethod
    async def list_by_state(
        session: AsyncSession,
        state: SessionState,
        limit: int = 100,
        offset: int = 0
    ) -> List[SessionSnapshotDB]:
        """List sessions by state with pagination."""
        result = await session.execute(
            select(SessionSnapshotDB)
            .where(SessionSnapshotDB.state == state)
            .order_by(SessionSnapshotDB.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_by_owner(
        session: AsyncSession,
        owner_agent: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[SessionSnapshotDB]:
        """List sessions by owner agent with pagination."""
        result = await session.execute(
            select(SessionSnapshotDB)
            .where(SessionSnapshotDB.owner_agent == owner_agent)
            .order_by(SessionSnapshotDB.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_expired(session: AsyncSession) -> List[SessionSnapshotDB]:
        """Get all expired sessions."""
        result = await session.execute(
            select(SessionSnapshotDB)
            .where(
                and_(
                    SessionSnapshotDB.expires_at.is_not(None),
                    SessionSnapshotDB.expires_at < datetime.utcnow(),
                    SessionSnapshotDB.state.notin_([SessionState.COMPLETED, SessionState.FAILED, SessionState.EXPIRED])
                )
            )
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_idle(session: AsyncSession, max_idle_seconds: int = 300) -> List[SessionSnapshotDB]:
        """Get sessions that have been idle longer than threshold."""
        idle_threshold = datetime.utcnow() - timedelta(seconds=max_idle_seconds)
        result = await session.execute(
            select(SessionSnapshotDB)
            .where(
                and_(
                    SessionSnapshotDB.state == SessionState.ACTIVE,
                    SessionSnapshotDB.updated_at < idle_threshold
                )
            )
        )
        return list(result.scalars().all())

    @staticmethod
    async def cleanup_old_sessions(
        session: AsyncSession,
        days_to_keep: int = 30
    ) -> int:
        """Delete terminal sessions older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        result = await session.execute(
            delete(SessionSnapshotDB)
            .where(
                and_(
                    SessionSnapshotDB.state.in_([SessionState.COMPLETED, SessionState.FAILED, SessionState.EXPIRED]),
                    SessionSnapshotDB.completed_at.is_not(None),
                    SessionSnapshotDB.completed_at < cutoff_date
                )
            )
        )
        return result.rowcount

    @staticmethod
    def to_snapshot(session_db: SessionSnapshotDB) -> SessionSnapshot:
        """Convert database model to SessionSnapshot."""
        from agentcore.a2a_protocol.models.session import SessionContext

        return SessionSnapshot(
            session_id=session_db.session_id,
            name=session_db.name,
            description=session_db.description,
            state=session_db.state,
            priority=session_db.priority,
            owner_agent=session_db.owner_agent,
            participant_agents=session_db.participant_agents or [],
            context=SessionContext.model_validate(session_db.context) if session_db.context else SessionContext(),
            task_ids=session_db.task_ids or [],
            artifact_ids=session_db.artifact_ids or [],
            created_at=session_db.created_at,
            updated_at=session_db.updated_at,
            expires_at=session_db.expires_at,
            completed_at=session_db.completed_at,
            timeout_seconds=session_db.timeout_seconds,
            max_idle_seconds=session_db.max_idle_seconds,
            tags=session_db.tags or [],
            metadata=session_db.session_metadata or {},
            checkpoint_interval_seconds=session_db.checkpoint_interval_seconds,
            last_checkpoint_at=session_db.last_checkpoint_at,
            checkpoint_count=session_db.checkpoint_count,
        )