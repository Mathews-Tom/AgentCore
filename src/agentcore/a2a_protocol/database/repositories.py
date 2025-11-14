"""
Database Repositories

Repository pattern for database operations on agents, tasks, and related entities.
"""

from datetime import UTC, datetime

import structlog
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.database.models import (
    AgentDB,
    AgentHealthMetricDB,
    EventSubscriptionDB,
    MessageQueueDB,
    SessionSnapshotDB,
    TaskDB,
)
from agentcore.a2a_protocol.models.agent import AgentCard, AgentStatus
from agentcore.a2a_protocol.models.session import (
    SessionPriority,
    SessionSnapshot,
    SessionState,
)
from agentcore.a2a_protocol.models.task import TaskDefinition, TaskExecution, TaskStatus

logger = structlog.get_logger()


class AgentRepository:
    """Repository for agent database operations."""

    @staticmethod
    async def create(session: AsyncSession, agent_card: AgentCard) -> AgentDB:
        """Create agent from AgentCard."""
        agent_db = AgentDB(
            id=agent_card.agent_id,
            name=agent_card.agent_name,
            version=agent_card.agent_version,
            status=agent_card.status,
            description=agent_card.description,
            capabilities=agent_card.capabilities,
            requirements=agent_card.requirements.model_dump(mode="json")
            if agent_card.requirements
            else None,
            agent_metadata=agent_card.metadata,
            endpoint=str(agent_card.endpoints[0].url) if agent_card.endpoints else None,
            current_load=0,
            max_load=10,
            created_at=agent_card.created_at,
            updated_at=agent_card.updated_at,
            last_seen=datetime.now(UTC),
        )
        session.add(agent_db)
        await session.flush()
        return agent_db

    @staticmethod
    async def get_by_id(session: AsyncSession, agent_id: str) -> AgentDB | None:
        """Get agent by ID."""
        result = await session.execute(select(AgentDB).where(AgentDB.id == agent_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all(
        session: AsyncSession, status: AgentStatus | None = None
    ) -> list[AgentDB]:
        """Get all agents, optionally filtered by status."""
        query = select(AgentDB)
        if status:
            query = query.where(AgentDB.status == status)
        result = await session.execute(query.order_by(AgentDB.created_at.desc()))
        return list(result.scalars().all())

    @staticmethod
    async def get_by_capabilities(
        session: AsyncSession,
        required_capabilities: list[str],
        status: AgentStatus | None = AgentStatus.ACTIVE,
    ) -> list[AgentDB]:
        """
        Get agents with required capabilities.

        Uses PostgreSQL JSON containment operator @>.
        """
        query = select(AgentDB).where(
            and_(
                AgentDB.status == status,
                AgentDB.capabilities.op("@>")(
                    required_capabilities
                ),  # JSON containment
            )
        )
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def update_status(
        session: AsyncSession, agent_id: str, status: AgentStatus
    ) -> bool:
        """Update agent status."""
        result = await session.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(status=status, updated_at=datetime.now(UTC))
        )
        return result.rowcount > 0

    @staticmethod
    async def update_load(
        session: AsyncSession, agent_id: str, load_delta: int
    ) -> bool:
        """Update agent load (increment or decrement)."""
        result = await session.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(
                current_load=AgentDB.current_load + load_delta,
                updated_at=datetime.now(UTC),
                last_seen=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def update_last_seen(session: AsyncSession, agent_id: str) -> bool:
        """Update agent last_seen timestamp."""
        result = await session.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(last_seen=datetime.now(UTC))
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
            select(AgentDB.status, func.count(AgentDB.id)).group_by(AgentDB.status)
        )
        return {status.value: count for status, count in result.all()}

    @staticmethod
    async def update_embedding(
        session: AsyncSession, agent_id: str, embedding: list[float]
    ) -> bool:
        """
        Update agent capability embedding (A2A-016).

        Args:
            session: Database session
            agent_id: Agent ID
            embedding: Vector embedding (384-dim for all-MiniLM-L6-v2)

        Returns:
            True if update successful
        """
        result = await session.execute(
            update(AgentDB)
            .where(AgentDB.id == agent_id)
            .values(capability_embedding=embedding, updated_at=datetime.now(UTC))
        )
        return result.rowcount > 0

    @staticmethod
    async def semantic_search(
        session: AsyncSession,
        query_embedding: list[float],
        similarity_threshold: float = 0.75,
        limit: int = 10,
        status: AgentStatus | None = AgentStatus.ACTIVE,
    ) -> list[tuple]:
        """
        Semantic search for agents using vector similarity (A2A-016).

        Uses cosine similarity via pgvector <=> operator.

        Args:
            session: Database session
            query_embedding: Query vector embedding
            similarity_threshold: Minimum similarity score (0.0-1.0)
            limit: Maximum number of results
            status: Filter by agent status

        Returns:
            List of tuples (agent, similarity_score)
        """
        try:
            # Import pgvector operator
            from pgvector.sqlalchemy import Vector

            # Cosine distance operator: <=> (smaller is more similar)
            # Convert to similarity: 1 - distance
            query = (
                select(
                    AgentDB,
                    (
                        1
                        - AgentDB.capability_embedding.cosine_distance(query_embedding)
                    ).label("similarity"),
                )
                .where(
                    and_(
                        AgentDB.status == status if status else True,
                        AgentDB.capability_embedding.isnot(None),
                        # Filter by similarity threshold
                        (
                            1
                            - AgentDB.capability_embedding.cosine_distance(
                                query_embedding
                            )
                        )
                        >= similarity_threshold,
                    )
                )
                .order_by(AgentDB.capability_embedding.cosine_distance(query_embedding))
                .limit(limit)
            )

            result = await session.execute(query)
            return list(result.all())

        except ImportError:
            logger.warning("pgvector not available, falling back to exact match")
            # Fallback to regular capability matching
            agents = await AgentRepository.get_all(session, status=status)
            return [(agent, 0.0) for agent in agents[:limit]]


class TaskRepository:
    """Repository for task database operations."""

    @staticmethod
    async def create(session: AsyncSession, task_def: TaskDefinition) -> TaskDB:
        """Create task from TaskDefinition."""
        task_db = TaskDB(
            id=task_def.task_id,
            name=task_def.title,
            description=task_def.description,
            status=TaskStatus.PENDING,
            priority=task_def.priority,
            required_capabilities=getattr(task_def.requirements, "capabilities", [])
            if task_def.requirements
            else [],
            parameters=task_def.parameters,
            depends_on=task_def.dependencies,
            task_metadata={},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(task_db)
        await session.flush()
        return task_db

    @staticmethod
    async def get_by_id(session: AsyncSession, task_id: str) -> TaskDB | None:
        """Get task by ID."""
        result = await session.execute(select(TaskDB).where(TaskDB.id == task_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all(
        session: AsyncSession,
        status: TaskStatus | None = None,
        agent_id: str | None = None,
    ) -> list[TaskDB]:
        """Get all tasks, optionally filtered by status and/or agent."""
        query = select(TaskDB)
        conditions = []
        if status:
            conditions.append(TaskDB.status == status)
        if agent_id:
            conditions.append(TaskDB.assigned_agent_id == agent_id)
        if conditions:
            query = query.where(and_(*conditions))
        result = await session.execute(
            query.order_by(TaskDB.priority.desc(), TaskDB.created_at)
        )
        return list(result.scalars().all())

    @staticmethod
    async def assign_to_agent(
        session: AsyncSession, task_id: str, agent_id: str
    ) -> bool:
        """Assign task to agent."""
        result = await session.execute(
            update(TaskDB)
            .where(TaskDB.id == task_id)
            .values(
                assigned_agent_id=agent_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def update_status(
        session: AsyncSession,
        task_id: str,
        status: TaskStatus,
        result: dict | None = None,
        error: str | None = None,
    ) -> bool:
        """Update task status and optionally result/error."""
        values = {"status": status, "updated_at": datetime.now(UTC)}
        if result is not None:
            values["result"] = result
        if error is not None:
            values["error"] = error
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            values["completed_at"] = datetime.now(UTC)

        result_obj = await session.execute(
            update(TaskDB).where(TaskDB.id == task_id).values(**values)
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
            select(TaskDB.status, func.count(TaskDB.id)).group_by(TaskDB.status)
        )
        return {status.value: count for status, count in result.all()}

    @staticmethod
    async def get_pending_tasks(
        session: AsyncSession, limit: int = 100
    ) -> list[TaskDB]:
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
        response_time_ms: float | None = None,
        status_code: int | None = None,
        error_message: str | None = None,
        cpu_percent: float | None = None,
        memory_mb: float | None = None,
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
            checked_at=datetime.now(UTC),
        )
        session.add(metric)
        await session.flush()
        return metric

    @staticmethod
    async def get_latest_metrics(
        session: AsyncSession, agent_id: str, limit: int = 10
    ) -> list[AgentHealthMetricDB]:
        """Get latest health metrics for agent."""
        result = await session.execute(
            select(AgentHealthMetricDB)
            .where(AgentHealthMetricDB.agent_id == agent_id)
            .order_by(AgentHealthMetricDB.checked_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_unhealthy_agents(session: AsyncSession) -> list[str]:
        """Get agent IDs with recent health failures."""
        # Get agents with failures in last check
        subquery = (
            select(
                AgentHealthMetricDB.agent_id,
                func.max(AgentHealthMetricDB.checked_at).label("last_check"),
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
                    AgentHealthMetricDB.checked_at == subquery.c.last_check,
                ),
            )
            .where(AgentHealthMetricDB.is_healthy == False)
        )
        return [row[0] for row in result.all()]

    @staticmethod
    async def cleanup_old_metrics(session: AsyncSession, days_to_keep: int = 7) -> int:
        """Delete health metrics older than specified days."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days_to_keep)
        result = await session.execute(
            delete(AgentHealthMetricDB).where(
                AgentHealthMetricDB.checked_at < cutoff_date
            )
        )
        return result.rowcount


# Import timedelta for cleanup_old_metrics
from datetime import timedelta


class SessionRepository:
    """Repository for session snapshot database operations."""

    @staticmethod
    def _strip_timezone(dt: datetime | None) -> datetime | None:
        """Strip timezone info for PostgreSQL TIMESTAMP WITHOUT TIME ZONE compatibility."""
        if dt is None:
            return None
        return dt.replace(tzinfo=None) if dt.tzinfo else dt

    @staticmethod
    def _add_timezone(dt: datetime | None) -> datetime | None:
        """Add UTC timezone to naive datetime when loading from database."""
        if dt is None:
            return None
        # If already has timezone, return as-is; otherwise assume UTC
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

    @staticmethod
    async def create(
        session: AsyncSession, snapshot: SessionSnapshot
    ) -> SessionSnapshotDB:
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
            created_at=SessionRepository._strip_timezone(snapshot.created_at),
            updated_at=SessionRepository._strip_timezone(snapshot.updated_at),
            expires_at=SessionRepository._strip_timezone(snapshot.expires_at),
            completed_at=SessionRepository._strip_timezone(snapshot.completed_at),
            timeout_seconds=snapshot.timeout_seconds,
            max_idle_seconds=snapshot.max_idle_seconds,
            tags=snapshot.tags,
            session_metadata=snapshot.metadata,
            checkpoint_interval_seconds=snapshot.checkpoint_interval_seconds,
            last_checkpoint_at=SessionRepository._strip_timezone(
                snapshot.last_checkpoint_at
            ),
            checkpoint_count=snapshot.checkpoint_count,
        )
        session.add(session_db)
        await session.flush()
        return session_db

    @staticmethod
    async def get_by_id(
        session: AsyncSession, session_id: str
    ) -> SessionSnapshotDB | None:
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
                updated_at=SessionRepository._strip_timezone(snapshot.updated_at),
                expires_at=SessionRepository._strip_timezone(snapshot.expires_at),
                completed_at=SessionRepository._strip_timezone(snapshot.completed_at),
                tags=snapshot.tags,
                session_metadata=snapshot.metadata,
                last_checkpoint_at=SessionRepository._strip_timezone(
                    snapshot.last_checkpoint_at
                ),
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
        session: AsyncSession, state: SessionState, limit: int = 100, offset: int = 0
    ) -> list[SessionSnapshotDB]:
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
        session: AsyncSession, owner_agent: str, limit: int = 100, offset: int = 0
    ) -> list[SessionSnapshotDB]:
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
    async def list_expired(session: AsyncSession) -> list[SessionSnapshotDB]:
        """Get all expired sessions."""
        result = await session.execute(
            select(SessionSnapshotDB).where(
                and_(
                    SessionSnapshotDB.expires_at.is_not(None),
                    SessionSnapshotDB.expires_at < datetime.now(UTC),
                    SessionSnapshotDB.state.notin_(
                        [
                            SessionState.COMPLETED,
                            SessionState.FAILED,
                            SessionState.EXPIRED,
                        ]
                    ),
                )
            )
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_idle(
        session: AsyncSession, max_idle_seconds: int = 300
    ) -> list[SessionSnapshotDB]:
        """Get sessions that have been idle longer than threshold."""
        idle_threshold = datetime.now(UTC) - timedelta(seconds=max_idle_seconds)
        result = await session.execute(
            select(SessionSnapshotDB).where(
                and_(
                    SessionSnapshotDB.state == SessionState.ACTIVE,
                    SessionSnapshotDB.updated_at < idle_threshold,
                )
            )
        )
        return list(result.scalars().all())

    @staticmethod
    async def cleanup_old_sessions(
        session: AsyncSession, days_to_keep: int = 30
    ) -> int:
        """Delete terminal sessions older than specified days."""
        cutoff_date = datetime.now(UTC) - timedelta(days=days_to_keep)
        result = await session.execute(
            delete(SessionSnapshotDB).where(
                and_(
                    SessionSnapshotDB.state.in_(
                        [
                            SessionState.COMPLETED,
                            SessionState.FAILED,
                            SessionState.EXPIRED,
                        ]
                    ),
                    SessionSnapshotDB.completed_at.is_not(None),
                    SessionSnapshotDB.completed_at < cutoff_date,
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
            context=SessionContext.model_validate(session_db.context)
            if session_db.context
            else SessionContext(),
            task_ids=session_db.task_ids or [],
            artifact_ids=session_db.artifact_ids or [],
            created_at=SessionRepository._add_timezone(session_db.created_at),
            updated_at=SessionRepository._add_timezone(session_db.updated_at),
            expires_at=SessionRepository._add_timezone(session_db.expires_at),
            completed_at=SessionRepository._add_timezone(session_db.completed_at),
            timeout_seconds=session_db.timeout_seconds,
            max_idle_seconds=session_db.max_idle_seconds,
            tags=session_db.tags or [],
            metadata=session_db.session_metadata or {},
            checkpoint_interval_seconds=session_db.checkpoint_interval_seconds,
            last_checkpoint_at=SessionRepository._add_timezone(
                session_db.last_checkpoint_at
            ),
            checkpoint_count=session_db.checkpoint_count,
        )


# ==================== Memory System Repositories ====================


class MemoryRepository:
    """Repository for memory database operations."""

    @staticmethod
    async def create(
        session: AsyncSession, memory_record: "MemoryRecord"
    ) -> "MemoryModel":
        """
        Create memory from MemoryRecord.

        Args:
            session: Database session
            memory_record: Pydantic memory record

        Returns:
            MemoryModel: Created memory ORM model

        Example:
            memory = await MemoryRepository.create(session, memory_record)
        """
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import MemoryModel

        # Extract UUIDs from ID strings
        def extract_uuid(id_str: str) -> UUID:
            """Extract UUID from prefixed ID string."""
            if "-" in id_str and not id_str.count("-") == 4:  # Has prefix
                return UUID(id_str.split("-", 1)[1])
            return UUID(id_str)

        memory_db = MemoryModel(
            memory_id=extract_uuid(memory_record.memory_id),
            memory_layer=memory_record.memory_layer,
            content=memory_record.content,
            summary=memory_record.summary,
            embedding=memory_record.embedding,
            agent_id=extract_uuid(memory_record.agent_id)
            if memory_record.agent_id
            else None,
            session_id=extract_uuid(memory_record.session_id)
            if memory_record.session_id
            else None,
            user_id=extract_uuid(memory_record.user_id)
            if memory_record.user_id
            else None,
            task_id=extract_uuid(memory_record.task_id)
            if memory_record.task_id
            else None,
            timestamp=memory_record.timestamp,
            entities=memory_record.entities,
            facts=memory_record.facts,
            keywords=memory_record.keywords,
            related_memory_ids=[
                extract_uuid(mid) for mid in memory_record.related_memory_ids
            ],
            parent_memory_id=extract_uuid(memory_record.parent_memory_id)
            if memory_record.parent_memory_id
            else None,
            relevance_score=memory_record.relevance_score,
            access_count=memory_record.access_count,
            last_accessed=memory_record.last_accessed,
            stage_id=extract_uuid(memory_record.stage_id)
            if memory_record.stage_id
            else None,
            is_critical=memory_record.is_critical,
            criticality_reason=memory_record.criticality_reason,
            actions=memory_record.actions,
            outcome=memory_record.outcome,
            success=memory_record.success,
        )
        session.add(memory_db)
        await session.flush()
        return memory_db

    @staticmethod
    async def get_by_id(
        session: AsyncSession, memory_id: str
    ) -> "MemoryModel | None":
        """Get memory by ID."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import MemoryModel

        result = await session.execute(
            select(MemoryModel).where(MemoryModel.memory_id == UUID(memory_id))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_stage_id(
        session: AsyncSession, stage_id: str, limit: int = 100
    ) -> list["MemoryModel"]:
        """
        Get memories filtered by stage_id.

        Args:
            session: Database session
            stage_id: Stage ID to filter by
            limit: Maximum number of results

        Returns:
            List of memory ORM models
        """
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import MemoryModel

        result = await session.execute(
            select(MemoryModel)
            .where(MemoryModel.stage_id == UUID(stage_id))
            .order_by(MemoryModel.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_by_agent_and_layer(
        session: AsyncSession,
        agent_id: str,
        layer: "MemoryLayer",
        limit: int = 100,
    ) -> list["MemoryModel"]:
        """Get memories by agent and layer."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import MemoryModel

        result = await session.execute(
            select(MemoryModel)
            .where(
                and_(
                    MemoryModel.agent_id == UUID(agent_id),
                    MemoryModel.memory_layer == layer,
                )
            )
            .order_by(MemoryModel.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_critical_memories(
        session: AsyncSession, agent_id: str, limit: int = 50
    ) -> list["MemoryModel"]:
        """Get critical memories for agent."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import MemoryModel

        result = await session.execute(
            select(MemoryModel)
            .where(
                and_(
                    MemoryModel.agent_id == UUID(agent_id),
                    MemoryModel.is_critical == True,
                )
            )
            .order_by(MemoryModel.relevance_score.desc(), MemoryModel.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_access(session: AsyncSession, memory_id: str) -> bool:
        """Update memory access tracking."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import MemoryModel

        result = await session.execute(
            update(MemoryModel)
            .where(MemoryModel.memory_id == UUID(memory_id))
            .values(
                access_count=MemoryModel.access_count + 1,
                last_accessed=datetime.now(UTC),
            )
        )
        return result.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, memory_id: str) -> bool:
        """Delete memory."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import MemoryModel

        result = await session.execute(
            delete(MemoryModel).where(MemoryModel.memory_id == UUID(memory_id))
        )
        return result.rowcount > 0


class StageMemoryRepository:
    """Repository for stage memory database operations."""

    @staticmethod
    async def create(
        session: AsyncSession, stage_memory: "StageMemory"
    ) -> "StageMemoryModel":
        """Create stage memory from StageMemory."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        # Extract UUIDs from ID strings (handles "stage-{uuid}", "task-{uuid}", etc.)
        def extract_uuid(id_str: str) -> UUID:
            """Extract UUID from prefixed ID string."""
            if "-" in id_str and not id_str.count("-") == 4:  # Has prefix
                return UUID(id_str.split("-", 1)[1])
            return UUID(id_str)

        stage_db = StageMemoryModel(
            stage_id=extract_uuid(stage_memory.stage_id),
            task_id=extract_uuid(stage_memory.task_id),
            agent_id=extract_uuid(stage_memory.agent_id),
            stage_type=stage_memory.stage_type,
            stage_summary=stage_memory.stage_summary,
            stage_insights=stage_memory.stage_insights,
            raw_memory_refs=[extract_uuid(ref) for ref in stage_memory.raw_memory_refs],
            relevance_score=stage_memory.relevance_score,
            compression_ratio=stage_memory.compression_ratio,
            compression_model=stage_memory.compression_model,
            quality_metrics={"quality_score": stage_memory.quality_score},
            created_at=stage_memory.created_at,
            updated_at=stage_memory.updated_at,
            completed_at=stage_memory.completed_at,
        )
        session.add(stage_db)
        await session.flush()
        return stage_db

    @staticmethod
    async def get_by_id(
        session: AsyncSession, stage_id: str
    ) -> "StageMemoryModel | None":
        """Get stage memory by ID."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        result = await session.execute(
            select(StageMemoryModel).where(StageMemoryModel.stage_id == UUID(stage_id))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_task_and_stage(
        session: AsyncSession, task_id: str, stage_type: "StageType"
    ) -> list["StageMemoryModel"]:
        """
        Get stage memories by task and stage type.

        Args:
            session: Database session
            task_id: Task ID to filter by
            stage_type: Stage type to filter by

        Returns:
            List of stage memory ORM models
        """
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        result = await session.execute(
            select(StageMemoryModel)
            .where(
                and_(
                    StageMemoryModel.task_id == UUID(task_id),
                    StageMemoryModel.stage_type == stage_type,
                )
            )
            .order_by(StageMemoryModel.created_at.desc())
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_by_task(
        session: AsyncSession, task_id: str
    ) -> list["StageMemoryModel"]:
        """Get all stage memories for a task."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        result = await session.execute(
            select(StageMemoryModel)
            .where(StageMemoryModel.task_id == UUID(task_id))
            .order_by(StageMemoryModel.created_at)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update(
        session: AsyncSession, stage_id: str, **updates: dict
    ) -> bool:
        """Update stage memory fields."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        updates["updated_at"] = datetime.now(UTC)
        result = await session.execute(
            update(StageMemoryModel)
            .where(StageMemoryModel.stage_id == UUID(stage_id))
            .values(**updates)
        )
        return result.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, stage_id: str) -> bool:
        """Delete stage memory."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        result = await session.execute(
            delete(StageMemoryModel).where(
                StageMemoryModel.stage_id == UUID(stage_id)
            )
        )
        return result.rowcount > 0


class TaskContextRepository:
    """Repository for task context database operations."""

    @staticmethod
    async def create(
        session: AsyncSession, task_context: "TaskContext"
    ) -> "TaskContextModel":
        """Create task context from TaskContext."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import TaskContextModel

        # Extract UUIDs from ID strings
        def extract_uuid(id_str: str) -> UUID:
            """Extract UUID from prefixed ID string."""
            if "-" in id_str and not id_str.count("-") == 4:  # Has prefix
                return UUID(id_str.split("-", 1)[1])
            return UUID(id_str)

        task_db = TaskContextModel(
            task_id=extract_uuid(task_context.task_id),
            agent_id=extract_uuid(task_context.agent_id),
            task_goal=task_context.task_goal,
            current_stage_id=extract_uuid(task_context.current_stage_id)
            if task_context.current_stage_id
            else None,
            task_progress_summary=task_context.task_progress_summary,
            critical_constraints=task_context.critical_constraints,
            performance_metrics=task_context.performance_metrics,
            created_at=task_context.created_at,
            updated_at=task_context.updated_at,
        )
        session.add(task_db)
        await session.flush()
        return task_db

    @staticmethod
    async def get_by_id(
        session: AsyncSession, task_id: str
    ) -> "TaskContextModel | None":
        """Get task context by ID."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import TaskContextModel

        result = await session.execute(
            select(TaskContextModel).where(TaskContextModel.task_id == UUID(task_id))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_current_stage(
        session: AsyncSession, task_id: str
    ) -> str | None:
        """
        Get current stage ID for a task.

        Args:
            session: Database session
            task_id: Task ID

        Returns:
            Current stage ID or None
        """
        task_context = await TaskContextRepository.get_by_id(session, task_id)
        if task_context and task_context.current_stage_id:
            return str(task_context.current_stage_id)
        return None

    @staticmethod
    async def update_progress(
        session: AsyncSession,
        task_id: str,
        progress_summary: str,
        current_stage_id: str | None = None,
    ) -> bool:
        """
        Update task progress summary and current stage.

        Args:
            session: Database session
            task_id: Task ID
            progress_summary: Updated progress summary
            current_stage_id: New current stage ID (optional)

        Returns:
            True if update successful
        """
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import TaskContextModel

        updates = {
            "task_progress_summary": progress_summary,
            "updated_at": datetime.now(UTC),
        }
        if current_stage_id:
            updates["current_stage_id"] = UUID(current_stage_id)

        result = await session.execute(
            update(TaskContextModel)
            .where(TaskContextModel.task_id == UUID(task_id))
            .values(**updates)
        )
        return result.rowcount > 0

    @staticmethod
    async def update_metrics(
        session: AsyncSession, task_id: str, metrics: dict
    ) -> bool:
        """Update task performance metrics."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import TaskContextModel

        result = await session.execute(
            update(TaskContextModel)
            .where(TaskContextModel.task_id == UUID(task_id))
            .values(performance_metrics=metrics, updated_at=datetime.now(UTC))
        )
        return result.rowcount > 0

    @staticmethod
    async def delete(session: AsyncSession, task_id: str) -> bool:
        """Delete task context."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import TaskContextModel

        result = await session.execute(
            delete(TaskContextModel).where(TaskContextModel.task_id == UUID(task_id))
        )
        return result.rowcount > 0


class ErrorRepository:
    """Repository for error record database operations."""

    @staticmethod
    async def create(
        session: AsyncSession, error_record: "ErrorRecord"
    ) -> "ErrorModel":
        """Create error record from ErrorRecord."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import ErrorModel

        # Extract UUIDs from ID strings
        def extract_uuid(id_str: str) -> UUID:
            """Extract UUID from prefixed ID string."""
            if "-" in id_str and not id_str.count("-") == 4:  # Has prefix
                return UUID(id_str.split("-", 1)[1])
            return UUID(id_str)

        error_db = ErrorModel(
            error_id=extract_uuid(error_record.error_id),
            task_id=extract_uuid(error_record.task_id),
            stage_id=extract_uuid(error_record.stage_id)
            if error_record.stage_id
            else None,
            agent_id=extract_uuid(error_record.agent_id),
            error_type=error_record.error_type,
            error_description=error_record.error_description,
            context_when_occurred=error_record.context_when_occurred,
            recovery_action=error_record.recovery_action,
            error_severity=error_record.error_severity,
            recorded_at=error_record.recorded_at,
        )
        session.add(error_db)
        await session.flush()
        return error_db

    @staticmethod
    async def get_by_id(
        session: AsyncSession, error_id: str
    ) -> "ErrorModel | None":
        """Get error record by ID."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import ErrorModel

        result = await session.execute(
            select(ErrorModel).where(ErrorModel.error_id == UUID(error_id))
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_recent_errors(
        session: AsyncSession,
        task_id: str,
        hours: int = 24,
        limit: int = 50,
    ) -> list["ErrorModel"]:
        """
        Get recent errors for a task within time window.

        Args:
            session: Database session
            task_id: Task ID to filter by
            hours: Time window in hours
            limit: Maximum number of results

        Returns:
            List of error ORM models
        """
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import ErrorModel

        cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
        result = await session.execute(
            select(ErrorModel)
            .where(
                and_(
                    ErrorModel.task_id == UUID(task_id),
                    ErrorModel.recorded_at >= cutoff_time,
                )
            )
            .order_by(ErrorModel.recorded_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def detect_patterns(
        session: AsyncSession,
        task_id: str,
        error_type: "ErrorType | None" = None,
        min_occurrences: int = 3,
    ) -> dict[str, int]:
        """
        Detect error patterns by counting error types.

        Args:
            session: Database session
            task_id: Task ID to analyze
            error_type: Specific error type to filter (optional)
            min_occurrences: Minimum occurrences to be considered a pattern

        Returns:
            Dictionary mapping error types to occurrence counts
        """
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import ErrorModel

        query = select(ErrorModel.error_type, func.count(ErrorModel.error_id)).where(
            ErrorModel.task_id == UUID(task_id)
        )

        if error_type:
            query = query.where(ErrorModel.error_type == error_type)

        query = query.group_by(ErrorModel.error_type).having(
            func.count(ErrorModel.error_id) >= min_occurrences
        )

        result = await session.execute(query)
        return {str(error_type): count for error_type, count in result.all()}

    @staticmethod
    async def get_by_severity(
        session: AsyncSession,
        task_id: str,
        min_severity: float = 0.7,
        limit: int = 50,
    ) -> list["ErrorModel"]:
        """Get errors above severity threshold."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import ErrorModel

        result = await session.execute(
            select(ErrorModel)
            .where(
                and_(
                    ErrorModel.task_id == UUID(task_id),
                    ErrorModel.error_severity >= min_severity,
                )
            )
            .order_by(ErrorModel.error_severity.desc(), ErrorModel.recorded_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete(session: AsyncSession, error_id: str) -> bool:
        """Delete error record."""
        from uuid import UUID

        from agentcore.a2a_protocol.database.memory_models import ErrorModel

        result = await session.execute(
            delete(ErrorModel).where(ErrorModel.error_id == UUID(error_id))
        )
        return result.rowcount > 0
