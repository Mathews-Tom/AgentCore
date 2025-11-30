"""
Workflow State Repository

Data access layer for workflow state persistence with optimized JSONB queries.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import and_, delete, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.orchestration.state.models import (
    WorkflowExecutionDB,
    WorkflowStateDB,
    WorkflowStateVersion,
    WorkflowStatus,
)

logger = structlog.get_logger()


class WorkflowStateRepository:
    """Repository for workflow state database operations."""

    @staticmethod
    async def create_execution(
        session: AsyncSession,
        execution_id: str,
        workflow_id: str,
        workflow_name: str,
        orchestration_pattern: str,
        workflow_definition: dict[str, Any],
        workflow_version: str = "1.0",
        input_data: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowExecutionDB:
        """
        Create a new workflow execution.

        Args:
            session: Database session
            execution_id: Unique execution identifier
            workflow_id: Workflow definition ID
            workflow_name: Workflow name
            orchestration_pattern: Pattern type (supervisor, saga, etc.)
            workflow_definition: Complete workflow definition
            workflow_version: Workflow version
            input_data: Input parameters
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            Created workflow execution
        """
        execution = WorkflowExecutionDB(
            execution_id=execution_id,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            workflow_version=workflow_version,
            orchestration_pattern=orchestration_pattern,
            status=WorkflowStatus.PENDING,
            workflow_definition=workflow_definition,
            execution_state={},
            allocated_agents={},
            task_states={},
            input_data=input_data or {},
            tags=tags or [],
            workflow_metadata=metadata or {},
        )
        session.add(execution)
        await session.flush()

        logger.info(
            "Created workflow execution",
            execution_id=execution_id,
            workflow_id=workflow_id,
            pattern=orchestration_pattern,
        )

        return execution

    @staticmethod
    async def get_execution(
        session: AsyncSession, execution_id: str
    ) -> WorkflowExecutionDB | None:
        """
        Get workflow execution by ID.

        Args:
            session: Database session
            execution_id: Execution identifier

        Returns:
            Workflow execution or None
        """
        result = await session.execute(
            select(WorkflowExecutionDB).where(
                WorkflowExecutionDB.execution_id == execution_id
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_executions(
        session: AsyncSession,
        workflow_id: str | None = None,
        status: WorkflowStatus | None = None,
        orchestration_pattern: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkflowExecutionDB]:
        """
        List workflow executions with filters.

        Args:
            session: Database session
            workflow_id: Filter by workflow ID
            status: Filter by status
            orchestration_pattern: Filter by pattern
            tags: Filter by tags (ANY match)
            limit: Maximum results
            offset: Results offset

        Returns:
            List of workflow executions
        """
        query = select(WorkflowExecutionDB)

        # Apply filters
        if workflow_id:
            query = query.where(WorkflowExecutionDB.workflow_id == workflow_id)
        if status:
            query = query.where(WorkflowExecutionDB.status == status)
        if orchestration_pattern:
            query = query.where(
                WorkflowExecutionDB.orchestration_pattern == orchestration_pattern
            )
        if tags:
            # Use PostgreSQL JSONB containment for tag filtering
            # For SQLite, skip tag filtering (not supported)
            from sqlalchemy.engine import Engine

            dialect_name = session.bind.dialect.name if session.bind else "postgresql"
            if dialect_name == "postgresql":
                query = query.where(WorkflowExecutionDB.tags.op("@>")(tags))

        # Order by created_at descending
        query = query.order_by(desc(WorkflowExecutionDB.created_at))

        # Apply pagination
        query = query.limit(limit).offset(offset)

        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def update_execution_status(
        session: AsyncSession,
        execution_id: str,
        status: WorkflowStatus,
        error_message: str | None = None,
        error_stack_trace: str | None = None,
    ) -> WorkflowExecutionDB | None:
        """
        Update workflow execution status.

        Args:
            session: Database session
            execution_id: Execution identifier
            status: New status
            error_message: Error message if failed
            error_stack_trace: Stack trace if failed

        Returns:
            Updated execution or None
        """
        execution = await WorkflowStateRepository.get_execution(session, execution_id)
        if not execution:
            return None

        execution.status = status

        # Update timing based on status
        # Use timezone-naive datetimes for PostgreSQL TIMESTAMP WITHOUT TIME ZONE
        if status == WorkflowStatus.EXECUTING and not execution.started_at:
            execution.started_at = datetime.now(UTC).replace(tzinfo=None)
        elif status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.COMPENSATED,
            WorkflowStatus.CANCELLED,
        ):
            execution.completed_at = datetime.now(UTC).replace(tzinfo=None)
            if execution.started_at:
                # Both timestamps are now timezone-naive (PostgreSQL TIMESTAMP WITHOUT TIME ZONE)
                started = execution.started_at
                if started.tzinfo is not None:
                    # Strip timezone if present for consistent calculation
                    started = started.replace(tzinfo=None)
                execution.duration_seconds = int(
                    (execution.completed_at - started).total_seconds()
                )

        # Update error info
        if error_message:
            execution.error_message = error_message
        if error_stack_trace:
            execution.error_stack_trace = error_stack_trace

        await session.flush()

        logger.info(
            "Updated workflow execution status",
            execution_id=execution_id,
            status=status,
        )

        return execution

    @staticmethod
    async def update_execution_state(
        session: AsyncSession,
        execution_id: str,
        execution_state: dict[str, Any],
        allocated_agents: dict[str, str] | None = None,
        task_states: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        create_snapshot: bool = True,
    ) -> WorkflowExecutionDB | None:
        """
        Update workflow execution state.

        Args:
            session: Database session
            execution_id: Execution identifier
            execution_state: New execution state
            allocated_agents: Updated agent allocation
            task_states: Updated task states
            output_data: Output data
            create_snapshot: Whether to create state snapshot

        Returns:
            Updated execution or None
        """
        execution = await WorkflowStateRepository.get_execution(session, execution_id)
        if not execution:
            return None

        # Track changed fields for audit
        changed_fields = []

        # Update execution state
        old_state = execution.execution_state
        execution.execution_state = execution_state
        if old_state != execution_state:
            changed_fields.append("execution_state")

        # Update agent allocation
        if allocated_agents is not None:
            old_agents = execution.allocated_agents
            execution.allocated_agents = allocated_agents
            if old_agents != allocated_agents:
                changed_fields.append("allocated_agents")

        # Update task states
        if task_states is not None:
            old_tasks = execution.task_states
            execution.task_states = task_states
            if old_tasks != task_states:
                changed_fields.append("task_states")

            # Update task counts
            execution.total_tasks = len(task_states)
            execution.completed_task_count = sum(
                1
                for state in task_states.values()
                if state.get("status") == "completed"
            )
            execution.failed_task_count = sum(
                1 for state in task_states.values() if state.get("status") == "failed"
            )

        # Update output data
        if output_data is not None:
            execution.output_data = output_data
            changed_fields.append("output_data")

        await session.flush()

        # Create state snapshot if requested
        if create_snapshot and changed_fields:
            await WorkflowStateRepository.create_state_snapshot(
                session=session,
                execution_id=execution_id,
                state_type="event",
                state_snapshot={
                    "execution_state": execution_state,
                    "allocated_agents": allocated_agents or execution.allocated_agents,
                    "task_states": task_states or execution.task_states,
                },
                changed_fields=changed_fields,
                change_reason="state_update",
            )

        logger.debug(
            "Updated workflow execution state",
            execution_id=execution_id,
            changed_fields=changed_fields,
        )

        return execution

    @staticmethod
    async def create_checkpoint(
        session: AsyncSession,
        execution_id: str,
        checkpoint_data: dict[str, Any],
    ) -> WorkflowExecutionDB | None:
        """
        Create workflow checkpoint.

        Args:
            session: Database session
            execution_id: Execution identifier
            checkpoint_data: Checkpoint data

        Returns:
            Updated execution or None
        """
        execution = await WorkflowStateRepository.get_execution(session, execution_id)
        if not execution:
            return None

        execution.checkpoint_data = checkpoint_data
        execution.checkpoint_count += 1
        execution.last_checkpoint_at = datetime.now(UTC)

        await session.flush()

        # Create state snapshot for checkpoint
        await WorkflowStateRepository.create_state_snapshot(
            session=session,
            execution_id=execution_id,
            state_type="checkpoint",
            state_snapshot=checkpoint_data,
            change_reason=f"checkpoint_{execution.checkpoint_count}",
        )

        logger.info(
            "Created workflow checkpoint",
            execution_id=execution_id,
            checkpoint_count=execution.checkpoint_count,
        )

        return execution

    @staticmethod
    async def create_state_snapshot(
        session: AsyncSession,
        execution_id: str,
        state_type: str,
        state_snapshot: dict[str, Any],
        changed_fields: list[str] | None = None,
        change_reason: str | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowStateDB:
        """
        Create state snapshot for audit trail.

        Args:
            session: Database session
            execution_id: Execution identifier
            state_type: Snapshot type (checkpoint, event, snapshot)
            state_snapshot: State data
            changed_fields: Changed fields
            change_reason: Reason for change
            created_by: Creator identifier
            metadata: Additional metadata

        Returns:
            Created state snapshot
        """
        # Get latest version
        result = await session.execute(
            select(func.max(WorkflowStateDB.version)).where(
                WorkflowStateDB.execution_id == execution_id
            )
        )
        max_version = result.scalar()
        next_version = (max_version or 0) + 1

        state = WorkflowStateDB(
            execution_id=execution_id,
            version=next_version,
            state_type=state_type,
            state_snapshot=state_snapshot,
            changed_fields=changed_fields,
            change_reason=change_reason,
            created_by=created_by,
            state_metadata=metadata or {},
        )

        session.add(state)
        await session.flush()

        logger.debug(
            "Created state snapshot",
            execution_id=execution_id,
            version=next_version,
            state_type=state_type,
        )

        return state

    @staticmethod
    async def get_state_history(
        session: AsyncSession,
        execution_id: str,
        state_type: str | None = None,
        limit: int = 100,
    ) -> list[WorkflowStateDB]:
        """
        Get state history for execution.

        Args:
            session: Database session
            execution_id: Execution identifier
            state_type: Filter by state type
            limit: Maximum results

        Returns:
            List of state snapshots
        """
        query = select(WorkflowStateDB).where(
            WorkflowStateDB.execution_id == execution_id
        )

        if state_type:
            query = query.where(WorkflowStateDB.state_type == state_type)

        query = query.order_by(desc(WorkflowStateDB.version)).limit(limit)

        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_state_at_version(
        session: AsyncSession, execution_id: str, version: int
    ) -> WorkflowStateDB | None:
        """
        Get state at specific version.

        Args:
            session: Database session
            execution_id: Execution identifier
            version: Version number

        Returns:
            State snapshot or None
        """
        result = await session.execute(
            select(WorkflowStateDB).where(
                and_(
                    WorkflowStateDB.execution_id == execution_id,
                    WorkflowStateDB.version == version,
                )
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def delete_execution(session: AsyncSession, execution_id: str) -> bool:
        """
        Delete workflow execution and all history.

        Args:
            session: Database session
            execution_id: Execution identifier

        Returns:
            True if deleted, False if not found
        """
        result = await session.execute(
            delete(WorkflowExecutionDB).where(
                WorkflowExecutionDB.execution_id == execution_id
            )
        )

        deleted = result.rowcount > 0

        if deleted:
            logger.info("Deleted workflow execution", execution_id=execution_id)

        return deleted

    @staticmethod
    async def get_execution_stats(
        session: AsyncSession,
        workflow_id: str | None = None,
        orchestration_pattern: str | None = None,
    ) -> dict[str, Any]:
        """
        Get workflow execution statistics.

        Args:
            session: Database session
            workflow_id: Filter by workflow ID
            orchestration_pattern: Filter by pattern

        Returns:
            Statistics dictionary
        """
        query = select(WorkflowExecutionDB)

        if workflow_id:
            query = query.where(WorkflowExecutionDB.workflow_id == workflow_id)
        if orchestration_pattern:
            query = query.where(
                WorkflowExecutionDB.orchestration_pattern == orchestration_pattern
            )

        result = await session.execute(query)
        executions = list(result.scalars().all())

        # Calculate statistics
        total = len(executions)
        if total == 0:
            return {
                "total_executions": 0,
                "by_status": {},
                "by_pattern": {},
                "avg_duration_seconds": None,
                "total_tasks": 0,
                "avg_tasks_per_execution": 0,
            }

        by_status: dict[str, int] = {}
        by_pattern: dict[str, int] = {}
        durations = []
        total_tasks = 0

        for execution in executions:
            # Count by status
            status = execution.status.value
            by_status[status] = by_status.get(status, 0) + 1

            # Count by pattern
            pattern = execution.orchestration_pattern
            by_pattern[pattern] = by_pattern.get(pattern, 0) + 1

            # Collect durations
            if execution.duration_seconds:
                durations.append(execution.duration_seconds)

            # Sum tasks
            total_tasks += execution.total_tasks

        return {
            "total_executions": total,
            "by_status": by_status,
            "by_pattern": by_pattern,
            "avg_duration_seconds": sum(durations) / len(durations)
            if durations
            else None,
            "total_tasks": total_tasks,
            "avg_tasks_per_execution": total_tasks / total if total > 0 else 0,
        }


class WorkflowVersionRepository:
    """Repository for workflow state versioning and migrations."""

    @staticmethod
    async def create_version(
        session: AsyncSession,
        version_id: str,
        schema_version: int,
        workflow_type: str,
        state_schema: dict[str, Any],
        description: str | None = None,
        migration_script: str | None = None,
    ) -> WorkflowStateVersion:
        """
        Create workflow state version.

        Args:
            session: Database session
            version_id: Version identifier
            schema_version: Schema version number
            workflow_type: Workflow type
            state_schema: JSON schema for state
            description: Version description
            migration_script: Migration script

        Returns:
            Created version
        """
        version = WorkflowStateVersion(
            version_id=version_id,
            schema_version=schema_version,
            workflow_type=workflow_type,
            state_schema=state_schema,
            description=description,
            migration_script=migration_script,
            is_active=True,
        )

        session.add(version)
        await session.flush()

        logger.info(
            "Created workflow state version",
            version_id=version_id,
            schema_version=schema_version,
            workflow_type=workflow_type,
        )

        return version

    @staticmethod
    async def get_latest_version(
        session: AsyncSession, workflow_type: str
    ) -> WorkflowStateVersion | None:
        """
        Get latest active version for workflow type.

        Args:
            session: Database session
            workflow_type: Workflow type

        Returns:
            Latest version or None
        """
        result = await session.execute(
            select(WorkflowStateVersion)
            .where(
                and_(
                    WorkflowStateVersion.workflow_type == workflow_type,
                    WorkflowStateVersion.is_active == True,
                )
            )
            .order_by(desc(WorkflowStateVersion.schema_version))
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def deprecate_version(
        session: AsyncSession, version_id: str
    ) -> WorkflowStateVersion | None:
        """
        Deprecate a workflow state version.

        Args:
            session: Database session
            version_id: Version identifier

        Returns:
            Updated version or None
        """
        result = await session.execute(
            select(WorkflowStateVersion).where(
                WorkflowStateVersion.version_id == version_id
            )
        )
        version = result.scalar_one_or_none()

        if version:
            version.is_active = False
            version.deprecated_at = datetime.now(UTC)
            await session.flush()

            logger.info("Deprecated workflow state version", version_id=version_id)

        return version
