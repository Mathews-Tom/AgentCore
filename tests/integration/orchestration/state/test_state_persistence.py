"""
Integration Tests for Workflow State Persistence

Tests for state persistence with real database.
- SQLite (default): Fast, in-memory, basic functionality
- PostgreSQL (USE_POSTGRES=1): Full features including JSONB operators

Usage:
    pytest tests/integration/orchestration/state/  # SQLite
    USE_POSTGRES=1 pytest tests/integration/orchestration/state/  # PostgreSQL
"""

import os
import pytest
from datetime import UTC, datetime
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from agentcore.orchestration.state.models import (
    WorkflowExecutionDB,
    WorkflowStateDB,
    WorkflowStateVersion,
    WorkflowStatus,
    Base,
)
from agentcore.orchestration.state.repository import (
    WorkflowStateRepository,
    WorkflowVersionRepository,
)

# Check if PostgreSQL testing is enabled
USE_POSTGRES = os.getenv("USE_POSTGRES", "0") == "1"

if USE_POSTGRES:
    try:
        from testcontainers.postgres import PostgresContainer
        POSTGRES_AVAILABLE = True
    except ImportError:
        POSTGRES_AVAILABLE = False
        pytest.skip(
            "testcontainers not available. Install with: uv add --dev 'testcontainers[postgres]'",
            allow_module_level=True
        )
else:
    POSTGRES_AVAILABLE = False


@pytest.fixture(scope="module")
def postgres_container():
    """Start PostgreSQL container if USE_POSTGRES=1."""
    if not USE_POSTGRES or not POSTGRES_AVAILABLE:
        pytest.skip("PostgreSQL testing not enabled")

    container = PostgresContainer("postgres:16-alpine")
    container.start()

    yield container

    container.stop()


@pytest.fixture
async def db_engine(postgres_container=None):
    """
    Create database engine for testing.

    - Default: SQLite in-memory (fast, basic features)
    - USE_POSTGRES=1: PostgreSQL container (full JSONB features)
    """
    if USE_POSTGRES and POSTGRES_AVAILABLE:
        # Use PostgreSQL container
        db_url = postgres_container.get_connection_url().replace("psycopg2", "asyncpg")
        engine = create_async_engine(db_url, echo=False)
    else:
        # Use SQLite in-memory (default)
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def db_session(db_engine):
    """Create database session for testing."""
    async_session = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session
        await session.rollback()


class TestWorkflowStateRepository:
    """Test workflow state repository operations."""

    @pytest.mark.asyncio
    async def test_create_execution(self, db_session: AsyncSession) -> None:
        """Test creating workflow execution."""
        execution_id = str(uuid4())
        workflow_id = str(uuid4())

        execution = await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=execution_id,
            workflow_id=workflow_id,
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
            workflow_version="1.0",
            input_data={"input": "value"},
            tags=["test"],
            metadata={"meta": "data"},
        )

        assert execution.execution_id == execution_id
        assert execution.workflow_id == workflow_id
        assert execution.workflow_name == "test_workflow"
        assert execution.orchestration_pattern == "saga"
        assert execution.status == WorkflowStatus.PENDING
        assert execution.workflow_definition == {"steps": []}
        assert execution.input_data == {"input": "value"}
        assert execution.tags == ["test"]
        assert execution.workflow_metadata == {"meta": "data"}

    @pytest.mark.asyncio
    async def test_get_execution(self, db_session: AsyncSession) -> None:
        """Test retrieving workflow execution."""
        execution_id = str(uuid4())
        workflow_id = str(uuid4())

        # Create execution
        await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=execution_id,
            workflow_id=workflow_id,
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await db_session.commit()

        # Retrieve execution
        execution = await WorkflowStateRepository.get_execution(
            db_session, execution_id
        )

        assert execution is not None
        assert execution.execution_id == execution_id
        assert execution.workflow_id == workflow_id

    @pytest.mark.asyncio
    async def test_list_executions_with_filters(
        self, db_session: AsyncSession
    ) -> None:
        """Test listing executions with filters."""
        workflow_id1 = str(uuid4())
        workflow_id2 = str(uuid4())

        # Create multiple executions
        exec1 = await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=str(uuid4()),
            workflow_id=workflow_id1,
            workflow_name="workflow1",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
            tags=["test", "priority"],
        )

        exec2 = await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=str(uuid4()),
            workflow_id=workflow_id2,
            workflow_name="workflow2",
            orchestration_pattern="supervisor",
            workflow_definition={"steps": []},
            tags=["test"],
        )

        await db_session.commit()

        # Test filter by workflow_id
        executions = await WorkflowStateRepository.list_executions(
            db_session, workflow_id=workflow_id1
        )
        assert len(executions) == 1
        assert executions[0].execution_id == exec1.execution_id

        # Test filter by orchestration_pattern
        executions = await WorkflowStateRepository.list_executions(
            db_session, orchestration_pattern="supervisor"
        )
        assert len(executions) == 1
        assert executions[0].execution_id == exec2.execution_id

        # Test filter by tags
        executions = await WorkflowStateRepository.list_executions(
            db_session, tags=["priority"]
        )
        assert len(executions) == 1
        assert executions[0].execution_id == exec1.execution_id

    @pytest.mark.asyncio
    async def test_update_execution_status(self, db_session: AsyncSession) -> None:
        """Test updating execution status."""
        execution_id = str(uuid4())

        # Create execution
        await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=execution_id,
            workflow_id=str(uuid4()),
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await db_session.commit()

        # Update to executing
        execution = await WorkflowStateRepository.update_execution_status(
            session=db_session,
            execution_id=execution_id,
            status=WorkflowStatus.EXECUTING,
        )

        assert execution is not None
        assert execution.status == WorkflowStatus.EXECUTING
        assert execution.started_at is not None

        await db_session.commit()

        # Update to completed
        execution = await WorkflowStateRepository.update_execution_status(
            session=db_session,
            execution_id=execution_id,
            status=WorkflowStatus.COMPLETED,
        )

        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.completed_at is not None
        assert execution.duration_seconds is not None
        assert execution.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_update_execution_state(self, db_session: AsyncSession) -> None:
        """Test updating execution state."""
        execution_id = str(uuid4())

        # Create execution
        await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=execution_id,
            workflow_id=str(uuid4()),
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await db_session.commit()

        # Update state
        new_state = {"current_step": 1, "data": "value"}
        allocated_agents = {"agent1": "agent_id_1"}
        task_states = {
            "task1": {"status": "completed"},
            "task2": {"status": "running"},
        }

        execution = await WorkflowStateRepository.update_execution_state(
            session=db_session,
            execution_id=execution_id,
            execution_state=new_state,
            allocated_agents=allocated_agents,
            task_states=task_states,
            create_snapshot=True,
        )

        assert execution is not None
        assert execution.execution_state == new_state
        assert execution.allocated_agents == allocated_agents
        assert execution.task_states == task_states
        assert execution.total_tasks == 2
        assert execution.completed_task_count == 1
        assert execution.failed_task_count == 0

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, db_session: AsyncSession) -> None:
        """Test creating workflow checkpoint."""
        execution_id = str(uuid4())

        # Create execution
        await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=execution_id,
            workflow_id=str(uuid4()),
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await db_session.commit()

        # Create checkpoint
        checkpoint_data = {"step": 5, "state": "checkpoint_data"}

        execution = await WorkflowStateRepository.create_checkpoint(
            session=db_session,
            execution_id=execution_id,
            checkpoint_data=checkpoint_data,
        )

        assert execution is not None
        assert execution.checkpoint_data == checkpoint_data
        assert execution.checkpoint_count == 1
        assert execution.last_checkpoint_at is not None

        await db_session.commit()

        # Create another checkpoint
        execution = await WorkflowStateRepository.create_checkpoint(
            session=db_session,
            execution_id=execution_id,
            checkpoint_data={"step": 10},
        )

        assert execution.checkpoint_count == 2

    @pytest.mark.asyncio
    async def test_state_history(self, db_session: AsyncSession) -> None:
        """Test state history and versioning."""
        execution_id = str(uuid4())

        # Create execution
        await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=execution_id,
            workflow_id=str(uuid4()),
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await db_session.commit()

        # Create multiple state snapshots
        await WorkflowStateRepository.create_state_snapshot(
            session=db_session,
            execution_id=execution_id,
            state_type="event",
            state_snapshot={"state": "v1"},
            change_reason="initial",
        )

        await WorkflowStateRepository.create_state_snapshot(
            session=db_session,
            execution_id=execution_id,
            state_type="checkpoint",
            state_snapshot={"state": "v2"},
            change_reason="checkpoint",
        )

        await db_session.commit()

        # Get all history
        history = await WorkflowStateRepository.get_state_history(
            db_session, execution_id
        )

        assert len(history) == 2
        assert history[0].version == 2  # Latest first
        assert history[1].version == 1

        # Get checkpoint history only
        checkpoints = await WorkflowStateRepository.get_state_history(
            db_session, execution_id, state_type="checkpoint"
        )

        assert len(checkpoints) == 1
        assert checkpoints[0].state_type == "checkpoint"

        # Get specific version
        state = await WorkflowStateRepository.get_state_at_version(
            db_session, execution_id, version=1
        )

        assert state is not None
        assert state.version == 1
        assert state.state_snapshot == {"state": "v1"}

    @pytest.mark.asyncio
    async def test_delete_execution(self, db_session: AsyncSession) -> None:
        """Test deleting workflow execution."""
        execution_id = str(uuid4())

        # Create execution with history
        await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=execution_id,
            workflow_id=str(uuid4()),
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await WorkflowStateRepository.create_state_snapshot(
            session=db_session,
            execution_id=execution_id,
            state_type="event",
            state_snapshot={"state": "data"},
        )

        await db_session.commit()

        # Delete execution
        deleted = await WorkflowStateRepository.delete_execution(
            db_session, execution_id
        )

        assert deleted is True

        await db_session.commit()

        # Verify deletion
        execution = await WorkflowStateRepository.get_execution(
            db_session, execution_id
        )
        assert execution is None

        # Verify cascade deletion of history
        history = await WorkflowStateRepository.get_state_history(
            db_session, execution_id
        )
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_execution_statistics(self, db_session: AsyncSession) -> None:
        """Test execution statistics."""
        workflow_id = str(uuid4())

        # Create multiple executions with different statuses
        exec1 = await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=str(uuid4()),
            workflow_id=workflow_id,
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await WorkflowStateRepository.update_execution_status(
            db_session, exec1.execution_id, WorkflowStatus.EXECUTING
        )
        await WorkflowStateRepository.update_execution_status(
            db_session, exec1.execution_id, WorkflowStatus.COMPLETED
        )

        exec2 = await WorkflowStateRepository.create_execution(
            session=db_session,
            execution_id=str(uuid4()),
            workflow_id=workflow_id,
            workflow_name="test_workflow",
            orchestration_pattern="saga",
            workflow_definition={"steps": []},
        )

        await WorkflowStateRepository.update_execution_status(
            db_session, exec2.execution_id, WorkflowStatus.FAILED
        )

        # Update task counts
        await WorkflowStateRepository.update_execution_state(
            db_session,
            exec1.execution_id,
            execution_state={},
            task_states={"task1": {"status": "completed"}},
        )

        await db_session.commit()

        # Get statistics
        stats = await WorkflowStateRepository.get_execution_stats(
            db_session, workflow_id=workflow_id, orchestration_pattern="saga"
        )

        assert stats["total_executions"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["failed"] == 1
        assert stats["by_pattern"]["saga"] == 2
        assert stats["avg_duration_seconds"] is not None
        assert stats["total_tasks"] == 1


class TestWorkflowVersionRepository:
    """Test workflow version repository operations."""

    @pytest.mark.asyncio
    async def test_create_version(self, db_session: AsyncSession) -> None:
        """Test creating workflow state version."""
        version_id = str(uuid4())

        version = await WorkflowVersionRepository.create_version(
            session=db_session,
            version_id=version_id,
            schema_version=1,
            workflow_type="saga",
            state_schema={"type": "object", "properties": {}},
            description="Initial version",
            migration_script="-- No migration needed",
        )

        assert version.version_id == version_id
        assert version.schema_version == 1
        assert version.workflow_type == "saga"
        assert version.is_active is True
        assert version.description == "Initial version"

    @pytest.mark.asyncio
    async def test_get_latest_version(self, db_session: AsyncSession) -> None:
        """Test getting latest active version."""
        # Create multiple versions
        await WorkflowVersionRepository.create_version(
            session=db_session,
            version_id=str(uuid4()),
            schema_version=1,
            workflow_type="saga",
            state_schema={"version": 1},
        )

        await WorkflowVersionRepository.create_version(
            session=db_session,
            version_id=str(uuid4()),
            schema_version=2,
            workflow_type="saga",
            state_schema={"version": 2},
        )

        await db_session.commit()

        # Get latest version
        latest = await WorkflowVersionRepository.get_latest_version(
            db_session, "saga"
        )

        assert latest is not None
        assert latest.schema_version == 2

    @pytest.mark.asyncio
    async def test_deprecate_version(self, db_session: AsyncSession) -> None:
        """Test deprecating workflow version."""
        version_id = str(uuid4())

        # Create version
        await WorkflowVersionRepository.create_version(
            session=db_session,
            version_id=version_id,
            schema_version=1,
            workflow_type="saga",
            state_schema={"version": 1},
        )

        await db_session.commit()

        # Deprecate version
        version = await WorkflowVersionRepository.deprecate_version(
            db_session, version_id
        )

        assert version is not None
        assert version.is_active is False
        assert version.deprecated_at is not None
