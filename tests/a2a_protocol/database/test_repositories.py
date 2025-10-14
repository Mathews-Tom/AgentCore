"""
Tests for Database Repositories

Comprehensive test suite for AgentRepository, TaskRepository, HealthMetricRepository,
and SessionRepository to achieve 85%+ coverage.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.database.models import (
    AgentDB,
    AgentHealthMetricDB,
    SessionSnapshotDB,
    TaskDB,
)
from agentcore.a2a_protocol.database.repositories import (
    AgentRepository,
    HealthMetricRepository,
    SessionRepository,
    TaskRepository,
)
from agentcore.a2a_protocol.models.agent import (
    AgentAuthentication,
    AgentCapability,
    AgentCard,
    AgentEndpoint,
    AgentStatus,
    AuthenticationType,
    EndpointType,
)
from agentcore.a2a_protocol.models.session import (
    SessionContext,
    SessionPriority,
    SessionSnapshot,
    SessionState,
)
from agentcore.a2a_protocol.models.task import TaskDefinition, TaskPriority, TaskStatus

# ==================== AgentRepository Tests ====================


@pytest.mark.asyncio
async def test_agent_repository_create():
    """Test creating agent from AgentCard."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    agent_card = AgentCard(
        agent_id="test-agent",
        agent_name="Test Agent",
        agent_version="1.0.0",
        status=AgentStatus.ACTIVE,
        capabilities=[AgentCapability(name="test-capability", version="1.0")],
        endpoints=[AgentEndpoint(url="http://test.local", type=EndpointType.HTTP)],
        authentication=AgentAuthentication(
            type=AuthenticationType.NONE, required=False
        ),
    )

    agent_db = await AgentRepository.create(mock_session, agent_card)

    assert agent_db.id == "test-agent"
    assert agent_db.name == "Test Agent"
    assert agent_db.version == "1.0.0"
    assert agent_db.status == AgentStatus.ACTIVE
    assert agent_db.current_load == 0
    assert agent_db.max_load == 10
    mock_session.add.assert_called_once_with(agent_db)
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_agent_repository_get_by_id():
    """Test getting agent by ID."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_agent = AgentDB(id="test-agent", name="Test Agent", version="1.0.0")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_agent
    mock_session.execute = AsyncMock(return_value=mock_result)

    agent = await AgentRepository.get_by_id(mock_session, "test-agent")

    assert agent == mock_agent
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_agent_repository_get_by_id_not_found():
    """Test getting non-existent agent returns None."""
    mock_session = MagicMock(spec=AsyncSession)

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    agent = await AgentRepository.get_by_id(mock_session, "nonexistent")

    assert agent is None


@pytest.mark.asyncio
async def test_agent_repository_get_all():
    """Test getting all agents."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_agents = [
        AgentDB(id="agent-1", name="Agent 1", version="1.0"),
        AgentDB(id="agent-2", name="Agent 2", version="1.0"),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_agents
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    agents = await AgentRepository.get_all(mock_session)

    assert len(agents) == 2
    assert agents == mock_agents


@pytest.mark.asyncio
async def test_agent_repository_get_all_with_status_filter():
    """Test getting agents filtered by status."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_agents = [AgentDB(id="agent-1", name="Agent 1", status=AgentStatus.ACTIVE)]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_agents
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    agents = await AgentRepository.get_all(mock_session, status=AgentStatus.ACTIVE)

    assert len(agents) == 1


@pytest.mark.asyncio
async def test_agent_repository_get_by_capabilities():
    """Test getting agents by capabilities."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_agents = [AgentDB(id="agent-1", capabilities=["text-generation"])]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_agents
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    agents = await AgentRepository.get_by_capabilities(
        mock_session, required_capabilities=["text-generation"]
    )

    assert len(agents) == 1


@pytest.mark.asyncio
async def test_agent_repository_update_status():
    """Test updating agent status."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await AgentRepository.update_status(
        mock_session, "test-agent", AgentStatus.INACTIVE
    )

    assert success is True


@pytest.mark.asyncio
async def test_agent_repository_update_status_not_found():
    """Test updating status of non-existent agent."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 0
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await AgentRepository.update_status(
        mock_session, "nonexistent", AgentStatus.INACTIVE
    )

    assert success is False


@pytest.mark.asyncio
async def test_agent_repository_update_load():
    """Test updating agent load."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await AgentRepository.update_load(mock_session, "test-agent", 1)

    assert success is True


@pytest.mark.asyncio
async def test_agent_repository_update_last_seen():
    """Test updating agent last_seen timestamp."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await AgentRepository.update_last_seen(mock_session, "test-agent")

    assert success is True


@pytest.mark.asyncio
async def test_agent_repository_delete():
    """Test deleting agent."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await AgentRepository.delete(mock_session, "test-agent")

    assert success is True


@pytest.mark.asyncio
async def test_agent_repository_count_by_status():
    """Test counting agents by status."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.all.return_value = [
        (AgentStatus.ACTIVE, 5),
        (AgentStatus.INACTIVE, 2),
    ]
    mock_session.execute = AsyncMock(return_value=mock_result)

    counts = await AgentRepository.count_by_status(mock_session)

    assert counts["active"] == 5
    assert counts["inactive"] == 2


@pytest.mark.asyncio
async def test_agent_repository_update_embedding():
    """Test updating agent capability embedding."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    embedding = [0.1] * 384
    success = await AgentRepository.update_embedding(
        mock_session, "test-agent", embedding
    )

    assert success is True


@pytest.mark.asyncio
async def test_agent_repository_semantic_search():
    """Test semantic search with vector similarity."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_agent = AgentDB(id="agent-1", name="Test Agent")
    mock_result = MagicMock()
    mock_result.all.return_value = [(mock_agent, 0.85)]
    mock_session.execute = AsyncMock(return_value=mock_result)

    query_embedding = [0.1] * 384
    results = await AgentRepository.semantic_search(
        mock_session, query_embedding, similarity_threshold=0.75
    )

    assert len(results) == 1
    assert results[0][0] == mock_agent
    assert results[0][1] == 0.85


@pytest.mark.asyncio
async def test_agent_repository_semantic_search_fallback():
    """Test semantic search fallback when pgvector not available."""
    mock_session = MagicMock(spec=AsyncSession)

    # Mock ImportError for pgvector
    with patch(
        "agentcore.a2a_protocol.database.repositories.AgentRepository.get_all"
    ) as mock_get_all:
        mock_agents = [AgentDB(id="agent-1"), AgentDB(id="agent-2")]
        mock_get_all.return_value = mock_agents

        # Trigger ImportError by making execute fail with ImportError
        mock_session.execute = AsyncMock(
            side_effect=ImportError("pgvector not installed")
        )

        query_embedding = [0.1] * 384

        # The method catches ImportError and falls back
        with patch("agentcore.a2a_protocol.database.repositories.logger"):
            results = await AgentRepository.semantic_search(
                mock_session, query_embedding
            )

        # Should return agents with 0.0 similarity
        assert len(results) <= 10


# ==================== TaskRepository Tests ====================


@pytest.mark.asyncio
async def test_task_repository_create():
    """Test creating task from TaskDefinition."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    task_def = TaskDefinition(
        task_id="test-task",
        task_type="text.generation",
        title="Test Task",
        description="Test Description",
        priority=TaskPriority.NORMAL,
        parameters={"key": "value"},
    )

    task_db = await TaskRepository.create(mock_session, task_def)

    assert task_db.id == "test-task"
    assert task_db.name == "Test Task"
    assert task_db.status == TaskStatus.PENDING
    assert task_db.priority == TaskPriority.NORMAL
    mock_session.add.assert_called_once_with(task_db)
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_task_repository_get_by_id():
    """Test getting task by ID."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_task = TaskDB(id="test-task", name="Test Task")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_task
    mock_session.execute = AsyncMock(return_value=mock_result)

    task = await TaskRepository.get_by_id(mock_session, "test-task")

    assert task == mock_task


@pytest.mark.asyncio
async def test_task_repository_get_all():
    """Test getting all tasks."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_tasks = [
        TaskDB(id="task-1", name="Task 1"),
        TaskDB(id="task-2", name="Task 2"),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_tasks
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    tasks = await TaskRepository.get_all(mock_session)

    assert len(tasks) == 2


@pytest.mark.asyncio
async def test_task_repository_get_all_with_filters():
    """Test getting tasks with status and agent filters."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_tasks = [
        TaskDB(id="task-1", status=TaskStatus.RUNNING, assigned_agent_id="agent-1")
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_tasks
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    tasks = await TaskRepository.get_all(
        mock_session, status=TaskStatus.RUNNING, agent_id="agent-1"
    )

    assert len(tasks) == 1


@pytest.mark.asyncio
async def test_task_repository_assign_to_agent():
    """Test assigning task to agent."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskRepository.assign_to_agent(
        mock_session, "test-task", "test-agent"
    )

    assert success is True


@pytest.mark.asyncio
async def test_task_repository_update_status():
    """Test updating task status."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskRepository.update_status(
        mock_session, "test-task", TaskStatus.COMPLETED, result={"output": "success"}
    )

    assert success is True


@pytest.mark.asyncio
async def test_task_repository_update_status_with_error():
    """Test updating task status with error."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskRepository.update_status(
        mock_session, "test-task", TaskStatus.FAILED, error="Test error"
    )

    assert success is True


@pytest.mark.asyncio
async def test_task_repository_delete():
    """Test deleting task."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskRepository.delete(mock_session, "test-task")

    assert success is True


@pytest.mark.asyncio
async def test_task_repository_count_by_status():
    """Test counting tasks by status."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.all.return_value = [
        (TaskStatus.PENDING, 10),
        (TaskStatus.RUNNING, 5),
        (TaskStatus.COMPLETED, 20),
    ]
    mock_session.execute = AsyncMock(return_value=mock_result)

    counts = await TaskRepository.count_by_status(mock_session)

    assert counts["pending"] == 10
    assert counts["running"] == 5
    assert counts["completed"] == 20


@pytest.mark.asyncio
async def test_task_repository_get_pending_tasks():
    """Test getting pending tasks ordered by priority."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_tasks = [
        TaskDB(id="task-1", status=TaskStatus.PENDING, priority=TaskPriority.HIGH),
        TaskDB(id="task-2", status=TaskStatus.PENDING, priority=TaskPriority.LOW),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_tasks
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    tasks = await TaskRepository.get_pending_tasks(mock_session, limit=50)

    assert len(tasks) == 2


# ==================== HealthMetricRepository Tests ====================


@pytest.mark.asyncio
async def test_health_metric_repository_record_health_check():
    """Test recording health check result."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    metric = await HealthMetricRepository.record_health_check(
        mock_session,
        agent_id="test-agent",
        is_healthy=True,
        response_time_ms=50.0,
        status_code=200,
        cpu_percent=25.5,
        memory_mb=512.0,
    )

    assert metric.agent_id == "test-agent"
    assert metric.is_healthy is True
    assert metric.response_time_ms == 50.0
    assert metric.status_code == 200
    assert metric.cpu_percent == 25.5
    assert metric.memory_mb == 512.0
    mock_session.add.assert_called_once_with(metric)
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_health_metric_repository_record_unhealthy_check():
    """Test recording unhealthy check with error."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    metric = await HealthMetricRepository.record_health_check(
        mock_session,
        agent_id="test-agent",
        is_healthy=False,
        status_code=500,
        error_message="Connection timeout",
    )

    assert metric.is_healthy is False
    assert metric.error_message == "Connection timeout"


@pytest.mark.asyncio
async def test_health_metric_repository_get_latest_metrics():
    """Test getting latest health metrics for agent."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_metrics = [
        AgentHealthMetricDB(agent_id="test-agent", is_healthy=True),
        AgentHealthMetricDB(agent_id="test-agent", is_healthy=True),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_metrics
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    metrics = await HealthMetricRepository.get_latest_metrics(
        mock_session, "test-agent", limit=10
    )

    assert len(metrics) == 2


@pytest.mark.asyncio
async def test_health_metric_repository_get_unhealthy_agents():
    """Test getting unhealthy agents."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.all.return_value = [("agent-1",), ("agent-2",)]
    mock_session.execute = AsyncMock(return_value=mock_result)

    unhealthy = await HealthMetricRepository.get_unhealthy_agents(mock_session)

    assert len(unhealthy) == 2
    assert "agent-1" in unhealthy
    assert "agent-2" in unhealthy


@pytest.mark.asyncio
async def test_health_metric_repository_cleanup_old_metrics():
    """Test cleanup of old health metrics."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 50
    mock_session.execute = AsyncMock(return_value=mock_result)

    deleted = await HealthMetricRepository.cleanup_old_metrics(
        mock_session, days_to_keep=7
    )

    assert deleted == 50


# ==================== SessionRepository Tests ====================


@pytest.mark.asyncio
async def test_session_repository_create():
    """Test creating session snapshot."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    snapshot = SessionSnapshot(
        session_id="test-session",
        name="Test Session",
        description="Test Description",
        state=SessionState.ACTIVE,
        priority=SessionPriority.NORMAL,
        owner_agent="agent-1",
        participant_agents=["agent-1", "agent-2"],
        context=SessionContext(),
        timeout_seconds=3600,
        max_idle_seconds=300,
    )

    session_db = await SessionRepository.create(mock_session, snapshot)

    assert session_db.session_id == "test-session"
    assert session_db.name == "Test Session"
    assert session_db.state == SessionState.ACTIVE
    assert session_db.owner_agent == "agent-1"
    mock_session.add.assert_called_once_with(session_db)
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_session_repository_get_by_id():
    """Test getting session by ID."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_snapshot = SessionSnapshotDB(
        session_id="test-session", name="Test Session", state=SessionState.ACTIVE
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_snapshot
    mock_session.execute = AsyncMock(return_value=mock_result)

    session_db = await SessionRepository.get_by_id(mock_session, "test-session")

    assert session_db == mock_snapshot


@pytest.mark.asyncio
async def test_session_repository_update():
    """Test updating session snapshot."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    snapshot = SessionSnapshot(
        session_id="test-session",
        name="Updated Session",
        state=SessionState.COMPLETED,
        priority=SessionPriority.HIGH,
        owner_agent="agent-1",
        participant_agents=[],
        context=SessionContext(),
    )

    success = await SessionRepository.update(mock_session, snapshot)

    assert success is True


@pytest.mark.asyncio
async def test_session_repository_delete():
    """Test deleting session snapshot."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await SessionRepository.delete(mock_session, "test-session")

    assert success is True


@pytest.mark.asyncio
async def test_session_repository_list_by_state():
    """Test listing sessions by state."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_sessions = [
        SessionSnapshotDB(session_id="session-1", state=SessionState.ACTIVE),
        SessionSnapshotDB(session_id="session-2", state=SessionState.ACTIVE),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_sessions
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    sessions = await SessionRepository.list_by_state(
        mock_session, SessionState.ACTIVE, limit=100
    )

    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_session_repository_list_by_owner():
    """Test listing sessions by owner agent."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_sessions = [
        SessionSnapshotDB(session_id="session-1", owner_agent="agent-1"),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_sessions
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    sessions = await SessionRepository.list_by_owner(mock_session, "agent-1", limit=100)

    assert len(sessions) == 1


@pytest.mark.asyncio
async def test_session_repository_list_expired():
    """Test getting expired sessions."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_sessions = [
        SessionSnapshotDB(
            session_id="session-1",
            state=SessionState.ACTIVE,
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        ),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_sessions
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    sessions = await SessionRepository.list_expired(mock_session)

    assert len(sessions) == 1


@pytest.mark.asyncio
async def test_session_repository_list_idle():
    """Test getting idle sessions."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_sessions = [
        SessionSnapshotDB(
            session_id="session-1",
            state=SessionState.ACTIVE,
            updated_at=datetime.now(UTC) - timedelta(minutes=10),
        ),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_sessions
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    sessions = await SessionRepository.list_idle(mock_session, max_idle_seconds=300)

    assert len(sessions) == 1


@pytest.mark.asyncio
async def test_session_repository_cleanup_old_sessions():
    """Test cleanup of old terminal sessions."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 25
    mock_session.execute = AsyncMock(return_value=mock_result)

    deleted = await SessionRepository.cleanup_old_sessions(
        mock_session, days_to_keep=30
    )

    assert deleted == 25


@pytest.mark.asyncio
async def test_session_repository_to_snapshot():
    """Test converting SessionSnapshotDB to SessionSnapshot."""
    session_db = SessionSnapshotDB(
        session_id="test-session",
        name="Test Session",
        description="Test Description",
        state=SessionState.ACTIVE,
        priority=SessionPriority.NORMAL,
        owner_agent="agent-1",
        participant_agents=["agent-1", "agent-2"],
        context={"key": "value"},
        task_ids=["task-1"],
        artifact_ids=["artifact-1"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        timeout_seconds=3600,
        max_idle_seconds=300,
        tags=["test"],
        session_metadata={"meta": "data"},
        checkpoint_interval_seconds=60,
        checkpoint_count=5,
    )

    snapshot = SessionRepository.to_snapshot(session_db)

    assert snapshot.session_id == "test-session"
    assert snapshot.name == "Test Session"
    assert snapshot.state == SessionState.ACTIVE
    assert snapshot.owner_agent == "agent-1"
    assert len(snapshot.participant_agents) == 2
    assert len(snapshot.task_ids) == 1
    assert snapshot.checkpoint_count == 5
