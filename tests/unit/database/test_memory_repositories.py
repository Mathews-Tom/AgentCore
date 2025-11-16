"""
Tests for Memory System Repositories

Comprehensive test suite for MemoryRepository, StageMemoryRepository,
TaskContextRepository, and ErrorRepository to achieve 90%+ coverage.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.database.memory_models import (
    ErrorModel,
    MemoryModel,
    StageMemoryModel,
    TaskContextModel,
)
from agentcore.a2a_protocol.database.repositories import (
    ErrorRepository,
    MemoryRepository,
    StageMemoryRepository,
    TaskContextRepository,
)
from agentcore.a2a_protocol.models.memory import (
    ErrorRecord,
    ErrorType,
    MemoryLayer,
    MemoryRecord,
    StageMemory,
    StageType,
    TaskContext,
)


# ==================== MemoryRepository Tests ====================


@pytest.mark.asyncio
async def test_memory_repository_create():
    """Test creating memory from MemoryRecord."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    memory_record = MemoryRecord(
        memory_id=f"mem-{uuid4()}",
        memory_layer=MemoryLayer.EPISODIC,
        content="Test memory content",
        summary="Test summary",
        embedding=[0.1] * 768,
        agent_id=str(uuid4()),
        task_id=str(uuid4()),
        entities=["entity1", "entity2"],
        facts=["fact1"],
        keywords=["keyword1"],
    )

    memory_db = await MemoryRepository.create(mock_session, memory_record)

    assert memory_db.memory_layer == MemoryLayer.EPISODIC
    assert memory_db.content == "Test memory content"
    assert memory_db.summary == "Test summary"
    assert len(memory_db.entities) == 2
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_memory_repository_get_by_id():
    """Test getting memory by ID."""
    mock_session = MagicMock(spec=AsyncSession)
    memory_id = str(uuid4())
    mock_memory = MemoryModel(
        memory_id=memory_id,
        memory_layer=MemoryLayer.SEMANTIC,
        content="Test content",
        summary="Test summary",
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_memory
    mock_session.execute = AsyncMock(return_value=mock_result)

    memory = await MemoryRepository.get_by_id(mock_session, memory_id)

    assert memory == mock_memory
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_memory_repository_get_by_id_not_found():
    """Test getting non-existent memory returns None."""
    mock_session = MagicMock(spec=AsyncSession)

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    memory = await MemoryRepository.get_by_id(mock_session, str(uuid4()))

    assert memory is None


@pytest.mark.asyncio
async def test_memory_repository_get_by_stage_id():
    """Test getting memories filtered by stage_id."""
    mock_session = MagicMock(spec=AsyncSession)
    stage_id = str(uuid4())
    mock_memories = [
        MemoryModel(
            memory_id=uuid4(),
            memory_layer=MemoryLayer.EPISODIC,
            content="Memory 1",
            summary="Summary 1",
            stage_id=stage_id,
        ),
        MemoryModel(
            memory_id=uuid4(),
            memory_layer=MemoryLayer.EPISODIC,
            content="Memory 2",
            summary="Summary 2",
            stage_id=stage_id,
        ),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_memories
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    memories = await MemoryRepository.get_by_stage_id(mock_session, stage_id)

    assert len(memories) == 2
    assert memories == mock_memories


@pytest.mark.asyncio
async def test_memory_repository_get_by_agent_and_layer():
    """Test getting memories by agent and layer."""
    mock_session = MagicMock(spec=AsyncSession)
    agent_id = str(uuid4())
    mock_memories = [
        MemoryModel(
            memory_id=uuid4(),
            memory_layer=MemoryLayer.SEMANTIC,
            content="Semantic memory",
            summary="Summary",
            agent_id=agent_id,
        )
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_memories
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    memories = await MemoryRepository.get_by_agent_and_layer(
        mock_session, agent_id, MemoryLayer.SEMANTIC
    )

    assert len(memories) == 1


@pytest.mark.asyncio
async def test_memory_repository_get_critical_memories():
    """Test getting critical memories for agent."""
    mock_session = MagicMock(spec=AsyncSession)
    agent_id = str(uuid4())
    mock_memories = [
        MemoryModel(
            memory_id=uuid4(),
            memory_layer=MemoryLayer.EPISODIC,
            content="Critical memory",
            summary="Important",
            agent_id=agent_id,
            is_critical=True,
            relevance_score=0.95,
        )
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_memories
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    memories = await MemoryRepository.get_critical_memories(mock_session, agent_id)

    assert len(memories) == 1
    assert memories[0].is_critical is True


@pytest.mark.asyncio
async def test_memory_repository_update_access():
    """Test updating memory access tracking."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await MemoryRepository.update_access(mock_session, str(uuid4()))

    assert success is True


@pytest.mark.asyncio
async def test_memory_repository_delete():
    """Test deleting memory."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await MemoryRepository.delete(mock_session, str(uuid4()))

    assert success is True


# ==================== StageMemoryRepository Tests ====================


@pytest.mark.asyncio
async def test_stage_memory_repository_create():
    """Test creating stage memory from StageMemory."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    stage_memory = StageMemory(
        stage_id=f"stage-{uuid4()}",
        task_id=str(uuid4()),
        agent_id=str(uuid4()),
        stage_type=StageType.PLANNING,
        stage_summary="Planning phase summary",
        stage_insights=["insight1", "insight2"],
        raw_memory_refs=[str(uuid4())],
        compression_ratio=10.2,
        compression_model="gpt-4.1-mini",
        quality_score=0.95,
    )

    stage_db = await StageMemoryRepository.create(mock_session, stage_memory)

    assert stage_db.stage_type == StageType.PLANNING
    assert stage_db.stage_summary == "Planning phase summary"
    assert len(stage_db.stage_insights) == 2
    assert stage_db.compression_ratio == 10.2
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_stage_memory_repository_get_by_id():
    """Test getting stage memory by ID."""
    mock_session = MagicMock(spec=AsyncSession)
    stage_id = str(uuid4())
    mock_stage = StageMemoryModel(
        stage_id=stage_id,
        task_id=uuid4(),
        agent_id=uuid4(),
        stage_type=StageType.EXECUTION,
        stage_summary="Execution summary",
        raw_memory_refs=[],
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_stage
    mock_session.execute = AsyncMock(return_value=mock_result)

    stage = await StageMemoryRepository.get_by_id(mock_session, stage_id)

    assert stage == mock_stage


@pytest.mark.asyncio
async def test_stage_memory_repository_get_by_task_and_stage():
    """Test getting stage memories by task and stage type."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_stages = [
        StageMemoryModel(
            stage_id=uuid4(),
            task_id=task_id,
            agent_id=uuid4(),
            stage_type=StageType.PLANNING,
            stage_summary="Planning 1",
            raw_memory_refs=[],
        ),
        StageMemoryModel(
            stage_id=uuid4(),
            task_id=task_id,
            agent_id=uuid4(),
            stage_type=StageType.PLANNING,
            stage_summary="Planning 2",
            raw_memory_refs=[],
        ),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_stages
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    stages = await StageMemoryRepository.get_by_task_and_stage(
        mock_session, task_id, StageType.PLANNING
    )

    assert len(stages) == 2


@pytest.mark.asyncio
async def test_stage_memory_repository_get_by_task():
    """Test getting all stage memories for a task."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_stages = [
        StageMemoryModel(
            stage_id=uuid4(),
            task_id=task_id,
            agent_id=uuid4(),
            stage_type=StageType.PLANNING,
            stage_summary="Planning",
            raw_memory_refs=[],
        ),
        StageMemoryModel(
            stage_id=uuid4(),
            task_id=task_id,
            agent_id=uuid4(),
            stage_type=StageType.EXECUTION,
            stage_summary="Execution",
            raw_memory_refs=[],
        ),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_stages
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    stages = await StageMemoryRepository.get_by_task(mock_session, task_id)

    assert len(stages) == 2


@pytest.mark.asyncio
async def test_stage_memory_repository_update():
    """Test updating stage memory fields."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await StageMemoryRepository.update(
        mock_session, str(uuid4()), stage_summary="Updated summary"
    )

    assert success is True


@pytest.mark.asyncio
async def test_stage_memory_repository_delete():
    """Test deleting stage memory."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await StageMemoryRepository.delete(mock_session, str(uuid4()))

    assert success is True


# ==================== TaskContextRepository Tests ====================


@pytest.mark.asyncio
async def test_task_context_repository_create():
    """Test creating task context from TaskContext."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    task_context = TaskContext(
        task_id=f"task-{uuid4()}",
        agent_id=str(uuid4()),
        task_goal="Implement authentication system",
        current_stage_id=str(uuid4()),
        task_progress_summary="Completed planning phase",
        critical_constraints=["Use JWT", "Redis storage"],
        performance_metrics={"error_rate": 0.05},
    )

    task_db = await TaskContextRepository.create(mock_session, task_context)

    assert task_db.task_goal == "Implement authentication system"
    assert len(task_db.critical_constraints) == 2
    assert task_db.performance_metrics["error_rate"] == 0.05
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_task_context_repository_get_by_id():
    """Test getting task context by ID."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_task = TaskContextModel(
        task_id=task_id,
        agent_id=uuid4(),
        task_goal="Test goal",
        task_progress_summary="In progress",
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_task
    mock_session.execute = AsyncMock(return_value=mock_result)

    task = await TaskContextRepository.get_by_id(mock_session, task_id)

    assert task == mock_task


@pytest.mark.asyncio
async def test_task_context_repository_get_current_stage():
    """Test getting current stage ID for a task."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    current_stage_id = uuid4()
    mock_task = TaskContextModel(
        task_id=task_id,
        agent_id=uuid4(),
        task_goal="Test goal",
        current_stage_id=current_stage_id,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_task
    mock_session.execute = AsyncMock(return_value=mock_result)

    stage_id = await TaskContextRepository.get_current_stage(mock_session, task_id)

    assert stage_id == str(current_stage_id)


@pytest.mark.asyncio
async def test_task_context_repository_get_current_stage_none():
    """Test getting current stage when none is set."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_task = TaskContextModel(
        task_id=task_id,
        agent_id=uuid4(),
        task_goal="Test goal",
        current_stage_id=None,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_task
    mock_session.execute = AsyncMock(return_value=mock_result)

    stage_id = await TaskContextRepository.get_current_stage(mock_session, task_id)

    assert stage_id is None


@pytest.mark.asyncio
async def test_task_context_repository_update_progress():
    """Test updating task progress summary and current stage."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskContextRepository.update_progress(
        mock_session,
        str(uuid4()),
        "Updated progress",
        current_stage_id=str(uuid4()),
    )

    assert success is True


@pytest.mark.asyncio
async def test_task_context_repository_update_progress_no_stage():
    """Test updating task progress without changing stage."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskContextRepository.update_progress(
        mock_session, str(uuid4()), "Updated progress"
    )

    assert success is True


@pytest.mark.asyncio
async def test_task_context_repository_update_metrics():
    """Test updating task performance metrics."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskContextRepository.update_metrics(
        mock_session, str(uuid4()), {"error_rate": 0.1, "progress_rate": 0.75}
    )

    assert success is True


@pytest.mark.asyncio
async def test_task_context_repository_delete():
    """Test deleting task context."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await TaskContextRepository.delete(mock_session, str(uuid4()))

    assert success is True


# ==================== ErrorRepository Tests ====================


@pytest.mark.asyncio
async def test_error_repository_create():
    """Test creating error record from ErrorRecord."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()

    error_record = ErrorRecord(
        error_id=f"err-{uuid4()}",
        task_id=str(uuid4()),
        stage_id=str(uuid4()),
        agent_id=str(uuid4()),
        error_type=ErrorType.INCORRECT_ACTION,
        error_description="Used wrong API endpoint",
        context_when_occurred="During token refresh",
        recovery_action="Corrected to /auth/refresh",
        error_severity=0.6,
    )

    error_db = await ErrorRepository.create(mock_session, error_record)

    assert error_db.error_type == ErrorType.INCORRECT_ACTION
    assert error_db.error_description == "Used wrong API endpoint"
    assert error_db.error_severity == 0.6
    mock_session.add.assert_called_once()
    mock_session.flush.assert_called_once()


@pytest.mark.asyncio
async def test_error_repository_get_by_id():
    """Test getting error record by ID."""
    mock_session = MagicMock(spec=AsyncSession)
    error_id = str(uuid4())
    mock_error = ErrorModel(
        error_id=error_id,
        task_id=uuid4(),
        agent_id=uuid4(),
        error_type=ErrorType.HALLUCINATION,
        error_description="Test error",
        error_severity=0.8,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_error
    mock_session.execute = AsyncMock(return_value=mock_result)

    error = await ErrorRepository.get_by_id(mock_session, error_id)

    assert error == mock_error


@pytest.mark.asyncio
async def test_error_repository_get_recent_errors():
    """Test getting recent errors for a task within time window."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_errors = [
        ErrorModel(
            error_id=uuid4(),
            task_id=task_id,
            agent_id=uuid4(),
            error_type=ErrorType.MISSING_INFO,
            error_description="Error 1",
            error_severity=0.5,
            recorded_at=datetime.now(UTC),
        ),
        ErrorModel(
            error_id=uuid4(),
            task_id=task_id,
            agent_id=uuid4(),
            error_type=ErrorType.INCORRECT_ACTION,
            error_description="Error 2",
            error_severity=0.7,
            recorded_at=datetime.now(UTC),
        ),
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_errors
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    errors = await ErrorRepository.get_recent_errors(mock_session, task_id, hours=24)

    assert len(errors) == 2


@pytest.mark.asyncio
async def test_error_repository_detect_patterns():
    """Test detecting error patterns by counting error types."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_result = MagicMock()
    mock_result.all.return_value = [
        (ErrorType.INCORRECT_ACTION, 5),
        (ErrorType.MISSING_INFO, 3),
    ]
    mock_session.execute = AsyncMock(return_value=mock_result)

    patterns = await ErrorRepository.detect_patterns(
        mock_session, task_id, min_occurrences=3
    )

    assert patterns["ErrorType.INCORRECT_ACTION"] == 5
    assert patterns["ErrorType.MISSING_INFO"] == 3


@pytest.mark.asyncio
async def test_error_repository_detect_patterns_filtered():
    """Test detecting error patterns with type filter."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_result = MagicMock()
    mock_result.all.return_value = [(ErrorType.HALLUCINATION, 4)]
    mock_session.execute = AsyncMock(return_value=mock_result)

    patterns = await ErrorRepository.detect_patterns(
        mock_session, task_id, error_type=ErrorType.HALLUCINATION, min_occurrences=3
    )

    assert patterns["ErrorType.HALLUCINATION"] == 4


@pytest.mark.asyncio
async def test_error_repository_get_by_severity():
    """Test getting errors above severity threshold."""
    mock_session = MagicMock(spec=AsyncSession)
    task_id = str(uuid4())
    mock_errors = [
        ErrorModel(
            error_id=uuid4(),
            task_id=task_id,
            agent_id=uuid4(),
            error_type=ErrorType.CONTEXT_DEGRADATION,
            error_description="Critical error",
            error_severity=0.9,
        )
    ]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = mock_errors
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    errors = await ErrorRepository.get_by_severity(
        mock_session, task_id, min_severity=0.7
    )

    assert len(errors) == 1
    assert errors[0].error_severity == 0.9


@pytest.mark.asyncio
async def test_error_repository_delete():
    """Test deleting error record."""
    mock_session = MagicMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.rowcount = 1
    mock_session.execute = AsyncMock(return_value=mock_result)

    success = await ErrorRepository.delete(mock_session, str(uuid4()))

    assert success is True
