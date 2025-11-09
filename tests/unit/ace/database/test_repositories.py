"""
Unit tests for ACE repository layer.

Tests repository methods using mocked AsyncSession.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.database import (
    ContextDeltaDB,
    ContextPlaybookDB,
    DeltaRepository,
    EvolutionStatusDB,
    EvolutionStatusRepository,
    ExecutionTraceDB,
    PlaybookRepository,
    TraceRepository,
)
from agentcore.ace.models.ace_models import EvolutionStatusType


class TestPlaybookRepository:
    """Tests for PlaybookRepository."""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Create mock AsyncSession."""
        session = MagicMock(spec=AsyncSession)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_create_playbook(self, mock_session: AsyncSession) -> None:
        """Test creating a playbook."""
        # Arrange
        agent_id = "agent-001"
        context = {"goal": "test goal"}
        metadata = {"version": "1.0"}

        # Act
        playbook = await PlaybookRepository.create(
            mock_session, agent_id, context, metadata
        )

        # Assert
        assert playbook.agent_id == agent_id
        assert playbook.context == context
        assert playbook.playbook_metadata == metadata
        assert playbook.version == 1
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_playbook_default_metadata(
        self, mock_session: AsyncSession
    ) -> None:
        """Test creating a playbook with default metadata."""
        # Arrange
        agent_id = "agent-001"
        context = {"goal": "test goal"}

        # Act
        playbook = await PlaybookRepository.create(mock_session, agent_id, context)

        # Assert
        assert playbook.playbook_metadata == {}

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session: AsyncSession) -> None:
        """Test getting playbook by ID when found."""
        # Arrange
        playbook_id = uuid4()
        expected_playbook = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "test"},
        )
        expected_playbook.playbook_id = playbook_id

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_playbook
        mock_session.execute.return_value = mock_result

        # Act
        result = await PlaybookRepository.get_by_id(mock_session, playbook_id)

        # Assert
        assert result == expected_playbook
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session: AsyncSession) -> None:
        """Test getting playbook by ID when not found."""
        # Arrange
        playbook_id = uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await PlaybookRepository.get_by_id(mock_session, playbook_id)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_agent_id_found(self, mock_session: AsyncSession) -> None:
        """Test getting latest playbook for agent when found."""
        # Arrange
        agent_id = "agent-001"
        expected_playbook = ContextPlaybookDB(
            agent_id=agent_id,
            context={"goal": "test"},
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_playbook
        mock_session.execute.return_value = mock_result

        # Act
        result = await PlaybookRepository.get_by_agent_id(mock_session, agent_id)

        # Assert
        assert result == expected_playbook

    @pytest.mark.asyncio
    async def test_get_by_agent_id_not_found(self, mock_session: AsyncSession) -> None:
        """Test getting latest playbook for agent when not found."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await PlaybookRepository.get_by_agent_id(mock_session, agent_id)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_update_context_success(self, mock_session: AsyncSession) -> None:
        """Test updating playbook context successfully."""
        # Arrange
        playbook_id = uuid4()
        new_context = {"goal": "updated goal"}
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        # Act
        success = await PlaybookRepository.update_context(
            mock_session, playbook_id, new_context
        )

        # Assert
        assert success is True
        mock_session.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_context_not_found(self, mock_session: AsyncSession) -> None:
        """Test updating playbook context when not found."""
        # Arrange
        playbook_id = uuid4()
        new_context = {"goal": "updated goal"}
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        # Act
        success = await PlaybookRepository.update_context(
            mock_session, playbook_id, new_context
        )

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_success(self, mock_session: AsyncSession) -> None:
        """Test deleting playbook successfully."""
        # Arrange
        playbook_id = uuid4()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        # Act
        success = await PlaybookRepository.delete(mock_session, playbook_id)

        # Assert
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self, mock_session: AsyncSession) -> None:
        """Test deleting playbook when not found."""
        # Arrange
        playbook_id = uuid4()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        # Act
        success = await PlaybookRepository.delete(mock_session, playbook_id)

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_list_by_agent(self, mock_session: AsyncSession) -> None:
        """Test listing playbooks for agent."""
        # Arrange
        agent_id = "agent-001"
        playbooks = [
            ContextPlaybookDB(agent_id=agent_id, context={"goal": "test1"}),
            ContextPlaybookDB(agent_id=agent_id, context={"goal": "test2"}),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = playbooks
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await PlaybookRepository.list_by_agent(mock_session, agent_id)

        # Assert
        assert result == playbooks
        assert len(result) == 2


class TestDeltaRepository:
    """Tests for DeltaRepository."""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Create mock AsyncSession."""
        session = MagicMock(spec=AsyncSession)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_create_delta(self, mock_session: AsyncSession) -> None:
        """Test creating a delta."""
        # Arrange
        playbook_id = uuid4()
        changes = {"temperature": 0.8}
        confidence = 0.85
        reasoning = "Test reasoning"

        # Act
        delta = await DeltaRepository.create(
            mock_session, playbook_id, changes, confidence, reasoning
        )

        # Assert
        assert delta.playbook_id == playbook_id
        assert delta.changes == changes
        assert delta.confidence == confidence
        assert delta.reasoning == reasoning
        assert delta.applied is False
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session: AsyncSession) -> None:
        """Test getting delta by ID when found."""
        # Arrange
        delta_id = uuid4()
        expected_delta = ContextDeltaDB(
            playbook_id=uuid4(),
            changes={"test": "value"},
            confidence=0.9,
            reasoning="test",
        )
        expected_delta.delta_id = delta_id

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_delta
        mock_session.execute.return_value = mock_result

        # Act
        result = await DeltaRepository.get_by_id(mock_session, delta_id)

        # Assert
        assert result == expected_delta

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session: AsyncSession) -> None:
        """Test getting delta by ID when not found."""
        # Arrange
        delta_id = uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await DeltaRepository.get_by_id(mock_session, delta_id)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_unapplied_by_playbook(self, mock_session: AsyncSession) -> None:
        """Test getting unapplied deltas for playbook."""
        # Arrange
        playbook_id = uuid4()
        deltas = [
            ContextDeltaDB(
                playbook_id=playbook_id,
                changes={"test": "1"},
                confidence=0.9,
                reasoning="test1",
            ),
            ContextDeltaDB(
                playbook_id=playbook_id,
                changes={"test": "2"},
                confidence=0.8,
                reasoning="test2",
            ),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = deltas
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await DeltaRepository.get_unapplied_by_playbook(
            mock_session, playbook_id
        )

        # Assert
        assert result == deltas
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_mark_applied_success(self, mock_session: AsyncSession) -> None:
        """Test marking delta as applied successfully."""
        # Arrange
        delta_id = uuid4()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        # Act
        success = await DeltaRepository.mark_applied(mock_session, delta_id)

        # Assert
        assert success is True

    @pytest.mark.asyncio
    async def test_mark_applied_not_found(self, mock_session: AsyncSession) -> None:
        """Test marking delta as applied when not found."""
        # Arrange
        delta_id = uuid4()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        # Act
        success = await DeltaRepository.mark_applied(mock_session, delta_id)

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_list_by_playbook(self, mock_session: AsyncSession) -> None:
        """Test listing deltas for playbook."""
        # Arrange
        playbook_id = uuid4()
        deltas = [
            ContextDeltaDB(
                playbook_id=playbook_id,
                changes={"test": "1"},
                confidence=0.9,
                reasoning="test1",
            ),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = deltas
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await DeltaRepository.list_by_playbook(mock_session, playbook_id)

        # Assert
        assert result == deltas

    @pytest.mark.asyncio
    async def test_count_by_playbook(self, mock_session: AsyncSession) -> None:
        """Test counting deltas for playbook."""
        # Arrange
        playbook_id = uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 5
        mock_session.execute.return_value = mock_result

        # Act
        count = await DeltaRepository.count_by_playbook(mock_session, playbook_id)

        # Assert
        assert count == 5

    @pytest.mark.asyncio
    async def test_count_by_playbook_applied_only(
        self, mock_session: AsyncSession
    ) -> None:
        """Test counting applied deltas only."""
        # Arrange
        playbook_id = uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 3
        mock_session.execute.return_value = mock_result

        # Act
        count = await DeltaRepository.count_by_playbook(
            mock_session, playbook_id, applied_only=True
        )

        # Assert
        assert count == 3


class TestTraceRepository:
    """Tests for TraceRepository."""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Create mock AsyncSession."""
        session = MagicMock(spec=AsyncSession)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_create_trace_success(self, mock_session: AsyncSession) -> None:
        """Test creating a successful trace."""
        # Arrange
        agent_id = "agent-001"
        execution_time = 2.5
        success = True
        task_id = "task-123"
        output_quality = 0.92

        # Act
        trace = await TraceRepository.create(
            mock_session,
            agent_id,
            execution_time,
            success,
            task_id,
            output_quality,
        )

        # Assert
        assert trace.agent_id == agent_id
        assert trace.execution_time == execution_time
        assert trace.success is True
        assert trace.task_id == task_id
        assert trace.output_quality == output_quality
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_trace_failure(self, mock_session: AsyncSession) -> None:
        """Test creating a failed trace."""
        # Arrange
        agent_id = "agent-001"
        execution_time = 1.0
        success = False
        error_message = "Task failed"

        # Act
        trace = await TraceRepository.create(
            mock_session,
            agent_id,
            execution_time,
            success,
            error_message=error_message,
        )

        # Assert
        assert trace.success is False
        assert trace.error_message == error_message

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, mock_session: AsyncSession) -> None:
        """Test getting trace by ID when found."""
        # Arrange
        trace_id = uuid4()
        expected_trace = ExecutionTraceDB(
            agent_id="agent-001",
            execution_time=1.0,
            success=True,
        )
        expected_trace.trace_id = trace_id

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_trace
        mock_session.execute.return_value = mock_result

        # Act
        result = await TraceRepository.get_by_id(mock_session, trace_id)

        # Assert
        assert result == expected_trace

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, mock_session: AsyncSession) -> None:
        """Test getting trace by ID when not found."""
        # Arrange
        trace_id = uuid4()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await TraceRepository.get_by_id(mock_session, trace_id)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_agent(self, mock_session: AsyncSession) -> None:
        """Test listing traces for agent."""
        # Arrange
        agent_id = "agent-001"
        traces = [
            ExecutionTraceDB(agent_id=agent_id, execution_time=1.0, success=True),
            ExecutionTraceDB(agent_id=agent_id, execution_time=2.0, success=False),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = traces
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await TraceRepository.list_by_agent(mock_session, agent_id)

        # Assert
        assert result == traces
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_by_agent_success_only(
        self, mock_session: AsyncSession
    ) -> None:
        """Test listing only successful traces."""
        # Arrange
        agent_id = "agent-001"
        traces = [
            ExecutionTraceDB(agent_id=agent_id, execution_time=1.0, success=True),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = traces
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await TraceRepository.list_by_agent(
            mock_session, agent_id, success_only=True
        )

        # Assert
        assert result == traces

    @pytest.mark.asyncio
    async def test_get_recent_for_evolution(self, mock_session: AsyncSession) -> None:
        """Test getting recent traces for evolution."""
        # Arrange
        agent_id = "agent-001"
        traces = [
            ExecutionTraceDB(agent_id=agent_id, execution_time=1.0, success=True)
            for _ in range(10)
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = traces
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await TraceRepository.get_recent_for_evolution(mock_session, agent_id)

        # Assert
        assert result == traces
        assert len(result) == 10

    @pytest.mark.asyncio
    async def test_count_by_agent(self, mock_session: AsyncSession) -> None:
        """Test counting traces for agent."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 15
        mock_session.execute.return_value = mock_result

        # Act
        count = await TraceRepository.count_by_agent(mock_session, agent_id)

        # Assert
        assert count == 15

    @pytest.mark.asyncio
    async def test_count_by_agent_success_only(
        self, mock_session: AsyncSession
    ) -> None:
        """Test counting successful traces only."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 12
        mock_session.execute.return_value = mock_result

        # Act
        count = await TraceRepository.count_by_agent(
            mock_session, agent_id, success_only=True
        )

        # Assert
        assert count == 12

    @pytest.mark.asyncio
    async def test_get_avg_execution_time(self, mock_session: AsyncSession) -> None:
        """Test getting average execution time."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 2.5
        mock_session.execute.return_value = mock_result

        # Act
        avg_time = await TraceRepository.get_avg_execution_time(mock_session, agent_id)

        # Assert
        assert avg_time == 2.5

    @pytest.mark.asyncio
    async def test_delete_old_traces(self, mock_session: AsyncSession) -> None:
        """Test deleting old traces."""
        # Arrange
        agent_id = "agent-001"
        cutoff_trace_id = uuid4()

        # Mock cutoff query
        mock_cutoff_result = MagicMock()
        mock_cutoff_result.scalar_one_or_none.return_value = cutoff_trace_id

        # Mock delete query
        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 5

        mock_session.execute.side_effect = [mock_cutoff_result, mock_delete_result]

        # Act
        deleted_count = await TraceRepository.delete_old_traces(
            mock_session, agent_id, keep_count=100
        )

        # Assert
        assert deleted_count == 5

    @pytest.mark.asyncio
    async def test_delete_old_traces_no_cutoff(
        self, mock_session: AsyncSession
    ) -> None:
        """Test deleting old traces when no cutoff exists."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        deleted_count = await TraceRepository.delete_old_traces(mock_session, agent_id)

        # Assert
        assert deleted_count == 0


class TestEvolutionStatusRepository:
    """Tests for EvolutionStatusRepository."""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Create mock AsyncSession."""
        session = MagicMock(spec=AsyncSession)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_create_status(self, mock_session: AsyncSession) -> None:
        """Test creating evolution status."""
        # Arrange
        agent_id = "agent-001"

        # Act
        status = await EvolutionStatusRepository.create(mock_session, agent_id)

        # Assert
        assert status.agent_id == agent_id
        assert status.pending_traces == 0
        assert status.deltas_generated == 0
        assert status.deltas_applied == 0
        assert status.total_cost == 0.0
        assert status.status == EvolutionStatusType.IDLE
        mock_session.add.assert_called_once()
        mock_session.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_by_agent_id_found(self, mock_session: AsyncSession) -> None:
        """Test getting status by agent ID when found."""
        # Arrange
        agent_id = "agent-001"
        expected_status = EvolutionStatusDB(agent_id=agent_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = expected_status
        mock_session.execute.return_value = mock_result

        # Act
        result = await EvolutionStatusRepository.get_by_agent_id(
            mock_session, agent_id
        )

        # Assert
        assert result == expected_status

    @pytest.mark.asyncio
    async def test_get_by_agent_id_not_found(
        self, mock_session: AsyncSession
    ) -> None:
        """Test getting status by agent ID when not found."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await EvolutionStatusRepository.get_by_agent_id(
            mock_session, agent_id
        )

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_exists(self, mock_session: AsyncSession) -> None:
        """Test get_or_create when status exists."""
        # Arrange
        agent_id = "agent-001"
        existing_status = EvolutionStatusDB(agent_id=agent_id)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_status
        mock_session.execute.return_value = mock_result

        # Act
        result = await EvolutionStatusRepository.get_or_create(mock_session, agent_id)

        # Assert
        assert result == existing_status
        mock_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_new(self, mock_session: AsyncSession) -> None:
        """Test get_or_create when status doesn't exist."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        result = await EvolutionStatusRepository.get_or_create(mock_session, agent_id)

        # Assert
        assert result.agent_id == agent_id
        mock_session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_pending_traces(self, mock_session: AsyncSession) -> None:
        """Test updating pending traces count."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        # Act
        success = await EvolutionStatusRepository.update_pending_traces(
            mock_session, agent_id, delta=5
        )

        # Assert
        assert success is True

    @pytest.mark.asyncio
    async def test_record_evolution(self, mock_session: AsyncSession) -> None:
        """Test recording evolution event."""
        # Arrange
        agent_id = "agent-001"
        deltas_generated = 3
        cost = 0.15
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        # Act
        success = await EvolutionStatusRepository.record_evolution(
            mock_session, agent_id, deltas_generated, cost
        )

        # Assert
        assert success is True

    @pytest.mark.asyncio
    async def test_increment_deltas_applied(self, mock_session: AsyncSession) -> None:
        """Test incrementing deltas applied count."""
        # Arrange
        agent_id = "agent-001"
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        # Act
        success = await EvolutionStatusRepository.increment_deltas_applied(
            mock_session, agent_id
        )

        # Assert
        assert success is True

    @pytest.mark.asyncio
    async def test_update_status(self, mock_session: AsyncSession) -> None:
        """Test updating evolution status."""
        # Arrange
        agent_id = "agent-001"
        new_status = EvolutionStatusType.PROCESSING
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        # Act
        success = await EvolutionStatusRepository.update_status(
            mock_session, agent_id, new_status
        )

        # Assert
        assert success is True

    @pytest.mark.asyncio
    async def test_list_all(self, mock_session: AsyncSession) -> None:
        """Test listing all evolution statuses."""
        # Arrange
        statuses = [
            EvolutionStatusDB(agent_id="agent-001"),
            EvolutionStatusDB(agent_id="agent-002"),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = statuses
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await EvolutionStatusRepository.list_all(mock_session)

        # Assert
        assert result == statuses
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_all_filtered(self, mock_session: AsyncSession) -> None:
        """Test listing evolution statuses filtered by status."""
        # Arrange
        statuses = [
            EvolutionStatusDB(
                agent_id="agent-001", status=EvolutionStatusType.PROCESSING
            ),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = statuses
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Act
        result = await EvolutionStatusRepository.list_all(
            mock_session, status=EvolutionStatusType.PROCESSING
        )

        # Assert
        assert result == statuses

    @pytest.mark.asyncio
    async def test_get_total_cost(self, mock_session: AsyncSession) -> None:
        """Test getting total cost across all agents."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = 5.75
        mock_session.execute.return_value = mock_result

        # Act
        total_cost = await EvolutionStatusRepository.get_total_cost(mock_session)

        # Assert
        assert total_cost == 5.75

    @pytest.mark.asyncio
    async def test_get_total_cost_no_data(self, mock_session: AsyncSession) -> None:
        """Test getting total cost when no data exists."""
        # Arrange
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = None
        mock_session.execute.return_value = mock_result

        # Act
        total_cost = await EvolutionStatusRepository.get_total_cost(mock_session)

        # Assert
        assert total_cost == 0.0
