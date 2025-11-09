"""
Unit tests for PlaybookManager service.

Tests playbook management business logic using mocked repositories.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.ace.database import ContextDeltaDB, ContextPlaybookDB
from agentcore.ace.models.ace_models import (
    PlaybookCreateRequest,
    PlaybookUpdateRequest,
)
from agentcore.ace.services import PlaybookManager


class TestPlaybookManager:
    """Tests for PlaybookManager service."""

    @pytest.fixture
    def mock_session(self) -> AsyncSession:
        """Create mock AsyncSession."""
        session = MagicMock(spec=AsyncSession)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def manager(self, mock_session: AsyncSession) -> PlaybookManager:
        """Create PlaybookManager instance with mocked session."""
        return PlaybookManager(mock_session)

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_create_playbook_success(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test creating a new playbook successfully."""
        # Arrange
        agent_id = "agent-001"
        context = {"goal": "test goal", "strategies": ["strategy1"]}
        metadata = {"version": "1.0"}

        request = PlaybookCreateRequest(
            agent_id=agent_id,
            context=context,
            metadata=metadata,
        )

        # Mock repository responses
        mock_repo.get_by_agent_id = AsyncMock(return_value=None)  # No existing playbook

        playbook_db = ContextPlaybookDB(
            agent_id=agent_id,
            context=context,
            version=1,
        )
        playbook_db.playbook_id = uuid4()
        playbook_db.created_at = datetime.now(UTC)
        playbook_db.updated_at = datetime.now(UTC)
        playbook_db.playbook_metadata = metadata

        mock_repo.create = AsyncMock(return_value=playbook_db)

        # Act
        response = await manager.create_playbook(request)

        # Assert
        assert response.agent_id == agent_id
        assert response.context == context
        assert response.version == 1
        assert response.metadata == metadata
        mock_repo.get_by_agent_id.assert_awaited_once_with(mock_session, agent_id)
        mock_repo.create.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_create_playbook_duplicate_agent(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test creating playbook for agent that already has one."""
        # Arrange
        agent_id = "agent-001"
        request = PlaybookCreateRequest(
            agent_id=agent_id,
            context={"goal": "test"},
        )

        # Mock existing playbook
        existing_playbook = ContextPlaybookDB(
            agent_id=agent_id,
            context={"goal": "existing"},
        )
        mock_repo.get_by_agent_id = AsyncMock(return_value=existing_playbook)

        # Act & Assert
        with pytest.raises(ValueError, match="already has an active playbook"):
            await manager.create_playbook(request)

        mock_repo.get_by_agent_id.assert_awaited_once()
        mock_repo.create.assert_not_called()

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_get_playbook_by_id_found(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test getting playbook by ID when found."""
        # Arrange
        playbook_id = uuid4()
        playbook_db = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "test"},
        )
        playbook_db.playbook_id = playbook_id
        playbook_db.created_at = datetime.now(UTC)
        playbook_db.updated_at = datetime.now(UTC)
        playbook_db.playbook_metadata = {}
        playbook_db.version = 1

        mock_repo.get_by_id = AsyncMock(return_value=playbook_db)

        # Act
        response = await manager.get_playbook(playbook_id=playbook_id)

        # Assert
        assert response is not None
        assert response.playbook_id == playbook_id
        assert response.agent_id == "agent-001"
        mock_repo.get_by_id.assert_awaited_once_with(mock_session, playbook_id)

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_get_playbook_by_id_not_found(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
    ) -> None:
        """Test getting playbook by ID when not found."""
        # Arrange
        playbook_id = uuid4()
        mock_repo.get_by_id = AsyncMock(return_value=None)

        # Act
        response = await manager.get_playbook(playbook_id=playbook_id)

        # Assert
        assert response is None

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_get_playbook_by_agent_id_found(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test getting playbook by agent ID when found."""
        # Arrange
        agent_id = "agent-001"
        playbook_db = ContextPlaybookDB(
            agent_id=agent_id,
            context={"goal": "test"},
        )
        playbook_db.playbook_id = uuid4()
        playbook_db.created_at = datetime.now(UTC)
        playbook_db.updated_at = datetime.now(UTC)
        playbook_db.playbook_metadata = {}
        playbook_db.version = 1

        mock_repo.get_by_agent_id = AsyncMock(return_value=playbook_db)

        # Act
        response = await manager.get_playbook(agent_id=agent_id)

        # Assert
        assert response is not None
        assert response.agent_id == agent_id
        mock_repo.get_by_agent_id.assert_awaited_once_with(mock_session, agent_id)

    @pytest.mark.asyncio
    async def test_get_playbook_no_params(self, manager: PlaybookManager) -> None:
        """Test getting playbook without parameters raises error."""
        # Act & Assert
        with pytest.raises(ValueError, match="Either playbook_id or agent_id"):
            await manager.get_playbook()

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_update_playbook_success(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test updating playbook successfully."""
        # Arrange
        playbook_id = uuid4()
        new_context = {"goal": "updated goal", "strategies": ["new strategy"]}
        request = PlaybookUpdateRequest(context=new_context)

        # Mock existing playbook
        existing_playbook = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "old goal"},
        )
        existing_playbook.playbook_id = playbook_id

        # Mock updated playbook
        updated_playbook = ContextPlaybookDB(
            agent_id="agent-001",
            context=new_context,
        )
        updated_playbook.playbook_id = playbook_id
        updated_playbook.version = 2
        updated_playbook.created_at = datetime.now(UTC)
        updated_playbook.updated_at = datetime.now(UTC)
        updated_playbook.playbook_metadata = {}

        mock_repo.get_by_id = AsyncMock(side_effect=[existing_playbook, updated_playbook])
        mock_repo.update_context = AsyncMock(return_value=True)

        # Act
        response = await manager.update_playbook(playbook_id, request)

        # Assert
        assert response.playbook_id == playbook_id
        assert response.context == new_context
        assert response.version == 2
        mock_repo.update_context.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_update_playbook_not_found(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
    ) -> None:
        """Test updating playbook that doesn't exist."""
        # Arrange
        playbook_id = uuid4()
        request = PlaybookUpdateRequest(context={"goal": "test"})

        mock_repo.get_by_id = AsyncMock(return_value=None)

        # Act & Assert
        with pytest.raises(ValueError, match="not found"):
            await manager.update_playbook(playbook_id, request)

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.DeltaRepository")
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_apply_delta_success(
        self,
        mock_playbook_repo: MagicMock,
        mock_delta_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test applying delta to playbook successfully."""
        # Arrange
        playbook_id = uuid4()
        delta_id = uuid4()

        # Mock playbook
        playbook_db = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "original", "temperature": 0.7},
        )
        playbook_db.playbook_id = playbook_id

        # Mock delta
        delta_db = ContextDeltaDB(
            playbook_id=playbook_id,
            changes={"temperature": 0.8, "strategies": ["new"]},
            confidence=0.9,
            reasoning="test",
        )
        delta_db.delta_id = delta_id
        delta_db.applied = False

        # Mock updated playbook
        updated_playbook = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "original", "temperature": 0.8, "strategies": ["new"]},
        )
        updated_playbook.playbook_id = playbook_id
        updated_playbook.version = 2
        updated_playbook.created_at = datetime.now(UTC)
        updated_playbook.updated_at = datetime.now(UTC)
        updated_playbook.playbook_metadata = {}

        mock_playbook_repo.get_by_id = AsyncMock(
            side_effect=[playbook_db, updated_playbook]
        )
        mock_delta_repo.get_by_id = AsyncMock(return_value=delta_db)
        mock_playbook_repo.update_context = AsyncMock(return_value=True)
        mock_delta_repo.mark_applied = AsyncMock(return_value=True)

        # Act
        response = await manager.apply_delta(playbook_id, delta_id)

        # Assert
        assert response.playbook_id == playbook_id
        assert response.context["temperature"] == 0.8
        assert "strategies" in response.context
        assert response.version == 2
        mock_delta_repo.mark_applied.assert_awaited_once()
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.DeltaRepository")
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_apply_delta_already_applied(
        self,
        mock_playbook_repo: MagicMock,
        mock_delta_repo: MagicMock,
        manager: PlaybookManager,
    ) -> None:
        """Test applying delta that was already applied."""
        # Arrange
        playbook_id = uuid4()
        delta_id = uuid4()

        playbook_db = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "test"},
        )
        playbook_db.playbook_id = playbook_id

        delta_db = ContextDeltaDB(
            playbook_id=playbook_id,
            changes={"test": "value"},
            confidence=0.9,
            reasoning="test",
        )
        delta_db.delta_id = delta_id
        delta_db.applied = True  # Already applied

        mock_playbook_repo.get_by_id = AsyncMock(return_value=playbook_db)
        mock_delta_repo.get_by_id = AsyncMock(return_value=delta_db)

        # Act & Assert
        with pytest.raises(ValueError, match="already applied"):
            await manager.apply_delta(playbook_id, delta_id)

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.DeltaRepository")
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_apply_delta_wrong_playbook(
        self,
        mock_playbook_repo: MagicMock,
        mock_delta_repo: MagicMock,
        manager: PlaybookManager,
    ) -> None:
        """Test applying delta to wrong playbook."""
        # Arrange
        playbook_id = uuid4()
        other_playbook_id = uuid4()
        delta_id = uuid4()

        playbook_db = ContextPlaybookDB(
            agent_id="agent-001",
            context={"goal": "test"},
        )
        playbook_db.playbook_id = playbook_id

        delta_db = ContextDeltaDB(
            playbook_id=other_playbook_id,  # Different playbook
            changes={"test": "value"},
            confidence=0.9,
            reasoning="test",
        )
        delta_db.delta_id = delta_id
        delta_db.applied = False

        mock_playbook_repo.get_by_id = AsyncMock(return_value=playbook_db)
        mock_delta_repo.get_by_id = AsyncMock(return_value=delta_db)

        # Act & Assert
        with pytest.raises(ValueError, match="does not belong to playbook"):
            await manager.apply_delta(playbook_id, delta_id)

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_compile_context_by_playbook_id(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test compiling context by playbook ID."""
        # Arrange
        playbook_id = uuid4()
        playbook_db = ContextPlaybookDB(
            agent_id="agent-001",
            context={
                "goal": "Complete task successfully",
                "strategies": {"planning": "Use step-by-step approach"},
                "failures": ["Error in step 3"],
            },
        )
        playbook_db.playbook_id = playbook_id
        playbook_db.version = 2
        playbook_db.created_at = datetime.now(UTC)
        playbook_db.updated_at = datetime.now(UTC)
        playbook_db.playbook_metadata = {}

        mock_repo.get_by_id = AsyncMock(return_value=playbook_db)

        # Act
        context_str = await manager.compile_context(playbook_id=playbook_id)

        # Assert
        assert "Agent Context (v2)" in context_str
        assert "Goal" in context_str
        assert "Complete task successfully" in context_str
        assert "Strategies" in context_str
        assert "planning" in context_str
        assert "Failures" in context_str
        assert "Error in step 3" in context_str

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_compile_context_not_found(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
    ) -> None:
        """Test compiling context when playbook not found."""
        # Arrange
        playbook_id = uuid4()
        mock_repo.get_by_id = AsyncMock(return_value=None)

        # Act & Assert
        with pytest.raises(ValueError, match="Playbook not found"):
            await manager.compile_context(playbook_id=playbook_id)

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_list_playbooks(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test listing playbooks for agent."""
        # Arrange
        agent_id = "agent-001"
        playbooks = [
            ContextPlaybookDB(agent_id=agent_id, context={"goal": "test1"}),
            ContextPlaybookDB(agent_id=agent_id, context={"goal": "test2"}),
        ]

        for i, pb in enumerate(playbooks):
            pb.playbook_id = uuid4()
            pb.version = i + 1
            pb.created_at = datetime.now(UTC)
            pb.updated_at = datetime.now(UTC)
            pb.playbook_metadata = {}

        mock_repo.list_by_agent = AsyncMock(return_value=playbooks)

        # Act
        responses = await manager.list_playbooks(agent_id, limit=10)

        # Assert
        assert len(responses) == 2
        assert all(r.agent_id == agent_id for r in responses)
        mock_repo.list_by_agent.assert_awaited_once_with(mock_session, agent_id, 10)

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_delete_playbook_success(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
        mock_session: AsyncSession,
    ) -> None:
        """Test deleting playbook successfully."""
        # Arrange
        playbook_id = uuid4()
        mock_repo.delete = AsyncMock(return_value=True)

        # Act
        result = await manager.delete_playbook(playbook_id)

        # Assert
        assert result is True
        mock_repo.delete.assert_awaited_once_with(mock_session, playbook_id)
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("agentcore.ace.services.playbook_manager.PlaybookRepository")
    async def test_delete_playbook_not_found(
        self,
        mock_repo: MagicMock,
        manager: PlaybookManager,
    ) -> None:
        """Test deleting playbook that doesn't exist."""
        # Arrange
        playbook_id = uuid4()
        mock_repo.delete = AsyncMock(return_value=False)

        # Act
        result = await manager.delete_playbook(playbook_id)

        # Assert
        assert result is False

    def test_apply_delta_changes_merge(self, manager: PlaybookManager) -> None:
        """Test _apply_delta_changes with dictionary merge."""
        # Arrange
        context = {
            "goal": "original",
            "strategies": {"planning": "step1"},
            "temperature": 0.7,
        }
        changes = {"strategies": {"execution": "step2"}, "temperature": 0.8}

        # Act
        result = manager._apply_delta_changes(context, changes)

        # Assert
        assert result["goal"] == "original"
        assert result["temperature"] == 0.8
        assert "planning" in result["strategies"]
        assert "execution" in result["strategies"]

    def test_apply_delta_changes_replace(self, manager: PlaybookManager) -> None:
        """Test _apply_delta_changes with value replacement."""
        # Arrange
        context = {"goal": "original", "temperature": 0.7}
        changes = {"goal": "updated", "new_field": "value"}

        # Act
        result = manager._apply_delta_changes(context, changes)

        # Assert
        assert result["goal"] == "updated"
        assert result["temperature"] == 0.7
        assert result["new_field"] == "value"
