"""
Unit Tests for StageManager

Tests COMPASS stage lifecycle management, compression triggers,
memory linking, and stage context retrieval.

Component ID: MEM-008
Ticket: MEM-008 (Implement StageManager Core)
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.database.memory_models import StageMemoryModel
from agentcore.a2a_protocol.models.memory import StageMemory, StageType
from agentcore.a2a_protocol.services.memory import CompressionTrigger, StageManager


@pytest.fixture
def mock_session() -> AsyncMock:
    """Create mock database session."""
    session = AsyncMock(spec=AsyncSession)
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def mock_compression_trigger() -> AsyncMock:
    """Create mock compression trigger."""
    trigger = AsyncMock(spec=CompressionTrigger)
    trigger.compress_stage = AsyncMock(
        return_value={
            "compression_ratio": 10.2,
            "quality_score": 0.97,
        }
    )
    return trigger


@pytest.fixture
def stage_manager(mock_compression_trigger: AsyncMock) -> StageManager:
    """Create StageManager instance."""
    return StageManager(compression_trigger=mock_compression_trigger)


@pytest.fixture
def stage_manager_no_compression() -> StageManager:
    """Create StageManager without compression trigger."""
    return StageManager(compression_trigger=None)


@pytest.fixture
def sample_stage_memory() -> StageMemory:
    """Create sample StageMemory for testing."""
    return StageMemory(
        stage_id=f"stage-{uuid4()}",
        task_id=f"task-{uuid4()}",
        agent_id=f"agent-{uuid4()}",
        stage_type=StageType.PLANNING,
        stage_summary="Planning authentication system",
        stage_insights=["Use JWT", "Store in Redis"],
        raw_memory_refs=[f"mem-{uuid4()}", f"mem-{uuid4()}"],
        compression_ratio=1.0,
        compression_model="none",
        quality_score=1.0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        completed_at=None,
    )


@pytest.fixture
def sample_stage_db(sample_stage_memory: StageMemory) -> StageMemoryModel:
    """Create sample StageMemoryModel for testing."""
    from uuid import UUID

    def extract_uuid(id_str: str) -> UUID:
        """Extract UUID from prefixed ID."""
        if "-" in id_str and not id_str.count("-") == 4:
            return UUID(id_str.split("-", 1)[1])
        return UUID(id_str)

    model = MagicMock(spec=StageMemoryModel)
    model.stage_id = extract_uuid(sample_stage_memory.stage_id)
    model.task_id = extract_uuid(sample_stage_memory.task_id)
    model.agent_id = extract_uuid(sample_stage_memory.agent_id)
    model.stage_type = sample_stage_memory.stage_type
    model.stage_summary = sample_stage_memory.stage_summary
    model.stage_insights = sample_stage_memory.stage_insights
    model.raw_memory_refs = [
        extract_uuid(ref) for ref in sample_stage_memory.raw_memory_refs
    ]
    model.relevance_score = sample_stage_memory.relevance_score
    model.compression_ratio = sample_stage_memory.compression_ratio
    model.compression_model = sample_stage_memory.compression_model
    model.quality_metrics = {"quality_score": sample_stage_memory.quality_score}
    model.created_at = sample_stage_memory.created_at
    model.updated_at = sample_stage_memory.updated_at
    model.completed_at = sample_stage_memory.completed_at
    model.to_pydantic = MagicMock(return_value=sample_stage_memory)
    return model


class TestStageCreation:
    """Test stage creation functionality."""

    @pytest.mark.asyncio
    async def test_create_stage_all_types(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test creating stages of all types."""
        task_id = f"task-{uuid4()}"
        agent_id = f"agent-{uuid4()}"

        for stage_type in StageType:
            with patch(
                "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
            ) as mock_repo:
                mock_repo.create = AsyncMock()

                stage = await stage_manager.create_stage(
                    session=mock_session,
                    task_id=task_id,
                    agent_id=agent_id,
                    stage_type=stage_type,
                )

                assert stage.task_id == task_id
                assert stage.agent_id == agent_id
                assert stage.stage_type == stage_type
                assert stage.completed_at is None
                assert stage.compression_ratio == 1.0
                assert stage.quality_score == 1.0
                assert len(stage.raw_memory_refs) == 0

                mock_repo.create.assert_called_once()
                mock_session.commit.assert_called_once()
                mock_session.reset_mock()

    @pytest.mark.asyncio
    async def test_create_stage_with_custom_summary(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test creating stage with custom summary."""
        task_id = f"task-{uuid4()}"
        agent_id = f"agent-{uuid4()}"
        custom_summary = "Custom planning summary"

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.create = AsyncMock()

            stage = await stage_manager.create_stage(
                session=mock_session,
                task_id=task_id,
                agent_id=agent_id,
                stage_type=StageType.PLANNING,
                stage_summary=custom_summary,
            )

            assert stage.stage_summary == custom_summary

    @pytest.mark.asyncio
    async def test_create_stage_generates_default_summary(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test creating stage generates default summary."""
        task_id = f"task-{uuid4()}"
        agent_id = f"agent-{uuid4()}"

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.create = AsyncMock()

            stage = await stage_manager.create_stage(
                session=mock_session,
                task_id=task_id,
                agent_id=agent_id,
                stage_type=StageType.EXECUTION,
            )

            assert "execution" in stage.stage_summary.lower()
            assert task_id in stage.stage_summary

    @pytest.mark.asyncio
    async def test_create_stage_invalid_inputs(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test creating stage with invalid inputs raises ValueError."""
        with pytest.raises(ValueError, match="task_id and agent_id are required"):
            await stage_manager.create_stage(
                session=mock_session,
                task_id="",
                agent_id="agent-123",
                stage_type=StageType.PLANNING,
            )

        with pytest.raises(ValueError, match="task_id and agent_id are required"):
            await stage_manager.create_stage(
                session=mock_session,
                task_id="task-123",
                agent_id="",
                stage_type=StageType.PLANNING,
            )


class TestStageCompletion:
    """Test stage completion and compression functionality."""

    @pytest.mark.asyncio
    async def test_complete_stage_triggers_compression(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        mock_compression_trigger: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test completing stage triggers compression."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)
            mock_repo.update = AsyncMock()

            completed_stage = await stage_manager.complete_stage(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
            )

            assert completed_stage.completed_at is not None
            assert completed_stage.compression_ratio == 10.2
            assert completed_stage.quality_score == 0.97

            mock_compression_trigger.compress_stage.assert_called_once_with(
                stage_id=sample_stage_memory.stage_id,
                raw_memory_ids=sample_stage_memory.raw_memory_refs,
            )
            mock_repo.update.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_stage_with_final_summary(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test completing stage with final summary."""
        final_summary = "Authentication planning completed successfully"

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)
            mock_repo.update = AsyncMock()

            completed_stage = await stage_manager.complete_stage(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
                final_summary=final_summary,
            )

            assert completed_stage.stage_summary == final_summary

    @pytest.mark.asyncio
    async def test_complete_stage_with_insights(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test completing stage with insights."""
        insights = ["JWT tokens chosen", "Redis for session storage"]

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)
            mock_repo.update = AsyncMock()

            completed_stage = await stage_manager.complete_stage(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
                stage_insights=insights,
            )

            assert completed_stage.stage_insights == insights

    @pytest.mark.asyncio
    async def test_complete_stage_no_compression_trigger(
        self,
        stage_manager_no_compression: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test completing stage without compression trigger."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)
            mock_repo.update = AsyncMock()

            completed_stage = await stage_manager_no_compression.complete_stage(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
            )

            assert completed_stage.completed_at is not None
            assert completed_stage.compression_ratio == 1.0
            assert completed_stage.quality_score == 1.0

    @pytest.mark.asyncio
    async def test_complete_stage_compression_failure_handled(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        mock_compression_trigger: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test completing stage handles compression failure gracefully."""
        mock_compression_trigger.compress_stage.side_effect = Exception(
            "Compression failed"
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)
            mock_repo.update = AsyncMock()

            completed_stage = await stage_manager.complete_stage(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
            )

            assert completed_stage.completed_at is not None
            assert completed_stage.compression_ratio == 1.0
            assert completed_stage.quality_score == 1.0

    @pytest.mark.asyncio
    async def test_complete_stage_not_found(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test completing non-existent stage raises ValueError."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(ValueError, match="Stage .+ not found"):
                await stage_manager.complete_stage(
                    session=mock_session,
                    stage_id=f"stage-{uuid4()}",
                )

    @pytest.mark.asyncio
    async def test_complete_stage_invalid_stage_id(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test completing stage with invalid stage_id raises ValueError."""
        with pytest.raises(ValueError, match="stage_id is required"):
            await stage_manager.complete_stage(
                session=mock_session,
                stage_id="",
            )


class TestMemoryLinking:
    """Test memory linking to stages."""

    @pytest.mark.asyncio
    async def test_link_memory_to_stage(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test linking memory to stage."""
        memory_id = f"mem-{uuid4()}"

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)
            mock_repo.update = AsyncMock()

            result = await stage_manager.link_memory_to_stage(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
                memory_id=memory_id,
            )

            assert result is True
            mock_repo.update.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_link_memory_duplicate_ignored(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test linking duplicate memory is ignored."""
        memory_id = sample_stage_memory.raw_memory_refs[0]

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)
            mock_repo.update = AsyncMock()

            result = await stage_manager.link_memory_to_stage(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
                memory_id=memory_id,
            )

            assert result is False
            mock_repo.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_link_memory_to_completed_stage_fails(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test linking memory to completed stage raises ValueError."""
        sample_stage_db.completed_at = datetime.now(UTC)
        memory_id = f"mem-{uuid4()}"

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)

            with pytest.raises(ValueError, match="Cannot link memory to completed stage"):
                await stage_manager.link_memory_to_stage(
                    session=mock_session,
                    stage_id=sample_stage_memory.stage_id,
                    memory_id=memory_id,
                )

    @pytest.mark.asyncio
    async def test_link_memory_stage_not_found(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test linking memory to non-existent stage raises ValueError."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            with pytest.raises(ValueError, match="Stage .+ not found"):
                await stage_manager.link_memory_to_stage(
                    session=mock_session,
                    stage_id=f"stage-{uuid4()}",
                    memory_id=f"mem-{uuid4()}",
                )

    @pytest.mark.asyncio
    async def test_link_memory_invalid_inputs(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test linking memory with invalid inputs raises ValueError."""
        with pytest.raises(ValueError, match="stage_id and memory_id are required"):
            await stage_manager.link_memory_to_stage(
                session=mock_session,
                stage_id="",
                memory_id="mem-123",
            )

        with pytest.raises(ValueError, match="stage_id and memory_id are required"):
            await stage_manager.link_memory_to_stage(
                session=mock_session,
                stage_id="stage-123",
                memory_id="",
            )


class TestStageRetrieval:
    """Test stage retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_current_stage(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test getting current active stage."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_task = AsyncMock(return_value=[sample_stage_db])

            current = await stage_manager.get_current_stage(
                session=mock_session,
                task_id=sample_stage_memory.task_id,
            )

            assert current is not None
            assert current.stage_id == sample_stage_memory.stage_id
            assert current.completed_at is None

    @pytest.mark.asyncio
    async def test_get_current_stage_none_if_all_completed(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test getting current stage returns None if all completed."""
        sample_stage_db.completed_at = datetime.now(UTC)

        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_task = AsyncMock(return_value=[sample_stage_db])

            current = await stage_manager.get_current_stage(
                session=mock_session,
                task_id=sample_stage_memory.task_id,
            )

            assert current is None

    @pytest.mark.asyncio
    async def test_get_stage_by_id(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test getting stage by ID."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)

            stage = await stage_manager.get_stage_by_id(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
            )

            assert stage is not None
            assert stage.stage_id == sample_stage_memory.stage_id

    @pytest.mark.asyncio
    async def test_get_stages_by_task(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test getting all stages for a task."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_task = AsyncMock(return_value=[sample_stage_db])

            stages = await stage_manager.get_stages_by_task(
                session=mock_session,
                task_id=sample_stage_memory.task_id,
            )

            assert len(stages) == 1
            assert stages[0].stage_id == sample_stage_memory.stage_id

    @pytest.mark.asyncio
    async def test_get_stages_by_type(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test getting stages by type."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_task_and_stage = AsyncMock(return_value=[sample_stage_db])

            stages = await stage_manager.get_stages_by_type(
                session=mock_session,
                task_id=sample_stage_memory.task_id,
                stage_type=StageType.PLANNING,
            )

            assert len(stages) == 1
            assert stages[0].stage_type == StageType.PLANNING

    @pytest.mark.asyncio
    async def test_get_stage_context(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
        sample_stage_db: StageMemoryModel,
        sample_stage_memory: StageMemory,
    ):
        """Test getting comprehensive stage context."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=sample_stage_db)

            context = await stage_manager.get_stage_context(
                session=mock_session,
                stage_id=sample_stage_memory.stage_id,
            )

            assert context["stage_id"] == sample_stage_memory.stage_id
            assert context["task_id"] == sample_stage_memory.task_id
            assert context["agent_id"] == sample_stage_memory.agent_id
            assert context["stage_type"] == sample_stage_memory.stage_type.value
            assert context["stage_summary"] == sample_stage_memory.stage_summary
            assert context["stage_insights"] == sample_stage_memory.stage_insights
            assert context["raw_memory_count"] == len(sample_stage_memory.raw_memory_refs)
            assert context["raw_memory_refs"] == sample_stage_memory.raw_memory_refs
            assert context["is_completed"] is False
            assert context["compression_ratio"] == sample_stage_memory.compression_ratio
            assert context["quality_score"] == sample_stage_memory.quality_score

    @pytest.mark.asyncio
    async def test_get_stage_context_not_found(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test getting context for non-existent stage returns empty dict."""
        with patch(
            "agentcore.a2a_protocol.services.memory.stage_manager.StageMemoryRepository"
        ) as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=None)

            context = await stage_manager.get_stage_context(
                session=mock_session,
                stage_id=f"stage-{uuid4()}",
            )

            assert context == {}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_retrieval_with_empty_task_id(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test retrieval methods handle empty task_id gracefully."""
        result = await stage_manager.get_current_stage(
            session=mock_session,
            task_id="",
        )
        assert result is None

        stages = await stage_manager.get_stages_by_task(
            session=mock_session,
            task_id="",
        )
        assert stages == []

        stages = await stage_manager.get_stages_by_type(
            session=mock_session,
            task_id="",
            stage_type=StageType.PLANNING,
        )
        assert stages == []

    @pytest.mark.asyncio
    async def test_get_stage_by_id_with_empty_id(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test getting stage by empty ID returns None."""
        result = await stage_manager.get_stage_by_id(
            session=mock_session,
            stage_id="",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_stage_context_with_empty_id(
        self,
        stage_manager: StageManager,
        mock_session: AsyncMock,
    ):
        """Test getting context with empty ID returns empty dict."""
        context = await stage_manager.get_stage_context(
            session=mock_session,
            stage_id="",
        )
        assert context == {}
