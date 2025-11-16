"""
Stage Manager for COMPASS Hierarchical Organization

Manages stage lifecycle, stage detection, and compression triggers for COMPASS
reasoning stages. Implements stage-aware memory organization with hierarchical
context management.

Component ID: MEM-008
Ticket: MEM-008 (Implement StageManager Core)
"""

from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.database.repositories import StageMemoryRepository
from agentcore.a2a_protocol.models.memory import StageMemory, StageType

logger = structlog.get_logger()


class CompressionTrigger(Protocol):
    """
    Protocol for compression service integration.

    This is a stub interface for MEM-012 (ContextCompressor).
    When MEM-012 is implemented, replace this with the actual service.
    """

    async def compress_stage(
        self, stage_id: str, raw_memory_ids: list[str]
    ) -> dict[str, float]:
        """
        Trigger stage compression.

        Args:
            stage_id: Stage to compress
            raw_memory_ids: Raw memory IDs to compress

        Returns:
            Compression metrics (ratio, quality_score)
        """
        ...


class StageManager:
    """
    Manages COMPASS stage lifecycle and compression triggers.

    Implements:
    - Stage creation (planning, execution, reflection, verification)
    - Stage completion with compression trigger
    - Memory linking to stages
    - Stage context retrieval
    - Hierarchical stage organization
    """

    def __init__(
        self,
        compression_trigger: CompressionTrigger | None = None,
    ):
        """
        Initialize StageManager.

        Args:
            compression_trigger: Optional compression service (stub for MEM-012)
        """
        self._compression_trigger = compression_trigger
        self._logger = logger.bind(component="stage_manager")

    async def create_stage(
        self,
        session: AsyncSession,
        task_id: str,
        agent_id: str,
        stage_type: StageType,
        stage_summary: str | None = None,
    ) -> StageMemory:
        """
        Create a new COMPASS reasoning stage.

        Args:
            session: Database session
            task_id: Parent task ID
            agent_id: Agent ID creating the stage
            stage_type: Stage type (planning, execution, reflection, verification)
            stage_summary: Optional initial summary

        Returns:
            Created StageMemory

        Raises:
            ValueError: If task_id or agent_id is invalid
        """
        if not task_id or not agent_id:
            raise ValueError("task_id and agent_id are required")

        stage_id = f"stage-{uuid4()}"

        # Default summary based on stage type
        if not stage_summary:
            stage_summary = f"Started {stage_type.value} stage for task {task_id}"

        stage_memory = StageMemory(
            stage_id=stage_id,
            task_id=task_id,
            agent_id=agent_id,
            stage_type=stage_type,
            stage_summary=stage_summary,
            stage_insights=[],
            raw_memory_refs=[],
            relevance_score=1.0,
            compression_ratio=1.0,
            compression_model="none",
            quality_score=1.0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            completed_at=None,
        )

        # Persist to database
        await StageMemoryRepository.create(session, stage_memory)
        await session.commit()

        self._logger.info(
            "stage_created",
            stage_id=stage_id,
            task_id=task_id,
            agent_id=agent_id,
            stage_type=stage_type.value,
        )

        return stage_memory

    async def complete_stage(
        self,
        session: AsyncSession,
        stage_id: str,
        final_summary: str | None = None,
        stage_insights: list[str] | None = None,
    ) -> StageMemory:
        """
        Complete a stage and trigger compression.

        Args:
            session: Database session
            stage_id: Stage to complete
            final_summary: Optional final summary
            stage_insights: Optional key insights from stage

        Returns:
            Updated StageMemory with compression metrics

        Raises:
            ValueError: If stage_id is invalid or stage not found
        """
        if not stage_id:
            raise ValueError("stage_id is required")

        # Retrieve stage
        stage_db = await StageMemoryRepository.get_by_id(session, stage_id)
        if not stage_db:
            raise ValueError(f"Stage {stage_id} not found")

        # Convert to Pydantic
        stage_memory = stage_db.to_pydantic()

        # Update summary if provided
        if final_summary:
            stage_memory.stage_summary = final_summary

        # Update insights if provided
        if stage_insights:
            stage_memory.stage_insights = stage_insights

        # Mark as completed
        stage_memory.completed_at = datetime.now(UTC)
        stage_memory.updated_at = datetime.now(UTC)

        # Trigger compression if we have raw memories and compression service
        compression_metrics = {
            "compression_ratio": 1.0,
            "quality_score": 1.0,
        }

        if stage_memory.raw_memory_refs and self._compression_trigger:
            try:
                compression_metrics = await self._compression_trigger.compress_stage(
                    stage_id=stage_id,
                    raw_memory_ids=stage_memory.raw_memory_refs,
                )
                self._logger.info(
                    "stage_compressed",
                    stage_id=stage_id,
                    compression_ratio=compression_metrics.get("compression_ratio", 1.0),
                    quality_score=compression_metrics.get("quality_score", 1.0),
                )
            except Exception as e:
                self._logger.warning(
                    "compression_failed",
                    stage_id=stage_id,
                    error=str(e),
                )

        # Update compression metrics
        stage_memory.compression_ratio = compression_metrics.get("compression_ratio", 1.0)
        stage_memory.quality_score = compression_metrics.get("quality_score", 1.0)

        # Persist updates
        await StageMemoryRepository.update(
            session,
            stage_id,
            stage_summary=stage_memory.stage_summary,
            stage_insights=stage_memory.stage_insights,
            compression_ratio=stage_memory.compression_ratio,
            quality_metrics={"quality_score": stage_memory.quality_score},
            completed_at=stage_memory.completed_at,
            updated_at=stage_memory.updated_at,
        )
        await session.commit()

        self._logger.info(
            "stage_completed",
            stage_id=stage_id,
            task_id=stage_memory.task_id,
            stage_type=stage_memory.stage_type.value,
            raw_memory_count=len(stage_memory.raw_memory_refs),
        )

        return stage_memory

    async def get_current_stage(
        self,
        session: AsyncSession,
        task_id: str,
    ) -> StageMemory | None:
        """
        Get the current active stage for a task.

        Args:
            session: Database session
            task_id: Task ID

        Returns:
            Current StageMemory or None if no active stage
        """
        if not task_id:
            return None

        # Get all stages for task
        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        stage_dbs: list[StageMemoryModel] = await StageMemoryRepository.get_by_task(session, task_id)

        # Find most recent incomplete stage
        for stage_db in reversed(stage_dbs):  # Reversed for most recent first
            if stage_db.completed_at is None:
                return stage_db.to_pydantic()

        return None

    async def get_stage_by_id(
        self,
        session: AsyncSession,
        stage_id: str,
    ) -> StageMemory | None:
        """
        Get a stage by ID.

        Args:
            session: Database session
            stage_id: Stage ID

        Returns:
            StageMemory or None if not found
        """
        if not stage_id:
            return None

        from agentcore.a2a_protocol.database.memory_models import StageMemoryModel

        stage_db: StageMemoryModel | None = await StageMemoryRepository.get_by_id(session, stage_id)
        return stage_db.to_pydantic() if stage_db else None

    async def get_stages_by_task(
        self,
        session: AsyncSession,
        task_id: str,
    ) -> list[StageMemory]:
        """
        Get all stages for a task.

        Args:
            session: Database session
            task_id: Task ID

        Returns:
            List of StageMemory in chronological order
        """
        if not task_id:
            return []

        stage_dbs = await StageMemoryRepository.get_by_task(session, task_id)
        return [stage_db.to_pydantic() for stage_db in stage_dbs]

    async def get_stages_by_type(
        self,
        session: AsyncSession,
        task_id: str,
        stage_type: StageType,
    ) -> list[StageMemory]:
        """
        Get all stages of a specific type for a task.

        Args:
            session: Database session
            task_id: Task ID
            stage_type: Stage type to filter

        Returns:
            List of matching StageMemory
        """
        if not task_id:
            return []

        stage_dbs = await StageMemoryRepository.get_by_task_and_stage(
            session, task_id, stage_type
        )
        return [stage_db.to_pydantic() for stage_db in stage_dbs]

    async def link_memory_to_stage(
        self,
        session: AsyncSession,
        stage_id: str,
        memory_id: str,
    ) -> bool:
        """
        Link a memory to a stage.

        Args:
            session: Database session
            stage_id: Stage ID
            memory_id: Memory ID to link

        Returns:
            True if successful

        Raises:
            ValueError: If stage not found or already completed
        """
        if not stage_id or not memory_id:
            raise ValueError("stage_id and memory_id are required")

        # Retrieve stage
        stage_db = await StageMemoryRepository.get_by_id(session, stage_id)
        if not stage_db:
            raise ValueError(f"Stage {stage_id} not found")

        # Check if stage is completed
        if stage_db.completed_at is not None:
            raise ValueError(f"Cannot link memory to completed stage {stage_id}")

        # Convert to Pydantic
        stage_memory = stage_db.to_pydantic()

        # Add memory reference if not already present
        if memory_id not in stage_memory.raw_memory_refs:
            stage_memory.raw_memory_refs.append(memory_id)
            stage_memory.updated_at = datetime.now(UTC)

            # Persist update
            await StageMemoryRepository.update(
                session,
                stage_id,
                raw_memory_refs=stage_memory.raw_memory_refs,
                updated_at=stage_memory.updated_at,
            )
            await session.commit()

            self._logger.debug(
                "memory_linked_to_stage",
                stage_id=stage_id,
                memory_id=memory_id,
                total_memories=len(stage_memory.raw_memory_refs),
            )

            return True

        return False

    async def get_stage_context(
        self,
        session: AsyncSession,
        stage_id: str,
    ) -> dict[str, object]:
        """
        Get comprehensive stage context for retrieval.

        Args:
            session: Database session
            stage_id: Stage ID

        Returns:
            Dictionary with stage metadata and context
        """
        if not stage_id:
            return {}

        stage_db = await StageMemoryRepository.get_by_id(session, stage_id)
        if not stage_db:
            return {}

        stage_memory = stage_db.to_pydantic()

        return {
            "stage_id": stage_memory.stage_id,
            "task_id": stage_memory.task_id,
            "agent_id": stage_memory.agent_id,
            "stage_type": stage_memory.stage_type.value,
            "stage_summary": stage_memory.stage_summary,
            "stage_insights": stage_memory.stage_insights,
            "raw_memory_count": len(stage_memory.raw_memory_refs),
            "raw_memory_refs": stage_memory.raw_memory_refs,
            "is_completed": stage_memory.completed_at is not None,
            "compression_ratio": stage_memory.compression_ratio,
            "quality_score": stage_memory.quality_score,
            "created_at": stage_memory.created_at.isoformat(),
            "completed_at": stage_memory.completed_at.isoformat()
            if stage_memory.completed_at
            else None,
        }


__all__ = ["StageManager", "CompressionTrigger"]
