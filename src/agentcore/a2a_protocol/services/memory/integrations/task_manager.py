"""
TaskManager Memory Integration

Provides artifact storage integration for TaskManager service.

Features:
- Task artifact storage in memory system
- Artifact retrieval by similarity
- Cross-task artifact linking
- Artifact versioning and history

Component ID: MEM-025
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

from agentcore.a2a_protocol.models.memory import MemoryLayer, MemoryRecord
from agentcore.a2a_protocol.models.task import TaskArtifact
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)

logger = structlog.get_logger(__name__)


class ArtifactMemoryRecord(BaseModel):
    """Memory record for task artifact."""

    artifact_name: str
    task_id: str
    execution_id: str
    memory_id: str
    content_hash: str
    stored_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retrieval_count: int = 0


class ArtifactMemoryStorage:
    """
    Task artifact persistence in memory layer.

    Integrates with TaskManager to provide:
    - Artifact storage in memory graph
    - Artifact retrieval by similarity
    - Artifact versioning and history
    - Cross-task artifact sharing

    Usage:
        storage = ArtifactMemoryStorage(
            hybrid_search=hybrid_search
        )

        # Store artifact
        memory_id = await storage.store_task_artifact(
            task_id="task-123",
            artifact_data={"type": "result", "content": "..."}
        )

        # Retrieve artifacts
        artifacts = await storage.retrieve_task_artifacts(
            task_id="task-123"
        )

        # Link artifacts
        await storage.link_artifacts(
            task_id_1="task-123",
            task_id_2="task-456",
            relationship="depends_on"
        )
    """

    def __init__(
        self,
        hybrid_search: HybridSearchService | None = None,
        retrieval_service: EnhancedRetrievalService | None = None,
    ):
        """
        Initialize ArtifactMemoryStorage.

        Args:
            hybrid_search: Hybrid search service for retrieval
            retrieval_service: Retrieval service for scoring
        """
        self.hybrid_search = hybrid_search
        self.retrieval = retrieval_service or EnhancedRetrievalService()

        self._artifact_records: dict[str, ArtifactMemoryRecord] = {}
        self._memory_store: dict[str, MemoryRecord] = {}
        self._task_artifacts: dict[str, list[str]] = {}  # task_id -> artifact_ids
        self._artifact_links: dict[str, list[tuple[str, str]]] = {}  # task_id -> [(linked_task, relationship)]

        self._logger = logger.bind(component="artifact_memory_storage")
        self._logger.info("initialized_artifact_memory_storage")

    async def store_task_artifact(
        self,
        task_id: str,
        artifact_data: dict[str, Any],
        execution_id: str | None = None,
        embedding: list[float] | None = None,
        agent_id: str = "system",
    ) -> str:
        """
        Store task artifact in memory.

        Args:
            task_id: Task identifier
            artifact_data: Artifact data to store
            execution_id: Optional execution identifier
            embedding: Optional embedding for artifact content
            agent_id: Agent ID

        Returns:
            Memory ID of stored artifact
        """
        # Extract artifact name and type
        artifact_name = artifact_data.get("name", f"artifact-{task_id}")
        artifact_type = artifact_data.get("type", "unknown")

        # Create content hash
        content_str = json.dumps(artifact_data, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        # Create memory record
        memory = MemoryRecord(
            memory_id=f"artifact-{task_id}-{artifact_name}-{datetime.now(UTC).isoformat()}",
            memory_layer=MemoryLayer.SEMANTIC,  # Artifacts are semantic memories
            content=content_str,
            summary=f"Artifact: {artifact_name} ({artifact_type})",
            embedding=embedding or [],
            agent_id=agent_id,
            session_id=None,
            task_id=task_id,
            keywords=[
                "artifact",
                artifact_type,
                artifact_name,
                task_id,
            ],
            is_critical=False,
        )

        # Store memory
        self._memory_store[memory.memory_id] = memory

        # Create artifact record
        artifact_record = ArtifactMemoryRecord(
            artifact_name=artifact_name,
            task_id=task_id,
            execution_id=execution_id or f"exec-{task_id}",
            memory_id=memory.memory_id,
            content_hash=content_hash,
        )
        self._artifact_records[artifact_name] = artifact_record

        # Track task artifacts
        if task_id not in self._task_artifacts:
            self._task_artifacts[task_id] = []
        self._task_artifacts[task_id].append(artifact_name)

        self._logger.info(
            "artifact_stored_in_memory",
            artifact_name=artifact_name,
            memory_id=memory.memory_id,
            task_id=task_id,
            artifact_type=artifact_type,
            content_hash=content_hash[:16],
        )

        return memory.memory_id

    async def retrieve_task_artifacts(
        self,
        task_id: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve all artifacts for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of artifact data dictionaries
        """
        artifact_ids = self._task_artifacts.get(task_id, [])
        artifacts = []

        for artifact_id in artifact_ids:
            record = self._artifact_records.get(artifact_id)
            if record:
                memory = self._memory_store.get(record.memory_id)
                if memory:
                    # Parse artifact data
                    try:
                        artifact_data = json.loads(memory.content)
                        artifacts.append(artifact_data)
                        record.retrieval_count += 1
                    except json.JSONDecodeError as e:
                        self._logger.error(
                            "artifact_parse_error",
                            artifact_id=artifact_id,
                            error=str(e),
                        )

        self._logger.info(
            "task_artifacts_retrieved",
            task_id=task_id,
            artifact_count=len(artifacts),
        )

        return artifacts

    async def link_artifacts(
        self,
        task_id_1: str,
        task_id_2: str,
        relationship: str,
    ) -> None:
        """
        Link artifacts between tasks.

        Args:
            task_id_1: First task identifier
            task_id_2: Second task identifier
            relationship: Relationship type (e.g., "depends_on", "related_to")
        """
        # Store bidirectional link
        if task_id_1 not in self._artifact_links:
            self._artifact_links[task_id_1] = []
        self._artifact_links[task_id_1].append((task_id_2, relationship))

        if task_id_2 not in self._artifact_links:
            self._artifact_links[task_id_2] = []
        self._artifact_links[task_id_2].append((task_id_1, f"inverse_{relationship}"))

        self._logger.info(
            "artifacts_linked",
            task_id_1=task_id_1,
            task_id_2=task_id_2,
            relationship=relationship,
        )

    async def store_artifact(
        self,
        task_id: str,
        execution_id: str,
        artifact: TaskArtifact,
        embedding: list[float] | None = None,
        agent_id: str = "system",
    ) -> str:
        """
        Store task artifact (legacy method for compatibility).

        Args:
            task_id: Task identifier
            execution_id: Execution identifier
            artifact: Task artifact to store
            embedding: Optional embedding for artifact content
            agent_id: Agent ID

        Returns:
            Memory ID of stored artifact
        """
        # Convert TaskArtifact to dict
        artifact_data = {
            "name": artifact.name,
            "type": artifact.type,
            "content": artifact.content,
            "metadata": artifact.metadata,
        }

        return await self.store_task_artifact(
            task_id=task_id,
            artifact_data=artifact_data,
            execution_id=execution_id,
            embedding=embedding,
            agent_id=agent_id,
        )

    async def retrieve_artifact(
        self,
        artifact_name: str,
    ) -> MemoryRecord | None:
        """
        Retrieve artifact from memory.

        Args:
            artifact_name: Artifact name identifier

        Returns:
            Memory record or None if not found
        """
        record = self._artifact_records.get(artifact_name)
        if not record:
            return None

        memory = self._memory_store.get(record.memory_id)
        if memory:
            record.retrieval_count += 1

        return memory

    async def find_similar_artifacts(
        self,
        query_embedding: list[float],
        limit: int = 5,
        task_id: str | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """
        Find similar artifacts based on embedding similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            task_id: Optional filter by task ID

        Returns:
            List of (memory, score) tuples
        """
        # Filter memories to artifacts only
        artifact_memories = [
            mem
            for mem in self._memory_store.values()
            if "artifact" in mem.keywords
            and (task_id is None or mem.task_id == task_id)
        ]

        if not artifact_memories:
            return []

        # Score and rank
        scored = await self.retrieval.retrieve_top_k(
            memories=artifact_memories,
            k=limit,
            query_embedding=query_embedding,
        )

        results = [(mem, score) for mem, score, _ in scored]

        self._logger.info(
            "similar_artifacts_found",
            query_dims=len(query_embedding),
            results_count=len(results),
            task_filter=task_id,
        )

        return results

    async def get_task_artifacts(
        self,
        task_id: str,
    ) -> list[MemoryRecord]:
        """
        Get all artifacts for a task (as memory records).

        Args:
            task_id: Task identifier

        Returns:
            List of artifact memory records
        """
        artifact_ids = self._task_artifacts.get(task_id, [])
        memories = []

        for artifact_id in artifact_ids:
            record = self._artifact_records.get(artifact_id)
            if record:
                memory = self._memory_store.get(record.memory_id)
                if memory:
                    memories.append(memory)

        return memories

    def get_storage_stats(self) -> dict[str, Any]:
        """
        Get artifact storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        return {
            "total_artifacts": len(self._artifact_records),
            "total_memories": len(self._memory_store),
            "tasks_with_artifacts": len(self._task_artifacts),
            "total_links": sum(len(links) for links in self._artifact_links.values()) // 2,
            "total_retrieval_count": sum(
                rec.retrieval_count for rec in self._artifact_records.values()
            ),
        }
