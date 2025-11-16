"""
Memory Storage Backend Service

Unified storage layer that coordinates PostgreSQL, Qdrant, and Neo4j for memory persistence.
Provides async methods for storing and retrieving memories across all backends.

This service acts as the "glue" between:
- JSON-RPC API layer (external interface)
- PostgreSQL repositories (relational data)
- Qdrant client (vector embeddings)
- Neo4j driver (knowledge graph)

Component ID: MEM-029
Ticket: MEM-029 (Production Storage Backend Integration)
"""

from __future__ import annotations

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels
from qdrant_client.models import PointStruct

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.database.connection import get_session
from agentcore.a2a_protocol.database.repositories import (
    ErrorRepository,
    MemoryRepository,
    StageMemoryRepository,
)
from agentcore.a2a_protocol.models.memory import (
    ErrorRecord,
    MemoryRecord,
    StageMemory,
)

logger = structlog.get_logger(__name__)


class StorageBackendService:
    """
    Unified storage backend for memory service.

    Coordinates storage across:
    - PostgreSQL: Relational data (MemoryModel, StageMemoryModel, ErrorModel)
    - Qdrant: Vector embeddings for semantic search
    - Neo4j: Knowledge graph for entity relationships

    Usage:
        backend = StorageBackendService()
        await backend.initialize()

        # Store memory with embedding
        memory_record = MemoryRecord(...)
        await backend.store_memory(memory_record)

        # Retrieve with vector search
        results = await backend.vector_search(query_embedding, limit=10)

        await backend.close()
    """

    def __init__(self) -> None:
        """Initialize storage backend service (connections not yet established)."""
        self._qdrant: AsyncQdrantClient | None = None
        self._neo4j_driver: AsyncDriver | None = None
        self._initialized = False
        self._logger = logger.bind(component="storage_backend")

    @property
    def qdrant(self) -> AsyncQdrantClient:
        """Get Qdrant client (raises if not initialized)."""
        if self._qdrant is None:
            raise RuntimeError("StorageBackendService not initialized. Call initialize() first.")
        return self._qdrant

    @property
    def neo4j_driver(self) -> AsyncDriver:
        """Get Neo4j driver (raises if not initialized)."""
        if self._neo4j_driver is None:
            raise RuntimeError("StorageBackendService not initialized. Call initialize() first.")
        return self._neo4j_driver

    async def initialize(self) -> None:
        """
        Initialize connections to all storage backends.

        Establishes connections to:
        - Qdrant vector database
        - Neo4j graph database
        - PostgreSQL (via existing connection module)

        Raises:
            RuntimeError: If already initialized
            ConnectionError: If connection to any backend fails
        """
        if self._initialized:
            self._logger.warning("StorageBackendService already initialized")
            return

        self._logger.info(
            "initializing_storage_backends",
            qdrant_url=settings.QDRANT_URL,
            neo4j_uri=settings.NEO4J_URI,
        )

        # Initialize Qdrant client
        self._qdrant = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT,
        )

        # Ensure Qdrant collection exists
        await self._ensure_qdrant_collection()

        # Initialize Neo4j driver
        self._neo4j_driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_lifetime=settings.NEO4J_MAX_CONNECTION_LIFETIME,
            max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
            connection_acquisition_timeout=settings.NEO4J_CONNECTION_ACQUISITION_TIMEOUT,
            encrypted=settings.NEO4J_ENCRYPTED,
        )

        # Verify Neo4j connection
        await self._verify_neo4j_connection()

        self._initialized = True
        self._logger.info("storage_backends_initialized_successfully")

    async def _ensure_qdrant_collection(self) -> None:
        """Ensure Qdrant collection exists with correct schema."""
        collection_name = settings.QDRANT_COLLECTION_NAME

        collections = await self._qdrant.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            self._logger.info(
                "creating_qdrant_collection",
                collection=collection_name,
                vector_size=settings.QDRANT_VECTOR_SIZE,
            )

            await self._qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.QDRANT_VECTOR_SIZE,
                    distance=qmodels.Distance[settings.QDRANT_DISTANCE.upper()],
                ),
            )

            # Create payload indexes for filtering
            await self._qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="agent_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            await self._qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="session_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            await self._qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="task_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
            await self._qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="memory_layer",
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

            self._logger.info("qdrant_collection_created", collection=collection_name)
        else:
            self._logger.info("qdrant_collection_exists", collection=collection_name)

    async def _verify_neo4j_connection(self) -> None:
        """Verify Neo4j connection is working."""
        async with self._neo4j_driver.session(database=settings.NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 AS test")
            record = await result.single()
            if record["test"] != 1:
                raise ConnectionError("Neo4j connection verification failed")
        self._logger.info("neo4j_connection_verified")

    async def close(self) -> None:
        """Close all storage backend connections."""
        if not self._initialized:
            return

        self._logger.info("closing_storage_backends")

        if self._qdrant:
            await self._qdrant.close()
            self._qdrant = None

        if self._neo4j_driver:
            await self._neo4j_driver.close()
            self._neo4j_driver = None

        self._initialized = False
        self._logger.info("storage_backends_closed")

    async def store_memory(self, memory: MemoryRecord) -> str:
        """
        Store memory record in PostgreSQL and optionally in Qdrant.

        Args:
            memory: MemoryRecord to store

        Returns:
            str: Memory ID of stored record

        If the memory has an embedding, it will also be stored in Qdrant
        for vector similarity search.
        """
        # Store in PostgreSQL
        async with get_session() as session:
            await MemoryRepository.create(session, memory)
            self._logger.info(
                "memory_stored_postgresql",
                memory_id=memory.memory_id,
                memory_layer=memory.memory_layer.value,
            )

        # Store in Qdrant if embedding exists
        if memory.embedding and len(memory.embedding) > 0:
            await self._store_memory_vector(memory)

        return memory.memory_id

    async def _store_memory_vector(self, memory: MemoryRecord) -> None:
        """Store memory embedding in Qdrant."""
        # Extract UUID from memory_id (e.g., "mem-uuid" -> "uuid")
        point_id = memory.memory_id.split("-", 1)[1] if "-" in memory.memory_id else memory.memory_id

        payload = {
            "memory_id": memory.memory_id,
            "agent_id": memory.agent_id or "",
            "session_id": memory.session_id or "",
            "task_id": memory.task_id or "",
            "memory_layer": memory.memory_layer.value,
            "content": memory.content[:1000],  # Truncate for payload
            "summary": memory.summary,
            "keywords": memory.keywords,
            "timestamp": memory.timestamp.isoformat(),
            "is_critical": memory.is_critical,
        }

        point = PointStruct(
            id=point_id,
            vector=memory.embedding,
            payload=payload,
        )

        await self._qdrant.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=[point],
            wait=True,
        )

        self._logger.info(
            "memory_stored_qdrant",
            memory_id=memory.memory_id,
            vector_size=len(memory.embedding),
        )

    async def store_stage_memory(self, stage_memory: StageMemory) -> str:
        """
        Store stage memory record in PostgreSQL.

        Args:
            stage_memory: StageMemory to store

        Returns:
            str: Stage ID of stored record
        """
        async with get_session() as session:
            await StageMemoryRepository.create(session, stage_memory)
            self._logger.info(
                "stage_memory_stored",
                stage_id=stage_memory.stage_id,
                task_id=stage_memory.task_id,
                compression_ratio=stage_memory.compression_ratio,
            )

        return stage_memory.stage_id

    async def store_error(self, error_record: ErrorRecord) -> str:
        """
        Store error record in PostgreSQL.

        Args:
            error_record: ErrorRecord to store

        Returns:
            str: Error ID of stored record
        """
        async with get_session() as session:
            await ErrorRepository.create(session, error_record)
            self._logger.info(
                "error_stored",
                error_id=error_record.error_id,
                task_id=error_record.task_id,
                error_type=error_record.error_type.value,
            )

        return error_record.error_id

    async def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter_agent_id: str | None = None,
        filter_session_id: str | None = None,
        filter_task_id: str | None = None,
        filter_memory_layer: str | None = None,
    ) -> list[tuple[str, float, dict]]:
        """
        Search Qdrant for semantically similar memories.

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results
            filter_agent_id: Optional filter by agent ID
            filter_session_id: Optional filter by session ID
            filter_task_id: Optional filter by task ID
            filter_memory_layer: Optional filter by memory layer

        Returns:
            List of tuples (memory_id, score, payload)
        """
        # Build filter conditions
        filter_conditions = []

        if filter_agent_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="agent_id",
                    match=qmodels.MatchValue(value=filter_agent_id),
                )
            )

        if filter_session_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="session_id",
                    match=qmodels.MatchValue(value=filter_session_id),
                )
            )

        if filter_task_id:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="task_id",
                    match=qmodels.MatchValue(value=filter_task_id),
                )
            )

        if filter_memory_layer:
            filter_conditions.append(
                qmodels.FieldCondition(
                    key="memory_layer",
                    match=qmodels.MatchValue(value=filter_memory_layer),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = qmodels.Filter(must=filter_conditions)

        # Execute search
        results = await self._qdrant.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        self._logger.info(
            "vector_search_completed",
            results_count=len(results),
            limit=limit,
            has_filters=len(filter_conditions) > 0,
        )

        return [(str(r.id), r.score, r.payload) for r in results]

    async def get_memory_by_id(self, memory_id: str) -> MemoryRecord | None:
        """
        Retrieve memory record from PostgreSQL by ID.

        Args:
            memory_id: Memory ID (with or without prefix)

        Returns:
            MemoryRecord or None if not found
        """
        # Extract UUID part if prefixed
        uuid_str = memory_id.split("-", 1)[1] if "-" in memory_id and memory_id.count("-") != 4 else memory_id

        async with get_session() as session:
            memory_model = await MemoryRepository.get_by_id(session, uuid_str)
            if memory_model:
                return memory_model.to_pydantic()
        return None

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all storage backends.

        Returns:
            Dictionary with health status for each backend
        """
        health = {
            "postgresql": False,
            "qdrant": False,
            "neo4j": False,
        }

        # Check PostgreSQL
        try:
            async with get_session() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
            health["postgresql"] = True
        except Exception as e:
            self._logger.error("postgresql_health_check_failed", error=str(e))

        # Check Qdrant
        if self._qdrant:
            try:
                await self._qdrant.get_collections()
                health["qdrant"] = True
            except Exception as e:
                self._logger.error("qdrant_health_check_failed", error=str(e))

        # Check Neo4j
        if self._neo4j_driver:
            try:
                async with self._neo4j_driver.session(database=settings.NEO4J_DATABASE) as session:
                    await session.run("RETURN 1")
                health["neo4j"] = True
            except Exception as e:
                self._logger.error("neo4j_health_check_failed", error=str(e))

        return health


# Global singleton instance (lazy initialization)
_storage_backend: StorageBackendService | None = None


def get_storage_backend() -> StorageBackendService:
    """Get or create global storage backend instance."""
    global _storage_backend
    if _storage_backend is None:
        _storage_backend = StorageBackendService()
    return _storage_backend


async def initialize_storage_backend() -> None:
    """Initialize the global storage backend (call during app startup)."""
    backend = get_storage_backend()
    await backend.initialize()


async def close_storage_backend() -> None:
    """Close the global storage backend (call during app shutdown)."""
    global _storage_backend
    if _storage_backend:
        await _storage_backend.close()
        _storage_backend = None


__all__ = [
    "StorageBackendService",
    "get_storage_backend",
    "initialize_storage_backend",
    "close_storage_backend",
]
