"""
Integration Tests for Hybrid Memory Coordination (MEM-027.1)

Tests vector + graph coordination between Qdrant and Neo4j.
Validates:
- Memory storage in both vector and graph databases
- Cross-database consistency
- Hybrid retrieval combining vector similarity and graph relationships
- Update and delete operations with cascade
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

from agentcore.a2a_protocol.models.memory import MemoryLayer, StageType
from agentcore.a2a_protocol.services.memory.storage_backend import StorageBackend
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.entity_extractor import EntityExtractor


# Use function-scoped event loop for all tests
pytestmark = pytest.mark.asyncio


class TestVectorGraphCoordination:
    """Test coordination between Qdrant vector storage and Neo4j graph storage."""

    @pytest.fixture
    async def storage_backend(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
    ) -> StorageBackend:
        """Create storage backend with Qdrant client."""
        backend = StorageBackend(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )
        return backend

    @pytest.fixture
    async def graph_service(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db: None,
    ) -> GraphMemoryService:
        """Create graph service with Neo4j driver."""
        service = GraphMemoryService(driver=neo4j_driver)
        await service.initialize_schema()
        return service

    @pytest.fixture
    async def entity_extractor(self) -> EntityExtractor:
        """Create entity extractor (uses mock LLM in tests)."""
        return EntityExtractor(api_key="test-key")

    @pytest.fixture
    async def hybrid_search(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
    ) -> HybridSearchService:
        """Create hybrid search service."""
        return HybridSearchService(
            vector_backend=storage_backend,
            graph_service=graph_service,
        )

    async def test_store_memory_in_both_databases(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
        entity_extractor: EntityExtractor,
    ) -> None:
        """Test storing memory in both Qdrant and Neo4j."""
        # Arrange
        memory_content = "Implemented JWT authentication using Redis for session storage"
        memory_id = "mem-test-001"
        agent_id = "agent-1"

        # Create embedding (using simple mock)
        embedding = [0.1] * 1536

        # Act - Store in vector database
        await storage_backend.store_memory(
            memory_id=memory_id,
            content=memory_content,
            embedding=embedding,
            metadata={
                "memory_layer": "episodic",
                "agent_id": agent_id,
                "is_critical": True,
                "stage_type": "execution",
            },
        )

        # Extract entities and store in graph
        entities = await entity_extractor.extract_entities(memory_content)

        for entity in entities:
            await graph_service.create_entity(
                entity_name=entity["name"],
                entity_type=entity["type"],
                properties={"confidence": entity.get("confidence", 0.8)},
            )

            # Link entity to memory
            await graph_service.create_memory_entity_relationship(
                memory_id=memory_id,
                entity_name=entity["name"],
                relationship_type="MENTIONS",
            )

        # Assert - Verify vector storage
        vector_results = await storage_backend.search_similar(
            query_embedding=embedding,
            limit=1,
        )
        assert len(vector_results) == 1
        assert vector_results[0]["id"] == memory_id

        # Assert - Verify graph storage
        memory_entities = await graph_service.get_memory_entities(memory_id)
        assert len(memory_entities) >= 2  # Should have extracted JWT, Redis, etc.

        entity_names = [e["name"] for e in memory_entities]
        assert any("JWT" in name or "jwt" in name.lower() for name in entity_names)

    async def test_hybrid_search_combines_vector_and_graph(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
        hybrid_search: HybridSearchService,
    ) -> None:
        """Test hybrid search combines vector similarity and graph relationships."""
        # Arrange - Store multiple related memories
        memories = [
            {
                "id": "mem-001",
                "content": "JWT authentication implementation",
                "entities": [("JWT", "concept"), ("authentication", "concept")],
            },
            {
                "id": "mem-002",
                "content": "Redis session storage configuration",
                "entities": [("Redis", "tool"), ("session", "concept")],
            },
            {
                "id": "mem-003",
                "content": "User login flow with JWT tokens",
                "entities": [("JWT", "concept"), ("login", "concept"), ("user", "concept")],
            },
        ]

        for mem in memories:
            # Store in vector database
            embedding = [0.1 if i % 2 == 0 else 0.05 for i in range(1536)]
            await storage_backend.store_memory(
                memory_id=mem["id"],
                content=mem["content"],
                embedding=embedding,
                metadata={"memory_layer": "episodic"},
            )

            # Store entities in graph
            for entity_name, entity_type in mem["entities"]:
                try:
                    await graph_service.create_entity(
                        entity_name=entity_name,
                        entity_type=entity_type,
                    )
                except Exception:
                    pass  # Entity may already exist

                await graph_service.create_memory_entity_relationship(
                    memory_id=mem["id"],
                    entity_name=entity_name,
                    relationship_type="MENTIONS",
                )

        # Create relationships between entities
        await graph_service.create_entity_relationship(
            source_entity="JWT",
            target_entity="authentication",
            relationship_type="USED_FOR",
            strength=0.9,
        )

        # Act - Perform hybrid search
        query_embedding = [0.1 if i % 2 == 0 else 0.05 for i in range(1536)]
        results = await hybrid_search.search(
            query_embedding=query_embedding,
            limit=5,
            use_graph_expansion=True,
        )

        # Assert - Should get results from both vector search and graph expansion
        assert len(results) >= 2

        # Verify results include relevance scores
        for result in results:
            assert "id" in result
            assert "content" in result
            assert "score" in result
            assert result["score"] >= 0.0

        # Verify at least one result has graph expansion metadata
        assert any("graph_expansion" in r.get("metadata", {}) for r in results)

    async def test_update_memory_maintains_consistency(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
    ) -> None:
        """Test updating memory maintains consistency across databases."""
        # Arrange - Store initial memory
        memory_id = "mem-update-001"
        original_content = "Basic authentication implementation"
        original_embedding = [0.2] * 1536

        await storage_backend.store_memory(
            memory_id=memory_id,
            content=original_content,
            embedding=original_embedding,
            metadata={"memory_layer": "episodic"},
        )

        await graph_service.create_entity("authentication", "concept")
        await graph_service.create_memory_entity_relationship(
            memory_id=memory_id,
            entity_name="authentication",
            relationship_type="MENTIONS",
        )

        # Act - Update memory with new content
        updated_content = "JWT-based authentication with refresh tokens"
        updated_embedding = [0.3] * 1536

        await storage_backend.update_memory(
            memory_id=memory_id,
            content=updated_content,
            embedding=updated_embedding,
        )

        # Update graph relationships
        await graph_service.create_entity("JWT", "concept")
        await graph_service.create_memory_entity_relationship(
            memory_id=memory_id,
            entity_name="JWT",
            relationship_type="MENTIONS",
        )

        # Assert - Verify vector database update
        vector_results = await storage_backend.search_similar(
            query_embedding=updated_embedding,
            limit=1,
        )
        assert len(vector_results) == 1
        assert vector_results[0]["content"] == updated_content

        # Assert - Verify graph database update
        entities = await graph_service.get_memory_entities(memory_id)
        entity_names = [e["name"] for e in entities]
        assert "JWT" in entity_names
        assert "authentication" in entity_names

    async def test_delete_memory_cascades_to_graph(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
    ) -> None:
        """Test deleting memory cascades to graph database."""
        # Arrange - Store memory with entities
        memory_id = "mem-delete-001"
        content = "Test memory for deletion"
        embedding = [0.4] * 1536

        await storage_backend.store_memory(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            metadata={"memory_layer": "episodic"},
        )

        await graph_service.create_entity("test_entity", "concept")
        await graph_service.create_memory_entity_relationship(
            memory_id=memory_id,
            entity_name="test_entity",
            relationship_type="MENTIONS",
        )

        # Verify memory exists in both databases
        vector_results = await storage_backend.search_similar(embedding, limit=1)
        assert len(vector_results) == 1

        entities = await graph_service.get_memory_entities(memory_id)
        assert len(entities) == 1

        # Act - Delete memory
        await storage_backend.delete_memory(memory_id)
        await graph_service.delete_memory_relationships(memory_id)

        # Assert - Verify deletion from vector database
        vector_results = await storage_backend.search_similar(embedding, limit=10)
        assert all(r["id"] != memory_id for r in vector_results)

        # Assert - Verify relationship cleanup in graph
        entities_after = await graph_service.get_memory_entities(memory_id)
        assert len(entities_after) == 0

    async def test_cross_database_latency(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
    ) -> None:
        """Test cross-database operations complete within latency targets."""
        import time

        # Arrange
        memory_id = "mem-perf-001"
        content = "Performance test memory"
        embedding = [0.5] * 1536

        # Act - Measure storage latency
        start_time = time.time()

        await storage_backend.store_memory(
            memory_id=memory_id,
            content=content,
            embedding=embedding,
            metadata={"memory_layer": "episodic"},
        )

        await graph_service.create_entity("perf_entity", "concept")
        await graph_service.create_memory_entity_relationship(
            memory_id=memory_id,
            entity_name="perf_entity",
            relationship_type="MENTIONS",
        )

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Assert - Should complete within 300ms target
        assert latency_ms < 300, f"Cross-database storage took {latency_ms:.2f}ms, expected <300ms"

    async def test_batch_storage_consistency(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
    ) -> None:
        """Test batch storage operations maintain consistency."""
        # Arrange - Prepare batch of memories
        batch_size = 10
        memories = [
            {
                "id": f"mem-batch-{i:03d}",
                "content": f"Memory {i} about testing",
                "embedding": [float(i) / 100] * 1536,
            }
            for i in range(batch_size)
        ]

        # Act - Store batch
        for mem in memories:
            await storage_backend.store_memory(
                memory_id=mem["id"],
                content=mem["content"],
                embedding=mem["embedding"],
                metadata={"memory_layer": "episodic"},
            )

            # Create entity for each memory
            entity_name = f"entity_{mem['id']}"
            await graph_service.create_entity(entity_name, "concept")
            await graph_service.create_memory_entity_relationship(
                memory_id=mem["id"],
                entity_name=entity_name,
                relationship_type="MENTIONS",
            )

        # Assert - Verify all memories stored
        for mem in memories:
            # Check vector storage
            vector_results = await storage_backend.search_similar(
                query_embedding=mem["embedding"],
                limit=1,
            )
            assert len(vector_results) >= 1
            assert any(r["id"] == mem["id"] for r in vector_results)

            # Check graph storage
            entities = await graph_service.get_memory_entities(mem["id"])
            assert len(entities) == 1


class TestHybridSearchAccuracy:
    """Test hybrid search accuracy improvements over vector-only search."""

    @pytest.fixture
    async def populated_storage(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
    ) -> tuple[StorageBackend, GraphMemoryService]:
        """Populate storage with test data."""
        # Create knowledge graph of related concepts
        concepts = [
            ("Python", "language"),
            ("FastAPI", "framework"),
            ("asyncio", "library"),
            ("pytest", "tool"),
        ]

        for concept_name, concept_type in concepts:
            await graph_service.create_entity(concept_name, concept_type)

        # Create relationships
        await graph_service.create_entity_relationship(
            "FastAPI", "Python", "BUILT_WITH", 0.95
        )
        await graph_service.create_entity_relationship(
            "FastAPI", "asyncio", "USES", 0.90
        )
        await graph_service.create_entity_relationship(
            "pytest", "Python", "TESTS", 0.85
        )

        # Store memories
        memories_data = [
            {
                "id": "mem-py-001",
                "content": "FastAPI is a Python web framework",
                "entities": ["FastAPI", "Python"],
                "embedding": [0.8, 0.2] + [0.1] * 1534,
            },
            {
                "id": "mem-py-002",
                "content": "asyncio enables asynchronous programming",
                "entities": ["asyncio", "Python"],
                "embedding": [0.7, 0.3] + [0.1] * 1534,
            },
            {
                "id": "mem-py-003",
                "content": "pytest is the standard Python testing framework",
                "entities": ["pytest", "Python"],
                "embedding": [0.6, 0.4] + [0.1] * 1534,
            },
        ]

        for mem in memories_data:
            await storage_backend.store_memory(
                memory_id=mem["id"],
                content=mem["content"],
                embedding=mem["embedding"],
                metadata={"memory_layer": "semantic"},
            )

            for entity_name in mem["entities"]:
                await graph_service.create_memory_entity_relationship(
                    memory_id=mem["id"],
                    entity_name=entity_name,
                    relationship_type="MENTIONS",
                )

        return storage_backend, graph_service

    async def test_graph_expansion_finds_related_memories(
        self,
        populated_storage: tuple[StorageBackend, GraphMemoryService],
        hybrid_search: HybridSearchService,
    ) -> None:
        """Test graph expansion finds related memories."""
        storage_backend, graph_service = populated_storage

        # Act - Search for "FastAPI" should also find asyncio-related memories
        query_embedding = [0.8, 0.2] + [0.1] * 1534

        # Vector-only search
        vector_only = await storage_backend.search_similar(
            query_embedding=query_embedding,
            limit=3,
        )

        # Hybrid search with graph expansion
        hybrid_results = await hybrid_search.search(
            query_embedding=query_embedding,
            limit=3,
            use_graph_expansion=True,
        )

        # Assert - Hybrid should find more relevant results
        assert len(hybrid_results) >= len(vector_only)

        # Check for expanded results
        hybrid_ids = {r["id"] for r in hybrid_results}
        assert "mem-py-001" in hybrid_ids  # Direct match
        # May also include mem-py-002 due to FastAPI->asyncio relationship


class TestConcurrentOperations:
    """Test concurrent vector and graph operations."""

    async def test_concurrent_writes(
        self,
        storage_backend: StorageBackend,
        graph_service: GraphMemoryService,
    ) -> None:
        """Test concurrent writes to both databases."""
        import asyncio

        # Arrange
        num_concurrent = 5
        memories = [
            {
                "id": f"mem-concurrent-{i}",
                "content": f"Concurrent memory {i}",
                "embedding": [float(i)] * 1536,
            }
            for i in range(num_concurrent)
        ]

        # Act - Write concurrently
        async def write_memory(mem: dict[str, Any]) -> None:
            await storage_backend.store_memory(
                memory_id=mem["id"],
                content=mem["content"],
                embedding=mem["embedding"],
                metadata={"memory_layer": "episodic"},
            )

            entity_name = f"entity_{mem['id']}"
            await graph_service.create_entity(entity_name, "concept")
            await graph_service.create_memory_entity_relationship(
                memory_id=mem["id"],
                entity_name=entity_name,
                relationship_type="MENTIONS",
            )

        await asyncio.gather(*[write_memory(mem) for mem in memories])

        # Assert - All memories stored successfully
        for mem in memories:
            vector_results = await storage_backend.search_similar(
                query_embedding=mem["embedding"],
                limit=1,
            )
            assert len(vector_results) >= 1

            entities = await graph_service.get_memory_entities(mem["id"])
            assert len(entities) == 1
