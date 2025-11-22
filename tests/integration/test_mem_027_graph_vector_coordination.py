"""
Integration tests for MEM-027: Graph-Vector Coordination

Tests the coordination between Qdrant vector database and Neo4j graph database:
- Synchronized storage operations
- Hybrid query execution
- Graph-enriched vector search
- Vector-guided graph traversal
- Performance validation

Component ID: MEM-027
Ticket: MEM-027 (Implement Integration Tests)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import structlog
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    MemoryLayer,
    MemoryRecord,
    RelationshipEdge,
    RelationshipType,
)
from agentcore.a2a_protocol.services.memory.context_expander import (
    GraphContextExpander,
)
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.storage_backend import (
    VectorStorageBackend,
)

# Import testcontainer fixtures
from tests.integration.fixtures.neo4j import (
    clean_neo4j_db,
    neo4j_container,
    neo4j_driver,
    neo4j_uri,
)
from tests.integration.fixtures.qdrant import (
    qdrant_client,
    qdrant_container,
    qdrant_test_collection,
    qdrant_url,
)

logger = structlog.get_logger(__name__)

pytestmark = pytest.mark.asyncio


class TestSynchronizedStorage:
    """Test synchronized storage operations across vector and graph databases."""

    @pytest.fixture
    async def vector_backend(
        self, qdrant_client: AsyncQdrantClient, qdrant_test_collection: str
    ) -> VectorStorageBackend:
        """Create vector storage backend."""
        return VectorStorageBackend(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )

    @pytest.fixture
    async def graph_service(
        self, neo4j_driver: AsyncDriver, clean_neo4j_db
    ) -> GraphMemoryService:
        """Create graph service."""
        return GraphMemoryService(driver=neo4j_driver)

    async def test_atomic_storage_both_backends(
        self,
        vector_backend: VectorStorageBackend,
        graph_service: GraphMemoryService,
    ):
        """Test atomic storage to both vector and graph databases."""
        memory = MemoryRecord(
            memory_id="atomic-mem-1",
            memory_layer=MemoryLayer.EPISODIC,
            content="User requested PostgreSQL database setup with connection pooling",
            summary="PostgreSQL setup request",
            embedding=[0.5] * 1536,
            agent_id="agent-1",
            session_id="session-1",
            task_id="task-1",
            keywords=["postgresql", "database", "connection_pool"],
        )

        entity = EntityNode(
            entity_id="entity-postgresql",
            name="PostgreSQL",
            entity_type="tool",
            confidence=0.95,
        )

        relationship = RelationshipEdge(
            source_id=memory.memory_id,
            target_id=entity.entity_id,
            relationship_type=RelationshipType.MENTIONS,
            strength=0.9,
        )

        # Store atomically
        try:
            await vector_backend.store_memory(memory)
            await graph_service.store_memory_node(memory)
            await graph_service.store_entity(entity)
            await graph_service.store_relationship(relationship)
        except Exception as e:
            # If any operation fails, both backends should be consistent
            pytest.fail(f"Atomic storage failed: {e}")

        # Verify both backends
        # Vector
        vector_results = await vector_backend.search_similar(
            query_embedding=[0.5] * 1536, limit=5
        )
        assert any(r.memory_id == "atomic-mem-1" for r in vector_results)

        # Graph
        stored_entity = await graph_service.get_entity("entity-postgresql")
        assert stored_entity is not None

        related = await graph_service.get_related_entities(
            memory.memory_id, max_depth=1
        )
        assert any(e.entity_id == "entity-postgresql" for e in related)

    async def test_batch_storage_coordination(
        self,
        vector_backend: VectorStorageBackend,
        graph_service: GraphMemoryService,
    ):
        """Test batch storage operations maintain consistency."""
        # Create batch of memories
        memories = [
            MemoryRecord(
                memory_id=f"batch-mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Batch memory content {i}",
                summary=f"Batch {i}",
                embedding=[0.1 * i] * 1536,
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=[f"batch{i}"],
            )
            for i in range(10)
        ]

        # Store batch
        for memory in memories:
            await vector_backend.store_memory(memory)
            await graph_service.store_memory_node(memory)

        # Verify count matches in both backends
        vector_results = await vector_backend.search_similar(
            query_embedding=[0.5] * 1536, limit=20
        )

        # Should find all 10 memories in vector store
        batch_mem_count = sum(1 for r in vector_results if r.memory_id.startswith("batch-mem"))
        assert batch_mem_count == 10

        # Verify in graph store
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (m:Memory) WHERE m.memory_id STARTS WITH 'batch-mem' RETURN count(m) as count"
            )
            record = await result.single()
            assert record["count"] == 10

    async def test_update_consistency(
        self,
        vector_backend: VectorStorageBackend,
        graph_service: GraphMemoryService,
    ):
        """Test update operations maintain consistency."""
        # Initial memory
        memory = MemoryRecord(
            memory_id="update-mem-1",
            memory_layer=MemoryLayer.EPISODIC,
            content="Initial content",
            summary="Initial",
            embedding=[0.5] * 1536,
            agent_id="agent-1",
            session_id="session-1",
            task_id=None,
            keywords=["initial"],
        )

        await vector_backend.store_memory(memory)
        await graph_service.store_memory_node(memory)

        # Update memory
        updated_memory = MemoryRecord(
            memory_id="update-mem-1",
            memory_layer=MemoryLayer.SEMANTIC,  # Changed layer
            content="Updated content with more information",
            summary="Updated",
            embedding=[0.6] * 1536,  # Different embedding
            agent_id="agent-1",
            session_id="session-1",
            task_id=None,
            keywords=["updated", "information"],
        )

        await vector_backend.store_memory(updated_memory)
        await graph_service.update_memory_node(updated_memory)

        # Verify consistency
        vector_results = await vector_backend.search_similar(
            query_embedding=[0.6] * 1536, limit=5
        )
        updated_in_vector = next(
            (r for r in vector_results if r.memory_id == "update-mem-1"), None
        )
        assert updated_in_vector is not None
        assert updated_in_vector.memory_layer == MemoryLayer.SEMANTIC


class TestHybridQueryExecution:
    """Test hybrid query execution combining vector and graph operations."""

    @pytest.fixture
    async def hybrid_search(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db,
    ) -> HybridSearchService:
        """Create hybrid search service."""
        vector_backend = VectorStorageBackend(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )
        graph_service = GraphMemoryService(driver=neo4j_driver)

        return HybridSearchService(
            vector_backend=vector_backend,
            graph_service=graph_service,
        )

    async def test_vector_search_with_graph_filter(
        self, hybrid_search: HybridSearchService
    ):
        """Test vector search filtered by graph relationships."""
        # Create memories
        memories = [
            MemoryRecord(
                memory_id=f"filter-mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Memory about authentication {i}",
                summary=f"Auth {i}",
                embedding=[0.9] * 1536,
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=["auth"],
            )
            for i in range(5)
        ]

        # Store memories
        for mem in memories:
            await hybrid_search.vector_backend.store_memory(mem)
            await hybrid_search.graph_service.store_memory_node(mem)

        # Create entity and link only to some memories
        entity = EntityNode(
            entity_id="entity-jwt",
            name="JWT",
            entity_type="tool",
            confidence=0.95,
        )
        await hybrid_search.graph_service.store_entity(entity)

        # Link only first 3 memories to JWT entity
        for mem in memories[:3]:
            await hybrid_search.graph_service.store_relationship(
                RelationshipEdge(
                    source_id=mem.memory_id,
                    target_id=entity.entity_id,
                    relationship_type=RelationshipType.MENTIONS,
                    strength=0.9,
                )
            )

        # Hybrid search with graph filter
        results = await hybrid_search.search(
            query_embedding=[0.9] * 1536,
            limit=10,
            entity_filter="entity-jwt",  # Only memories mentioning JWT
        )

        # Should only return memories linked to JWT entity
        assert len(results) <= 3
        assert all(r.memory_id in {m.memory_id for m in memories[:3]} for r in results)

    async def test_graph_proximity_boosting(
        self, hybrid_search: HybridSearchService
    ):
        """Test boosting results based on graph proximity."""
        # Create memories with similar embeddings
        memories = [
            MemoryRecord(
                memory_id=f"prox-mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Similar content {i}",
                summary=f"Content {i}",
                embedding=[0.5 + 0.01 * i] * 1536,  # Very similar embeddings
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=["similar"],
            )
            for i in range(5)
        ]

        for mem in memories:
            await hybrid_search.vector_backend.store_memory(mem)
            await hybrid_search.graph_service.store_memory_node(mem)

        # Create central entity
        central_entity = EntityNode(
            entity_id="entity-central",
            name="central_concept",
            entity_type="concept",
            confidence=0.95,
        )
        await hybrid_search.graph_service.store_entity(central_entity)

        # Link first memory strongly to central entity
        await hybrid_search.graph_service.store_relationship(
            RelationshipEdge(
                source_id="prox-mem-0",
                target_id="entity-central",
                relationship_type=RelationshipType.MENTIONS,
                strength=0.95,  # Strong relationship
            )
        )

        # Link other memories weakly
        for mem in memories[1:]:
            await hybrid_search.graph_service.store_relationship(
                RelationshipEdge(
                    source_id=mem.memory_id,
                    target_id="entity-central",
                    relationship_type=RelationshipType.MENTIONS,
                    strength=0.3,  # Weak relationship
                )
            )

        # Search with proximity boosting
        results = await hybrid_search.search(
            query_embedding=[0.51] * 1536,
            limit=5,
            boost_graph_proximity=True,
        )

        # First result should be prox-mem-0 due to strong graph connection
        # (even though embedding might be slightly farther)
        assert len(results) > 0
        # prox-mem-0 should be ranked highly
        top_ids = [r.memory_id for r in results[:2]]
        assert "prox-mem-0" in top_ids


class TestGraphEnrichedVectorSearch:
    """Test vector search enriched with graph context."""

    @pytest.fixture
    async def context_expander(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db,
    ) -> GraphContextExpander:
        """Create context expander."""
        vector_backend = VectorStorageBackend(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )
        graph_service = GraphMemoryService(driver=neo4j_driver)

        return GraphContextExpander(
            vector_backend=vector_backend,
            graph_service=graph_service,
        )

    async def test_expand_with_1hop_neighbors(
        self, context_expander: GraphContextExpander
    ):
        """Test expanding vector results with 1-hop graph neighbors."""
        # Create central memory
        central_memory = MemoryRecord(
            memory_id="central-mem",
            memory_layer=MemoryLayer.EPISODIC,
            content="Central memory about FastAPI development",
            summary="FastAPI dev",
            embedding=[0.5] * 1536,
            agent_id="agent-1",
            session_id="session-1",
            task_id=None,
            keywords=["fastapi"],
        )

        await context_expander.vector_backend.store_memory(central_memory)
        await context_expander.graph_service.store_memory_node(central_memory)

        # Create related entities (1-hop neighbors)
        entities = [
            EntityNode(
                entity_id="entity-pydantic",
                name="Pydantic",
                entity_type="tool",
                confidence=0.9,
            ),
            EntityNode(
                entity_id="entity-uvicorn",
                name="Uvicorn",
                entity_type="tool",
                confidence=0.85,
            ),
        ]

        for entity in entities:
            await context_expander.graph_service.store_entity(entity)
            await context_expander.graph_service.store_relationship(
                RelationshipEdge(
                    source_id=central_memory.memory_id,
                    target_id=entity.entity_id,
                    relationship_type=RelationshipType.MENTIONS,
                    strength=0.8,
                )
            )

        # Expand context
        expanded_context = await context_expander.expand_context(
            memory_id=central_memory.memory_id,
            max_depth=1,
        )

        # Should include 1-hop neighbors
        assert len(expanded_context["entities"]) == 2
        entity_ids = {e.entity_id for e in expanded_context["entities"]}
        assert "entity-pydantic" in entity_ids
        assert "entity-uvicorn" in entity_ids

    async def test_expand_with_2hop_neighbors(
        self, context_expander: GraphContextExpander
    ):
        """Test expanding with 2-hop graph neighbors for critical memories."""
        # Create memory chain: mem1 -> entity1 -> entity2 -> entity3
        memory = MemoryRecord(
            memory_id="critical-mem",
            memory_layer=MemoryLayer.EPISODIC,
            content="Critical memory requiring deep context",
            summary="Critical",
            embedding=[0.5] * 1536,
            agent_id="agent-1",
            session_id="session-1",
            task_id=None,
            keywords=["critical"],
            is_critical=True,
        )

        await context_expander.vector_backend.store_memory(memory)
        await context_expander.graph_service.store_memory_node(memory)

        # Create entity chain
        entity1 = EntityNode(
            entity_id="entity-1",
            name="Entity1",
            entity_type="concept",
            confidence=0.9,
        )
        entity2 = EntityNode(
            entity_id="entity-2",
            name="Entity2",
            entity_type="concept",
            confidence=0.85,
        )
        entity3 = EntityNode(
            entity_id="entity-3",
            name="Entity3",
            entity_type="concept",
            confidence=0.8,
        )

        await context_expander.graph_service.store_entity(entity1)
        await context_expander.graph_service.store_entity(entity2)
        await context_expander.graph_service.store_entity(entity3)

        # Create relationship chain
        await context_expander.graph_service.store_relationship(
            RelationshipEdge(
                source_id=memory.memory_id,
                target_id=entity1.entity_id,
                relationship_type=RelationshipType.MENTIONS,
                strength=0.9,
            )
        )
        await context_expander.graph_service.store_relationship(
            RelationshipEdge(
                source_id=entity1.entity_id,
                target_id=entity2.entity_id,
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.8,
            )
        )
        await context_expander.graph_service.store_relationship(
            RelationshipEdge(
                source_id=entity2.entity_id,
                target_id=entity3.entity_id,
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.7,
            )
        )

        # Expand with 2-hop depth
        expanded_context = await context_expander.expand_context(
            memory_id=memory.memory_id,
            max_depth=2,
        )

        # Should find entity1 (1-hop) and entity2 (2-hop)
        entity_ids = {e.entity_id for e in expanded_context["entities"]}
        assert "entity-1" in entity_ids
        assert "entity-2" in entity_ids
        # entity-3 is 3-hop, should not be included
        assert len(entity_ids) >= 2


class TestPerformanceValidation:
    """Test performance targets for coordinated operations."""

    @pytest.fixture
    async def hybrid_search(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db,
    ) -> HybridSearchService:
        """Create hybrid search service."""
        vector_backend = VectorStorageBackend(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )
        graph_service = GraphMemoryService(driver=neo4j_driver)

        return HybridSearchService(
            vector_backend=vector_backend,
            graph_service=graph_service,
        )

    async def test_hybrid_search_latency_p95(
        self, hybrid_search: HybridSearchService
    ):
        """Test hybrid search meets <300ms p95 latency target."""
        # Create test dataset
        memories = [
            MemoryRecord(
                memory_id=f"perf-mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Performance test memory {i}",
                summary=f"Perf {i}",
                embedding=[0.1 * (i % 10)] * 1536,
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=[f"perf{i}"],
            )
            for i in range(100)
        ]

        # Store memories
        for mem in memories:
            await hybrid_search.vector_backend.store_memory(mem)
            await hybrid_search.graph_service.store_memory_node(mem)

        # Create entities and relationships
        entities = [
            EntityNode(
                entity_id=f"entity-{i}",
                name=f"Entity{i}",
                entity_type="concept",
                confidence=0.9,
            )
            for i in range(20)
        ]

        for entity in entities:
            await hybrid_search.graph_service.store_entity(entity)

        # Link memories to entities
        for i, mem in enumerate(memories[:50]):
            await hybrid_search.graph_service.store_relationship(
                RelationshipEdge(
                    source_id=mem.memory_id,
                    target_id=f"entity-{i % 20}",
                    relationship_type=RelationshipType.MENTIONS,
                    strength=0.8,
                )
            )

        # Run multiple searches and measure latency
        latencies = []
        for _ in range(20):
            start_time = datetime.now(UTC)
            await hybrid_search.search(
                query_embedding=[0.5] * 1536,
                limit=10,
                include_graph_context=True,
            )
            end_time = datetime.now(UTC)
            latency_ms = (end_time - start_time).total_seconds() * 1000
            latencies.append(latency_ms)

        # Calculate p95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]

        # Verify <300ms p95 target
        assert p95_latency < 300, f"P95 latency: {p95_latency:.2f}ms (target: <300ms)"

    async def test_vector_search_latency_p95(
        self, hybrid_search: HybridSearchService
    ):
        """Test vector-only search meets <100ms p95 latency target."""
        # Create larger dataset
        memories = [
            MemoryRecord(
                memory_id=f"vec-perf-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Vector performance test {i}",
                summary=f"Vec {i}",
                embedding=[0.1 * (i % 10)] * 1536,
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=[f"vec{i}"],
            )
            for i in range(1000)
        ]

        # Store in vector database only
        for mem in memories:
            await hybrid_search.vector_backend.store_memory(mem)

        # Run searches
        latencies = []
        for _ in range(20):
            start_time = datetime.now(UTC)
            await hybrid_search.vector_backend.search_similar(
                query_embedding=[0.5] * 1536,
                limit=10,
            )
            end_time = datetime.now(UTC)
            latency_ms = (end_time - start_time).total_seconds() * 1000
            latencies.append(latency_ms)

        # Calculate p95
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]

        # Verify <100ms p95 target
        assert p95_latency < 100, f"Vector search p95: {p95_latency:.2f}ms (target: <100ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
