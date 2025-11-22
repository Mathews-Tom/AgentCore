"""
Integration tests for MEM-027: Hybrid Memory Service

Comprehensive integration tests for the hybrid memory architecture:
- Real Qdrant instance (testcontainers)
- Real Neo4j instance (testcontainers)
- Vector + graph coordination
- ECL pipeline end-to-end
- Memify operations
- Hybrid search accuracy
- Memory persistence across restarts
- Stage compression pipeline
- Error tracking workflow

Component ID: MEM-027
Ticket: MEM-027 (Implement Integration Tests)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    ErrorRecord,
    ErrorType,
    MemoryLayer,
    MemoryRecord,
    RelationshipEdge,
    RelationshipType,
    StageMemory,
    StageType,
)
from agentcore.a2a_protocol.services.memory.context_compressor import (
    ContextCompressor,
)
from agentcore.a2a_protocol.services.memory.entity_extractor import (
    EntityExtractor,
)
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.hybrid_search import HybridSearchService
from agentcore.a2a_protocol.services.memory.memify_optimizer import MemifyOptimizer
from agentcore.a2a_protocol.services.memory.relationship_detector import (
    RelationshipDetector,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)
from agentcore.a2a_protocol.services.memory.stage_manager import StageManager
from agentcore.a2a_protocol.services.memory.storage_backend import (
    VectorStorageBackend,
)

# Import test helpers
from tests.integration.helpers.ecl_test_helper import ECLPipeline

# Import testcontainer fixtures
from tests.integration.fixtures.neo4j import (
    clean_neo4j_db,
    neo4j_container,
    neo4j_driver,
    neo4j_session_with_sample_graph,
    neo4j_uri,
)
from tests.integration.fixtures.qdrant import (
    qdrant_client,
    qdrant_container,
    qdrant_sample_points,
    qdrant_test_collection,
    qdrant_url,
)

logger = structlog.get_logger(__name__)


# Mark all tests in this module as integration tests requiring real backends
pytestmark = pytest.mark.asyncio


class TestVectorGraphCoordination:
    """Test coordination between Qdrant vector store and Neo4j graph database."""

    @pytest.fixture
    async def vector_backend(
        self, qdrant_client: AsyncQdrantClient, qdrant_test_collection: str
    ) -> VectorStorageBackend:
        """Create vector storage backend with real Qdrant."""
        return VectorStorageBackend(
            qdrant_client=qdrant_client, collection_name=qdrant_test_collection
        )

    @pytest.fixture
    async def graph_service(
        self, neo4j_driver: AsyncDriver, clean_neo4j_db
    ) -> GraphMemoryService:
        """Create graph service with real Neo4j."""
        return GraphMemoryService(driver=neo4j_driver)

    async def test_store_memory_in_both_backends(
        self,
        vector_backend: VectorStorageBackend,
        graph_service: GraphMemoryService,
    ):
        """Test storing memory in both vector and graph databases."""
        # Create memory record
        memory = MemoryRecord(
            memory_id="hybrid-mem-1",
            memory_layer=MemoryLayer.EPISODIC,
            content="User implemented JWT authentication using PyJWT library",
            summary="JWT authentication implementation",
            embedding=[0.1] * 1536,
            agent_id="agent-1",
            session_id="session-1",
            task_id="task-1",
            keywords=["authentication", "jwt", "security"],
            is_critical=True,
        )

        # Store in vector database
        await vector_backend.store_memory(memory)

        # Store in graph database
        await graph_service.store_memory_node(memory)

        # Verify vector storage
        vector_results = await vector_backend.search_similar(
            query_embedding=[0.1] * 1536, limit=5
        )
        assert len(vector_results) > 0
        assert any(r.memory_id == "hybrid-mem-1" for r in vector_results)

        # Verify graph storage
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (m:Memory {memory_id: $memory_id}) RETURN m.memory_id as id",
                memory_id="hybrid-mem-1",
            )
            record = await result.single()
            assert record is not None
            assert record["id"] == "hybrid-mem-1"

    async def test_vector_search_with_graph_enrichment(
        self,
        vector_backend: VectorStorageBackend,
        graph_service: GraphMemoryService,
    ):
        """Test vector search followed by graph-based context enrichment."""
        # Store multiple memories with relationships
        memories = [
            MemoryRecord(
                memory_id=f"mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Memory content {i}",
                summary=f"Summary {i}",
                embedding=[0.1 * i] * 1536,
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=[f"keyword{i}"],
            )
            for i in range(5)
        ]

        # Store in both backends
        for mem in memories:
            await vector_backend.store_memory(mem)
            await graph_service.store_memory_node(mem)

        # Create entity and link to memories
        entity = EntityNode(
            entity_id="entity-auth",
            name="authentication",
            entity_type="concept",
            confidence=0.9,
        )
        await graph_service.store_entity(entity)

        # Link memories to entity
        for mem in memories[:3]:
            relationship = RelationshipEdge(
                source_id=mem.memory_id,
                target_id=entity.entity_id,
                relationship_type=RelationshipType.MENTIONS,
                strength=0.8,
            )
            await graph_service.store_relationship(relationship)

        # Perform vector search
        vector_results = await vector_backend.search_similar(
            query_embedding=[0.2] * 1536, limit=3
        )

        # Enrich with graph context
        for result in vector_results:
            related_entities = await graph_service.get_related_entities(
                result.memory_id, max_depth=1
            )
            # Should find the authentication entity
            assert len(related_entities) > 0
            assert any(e.entity_id == "entity-auth" for e in related_entities)

    async def test_graph_traversal_performance(
        self, graph_service: GraphMemoryService
    ):
        """Test graph traversal meets performance targets (<200ms for 2-hop)."""
        # Create a small graph for traversal
        entities = [
            EntityNode(
                entity_id=f"entity-{i}",
                name=f"Entity {i}",
                entity_type="concept",
                confidence=0.9,
            )
            for i in range(10)
        ]

        for entity in entities:
            await graph_service.store_entity(entity)

        # Create relationships (chain)
        for i in range(len(entities) - 1):
            relationship = RelationshipEdge(
                source_id=entities[i].entity_id,
                target_id=entities[i + 1].entity_id,
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.8,
            )
            await graph_service.store_relationship(relationship)

        # Time 2-hop traversal
        start_time = datetime.now(UTC)
        related = await graph_service.get_related_entities("entity-0", max_depth=2)
        end_time = datetime.now(UTC)

        # Check performance target: <200ms
        duration_ms = (end_time - start_time).total_seconds() * 1000
        assert duration_ms < 200, f"Graph traversal took {duration_ms}ms (target: <200ms)"

        # Verify correctness: should find 2-hop neighbors
        assert len(related) >= 2


class TestECLPipelineE2E:
    """Test ECL (Extract-Contextualize-Link) pipeline end-to-end."""

    @pytest.fixture
    async def ecl_pipeline(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db,
    ) -> ECLPipeline:
        """Create ECL pipeline with real backends."""
        vector_backend = VectorStorageBackend(
            qdrant_client=qdrant_client, collection_name=qdrant_test_collection
        )
        graph_service = GraphMemoryService(driver=neo4j_driver)
        entity_extractor = EntityExtractor()
        relationship_detector = RelationshipDetector()

        return ECLPipeline(
            vector_backend=vector_backend,
            graph_service=graph_service,
            entity_extractor=entity_extractor,
            relationship_detector=relationship_detector,
        )

    async def test_ecl_pipeline_full_flow(self, ecl_pipeline: ECLPipeline):
        """Test complete ECL pipeline: extract, contextualize, load."""
        # Input: raw conversation data
        conversation = [
            {
                "role": "user",
                "content": "I need to implement JWT authentication for my API using Python",
            },
            {
                "role": "assistant",
                "content": "I'll help you implement JWT authentication. You can use the PyJWT library for this.",
            },
            {
                "role": "user",
                "content": "How do I validate the JWT tokens?",
            },
        ]

        # Run ECL pipeline
        result = await ecl_pipeline.process(
            conversation_data=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        # Verify extract phase: memories created
        assert result["memories_created"] > 0
        assert len(result["extracted_entities"]) > 0

        # Verify contextualize phase: entities extracted
        entities = result["extracted_entities"]
        entity_names = [e.name.lower() for e in entities]
        # Should extract key concepts like "authentication", "jwt", etc.
        assert any("auth" in name or "jwt" in name for name in entity_names)

        # Verify link phase: relationships created
        assert result["relationships_created"] > 0

        # Verify load phase: data stored in both backends
        # Check vector store
        memories = result["memory_records"]
        assert len(memories) > 0

        # Check graph store - verify entities and relationships exist
        graph_service = ecl_pipeline.graph_service
        for entity in entities:
            stored_entity = await graph_service.get_entity(entity.entity_id)
            assert stored_entity is not None
            assert stored_entity.name == entity.name

    async def test_ecl_pipeline_entity_extraction_accuracy(
        self, ecl_pipeline: ECLPipeline
    ):
        """Test entity extraction accuracy meets target (80%+)."""
        # Test data with known entities
        conversation = [
            {
                "role": "user",
                "content": "I'm working with PostgreSQL database and Redis cache. "
                "The API uses FastAPI framework and handles OAuth2 authentication.",
            }
        ]

        result = await ecl_pipeline.process(
            conversation_data=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        entities = result["extracted_entities"]
        entity_names = {e.name.lower() for e in entities}

        # Expected entities (at least 4 out of 5 for 80% accuracy)
        expected = {"postgresql", "redis", "fastapi", "oauth2", "database"}
        matched = sum(1 for exp in expected if any(exp in name for name in entity_names))

        accuracy = matched / len(expected)
        assert accuracy >= 0.8, f"Entity extraction accuracy: {accuracy:.2%} (target: 80%+)"

    async def test_ecl_pipeline_relationship_detection(
        self, ecl_pipeline: ECLPipeline
    ):
        """Test relationship detection between entities."""
        conversation = [
            {
                "role": "user",
                "content": "FastAPI uses Pydantic for data validation. "
                "The authentication system relies on JWT tokens.",
            }
        ]

        result = await ecl_pipeline.process(
            conversation_data=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        relationships = result["extracted_relationships"]

        # Should detect relationships like FastAPI USES Pydantic
        assert len(relationships) > 0

        # Check relationship types
        relationship_types = {r.relationship_type for r in relationships}
        assert RelationshipType.RELATES_TO in relationship_types or RelationshipType.USES in relationship_types


class TestMemifyOperations:
    """Test Memify graph optimization operations."""

    @pytest.fixture
    async def memify_optimizer(
        self, neo4j_driver: AsyncDriver, clean_neo4j_db
    ) -> MemifyOptimizer:
        """Create Memify optimizer with real Neo4j."""
        graph_service = GraphMemoryService(driver=neo4j_driver)
        return MemifyOptimizer(graph_service=graph_service)

    async def test_entity_consolidation(
        self, memify_optimizer: MemifyOptimizer, neo4j_driver: AsyncDriver
    ):
        """Test entity consolidation merges similar entities."""
        # Create duplicate/similar entities
        entities = [
            EntityNode(
                entity_id="entity-auth-1",
                name="authentication",
                entity_type="concept",
                confidence=0.9,
            ),
            EntityNode(
                entity_id="entity-auth-2",
                name="auth",  # Similar to authentication
                entity_type="concept",
                confidence=0.85,
            ),
            EntityNode(
                entity_id="entity-jwt-1",
                name="JWT",
                entity_type="tool",
                confidence=0.95,
            ),
        ]

        graph_service = memify_optimizer.graph_service
        for entity in entities:
            await graph_service.store_entity(entity)

        # Run consolidation
        consolidation_result = await memify_optimizer.consolidate_entities(
            similarity_threshold=0.85
        )

        # Verify consolidation occurred
        assert consolidation_result["entities_merged"] > 0
        assert consolidation_result["duplicate_rate"] < 0.05  # <5% duplicates target

        # Verify similar entities were merged
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity) WHERE e.name IN ['authentication', 'auth'] RETURN count(e) as count"
            )
            record = await result.single()
            # Should have merged to 1 entity
            assert record["count"] <= 1

    async def test_relationship_pruning(
        self, memify_optimizer: MemifyOptimizer, neo4j_driver: AsyncDriver
    ):
        """Test relationship pruning removes low-value edges."""
        graph_service = memify_optimizer.graph_service

        # Create entities
        entities = [
            EntityNode(
                entity_id=f"entity-{i}",
                name=f"Entity {i}",
                entity_type="concept",
                confidence=0.9,
            )
            for i in range(5)
        ]
        for entity in entities:
            await graph_service.store_entity(entity)

        # Create relationships with varying access counts
        relationships = [
            RelationshipEdge(
                source_id="entity-0",
                target_id="entity-1",
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.9,
                metadata={"access_count": 10},  # High access
            ),
            RelationshipEdge(
                source_id="entity-0",
                target_id="entity-2",
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.5,
                metadata={"access_count": 1},  # Low access (should be pruned)
            ),
            RelationshipEdge(
                source_id="entity-1",
                target_id="entity-3",
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.8,
                metadata={"access_count": 5},  # Medium access
            ),
        ]

        for rel in relationships:
            await graph_service.store_relationship(rel)

        # Run pruning (threshold: access_count < 2)
        pruning_result = await memify_optimizer.prune_relationships(
            min_access_count=2
        )

        # Verify low-value relationship was removed
        assert pruning_result["relationships_pruned"] > 0

        # Verify high-value relationships remain
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
            )
            record = await result.single()
            # Should have 2 relationships left (removed the one with access_count=1)
            assert record["count"] == 2

    async def test_memify_pattern_detection(self, memify_optimizer: MemifyOptimizer):
        """Test pattern detection identifies frequently traversed paths."""
        graph_service = memify_optimizer.graph_service

        # Create a graph with a common pattern
        entities = [
            EntityNode(
                entity_id=f"entity-{i}",
                name=f"Entity {i}",
                entity_type="concept",
                confidence=0.9,
            )
            for i in range(6)
        ]
        for entity in entities:
            await graph_service.store_entity(entity)

        # Create common pattern: 0->1->2 (traversed 3 times)
        # And uncommon pattern: 3->4->5 (traversed once)
        for _ in range(3):
            await graph_service.store_relationship(
                RelationshipEdge(
                    source_id="entity-0",
                    target_id="entity-1",
                    relationship_type=RelationshipType.RELATES_TO,
                    strength=0.9,
                )
            )
            await graph_service.store_relationship(
                RelationshipEdge(
                    source_id="entity-1",
                    target_id="entity-2",
                    relationship_type=RelationshipType.RELATES_TO,
                    strength=0.9,
                )
            )

        # Create uncommon pattern
        await graph_service.store_relationship(
            RelationshipEdge(
                source_id="entity-3",
                target_id="entity-4",
                relationship_type=RelationshipType.RELATES_TO,
                strength=0.5,
            )
        )

        # Detect patterns
        patterns = await memify_optimizer.detect_patterns(min_frequency=2)

        # Should detect the common pattern
        assert len(patterns) > 0
        # Pattern should involve entity-0, entity-1, entity-2
        pattern_entities = {
            entity_id for pattern in patterns for entity_id in pattern["path"]
        }
        assert "entity-0" in pattern_entities or "entity-1" in pattern_entities

    async def test_memify_consolidation_accuracy(
        self, memify_optimizer: MemifyOptimizer
    ):
        """Test Memify consolidation meets accuracy target (90%+)."""
        graph_service = memify_optimizer.graph_service

        # Create ground truth: 10 unique entities + 2 duplicates of each (30 total)
        unique_entities = [
            f"concept_{i}" for i in range(10)
        ]

        entities_to_create = []
        for concept in unique_entities:
            # Original
            entities_to_create.append(
                EntityNode(
                    entity_id=f"entity-{concept}-0",
                    name=concept,
                    entity_type="concept",
                    confidence=0.9,
                )
            )
            # Duplicate 1 (slight variation)
            entities_to_create.append(
                EntityNode(
                    entity_id=f"entity-{concept}-1",
                    name=f"{concept}_v1",  # Variation
                    entity_type="concept",
                    confidence=0.85,
                )
            )
            # Duplicate 2 (slight variation)
            entities_to_create.append(
                EntityNode(
                    entity_id=f"entity-{concept}-2",
                    name=f"{concept}_v2",  # Variation
                    entity_type="concept",
                    confidence=0.8,
                )
            )

        for entity in entities_to_create:
            await graph_service.store_entity(entity)

        # Run consolidation
        result = await memify_optimizer.consolidate_entities(
            similarity_threshold=0.8
        )

        # Calculate accuracy: should merge ~20 entities (keeping 10 unique)
        # Accuracy = correctly merged / total duplicates
        # Target: 90%+ consolidation accuracy
        total_entities = len(entities_to_create)
        entities_after = total_entities - result["entities_merged"]

        # Ideally should have ~10 entities left (perfect consolidation)
        # Allow some tolerance: 10-12 entities is good
        assert entities_after <= 12, f"Too many entities remaining: {entities_after} (expected ~10)"

        # Check consolidation accuracy metric from result
        assert result["consolidation_accuracy"] >= 0.9, \
            f"Consolidation accuracy: {result['consolidation_accuracy']:.2%} (target: 90%+)"


class TestHybridSearchAccuracy:
    """Test hybrid search combining vector similarity and graph traversal."""

    @pytest.fixture
    async def hybrid_search(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db,
    ) -> HybridSearchService:
        """Create hybrid search service with real backends."""
        vector_backend = VectorStorageBackend(
            qdrant_client=qdrant_client, collection_name=qdrant_test_collection
        )
        graph_service = GraphMemoryService(driver=neo4j_driver)

        return HybridSearchService(
            vector_backend=vector_backend,
            graph_service=graph_service,
        )

    async def test_hybrid_search_performance(
        self, hybrid_search: HybridSearchService
    ):
        """Test hybrid search meets performance target (<300ms p95)."""
        # Create test data
        memories = [
            MemoryRecord(
                memory_id=f"mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Memory about authentication and security {i}",
                summary=f"Auth memory {i}",
                embedding=[0.1 * i] * 1536,
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=["auth", "security"],
            )
            for i in range(10)
        ]

        # Store memories
        for mem in memories:
            await hybrid_search.vector_backend.store_memory(mem)
            await hybrid_search.graph_service.store_memory_node(mem)

        # Create entities and link
        entity = EntityNode(
            entity_id="entity-auth",
            name="authentication",
            entity_type="concept",
            confidence=0.9,
        )
        await hybrid_search.graph_service.store_entity(entity)

        for mem in memories[:5]:
            await hybrid_search.graph_service.store_relationship(
                RelationshipEdge(
                    source_id=mem.memory_id,
                    target_id=entity.entity_id,
                    relationship_type=RelationshipType.MENTIONS,
                    strength=0.8,
                )
            )

        # Time hybrid search
        start_time = datetime.now(UTC)
        results = await hybrid_search.search(
            query_embedding=[0.5] * 1536,
            limit=5,
            include_graph_context=True,
        )
        end_time = datetime.now(UTC)

        # Check performance target: <300ms
        duration_ms = (end_time - start_time).total_seconds() * 1000
        assert duration_ms < 300, f"Hybrid search took {duration_ms}ms (target: <300ms)"

        # Verify results
        assert len(results) > 0

    async def test_hybrid_search_accuracy(
        self, hybrid_search: HybridSearchService
    ):
        """Test hybrid search retrieval accuracy (90%+ target)."""
        # Create test dataset with known ground truth
        # 10 memories about "authentication", 5 about "database"
        auth_memories = [
            MemoryRecord(
                memory_id=f"auth-mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Authentication implementation using JWT tokens {i}",
                summary=f"Auth {i}",
                embedding=[0.9] * 1536,  # Similar vector
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=["auth", "jwt"],
            )
            for i in range(10)
        ]

        db_memories = [
            MemoryRecord(
                memory_id=f"db-mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Database schema design for users table {i}",
                summary=f"DB {i}",
                embedding=[0.1] * 1536,  # Different vector
                agent_id="agent-1",
                session_id="session-1",
                task_id=None,
                keywords=["database", "schema"],
            )
            for i in range(5)
        ]

        # Store all memories
        for mem in auth_memories + db_memories:
            await hybrid_search.vector_backend.store_memory(mem)
            await hybrid_search.graph_service.store_memory_node(mem)

        # Create entities
        auth_entity = EntityNode(
            entity_id="entity-auth",
            name="authentication",
            entity_type="concept",
            confidence=0.95,
        )
        db_entity = EntityNode(
            entity_id="entity-db",
            name="database",
            entity_type="concept",
            confidence=0.9,
        )

        await hybrid_search.graph_service.store_entity(auth_entity)
        await hybrid_search.graph_service.store_entity(db_entity)

        # Link memories to entities
        for mem in auth_memories:
            await hybrid_search.graph_service.store_relationship(
                RelationshipEdge(
                    source_id=mem.memory_id,
                    target_id=auth_entity.entity_id,
                    relationship_type=RelationshipType.MENTIONS,
                    strength=0.9,
                )
            )

        for mem in db_memories:
            await hybrid_search.graph_service.store_relationship(
                RelationshipEdge(
                    source_id=mem.memory_id,
                    target_id=db_entity.entity_id,
                    relationship_type=RelationshipType.MENTIONS,
                    strength=0.9,
                )
            )

        # Query for authentication memories
        results = await hybrid_search.search(
            query_embedding=[0.9] * 1536,  # Similar to auth memories
            limit=10,
            include_graph_context=True,
        )

        # Calculate precision: how many returned results are relevant (auth memories)
        relevant_results = [r for r in results if r.memory_id.startswith("auth-mem")]
        precision = len(relevant_results) / len(results) if results else 0

        # Target: 90%+ precision
        assert precision >= 0.9, f"Hybrid search precision: {precision:.2%} (target: 90%+)"


class TestMemoryPersistence:
    """Test memory persistence across container restarts."""

    async def test_qdrant_persistence(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
    ):
        """Test memory persistence in Qdrant across restarts."""
        # Store memory
        vector_backend = VectorStorageBackend(
            qdrant_client=qdrant_client, collection_name=qdrant_test_collection
        )

        memory = MemoryRecord(
            memory_id="persist-mem-1",
            memory_layer=MemoryLayer.EPISODIC,
            content="Test persistence",
            summary="Persistence test",
            embedding=[0.5] * 1536,
            agent_id="agent-1",
            session_id="session-1",
            task_id=None,
            keywords=["test"],
        )

        await vector_backend.store_memory(memory)

        # Verify stored
        results = await vector_backend.search_similar(
            query_embedding=[0.5] * 1536, limit=5
        )
        assert any(r.memory_id == "persist-mem-1" for r in results)

        # Note: Full restart test would require container restart
        # which is complex in pytest. This test validates basic persistence.

    async def test_neo4j_persistence(
        self, neo4j_driver: AsyncDriver, clean_neo4j_db
    ):
        """Test graph persistence in Neo4j."""
        graph_service = GraphMemoryService(driver=neo4j_driver)

        # Store entity
        entity = EntityNode(
            entity_id="persist-entity-1",
            name="persistent_entity",
            entity_type="concept",
            confidence=0.9,
        )

        await graph_service.store_entity(entity)

        # Verify stored
        retrieved = await graph_service.get_entity("persist-entity-1")
        assert retrieved is not None
        assert retrieved.name == "persistent_entity"


class TestStageCompressionPipeline:
    """Test stage compression pipeline integration."""

    @pytest.fixture
    async def stage_manager(self) -> StageManager:
        """Create stage manager."""
        return StageManager()

    @pytest.fixture
    async def context_compressor(self) -> ContextCompressor:
        """Create context compressor."""
        # Mock LLM for testing (real LLM would be too slow/expensive)
        return ContextCompressor(llm_client=None, use_mock=True)

    async def test_stage_completion_triggers_compression(
        self,
        stage_manager: StageManager,
        context_compressor: ContextCompressor,
    ):
        """Test stage completion triggers compression."""
        # Create stage with memories
        stage = StageMemory(
            stage_id="stage-1",
            task_id="task-1",
            agent_id="agent-1",
            stage_type=StageType.EXECUTION,
            raw_memory_ids=["mem-1", "mem-2", "mem-3"],
            compressed_summary=None,
        )

        await stage_manager.create_stage(stage)

        # Complete stage - should trigger compression
        await stage_manager.complete_stage(
            stage_id="stage-1",
            compressor=context_compressor,
        )

        # Verify compression occurred
        completed_stage = await stage_manager.get_stage("stage-1")
        assert completed_stage.compressed_summary is not None
        assert len(completed_stage.compressed_summary) > 0

    async def test_compression_quality_validation(
        self,
        context_compressor: ContextCompressor,
    ):
        """Test compression quality meets targets (95%+ fact retention)."""
        # Original content with known facts
        original_content = """
        User implemented JWT authentication using PyJWT library.
        The implementation includes token generation, validation, and refresh logic.
        Security best practices: HS256 algorithm, secret key rotation, expiration checks.
        """

        compressed = await context_compressor.compress_stage(
            memories=[original_content],
            stage_type=StageType.EXECUTION,
        )

        # Verify compression occurred (should be shorter)
        assert len(compressed) < len(original_content)

        # Verify key facts retained
        key_facts = ["jwt", "authentication", "pyjwt", "validation", "security"]
        retained_facts = sum(
            1 for fact in key_facts if fact.lower() in compressed.lower()
        )

        retention_rate = retained_facts / len(key_facts)
        assert retention_rate >= 0.95, \
            f"Fact retention: {retention_rate:.2%} (target: 95%+)"


class TestErrorTrackingWorkflow:
    """Test error tracking workflow integration."""

    @pytest.fixture
    async def error_tracker(self) -> ErrorTracker:
        """Create error tracker."""
        return ErrorTracker()

    async def test_error_recording_and_pattern_detection(
        self,
        error_tracker: ErrorTracker,
    ):
        """Test error recording and pattern detection."""
        # Record multiple similar errors
        errors = [
            ErrorRecord(
                error_id=f"error-{i}",
                error_type=ErrorType.HALLUCINATION,
                content=f"Model generated incorrect fact about authentication {i}",
                severity=0.8,
                context={"stage": "execution", "task_id": f"task-{i}"},
                agent_id="agent-1",
                session_id="session-1",
                task_id=f"task-{i}",
            )
            for i in range(5)
        ]

        for error in errors:
            await error_tracker.record_error(error)

        # Detect patterns
        patterns = await error_tracker.detect_patterns(agent_id="agent-1")

        # Should detect hallucination pattern
        assert len(patterns) > 0
        assert any(p["error_type"] == ErrorType.HALLUCINATION for p in patterns)
        assert any(p["frequency"] >= 3 for p in patterns)  # 5 errors should trigger pattern

    async def test_error_severity_scoring(
        self,
        error_tracker: ErrorTracker,
    ):
        """Test error severity scoring (0-1 scale)."""
        # Record errors with different severities
        critical_error = ErrorRecord(
            error_id="error-critical",
            error_type=ErrorType.INCORRECT_ACTION,
            content="Agent deleted user data without confirmation",
            severity=1.0,  # Maximum severity
            context={"impact": "data_loss"},
            agent_id="agent-1",
            session_id="session-1",
            task_id="task-1",
        )

        minor_error = ErrorRecord(
            error_id="error-minor",
            error_type=ErrorType.MISSING_INFO,
            content="Response lacked some optional details",
            severity=0.2,  # Low severity
            context={"impact": "minor"},
            agent_id="agent-1",
            session_id="session-1",
            task_id="task-1",
        )

        await error_tracker.record_error(critical_error)
        await error_tracker.record_error(minor_error)

        # Query errors by severity
        critical_errors = await error_tracker.get_errors_by_severity(
            min_severity=0.8
        )
        minor_errors = await error_tracker.get_errors_by_severity(
            max_severity=0.3
        )

        # Verify filtering
        assert len(critical_errors) >= 1
        assert len(minor_errors) >= 1
        assert all(e.severity >= 0.8 for e in critical_errors)
        assert all(e.severity <= 0.3 for e in minor_errors)


class TestCodeCoverage:
    """Verify 90%+ code coverage for memory service components."""

    async def test_coverage_placeholder(self):
        """Placeholder for coverage validation.

        Coverage is measured by pytest-cov during test execution.
        Run: pytest --cov=agentcore.a2a_protocol.services.memory --cov-report=term-missing

        Target: 90%+ coverage for all memory service modules
        """
        # This test exists to document coverage requirements
        # Actual coverage is measured by pytest-cov plugin
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
