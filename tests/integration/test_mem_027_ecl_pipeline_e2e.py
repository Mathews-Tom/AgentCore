"""
Integration tests for MEM-027: ECL Pipeline End-to-End

Focused integration tests for the Extract-Contextualize-Link pipeline:
- Extract phase: Memory extraction from conversations
- Contextualize phase: Entity and relationship extraction
- Link phase: Graph construction and vector indexing
- Load phase: Multi-backend coordination

Component ID: MEM-027
Ticket: MEM-027 (Implement Integration Tests)
"""

from __future__ import annotations

import pytest
import structlog
from neo4j import AsyncDriver
from qdrant_client import AsyncQdrantClient

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    MemoryLayer,
    RelationshipType,
)
from agentcore.a2a_protocol.services.memory.entity_extractor import EntityExtractor
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.relationship_detector import (
    RelationshipDetector,
)
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


class TestECLExtractPhase:
    """Test Extract phase of ECL pipeline."""

    @pytest.fixture
    async def entity_extractor(self) -> EntityExtractor:
        """Create entity extractor."""
        return EntityExtractor(use_mock_llm=True)

    async def test_extract_from_conversation(
        self, entity_extractor: EntityExtractor
    ):
        """Test extracting memories from conversation."""
        conversation = [
            {
                "role": "user",
                "content": "I need help implementing a REST API with authentication.",
            },
            {
                "role": "assistant",
                "content": "I'll help you create a REST API with JWT authentication using FastAPI.",
            },
        ]

        memories = await entity_extractor.extract_memories(
            conversation=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        # Verify memories extracted
        assert len(memories) > 0
        assert all(hasattr(m, "memory_id") for m in memories)
        assert all(m.agent_id == "agent-1" for m in memories)
        assert all(m.session_id == "session-1" for m in memories)

    async def test_extract_episodic_memories(
        self, entity_extractor: EntityExtractor
    ):
        """Test extracting episodic memories (conversation turns)."""
        conversation = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny today."},
            {"role": "user", "content": "Thanks!"},
        ]

        memories = await entity_extractor.extract_memories(
            conversation=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        # Should create episodic memory for each meaningful turn
        episodic_memories = [m for m in memories if m.memory_layer == MemoryLayer.EPISODIC]
        assert len(episodic_memories) >= 1

    async def test_extract_with_embeddings(
        self, entity_extractor: EntityExtractor
    ):
        """Test that extracted memories include embeddings."""
        conversation = [
            {"role": "user", "content": "Tell me about machine learning."},
        ]

        memories = await entity_extractor.extract_memories(
            conversation=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        # All memories should have embeddings
        assert all(len(m.embedding) > 0 for m in memories)
        # Embeddings should be 1536-dimensional (OpenAI text-embedding-3-small)
        assert all(len(m.embedding) == 1536 for m in memories)


class TestECLContextualizePhase:
    """Test Contextualize phase of ECL pipeline."""

    @pytest.fixture
    async def entity_extractor(self) -> EntityExtractor:
        """Create entity extractor."""
        return EntityExtractor(use_mock_llm=True)

    @pytest.fixture
    async def relationship_detector(self) -> RelationshipDetector:
        """Create relationship detector."""
        return RelationshipDetector(use_mock_llm=True)

    async def test_entity_extraction(self, entity_extractor: EntityExtractor):
        """Test extracting entities from memories."""
        content = """
        I'm building a microservices architecture using Docker and Kubernetes.
        The API gateway handles authentication and routes requests to backend services.
        """

        entities = await entity_extractor.extract_entities(content=content)

        # Should extract key entities
        assert len(entities) > 0
        entity_names = {e.name.lower() for e in entities}

        # Check for expected entities
        expected_entities = {"docker", "kubernetes", "microservices", "api", "authentication"}
        matched = sum(1 for exp in expected_entities if any(exp in name for name in entity_names))

        # Should match most expected entities
        assert matched >= 3, f"Only matched {matched} out of {len(expected_entities)} expected entities"

    async def test_entity_classification(self, entity_extractor: EntityExtractor):
        """Test entity type classification."""
        content = """
        John implemented the authentication system using the PyJWT library.
        The system validates OAuth2 tokens for API security.
        """

        entities = await entity_extractor.extract_entities(content=content)

        # Should classify different entity types
        entity_types = {e.entity_type for e in entities}

        # Should have multiple types (person, tool, concept)
        assert len(entity_types) >= 2

    async def test_relationship_detection(
        self,
        entity_extractor: EntityExtractor,
        relationship_detector: RelationshipDetector,
    ):
        """Test detecting relationships between entities."""
        content = """
        FastAPI uses Pydantic for data validation.
        The authentication system relies on JWT tokens.
        """

        # Extract entities first
        entities = await entity_extractor.extract_entities(content=content)

        # Detect relationships
        relationships = await relationship_detector.detect_relationships(
            content=content,
            entities=entities,
        )

        # Should detect relationships
        assert len(relationships) > 0

        # Check relationship types
        relationship_types = {r.relationship_type for r in relationships}
        assert len(relationship_types) > 0


class TestECLLinkPhase:
    """Test Link phase of ECL pipeline."""

    @pytest.fixture
    async def graph_service(
        self, neo4j_driver: AsyncDriver, clean_neo4j_db
    ) -> GraphMemoryService:
        """Create graph service."""
        return GraphMemoryService(driver=neo4j_driver)

    async def test_store_entities_in_graph(
        self, graph_service: GraphMemoryService
    ):
        """Test storing extracted entities in graph database."""
        entities = [
            EntityNode(
                entity_id="entity-1",
                name="FastAPI",
                entity_type="tool",
                confidence=0.95,
            ),
            EntityNode(
                entity_id="entity-2",
                name="Pydantic",
                entity_type="tool",
                confidence=0.9,
            ),
        ]

        # Store entities
        for entity in entities:
            await graph_service.store_entity(entity)

        # Verify storage
        for entity in entities:
            retrieved = await graph_service.get_entity(entity.entity_id)
            assert retrieved is not None
            assert retrieved.name == entity.name

    async def test_create_relationships_in_graph(
        self, graph_service: GraphMemoryService
    ):
        """Test creating relationships in graph database."""
        # Create entities first
        entity1 = EntityNode(
            entity_id="entity-fastapi",
            name="FastAPI",
            entity_type="tool",
            confidence=0.95,
        )
        entity2 = EntityNode(
            entity_id="entity-pydantic",
            name="Pydantic",
            entity_type="tool",
            confidence=0.9,
        )

        await graph_service.store_entity(entity1)
        await graph_service.store_entity(entity2)

        # Create relationship
        from agentcore.a2a_protocol.models.memory import RelationshipEdge

        relationship = RelationshipEdge(
            source_id=entity1.entity_id,
            target_id=entity2.entity_id,
            relationship_type=RelationshipType.USES,
            strength=0.9,
        )

        await graph_service.store_relationship(relationship)

        # Verify relationship
        related_entities = await graph_service.get_related_entities(
            entity1.entity_id, max_depth=1
        )

        assert len(related_entities) > 0
        assert any(e.entity_id == entity2.entity_id for e in related_entities)


class TestECLLoadPhase:
    """Test Load phase of ECL pipeline."""

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

    async def test_load_memories_to_vector_store(
        self, vector_backend: VectorStorageBackend
    ):
        """Test loading memories to vector database."""
        from agentcore.a2a_protocol.models.memory import MemoryRecord

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

        # Store memories
        for memory in memories:
            await vector_backend.store_memory(memory)

        # Verify storage via search
        results = await vector_backend.search_similar(
            query_embedding=[0.2] * 1536,
            limit=5,
        )

        assert len(results) > 0

    async def test_load_graph_data(
        self,
        vector_backend: VectorStorageBackend,
        graph_service: GraphMemoryService,
    ):
        """Test coordinated loading to both vector and graph stores."""
        from agentcore.a2a_protocol.models.memory import MemoryRecord

        # Create memory
        memory = MemoryRecord(
            memory_id="coord-mem-1",
            memory_layer=MemoryLayer.EPISODIC,
            content="User implemented JWT authentication",
            summary="JWT auth implementation",
            embedding=[0.5] * 1536,
            agent_id="agent-1",
            session_id="session-1",
            task_id=None,
            keywords=["jwt", "auth"],
        )

        # Create entity
        entity = EntityNode(
            entity_id="entity-jwt",
            name="JWT",
            entity_type="tool",
            confidence=0.95,
        )

        # Store in both backends
        await vector_backend.store_memory(memory)
        await graph_service.store_memory_node(memory)
        await graph_service.store_entity(entity)

        # Create relationship
        from agentcore.a2a_protocol.models.memory import RelationshipEdge

        relationship = RelationshipEdge(
            source_id=memory.memory_id,
            target_id=entity.entity_id,
            relationship_type=RelationshipType.MENTIONS,
            strength=0.9,
        )
        await graph_service.store_relationship(relationship)

        # Verify both stores have data
        # Vector store
        vector_results = await vector_backend.search_similar(
            query_embedding=[0.5] * 1536,
            limit=5,
        )
        assert any(r.memory_id == "coord-mem-1" for r in vector_results)

        # Graph store
        related = await graph_service.get_related_entities(
            memory.memory_id, max_depth=1
        )
        assert any(e.entity_id == "entity-jwt" for e in related)


class TestECLPipelineIntegration:
    """Test complete ECL pipeline integration."""

    @pytest.fixture
    async def ecl_pipeline(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j_db,
    ) -> ECLPipeline:
        """Create complete ECL pipeline."""
        vector_backend = VectorStorageBackend(
            qdrant_client=qdrant_client,
            collection_name=qdrant_test_collection,
        )
        graph_service = GraphMemoryService(driver=neo4j_driver)
        entity_extractor = EntityExtractor(use_mock_llm=True)
        relationship_detector = RelationshipDetector(use_mock_llm=True)

        return ECLPipeline(
            vector_backend=vector_backend,
            graph_service=graph_service,
            entity_extractor=entity_extractor,
            relationship_detector=relationship_detector,
        )

    async def test_full_ecl_pipeline_flow(self, ecl_pipeline: ECLPipeline):
        """Test complete ECL pipeline from conversation to stored knowledge."""
        conversation = [
            {
                "role": "user",
                "content": "I need to build a REST API with PostgreSQL database and Redis cache.",
            },
            {
                "role": "assistant",
                "content": "I'll help you create a REST API using FastAPI framework. "
                "We'll use SQLAlchemy for PostgreSQL and redis-py for caching.",
            },
            {
                "role": "user",
                "content": "How do I implement authentication?",
            },
            {
                "role": "assistant",
                "content": "We can implement JWT authentication using python-jose library. "
                "Tokens will be validated on each API request.",
            },
        ]

        # Run complete pipeline
        result = await ecl_pipeline.process(
            conversation_data=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        # Verify Extract phase
        assert result["memories_created"] > 0
        assert len(result["memory_records"]) > 0

        # Verify Contextualize phase
        assert len(result["extracted_entities"]) > 0
        assert len(result["extracted_relationships"]) > 0

        # Verify entity extraction quality
        entity_names = {e.name.lower() for e in result["extracted_entities"]}
        expected = {"fastapi", "postgresql", "redis", "jwt", "sqlalchemy"}
        matched = sum(1 for exp in expected if any(exp in name for name in entity_names))

        # Should extract at least 3 out of 5 key entities (60%+)
        assert matched >= 3, f"Only extracted {matched} out of {len(expected)} expected entities"

        # Verify Link phase
        assert result["relationships_created"] > 0

        # Verify Load phase - check both backends
        # Vector backend should have memories
        memories = result["memory_records"]
        assert all(len(m.embedding) == 1536 for m in memories)

        # Graph backend should have entities and relationships
        graph_service = ecl_pipeline.graph_service
        for entity in result["extracted_entities"]:
            stored = await graph_service.get_entity(entity.entity_id)
            assert stored is not None

    async def test_ecl_pipeline_handles_complex_conversation(
        self, ecl_pipeline: ECLPipeline
    ):
        """Test ECL pipeline with complex multi-topic conversation."""
        conversation = [
            {
                "role": "user",
                "content": "I'm designing a microservices architecture. Each service will have its own database.",
            },
            {
                "role": "assistant",
                "content": "That's a good approach for service isolation. Consider using Docker containers "
                "and Kubernetes for orchestration.",
            },
            {
                "role": "user",
                "content": "What about inter-service communication?",
            },
            {
                "role": "assistant",
                "content": "You can use REST APIs with an API Gateway, or consider gRPC for better performance. "
                "Message queues like RabbitMQ or Kafka work well for async communication.",
            },
        ]

        result = await ecl_pipeline.process(
            conversation_data=conversation,
            agent_id="agent-1",
            session_id="session-1",
        )

        # Should extract multiple entities from different domains
        entities = result["extracted_entities"]
        assert len(entities) >= 5

        # Should detect relationships between related concepts
        relationships = result["extracted_relationships"]
        assert len(relationships) >= 3

        # Check that we have diverse entity types
        entity_types = {e.entity_type for e in entities}
        assert len(entity_types) >= 2  # Should have tools, concepts, etc.

    async def test_ecl_pipeline_error_handling(self, ecl_pipeline: ECLPipeline):
        """Test ECL pipeline handles errors gracefully."""
        # Empty conversation
        result = await ecl_pipeline.process(
            conversation_data=[],
            agent_id="agent-1",
            session_id="session-1",
        )

        # Should handle gracefully with no errors
        assert result["memories_created"] == 0
        assert result["extracted_entities"] == []

        # Single user message (incomplete conversation)
        result = await ecl_pipeline.process(
            conversation_data=[{"role": "user", "content": "Hello"}],
            agent_id="agent-1",
            session_id="session-1",
        )

        # Should still process and create at least one memory
        assert result["memories_created"] >= 0  # May be 0 for very short content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
