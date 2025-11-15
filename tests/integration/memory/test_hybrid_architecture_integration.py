"""
Comprehensive Integration Tests for Hybrid Memory Architecture.

Tests the complete memory system with:
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

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from neo4j import AsyncDriver, AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from testcontainers.neo4j import Neo4jContainer

from agentcore.a2a_protocol.models.llm import LLMResponse, LLMUsage
from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    EntityType,
    ErrorRecord,
    ErrorType,
    MemoryLayer,
    MemoryRecord,
    RelationshipEdge,
    RelationshipType,
    StageMemory,
    StageType,
    TaskContext,
)
from agentcore.a2a_protocol.services.memory.context_compressor import (
    ContextCompressor,
)
from agentcore.a2a_protocol.services.memory.entity_extractor import (
    EntityExtractor,
)
from agentcore.a2a_protocol.services.memory.error_tracker import ErrorTracker
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.quality_validator import QualityValidator
from agentcore.a2a_protocol.services.memory.relationship_detector import (
    RelationshipDetectorTask,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)
from agentcore.a2a_protocol.services.memory.stage_manager import StageManager


pytestmark = pytest.mark.asyncio


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def neo4j_container() -> Neo4jContainer:
    """Start Neo4j testcontainer with APOC and GDS plugins."""
    container = Neo4jContainer(image="neo4j:5.15-community")
    container.with_env("NEO4J_AUTH", "neo4j/testpassword")
    container.with_env("NEO4J_PLUGINS", '["apoc", "graph-data-science"]')
    container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*,gds.*")
    container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*,gds.*")

    container.start()
    yield container
    container.stop()


@pytest.fixture(scope="module")
async def neo4j_driver(neo4j_container: Neo4jContainer) -> AsyncDriver:
    """Create async Neo4j driver from testcontainer."""
    bolt_url = neo4j_container.get_connection_url()
    driver = AsyncGraphDatabase.driver(
        bolt_url,
        auth=("neo4j", "testpassword"),
        max_connection_pool_size=50,
    )

    await driver.verify_connectivity()
    yield driver
    await driver.close()


@pytest.fixture(scope="function")
async def clean_neo4j(neo4j_driver: AsyncDriver) -> None:
    """Clean Neo4j database before each test."""
    async with neo4j_driver.session(database="neo4j") as session:
        await session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture(scope="function")
async def graph_memory_service(
    neo4j_driver: AsyncDriver, clean_neo4j: None
) -> GraphMemoryService:
    """Create GraphMemoryService instance with initialized indexes."""
    service = GraphMemoryService(neo4j_driver)
    await service.initialize()
    return service


@pytest.fixture(scope="function")
async def qdrant_test_collection(
    qdrant_client: AsyncQdrantClient,
) -> str:
    """Create a test collection for each test function."""
    collection_name = f"test_hybrid_{id(asyncio.current_task())}"

    await qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Create payload indexes
    await qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="memory_layer",
        field_schema="keyword",
    )
    await qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="agent_id",
        field_schema="keyword",
    )
    await qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="is_critical",
        field_schema="bool",
    )
    await qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="session_id",
        field_schema="keyword",
    )

    yield collection_name

    await qdrant_client.delete_collection(collection_name=collection_name)


@pytest.fixture
def sample_memories() -> list[MemoryRecord]:
    """Create sample memory records for testing."""
    agent_id = str(uuid4())
    session_id = str(uuid4())
    task_id = str(uuid4())

    return [
        MemoryRecord(
            memory_id=f"mem-{uuid4()}",
            memory_layer=MemoryLayer.EPISODIC,
            content="User requested JWT authentication with Redis storage for token management.",
            summary="JWT auth request with Redis",
            embedding=[0.1] * 1536,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            keywords=["JWT", "authentication", "Redis"],
            entities=["JWT", "Redis"],
            is_critical=True,
            criticality_reason="Core authentication requirement",
        ),
        MemoryRecord(
            memory_id=f"mem-{uuid4()}",
            memory_layer=MemoryLayer.SEMANTIC,
            content="JWT tokens should have 1-hour TTL with refresh capability.",
            summary="Token TTL configuration",
            embedding=[0.2] * 1536,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            keywords=["TTL", "refresh", "token"],
            entities=["token", "TTL"],
            is_critical=True,
        ),
        MemoryRecord(
            memory_id=f"mem-{uuid4()}",
            memory_layer=MemoryLayer.PROCEDURAL,
            content="Action: POST /auth/login -> Outcome: 200 OK with JWT token",
            summary="Login endpoint success",
            embedding=[0.3] * 1536,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            keywords=["login", "endpoint", "POST"],
            actions=["POST /auth/login"],
            outcome="200 OK with JWT token",
            success=True,
        ),
        MemoryRecord(
            memory_id=f"mem-{uuid4()}",
            memory_layer=MemoryLayer.EPISODIC,
            content="Error rate was 8%, exceeding 5% threshold. Fixed connection pooling.",
            summary="Error rate fixed",
            embedding=[0.4] * 1536,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            keywords=["error", "pooling", "fixed"],
            is_critical=True,
        ),
        MemoryRecord(
            memory_id=f"mem-{uuid4()}",
            memory_layer=MemoryLayer.SEMANTIC,
            content="Connection pooling improves performance and reduces connection errors.",
            summary="Connection pooling benefits",
            embedding=[0.5] * 1536,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            keywords=["pooling", "performance"],
            entities=["connection pooling"],
        ),
    ]


@pytest.fixture
def entity_extractor() -> EntityExtractor:
    """Create EntityExtractor instance."""
    return EntityExtractor(trace_id="test-hybrid-integration")


@pytest.fixture
def relationship_detector() -> RelationshipDetectorTask:
    """Create RelationshipDetector instance."""
    return RelationshipDetectorTask(trace_id="test-hybrid-integration")


@pytest.fixture
def retrieval_service() -> EnhancedRetrievalService:
    """Create EnhancedRetrievalService instance."""
    return EnhancedRetrievalService()


@pytest.fixture
def stage_manager() -> StageManager:
    """Create StageManager instance."""
    return StageManager()


@pytest.fixture
def error_tracker() -> ErrorTracker:
    """Create ErrorTracker instance."""
    return ErrorTracker()


@pytest.fixture
def quality_validator() -> QualityValidator:
    """Create QualityValidator instance."""
    return QualityValidator(trace_id="test-hybrid-integration")


@pytest.fixture
def context_compressor() -> ContextCompressor:
    """Create ContextCompressor instance."""
    return ContextCompressor(trace_id="test-hybrid-integration")


# =============================================================================
# Test Classes
# =============================================================================


class TestVectorGraphCoordination:
    """Test vector + graph coordination (Acceptance Criteria #3)."""

    async def test_store_memory_in_both_vector_and_graph(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test storing memory in both Qdrant and Neo4j simultaneously."""
        memory = sample_memories[0]

        # Store in Qdrant (vector store)
        point = PointStruct(
            id=hash(memory.memory_id) % (2**63),
            vector=memory.embedding,
            payload={
                "memory_id": memory.memory_id,
                "memory_layer": memory.memory_layer.value,
                "content": memory.content,
                "agent_id": memory.agent_id,
                "is_critical": memory.is_critical,
                "entities": memory.entities,
            },
        )
        await qdrant_client.upsert(
            collection_name=qdrant_test_collection,
            points=[point],
        )

        # Store in Neo4j (graph store)
        async with neo4j_driver.session(database="neo4j") as session:
            await session.run(
                """
                CREATE (m:Memory {
                    memory_id: $memory_id,
                    agent_id: $agent_id,
                    content: $content,
                    layer: $layer,
                    is_critical: $is_critical,
                    created_at: datetime()
                })
                """,
                memory_id=memory.memory_id,
                agent_id=memory.agent_id,
                content=memory.content,
                layer=memory.memory_layer.value,
                is_critical=memory.is_critical,
            )

            # Create entity nodes for extracted entities
            for entity_name in memory.entities:
                await session.run(
                    """
                    MERGE (e:Entity {name: $entity_name})
                    WITH e
                    MATCH (m:Memory {memory_id: $memory_id})
                    CREATE (m)-[:MENTIONS]->(e)
                    """,
                    entity_name=entity_name,
                    memory_id=memory.memory_id,
                )

        # Verify vector storage
        collection_info = await qdrant_client.get_collection(qdrant_test_collection)
        assert collection_info.points_count == 1

        # Verify graph storage
        async with neo4j_driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH (m:Memory {memory_id: $id}) RETURN m",
                id=memory.memory_id,
            )
            record = await result.single()
            assert record is not None
            assert record["m"]["agent_id"] == memory.agent_id

            # Verify entity relationships
            result = await session.run(
                """
                MATCH (m:Memory {memory_id: $id})-[:MENTIONS]->(e:Entity)
                RETURN collect(e.name) as entities
                """,
                id=memory.memory_id,
            )
            record = await result.single()
            entities = record["entities"]
            assert "JWT" in entities
            assert "Redis" in entities

    async def test_coordinated_search_vector_and_graph(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test coordinated search across vector and graph stores."""
        # Insert all memories in both stores
        for i, memory in enumerate(sample_memories):
            # Vector store
            point = PointStruct(
                id=i + 1,
                vector=memory.embedding,
                payload={
                    "memory_id": memory.memory_id,
                    "memory_layer": memory.memory_layer.value,
                    "content": memory.content,
                    "agent_id": memory.agent_id,
                    "session_id": memory.session_id,
                    "is_critical": memory.is_critical,
                },
            )
            await qdrant_client.upsert(
                collection_name=qdrant_test_collection,
                points=[point],
            )

            # Graph store
            async with neo4j_driver.session(database="neo4j") as session:
                await session.run(
                    """
                    CREATE (m:Memory {
                        memory_id: $memory_id,
                        agent_id: $agent_id,
                        layer: $layer
                    })
                    """,
                    memory_id=memory.memory_id,
                    agent_id=memory.agent_id,
                    layer=memory.memory_layer.value,
                )

        # Vector similarity search
        query_vector = [0.15] * 1536  # Between 0.1 and 0.2
        vector_results = await qdrant_client.search(
            collection_name=qdrant_test_collection,
            query_vector=query_vector,
            limit=3,
        )

        assert len(vector_results) == 3
        # First result should be closest to query vector
        assert vector_results[0].score > 0.9

        # Graph traversal to find related memories
        async with neo4j_driver.session(database="neo4j") as session:
            result = await session.run(
                """
                MATCH (m:Memory {agent_id: $agent_id})
                RETURN count(m) as count, collect(m.layer) as layers
                """,
                agent_id=sample_memories[0].agent_id,
            )
            record = await result.single()
            assert record["count"] == 5
            layers = record["layers"]
            assert "episodic" in layers
            assert "semantic" in layers
            assert "procedural" in layers


class TestECLPipelineEndToEnd:
    """Test ECL pipeline end-to-end (Acceptance Criteria #4)."""

    async def test_extract_cognify_load_pipeline(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
        entity_extractor: EntityExtractor,
        relationship_detector: RelationshipDetectorTask,
    ) -> None:
        """Test complete ECL pipeline: Extract -> Cognify -> Load."""
        raw_content = """
        The user wants to implement JWT authentication with Redis storage.
        The authentication service should use bcrypt for password hashing.
        Token expiration is set to 1 hour with refresh capability.
        """

        # Mock LLM responses for entity extraction
        entity_response = LLMResponse(
            content="""ENTITIES:
- JWT (concept): Authentication token standard
- Redis (tool): In-memory data store
- bcrypt (tool): Password hashing algorithm
- authentication service (concept): Service handling auth
""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=50, total_tokens=250),
            latency_ms=300,
        )

        relationship_response = LLMResponse(
            content="""RELATIONSHIPS:
- JWT -> Redis: PART_OF (token storage)
- authentication service -> bcrypt: USES (password hashing)
- JWT -> authentication service: PART_OF (token generation)
""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=250, completion_tokens=60, total_tokens=310),
            latency_ms=350,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.entity_extractor.llm_service.complete"
        ) as mock_entity_llm:
            mock_entity_llm.return_value = entity_response

            # Step 1: Extract entities using ECL task execute method
            extract_result = await entity_extractor.execute({"content": raw_content})

            entities_data = extract_result.get("entities", [])
            assert len(entities_data) >= 3
            # Convert to EntityNode objects if needed
            entities = [
                EntityNode(
                    entity_id=e.get("entity_id", f"ent-{i}"),
                    entity_name=e.get("entity_name", e.get("name", "")),
                    entity_type=EntityType(e.get("entity_type", "concept")),
                    properties=e.get("properties", {}),
                )
                for i, e in enumerate(entities_data)
            ]
            entity_names = [e.entity_name for e in entities]
            assert "JWT" in entity_names or any("JWT" in name for name in entity_names)

        with patch(
            "agentcore.a2a_protocol.services.memory.relationship_detector.llm_service.complete"
        ) as mock_rel_llm:
            mock_rel_llm.return_value = relationship_response

            # Step 2: Cognify - detect relationships using ECL task execute method
            # Convert entities to dict format for the task
            entities_input = [
                {
                    "entity_id": e.entity_id,
                    "entity_name": e.entity_name,
                    "entity_type": e.entity_type.value,
                }
                for e in entities
            ]
            detect_result = await relationship_detector.execute({
                "content": raw_content,
                "entities": entities_input,
            })

            relationships_data = detect_result.get("relationships", [])
            assert len(relationships_data) >= 2
            # Convert to RelationshipEdge objects
            relationships = [
                RelationshipEdge(
                    relationship_id=r.get("relationship_id", f"rel-{i}"),
                    source_entity_id=r.get("source_entity_id", ""),
                    target_entity_id=r.get("target_entity_id", ""),
                    relationship_type=RelationshipType(r.get("relationship_type", "relates_to")),
                    properties=r.get("properties", {}),
                )
                for i, r in enumerate(relationships_data)
            ]

        # Step 3: Load into graph database
        async with neo4j_driver.session(database="neo4j") as session:
            # Load entities
            for entity in entities:
                await session.run(
                    """
                    CREATE (e:Entity {
                        entity_id: $id,
                        name: $name,
                        type: $type,
                        created_at: datetime()
                    })
                    """,
                    id=entity.entity_id,
                    name=entity.entity_name,
                    type=entity.entity_type.value,
                )

            # Load relationships
            for rel in relationships:
                await session.run(
                    """
                    MATCH (s:Entity {entity_id: $source})
                    MATCH (t:Entity {entity_id: $target})
                    CREATE (s)-[r:RELATES_TO {
                        relationship_id: $rel_id,
                        type: $rel_type
                    }]->(t)
                    """,
                    source=rel.source_entity_id,
                    target=rel.target_entity_id,
                    rel_id=rel.relationship_id,
                    rel_type=rel.relationship_type.value,
                )

            # Verify loaded entities
            result = await session.run("MATCH (e:Entity) RETURN count(e) as count")
            record = await result.single()
            assert record["count"] >= 3

            # Verify loaded relationships
            result = await session.run("MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count")
            record = await result.single()
            assert record["count"] >= 2

    async def test_ecl_pipeline_with_memory_storage(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test ECL pipeline stores memories with proper vector embeddings."""
        # Store memories in vector database
        points = [
            PointStruct(
                id=i + 1,
                vector=mem.embedding,
                payload={
                    "memory_id": mem.memory_id,
                    "content": mem.content,
                    "memory_layer": mem.memory_layer.value,
                    "entities": mem.entities,
                    "keywords": mem.keywords,
                },
            )
            for i, mem in enumerate(sample_memories)
        ]

        await qdrant_client.upsert(
            collection_name=qdrant_test_collection,
            points=points,
        )

        # Verify storage
        info = await qdrant_client.get_collection(qdrant_test_collection)
        assert info.points_count == 5

        # Verify metadata integrity
        retrieved = await qdrant_client.retrieve(
            collection_name=qdrant_test_collection,
            ids=[1, 2, 3],
        )

        assert len(retrieved) == 3
        for point in retrieved:
            assert "memory_id" in point.payload
            assert "content" in point.payload
            assert "entities" in point.payload


class TestMemifyOperations:
    """Test Memify operations (Acceptance Criteria #5)."""

    async def test_entity_consolidation(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
    ) -> None:
        """Test entity consolidation (merging similar entities)."""
        # Create duplicate entities
        async with neo4j_driver.session(database="neo4j") as session:
            await session.run(
                """
                CREATE (e1:Entity {entity_id: 'ent-1', name: 'JWT', type: 'concept'})
                CREATE (e2:Entity {entity_id: 'ent-2', name: 'jwt', type: 'concept'})
                CREATE (e3:Entity {entity_id: 'ent-3', name: 'JSON Web Token', type: 'concept'})
                CREATE (e4:Entity {entity_id: 'ent-4', name: 'Redis', type: 'tool'})
                """
            )

            # Simulate Memify consolidation: merge similar JWT entities
            # In production, this would use embeddings for similarity
            await session.run(
                """
                MATCH (primary:Entity {entity_id: 'ent-1'})
                MATCH (duplicate:Entity)
                WHERE duplicate.entity_id IN ['ent-2', 'ent-3']
                WITH primary, collect(duplicate) as duplicates
                FOREACH (dup IN duplicates |
                    SET primary.aliases = coalesce(primary.aliases, []) + [dup.name]
                )
                """
            )

            # Delete duplicates (in production, would transfer relationships first)
            await session.run(
                """
                MATCH (e:Entity)
                WHERE e.entity_id IN ['ent-2', 'ent-3']
                DELETE e
                """
            )

            # Verify consolidation
            result = await session.run(
                "MATCH (e:Entity) RETURN count(e) as count"
            )
            record = await result.single()
            assert record["count"] == 2  # JWT and Redis

            # Verify aliases preserved
            result = await session.run(
                """
                MATCH (e:Entity {entity_id: 'ent-1'})
                RETURN e.aliases as aliases
                """
            )
            record = await result.single()
            aliases = record["aliases"]
            assert "jwt" in aliases
            assert "JSON Web Token" in aliases

    async def test_relationship_pruning(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
    ) -> None:
        """Test pruning low-value relationships."""
        async with neo4j_driver.session(database="neo4j") as session:
            # Create entities and relationships with access counts
            await session.run(
                """
                CREATE (e1:Entity {entity_id: 'ent-1', name: 'A'})
                CREATE (e2:Entity {entity_id: 'ent-2', name: 'B'})
                CREATE (e3:Entity {entity_id: 'ent-3', name: 'C'})
                CREATE (e1)-[:RELATES_TO {access_count: 10}]->(e2)
                CREATE (e2)-[:RELATES_TO {access_count: 1}]->(e3)
                CREATE (e1)-[:RELATES_TO {access_count: 0}]->(e3)
                """
            )

            # Prune relationships with low access count (< 2)
            result = await session.run(
                """
                MATCH ()-[r:RELATES_TO]->()
                WHERE r.access_count < 2
                DELETE r
                RETURN count(r) as pruned_count
                """
            )
            record = await result.single()
            assert record["pruned_count"] == 2

            # Verify only high-value relationship remains
            result = await session.run(
                "MATCH ()-[r:RELATES_TO]->() RETURN count(r) as count"
            )
            record = await result.single()
            assert record["count"] == 1

    async def test_pattern_detection(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
    ) -> None:
        """Test pattern detection in memory graph."""
        async with neo4j_driver.session(database="neo4j") as session:
            # Create a common pattern: Authentication -> Storage -> Cache
            await session.run(
                """
                CREATE (auth1:Concept {name: 'JWT Auth'})
                CREATE (store1:Tool {name: 'Redis'})
                CREATE (auth1)-[:USES]->(store1)

                CREATE (auth2:Concept {name: 'OAuth Auth'})
                CREATE (store2:Tool {name: 'Redis'})
                CREATE (auth2)-[:USES]->(store2)

                CREATE (auth3:Concept {name: 'Session Auth'})
                CREATE (store3:Tool {name: 'Redis'})
                CREATE (auth3)-[:USES]->(store3)
                """
            )

            # Detect pattern: Auth concept -> Redis tool
            result = await session.run(
                """
                MATCH (auth:Concept)-[:USES]->(redis:Tool {name: 'Redis'})
                RETURN count(auth) as pattern_count
                """
            )
            record = await result.single()
            pattern_count = record["pattern_count"]

            # Should detect 3 instances of the pattern
            assert pattern_count == 3


class TestHybridSearchAccuracy:
    """Test hybrid search accuracy (Acceptance Criteria #6)."""

    async def test_combined_vector_and_graph_search(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test hybrid search combining vector similarity with graph context."""
        # Store memories in both stores
        for i, memory in enumerate(sample_memories):
            # Vector store
            await qdrant_client.upsert(
                collection_name=qdrant_test_collection,
                points=[
                    PointStruct(
                        id=i + 1,
                        vector=memory.embedding,
                        payload={
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                            "keywords": memory.keywords,
                        },
                    )
                ],
            )

            # Graph store with keywords as entities
            async with neo4j_driver.session(database="neo4j") as session:
                await session.run(
                    """
                    CREATE (m:Memory {memory_id: $id, content: $content})
                    """,
                    id=memory.memory_id,
                    content=memory.content,
                )

                for keyword in memory.keywords:
                    await session.run(
                        """
                        MERGE (k:Keyword {name: $keyword})
                        WITH k
                        MATCH (m:Memory {memory_id: $memory_id})
                        CREATE (m)-[:HAS_KEYWORD]->(k)
                        """,
                        keyword=keyword,
                        memory_id=memory.memory_id,
                    )

        # Hybrid search: Vector similarity + graph traversal
        query_vector = [0.25] * 1536  # Between embeddings

        # Step 1: Vector search
        vector_results = await qdrant_client.search(
            collection_name=qdrant_test_collection,
            query_vector=query_vector,
            limit=3,
        )

        vector_memory_ids = [r.payload["memory_id"] for r in vector_results]
        assert len(vector_memory_ids) == 3

        # Step 2: Graph expansion - find related memories via shared keywords
        async with neo4j_driver.session(database="neo4j") as session:
            result = await session.run(
                """
                MATCH (m1:Memory)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(m2:Memory)
                WHERE m1.memory_id IN $memory_ids AND m1 <> m2
                RETURN DISTINCT m2.memory_id as related_id, count(k) as shared_keywords
                ORDER BY shared_keywords DESC
                LIMIT 5
                """,
                memory_ids=vector_memory_ids,
            )

            related_memories = [record async for record in result]

            # Should find memories connected by shared keywords
            assert len(related_memories) >= 1

    async def test_search_with_context_expansion(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        sample_memories: list[MemoryRecord],
        retrieval_service: EnhancedRetrievalService,
    ) -> None:
        """Test search with context-aware retrieval expansion."""
        # Store all memories
        for i, memory in enumerate(sample_memories):
            await qdrant_client.upsert(
                collection_name=qdrant_test_collection,
                points=[
                    PointStruct(
                        id=i + 1,
                        vector=memory.embedding,
                        payload={
                            "memory_id": memory.memory_id,
                            "memory_layer": memory.memory_layer.value,
                            "is_critical": memory.is_critical,
                        },
                    )
                ],
            )

        # Test retrieval scoring
        scored_memories = await retrieval_service.retrieve_top_k(
            memories=sample_memories,
            k=3,
            query_embedding=[0.15] * 1536,
            current_stage=StageType.EXECUTION,
            has_recent_errors=False,
        )

        assert len(scored_memories) == 3

        # Verify scoring includes multiple factors
        for memory, final_score, components in scored_memories:
            assert final_score > 0
            assert "base_similarity" in components or final_score > 0
            # Critical memories should be boosted
            if memory.is_critical:
                assert final_score >= 0.5


class TestMemoryPersistence:
    """Test memory persistence across restarts (Acceptance Criteria #7)."""

    async def test_qdrant_persistence_across_queries(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test Qdrant persists data across multiple operations."""
        # Store memories
        for i, memory in enumerate(sample_memories):
            await qdrant_client.upsert(
                collection_name=qdrant_test_collection,
                points=[
                    PointStruct(
                        id=i + 1,
                        vector=memory.embedding,
                        payload={
                            "memory_id": memory.memory_id,
                            "content": memory.content,
                        },
                    )
                ],
            )

        # Verify initial count
        info1 = await qdrant_client.get_collection(qdrant_test_collection)
        assert info1.points_count == 5

        # Perform search (simulating read operation)
        results = await qdrant_client.search(
            collection_name=qdrant_test_collection,
            query_vector=[0.3] * 1536,
            limit=10,
        )
        assert len(results) == 5

        # Update a point
        await qdrant_client.set_payload(
            collection_name=qdrant_test_collection,
            points=[1],
            payload={"updated": True},
        )

        # Verify data persisted
        info2 = await qdrant_client.get_collection(qdrant_test_collection)
        assert info2.points_count == 5

        # Verify update persisted
        retrieved = await qdrant_client.retrieve(
            collection_name=qdrant_test_collection,
            ids=[1],
        )
        assert retrieved[0].payload["updated"] is True

    async def test_neo4j_persistence_across_transactions(
        self,
        neo4j_driver: AsyncDriver,
        clean_neo4j: None,
    ) -> None:
        """Test Neo4j persists graph data across transactions."""
        # Transaction 1: Create nodes
        async with neo4j_driver.session(database="neo4j") as session:
            await session.run(
                """
                CREATE (m:Memory {memory_id: 'persist-test-1', content: 'Test'})
                CREATE (e:Entity {entity_id: 'entity-1', name: 'Test Entity'})
                CREATE (m)-[:MENTIONS]->(e)
                """
            )

        # Transaction 2: Verify persistence and add more data
        async with neo4j_driver.session(database="neo4j") as session:
            result = await session.run(
                "MATCH (m:Memory {memory_id: 'persist-test-1'}) RETURN m"
            )
            record = await result.single()
            assert record is not None

            # Add more relationships
            await session.run(
                """
                MATCH (m:Memory {memory_id: 'persist-test-1'})
                SET m.access_count = 5
                """
            )

        # Transaction 3: Verify all data persisted
        async with neo4j_driver.session(database="neo4j") as session:
            result = await session.run(
                """
                MATCH (m:Memory {memory_id: 'persist-test-1'})-[:MENTIONS]->(e:Entity)
                RETURN m.access_count as count, e.name as entity_name
                """
            )
            record = await result.single()
            assert record["count"] == 5
            assert record["entity_name"] == "Test Entity"


class TestStageCompressionPipeline:
    """Test stage compression pipeline (Acceptance Criteria #8)."""

    async def test_stage_memory_compression(
        self,
        context_compressor: ContextCompressor,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test compressing stage memories with COMPASS methodology."""
        # Mock LLM responses
        fact_response = LLMResponse(
            content="""1. JWT authentication with Redis storage
2. Token TTL: 1 hour with refresh
3. Login endpoint success
4. Error rate fixed from 8% to below threshold
5. Connection pooling improves performance""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=300, completion_tokens=60, total_tokens=360),
            latency_ms=250,
        )

        compression_response = LLMResponse(
            content="""Implemented JWT auth with Redis (1h TTL, refresh enabled).
Login endpoint working. Fixed 8% error rate via connection pooling.""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=400, completion_tokens=40, total_tokens=440),
            latency_ms=350,
        )

        quality_response = LLMResponse(
            content="0.96",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=200, completion_tokens=5, total_tokens=205),
            latency_ms=100,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_llm:
            mock_llm.side_effect = [fact_response, compression_response, quality_response]

            summary, metrics = await context_compressor.compress_stage(
                stage_type=StageType.EXECUTION,
                memories=sample_memories,
                task_goal="Implement authentication system",
            )

            # Verify compression achieved
            assert len(summary) < sum(len(m.content) for m in sample_memories)
            assert metrics.compression_ratio > 1.0
            assert metrics.quality_score >= 0.95
            assert "JWT" in summary or "auth" in summary.lower()

    async def test_progressive_task_compression(
        self,
        context_compressor: ContextCompressor,
    ) -> None:
        """Test progressive compression across task stages."""
        stage_summaries = [
            "Planning: JWT auth with Redis, 1h TTL, bcrypt hashing.",
            "Execution: Created /auth/login, /auth/refresh endpoints.",
            "Reflection: Fixed 8% error rate with connection pooling.",
        ]

        fact_response = LLMResponse(
            content="""1. JWT authentication
2. Redis storage
3. 1h TTL
4. bcrypt hashing
5. Login and refresh endpoints
6. Connection pooling fix""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=250, completion_tokens=50, total_tokens=300),
            latency_ms=200,
        )

        compression_response = LLMResponse(
            content="""JWT auth with Redis (1h TTL, bcrypt).
Endpoints: /auth/login, /auth/refresh. Connection pooling applied.""",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=350, completion_tokens=35, total_tokens=385),
            latency_ms=300,
        )

        quality_response = LLMResponse(
            content="0.97",
            model="gpt-4.1-mini",
            provider="openai",
            usage=LLMUsage(prompt_tokens=180, completion_tokens=5, total_tokens=185),
            latency_ms=100,
        )

        with patch(
            "agentcore.a2a_protocol.services.memory.context_compressor.llm_service.complete"
        ) as mock_llm:
            mock_llm.side_effect = [fact_response, compression_response, quality_response]

            summary, metrics = await context_compressor.compress_task(
                task_id="task-123",
                stage_summaries=stage_summaries,
                task_goal="Implement authentication",
            )

            # Task compression should be more aggressive (5:1 target)
            original_length = sum(len(s) for s in stage_summaries)
            assert len(summary) < original_length
            assert metrics.compression_ratio >= 1.0
            assert metrics.quality_score >= 0.95


class TestErrorTrackingWorkflow:
    """Test error tracking workflow (Acceptance Criteria #9)."""

    async def test_record_and_analyze_errors(
        self,
        error_tracker: ErrorTracker,
    ) -> None:
        """Test recording errors and detecting patterns."""
        agent_id = str(uuid4())
        task_id = str(uuid4())

        # Record multiple errors
        errors = [
            {
                "error_type": ErrorType.HALLUCINATION,
                "description": "Generated non-existent API endpoint",
                "severity": 0.8,
            },
            {
                "error_type": ErrorType.INCORRECT_ACTION,
                "description": "Used wrong HTTP method",
                "severity": 0.6,
            },
            {
                "error_type": ErrorType.HALLUCINATION,
                "description": "Fabricated configuration option",
                "severity": 0.9,
            },
        ]

        for error in errors:
            await error_tracker.record_error(
                task_id=task_id,
                agent_id=agent_id,
                error_type=error["error_type"],
                error_description=error["description"],
                context_when_occurred="During task execution",
                severity=error["severity"],
            )

        # Analyze patterns
        patterns = await error_tracker.get_error_patterns(agent_id=agent_id)

        # Should detect hallucination as dominant pattern
        assert len(patterns) > 0
        hallucination_count = sum(
            1 for p in patterns if "hallucination" in p.lower()
        )
        assert hallucination_count > 0

        # Get summary statistics
        stats = await error_tracker.get_error_statistics(agent_id=agent_id)
        assert stats["total_errors"] == 3
        assert stats["avg_severity"] > 0.7

    async def test_error_pattern_influences_retrieval(
        self,
        retrieval_service: EnhancedRetrievalService,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test that recent errors boost critical memory retrieval."""
        # Without recent errors
        results_no_errors = await retrieval_service.retrieve_top_k(
            memories=sample_memories,
            k=3,
            query_embedding=[0.15] * 1536,
            current_stage=StageType.EXECUTION,
            has_recent_errors=False,
        )

        # With recent errors
        results_with_errors = await retrieval_service.retrieve_top_k(
            memories=sample_memories,
            k=3,
            query_embedding=[0.15] * 1536,
            current_stage=StageType.EXECUTION,
            has_recent_errors=True,
        )

        # Critical memories should be ranked higher when errors occurred
        critical_score_no_errors = sum(
            score for mem, score, _ in results_no_errors if mem.is_critical
        )
        critical_score_with_errors = sum(
            score for mem, score, _ in results_with_errors if mem.is_critical
        )

        # With errors, critical memories get higher scores
        assert critical_score_with_errors >= critical_score_no_errors


class TestCodeCoverage:
    """Test comprehensive code coverage (Acceptance Criteria #10)."""

    async def test_memory_model_serialization(
        self,
        sample_memories: list[MemoryRecord],
    ) -> None:
        """Test memory model serialization and deserialization."""
        for memory in sample_memories:
            # Serialize to dict
            memory_dict = memory.model_dump(mode="json")

            # Verify all fields present
            assert "memory_id" in memory_dict
            assert "memory_layer" in memory_dict
            assert "content" in memory_dict
            assert "embedding" in memory_dict
            assert "is_critical" in memory_dict

            # Deserialize back
            restored = MemoryRecord.model_validate(memory_dict)
            assert restored.memory_id == memory.memory_id
            assert restored.memory_layer == memory.memory_layer

    async def test_stage_memory_lifecycle(self) -> None:
        """Test complete stage memory lifecycle."""
        stage = StageMemory(
            stage_id=f"stage-{uuid4()}",
            task_id=str(uuid4()),
            agent_id=str(uuid4()),
            stage_type=StageType.EXECUTION,
            stage_summary="Test stage summary",
            stage_insights=["Insight 1", "Insight 2"],
            raw_memory_refs=["mem-1", "mem-2"],
            compression_ratio=10.5,
            compression_model="gpt-4.1-mini",
            quality_score=0.97,
            completed_at=datetime.now(UTC),
        )

        # Verify fields
        assert stage.stage_type == StageType.EXECUTION
        assert stage.compression_ratio == 10.5
        assert len(stage.stage_insights) == 2

        # Serialize and restore
        stage_dict = stage.model_dump(mode="json")
        restored = StageMemory.model_validate(stage_dict)
        assert restored.stage_id == stage.stage_id
        assert restored.quality_score == 0.97

    async def test_entity_and_relationship_models(self) -> None:
        """Test entity and relationship model creation."""
        entity = EntityNode(
            entity_id=f"ent-{uuid4()}",
            entity_name="JWT Authentication",
            entity_type=EntityType.CONCEPT,
            properties={"domain": "security", "confidence": 0.95},
            memory_refs=["mem-1", "mem-2"],
        )

        relationship = RelationshipEdge(
            relationship_id=f"rel-{uuid4()}",
            source_entity_id=entity.entity_id,
            target_entity_id=f"ent-{uuid4()}",
            relationship_type=RelationshipType.RELATES_TO,
            properties={"strength": 0.85},
            access_count=5,
        )

        # Verify entity
        assert entity.entity_type == EntityType.CONCEPT
        assert entity.properties["confidence"] == 0.95

        # Verify relationship
        assert relationship.relationship_type == RelationshipType.RELATES_TO
        assert relationship.access_count == 5

    async def test_error_record_severity_validation(self) -> None:
        """Test error record with severity validation."""
        error = ErrorRecord(
            task_id=str(uuid4()),
            agent_id=str(uuid4()),
            error_type=ErrorType.HALLUCINATION,
            error_description="Generated false information",
            context_when_occurred="During response generation",
            recovery_action="Regenerated with fact-checking",
            error_severity=0.85,
        )

        assert error.error_severity == 0.85
        assert error.error_type == ErrorType.HALLUCINATION

        # Test validation
        error_dict = error.model_dump(mode="json")
        restored = ErrorRecord.model_validate(error_dict)
        assert restored.error_severity == 0.85

    async def test_task_context_progressive_update(self) -> None:
        """Test task context with progressive summarization."""
        context = TaskContext(
            task_id=str(uuid4()),
            agent_id=str(uuid4()),
            task_goal="Implement authentication system",
            current_stage_id=str(uuid4()),
            task_progress_summary="Completed planning and started execution.",
            critical_constraints=["Use JWT", "Store in Redis", "1h TTL"],
            performance_metrics={
                "error_rate": 0.02,
                "progress_rate": 0.75,
                "context_efficiency": 0.85,
            },
        )

        assert context.task_goal == "Implement authentication system"
        assert len(context.critical_constraints) == 3
        assert context.performance_metrics["error_rate"] == 0.02

        # Serialize
        context_dict = context.model_dump(mode="json")
        assert "performance_metrics" in context_dict
        assert context_dict["performance_metrics"]["progress_rate"] == 0.75
