"""
Integration Tests for GraphMemoryService with Real Neo4j

Tests graph memory operations against real Neo4j instance using testcontainers.
Validates end-to-end workflows, performance benchmarks, and data integrity.

Component ID: MEM-017
Ticket: MEM-017 (Implement GraphMemoryService - Neo4j Integration)

Performance targets:
- <200ms graph traversal (p95, 2-hop)
- 90%+ entity consolidation accuracy
- 95%+ relationship relevance after pruning
"""

import json
import time
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
from testcontainers.neo4j import Neo4jContainer

from agentcore.a2a_protocol.models.memory import (
    EntityNode,
    EntityType,
    MemoryLayer,
    MemoryRecord,
    RelationshipType,
)
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService


# Use module-scoped event loop for all tests (matches fixture scope)
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def neo4j_container():
    """Start Neo4j test container."""
    container = Neo4jContainer(image="neo4j:5.15-community", password="testpassword")
    container.with_env("NEO4J_PLUGINS", '["apoc"]')  # Enable APOC plugin

    container.start()

    yield container

    container.stop()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_driver(neo4j_container) -> AsyncGenerator[AsyncDriver, None]:
    """Create Neo4j async driver for tests."""
    uri = neo4j_container.get_connection_url()
    driver = AsyncGraphDatabase.driver(
        uri,
        auth=("neo4j", "testpassword"),
    )

    await driver.verify_connectivity()
    yield driver

    await driver.close()


@pytest_asyncio.fixture(scope="function", loop_scope="module")
async def graph_service(neo4j_driver) -> AsyncGenerator[GraphMemoryService, None]:
    """Create GraphMemoryService with real Neo4j."""
    service = GraphMemoryService(neo4j_driver)
    await service.initialize()

    yield service

    # Cleanup: delete all nodes and relationships
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    # Note: Don't close the service here as it would close the shared module-scoped driver


@pytest.fixture
def sample_memory():
    """Sample memory record."""
    return MemoryRecord(
        memory_id=f"mem-{uuid4()}",
        memory_layer=MemoryLayer.SEMANTIC,
        content="Implement JWT authentication using Redis for token storage",
        summary="JWT auth with Redis storage",
        embedding=[0.1] * 768,
        agent_id="agent-test",
        session_id="session-test",
        task_id="task-test",
        timestamp=datetime.now(UTC),
        is_critical=True,
        relevance_score=0.9,
    )


@pytest.fixture
def sample_entity():
    """Sample entity node."""
    return EntityNode(
        entity_id=f"ent-{uuid4()}",
        entity_name="JWT Token",
        entity_type=EntityType.CONCEPT,
        properties={"domain": "authentication", "confidence": 0.95},
        memory_refs=[],
    )


class TestGraphMemoryServiceInitialization:
    """Test service initialization with real Neo4j."""


    async def test_initialize_creates_indexes(self, graph_service, neo4j_driver):
        """Test that indexes are created successfully."""
        # Verify indexes exist
        async with neo4j_driver.session() as session:
            result = await session.run("SHOW INDEXES")
            indexes = [record async for record in result]

            # Neo4j 5.x returns indexes with different keys
            # Check for at least some indexes (system + user-created)
            assert len(indexes) >= 5, f"Expected at least 5 indexes, got {len(indexes)}"

            # Verify we have custom indexes by checking labelsOrTypes
            custom_indexes = [
                idx for idx in indexes
                if idx.get("labelsOrTypes") and any(
                    label in str(idx.get("labelsOrTypes", []))
                    for label in ["Memory", "Entity", "Concept"]
                )
            ]
            assert len(custom_indexes) >= 1, "Expected at least one custom index on Memory/Entity/Concept"


    async def test_initialize_creates_constraints(self, graph_service, neo4j_driver):
        """Test that uniqueness constraints are created."""
        async with neo4j_driver.session() as session:
            result = await session.run("SHOW CONSTRAINTS")
            constraints = [record async for record in result]

            # Should have at least one constraint
            assert len(constraints) >= 1, f"Expected at least 1 constraint, got {len(constraints)}"

            # Check that we have a uniqueness constraint on entity_id
            has_entity_id_constraint = any(
                "entity_id" in str(c.get("properties", [])).lower() or
                "entity_id" in str(c.get("name", "")).lower()
                for c in constraints
            )
            assert has_entity_id_constraint, "Expected constraint on entity_id"


class TestMemoryNodeOperations:
    """Test memory node storage and retrieval."""


    async def test_store_and_retrieve_memory_node(
        self, graph_service, sample_memory, neo4j_driver
    ):
        """Test end-to-end memory node storage and retrieval."""
        # Store memory
        memory_id = await graph_service.store_memory_node(sample_memory)
        assert memory_id == sample_memory.memory_id

        # Verify stored in Neo4j
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (m:Memory {memory_id: $memory_id}) RETURN m",
                {"memory_id": memory_id},
            )
            record = await result.single()

            assert record is not None
            mem_data = dict(record["m"])
            assert mem_data["content"] == sample_memory.content
            assert mem_data["is_critical"] == sample_memory.is_critical


    async def test_store_multiple_memory_nodes(self, graph_service):
        """Test storing multiple memory nodes."""
        memories = [
            MemoryRecord(
                memory_id=f"mem-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Memory {i}",
                summary=f"Summary {i}",
                embedding=[0.1] * 768,
                agent_id="agent-test",
                task_id="task-test",
                timestamp=datetime.now(UTC),
            )
            for i in range(10)
        ]

        # Store all memories
        for memory in memories:
            memory_id = await graph_service.store_memory_node(memory)
            assert memory_id == memory.memory_id


class TestEntityNodeOperations:
    """Test entity node storage and retrieval."""


    async def test_store_and_retrieve_entity_node(
        self, graph_service, sample_entity, neo4j_driver
    ):
        """Test end-to-end entity node storage."""
        # Store entity
        entity_id = await graph_service.store_entity_node(sample_entity)
        assert entity_id == sample_entity.entity_id

        # Verify stored in Neo4j
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity {entity_id: $entity_id}) RETURN e",
                {"entity_id": entity_id},
            )
            record = await result.single()

            assert record is not None
            ent_data = dict(record["e"])
            assert ent_data["entity_name"] == sample_entity.entity_name
            assert ent_data["entity_type"] == sample_entity.entity_type.value


    async def test_store_concept_node(self, graph_service, neo4j_driver):
        """Test storing concept nodes."""
        concept_id = await graph_service.store_concept_node(
            name="authentication",
            properties={"domain": "security"},
            memory_refs=["mem-001"],
        )

        assert concept_id.startswith("concept-")

        # Verify stored
        async with neo4j_driver.session() as session:
            result = await session.run(
                "MATCH (c:Concept {concept_id: $concept_id}) RETURN c",
                {"concept_id": concept_id},
            )
            record = await result.single()

            assert record is not None
            concept_data = dict(record["c"])
            assert concept_data["name"] == "authentication"


class TestRelationshipOperations:
    """Test relationship creation and traversal."""


    async def test_create_mention_relationship(
        self, graph_service, sample_memory, sample_entity, neo4j_driver
    ):
        """Test creating MENTIONS relationship."""
        # Store nodes first
        memory_id = await graph_service.store_memory_node(sample_memory)
        entity_id = await graph_service.store_entity_node(sample_entity)

        # Create relationship
        rel_id = await graph_service.create_mention_relationship(
            memory_id=memory_id,
            entity_id=entity_id,
            properties={"confidence": 0.95},
        )

        assert rel_id.startswith("rel-")

        # Verify relationship exists
        async with neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (m:Memory {memory_id: $memory_id})-[r:MENTIONS]->(e:Entity {entity_id: $entity_id})
                RETURN r
                """,
                {"memory_id": memory_id, "entity_id": entity_id},
            )
            record = await result.single()

            assert record is not None
            rel_data = dict(record["r"])
            # Properties are serialized as JSON string to avoid Neo4j Map type issues
            properties = json.loads(rel_data["properties_json"])
            assert properties["confidence"] == 0.95


    async def test_create_temporal_relationships(self, graph_service):
        """Test creating FOLLOWS/PRECEDES temporal relationships."""
        # Create two memories
        mem1 = MemoryRecord(
            memory_id="mem-001",
            memory_layer=MemoryLayer.EPISODIC,
            content="First step",
            summary="Step 1",
            embedding=[0.1] * 768,
            agent_id="agent-test",
            task_id="task-test",
            timestamp=datetime.now(UTC),
        )

        mem2 = MemoryRecord(
            memory_id="mem-002",
            memory_layer=MemoryLayer.EPISODIC,
            content="Second step",
            summary="Step 2",
            embedding=[0.1] * 768,
            agent_id="agent-test",
            task_id="task-test",
            timestamp=datetime.now(UTC),
        )

        await graph_service.store_memory_node(mem1)
        await graph_service.store_memory_node(mem2)

        # Create FOLLOWS relationship (mem1 -> mem2)
        rel_id = await graph_service.create_relationship(
            from_id="mem-001",
            to_id="mem-002",
            rel_type=RelationshipType.FOLLOWS,
            from_label="Memory",
            to_label="Memory",
        )

        assert rel_id.startswith("rel-")


    async def test_create_hierarchical_relationships(self, graph_service):
        """Test creating PART_OF hierarchical relationships."""
        # Create parent and child entities
        parent = EntityNode(
            entity_id="ent-parent",
            entity_name="Authentication System",
            entity_type=EntityType.CONCEPT,
        )

        child = EntityNode(
            entity_id="ent-child",
            entity_name="JWT Handler",
            entity_type=EntityType.TOOL,
        )

        await graph_service.store_entity_node(parent)
        await graph_service.store_entity_node(child)

        # Create PART_OF relationship
        rel_id = await graph_service.create_relationship(
            from_id="ent-child",
            to_id="ent-parent",
            rel_type=RelationshipType.PART_OF,
        )

        assert rel_id.startswith("rel-")


class TestGraphTraversal:
    """Test graph traversal and path finding."""


    async def test_traverse_graph_single_hop(self, graph_service):
        """Test 1-hop graph traversal."""
        # Create entity chain: ent1 -> ent2
        ent1 = EntityNode(
            entity_id="ent-001",
            entity_name="Entity 1",
            entity_type=EntityType.CONCEPT,
        )
        ent2 = EntityNode(
            entity_id="ent-002",
            entity_name="Entity 2",
            entity_type=EntityType.CONCEPT,
        )

        await graph_service.store_entity_node(ent1)
        await graph_service.store_entity_node(ent2)
        await graph_service.create_relationship(
            "ent-001", "ent-002", RelationshipType.RELATES_TO
        )

        # Traverse from ent1
        paths = await graph_service.traverse_graph(
            start_id="ent-001",
            max_depth=1,
        )

        assert len(paths) >= 1
        assert any(
            any(node.get("entity_id") == "ent-002" for node in path["nodes"])
            for path in paths
        )


    async def test_traverse_graph_multi_hop(self, graph_service):
        """Test 2-hop graph traversal."""
        # Create entity chain: ent1 -> ent2 -> ent3
        entities = [
            EntityNode(
                entity_id=f"ent-{i}",
                entity_name=f"Entity {i}",
                entity_type=EntityType.CONCEPT,
            )
            for i in range(1, 4)
        ]

        for entity in entities:
            await graph_service.store_entity_node(entity)

        await graph_service.create_relationship(
            "ent-1", "ent-2", RelationshipType.RELATES_TO
        )
        await graph_service.create_relationship(
            "ent-2", "ent-3", RelationshipType.RELATES_TO
        )

        # Traverse from ent1 with depth 2
        paths = await graph_service.traverse_graph(
            start_id="ent-1",
            max_depth=2,
        )

        # Should find path to ent-3
        assert any(
            any(node.get("entity_id") == "ent-3" for node in path["nodes"])
            for path in paths
        )


    async def test_find_related_entities(self, graph_service):
        """Test finding directly related entities."""
        # Create central entity with multiple connections
        central = EntityNode(
            entity_id="ent-central",
            entity_name="Central Entity",
            entity_type=EntityType.CONCEPT,
        )

        related = [
            EntityNode(
                entity_id=f"ent-rel-{i}",
                entity_name=f"Related {i}",
                entity_type=EntityType.CONCEPT,
            )
            for i in range(5)
        ]

        await graph_service.store_entity_node(central)
        for entity in related:
            await graph_service.store_entity_node(entity)
            await graph_service.create_relationship(
                "ent-central", entity.entity_id, RelationshipType.RELATES_TO
            )

        # Find related entities
        entities = await graph_service.find_related_entities(
            entity_id="ent-central",
            rel_type=RelationshipType.RELATES_TO,
        )

        assert len(entities) == 5
        assert all(isinstance(e, EntityNode) for e in entities)


class TestTemporalOperations:
    """Test temporal relationship queries."""


    async def test_get_temporal_sequence(self, graph_service):
        """Test retrieving temporal sequence of memories."""
        task_id = "task-temporal"

        # Create sequence of memories with FOLLOWS relationships
        memories = []
        for i in range(5):
            mem = MemoryRecord(
                memory_id=f"mem-seq-{i}",
                memory_layer=MemoryLayer.EPISODIC,
                content=f"Step {i}",
                summary=f"Step {i}",
                embedding=[0.1] * 768,
                agent_id="agent-test",
                task_id=task_id,
                timestamp=datetime.now(UTC),
            )
            await graph_service.store_memory_node(mem)
            memories.append(mem)

        # Create temporal relationships
        for i in range(len(memories) - 1):
            await graph_service.create_relationship(
                from_id=f"mem-seq-{i}",
                to_id=f"mem-seq-{i+1}",
                rel_type=RelationshipType.FOLLOWS,
                from_label="Memory",
                to_label="Memory",
            )

        # Retrieve sequence
        sequence = await graph_service.get_temporal_sequence(task_id=task_id)

        assert len(sequence) == 5


    async def test_find_memories_by_entity(self, graph_service):
        """Test finding memories that mention an entity."""
        entity = EntityNode(
            entity_id="ent-search",
            entity_name="Search Entity",
            entity_type=EntityType.CONCEPT,
        )
        await graph_service.store_entity_node(entity)

        # Create memories that mention this entity
        for i in range(3):
            mem = MemoryRecord(
                memory_id=f"mem-mention-{i}",
                memory_layer=MemoryLayer.SEMANTIC,
                content=f"Memory {i} mentions entity",
                summary=f"Mention {i}",
                embedding=[0.1] * 768,
                agent_id="agent-test",
                timestamp=datetime.now(UTC),
            )
            await graph_service.store_memory_node(mem)
            await graph_service.create_mention_relationship(
                memory_id=mem.memory_id,
                entity_id=entity.entity_id,
            )

        # Find memories
        memories = await graph_service.find_memories_by_entity(
            entity_id="ent-search"
        )

        assert len(memories) == 3


class TestPerformance:
    """Test performance requirements."""


    async def test_two_hop_traversal_performance(self, graph_service):
        """Test p95 latency <200ms for 2-hop graph traversal."""
        # Create graph with multiple paths
        for i in range(20):
            ent = EntityNode(
                entity_id=f"ent-perf-{i}",
                entity_name=f"Perf Entity {i}",
                entity_type=EntityType.CONCEPT,
            )
            await graph_service.store_entity_node(ent)

        # Create relationships
        for i in range(19):
            await graph_service.create_relationship(
                f"ent-perf-{i}", f"ent-perf-{i+1}", RelationshipType.RELATES_TO
            )

        # Measure traversal latency (multiple runs for p95)
        latencies = []
        for _ in range(20):
            start_time = time.time()
            await graph_service.traverse_graph(
                start_id="ent-perf-0",
                max_depth=2,
            )
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate p95
        latencies.sort()
        p95_latency = latencies[int(len(latencies) * 0.95)]

        assert (
            p95_latency < 200
        ), f"p95 latency {p95_latency:.2f}ms exceeds 200ms target"


    async def test_concurrent_operations(self, graph_service):
        """Test concurrent memory and entity operations."""
        import asyncio

        # Create operations concurrently
        tasks = []

        for i in range(10):
            mem = MemoryRecord(
                memory_id=f"mem-concurrent-{i}",
                memory_layer=MemoryLayer.SEMANTIC,
                content=f"Concurrent memory {i}",
                summary=f"Summary {i}",
                embedding=[0.1] * 768,
                agent_id="agent-test",
                timestamp=datetime.now(UTC),
            )
            tasks.append(graph_service.store_memory_node(mem))

        for i in range(10):
            ent = EntityNode(
                entity_id=f"ent-concurrent-{i}",
                entity_name=f"Concurrent Entity {i}",
                entity_type=EntityType.CONCEPT,
            )
            tasks.append(graph_service.store_entity_node(ent))

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(results)


class TestUtilityOperations:
    """Test utility methods."""


    async def test_update_relationship_access(self, graph_service):
        """Test incrementing relationship access count."""
        # Create entities and relationship
        ent1 = EntityNode(
            entity_id="ent-access-1",
            entity_name="Entity 1",
            entity_type=EntityType.CONCEPT,
        )
        ent2 = EntityNode(
            entity_id="ent-access-2",
            entity_name="Entity 2",
            entity_type=EntityType.CONCEPT,
        )

        await graph_service.store_entity_node(ent1)
        await graph_service.store_entity_node(ent2)

        rel_id = await graph_service.create_relationship(
            "ent-access-1", "ent-access-2", RelationshipType.RELATES_TO
        )

        # Update access count
        success = await graph_service.update_relationship_access(rel_id)
        assert success is True


    async def test_get_node_degree(self, graph_service):
        """Test getting node connection degree."""
        # Create central node with multiple connections
        central = EntityNode(
            entity_id="ent-degree",
            entity_name="Central",
            entity_type=EntityType.CONCEPT,
        )
        await graph_service.store_entity_node(central)

        # Create connected nodes
        for i in range(5):
            ent = EntityNode(
                entity_id=f"ent-conn-{i}",
                entity_name=f"Connected {i}",
                entity_type=EntityType.CONCEPT,
            )
            await graph_service.store_entity_node(ent)
            await graph_service.create_relationship(
                "ent-degree", f"ent-conn-{i}", RelationshipType.RELATES_TO
            )

        # Get degree
        degree = await graph_service.get_node_degree("ent-degree")
        assert degree == 5


    async def test_find_shortest_path(self, graph_service):
        """Test finding shortest path between entities."""
        # Create path: ent1 -> ent2 -> ent3
        entities = [
            EntityNode(
                entity_id=f"ent-path-{i}",
                entity_name=f"Path Entity {i}",
                entity_type=EntityType.CONCEPT,
            )
            for i in range(3)
        ]

        for entity in entities:
            await graph_service.store_entity_node(entity)

        await graph_service.create_relationship(
            "ent-path-0", "ent-path-1", RelationshipType.RELATES_TO
        )
        await graph_service.create_relationship(
            "ent-path-1", "ent-path-2", RelationshipType.RELATES_TO
        )

        # Find shortest path
        path = await graph_service.find_shortest_path("ent-path-0", "ent-path-2")

        assert path is not None
        assert path["length"] == 2
        assert len(path["nodes"]) == 3
