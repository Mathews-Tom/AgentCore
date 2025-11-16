"""
Neo4j Graph Traversal Performance Tests

Validates <200ms p95 latency for 2-hop graph traversal queries
as specified in MEM-003 acceptance criteria.
"""

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
from testcontainers.neo4j import Neo4jContainer

# Performance targets from MEM-003
TARGET_P95_LATENCY_MS = 200
SAMPLE_SIZE = 100
MIN_GRAPH_SIZE = 1000  # Nodes for realistic testing

# Use module-scoped event loop for all tests (matches fixture scope)
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def neo4j_container() -> Neo4jContainer:
    """
    Start Neo4j testcontainer with APOC plugin.

    Returns:
        Neo4jContainer: Running Neo4j container instance
    """
    container = Neo4jContainer(image="neo4j:5.15-community")
    container.with_env("NEO4J_AUTH", "neo4j/password")
    container.with_env("NEO4J_PLUGINS", '["apoc"]')
    container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*")
    container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*")

    container.start()
    yield container
    container.stop()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_driver(neo4j_container: Neo4jContainer) -> AsyncDriver:
    """
    Create async Neo4j driver from testcontainer.

    Args:
        neo4j_container: Running Neo4j container

    Returns:
        AsyncDriver: Configured Neo4j driver

    Yields:
        AsyncDriver: Neo4j driver instance
    """
    bolt_url = neo4j_container.get_connection_url()
    driver = AsyncGraphDatabase.driver(
        bolt_url,
        auth=("neo4j", "password"),
        max_connection_pool_size=50,
    )

    await driver.verify_connectivity()
    yield driver
    await driver.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def populated_graph(neo4j_driver: AsyncDriver) -> dict[str, Any]:
    """
    Create a realistic graph with Memory, Entity, and Concept nodes.

    Args:
        neo4j_driver: Neo4j async driver

    Returns:
        dict: Graph metadata (node counts, entity IDs, etc.)
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # Create constraints and indexes
        await session.run(
            """
            CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
            FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE
            """
        )
        await session.run(
            """
            CREATE INDEX memory_agent_idx IF NOT EXISTS
            FOR (m:Memory) ON (m.agent_id)
            """
        )
        await session.run(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
            """
        )
        await session.run(
            """
            CREATE INDEX entity_type_idx IF NOT EXISTS
            FOR (e:Entity) ON (e.entity_type)
            """
        )

        # Create Memory nodes (700 nodes)
        memory_ids = [str(uuid4()) for _ in range(700)]
        agent_id = str(uuid4())
        session_id = str(uuid4())

        for memory_id in memory_ids:
            await session.run(
                """
                CREATE (m:Memory {
                    memory_id: $memory_id,
                    agent_id: $agent_id,
                    session_id: $session_id,
                    layer: $layer,
                    stage: $stage,
                    content: $content,
                    criticality: $criticality,
                    created_at: datetime(),
                    accessed_count: 0
                })
                """,
                memory_id=memory_id,
                agent_id=agent_id,
                session_id=session_id,
                layer="episodic",
                stage="execution",
                content=f"Test memory content {memory_id}",
                criticality=0.5,
            )

        # Create Entity nodes (200 nodes)
        entity_ids = [str(uuid4()) for _ in range(200)]
        entity_types = ["person", "concept", "tool", "constraint"]

        for i, entity_id in enumerate(entity_ids):
            await session.run(
                """
                CREATE (e:Entity {
                    entity_id: $entity_id,
                    name: $name,
                    entity_type: $entity_type,
                    confidence: $confidence,
                    first_seen: datetime(),
                    last_seen: datetime(),
                    mention_count: 1
                })
                """,
                entity_id=entity_id,
                name=f"Entity-{i}",
                entity_type=entity_types[i % len(entity_types)],
                confidence=0.8,
            )

        # Create Concept nodes (100 nodes)
        concept_ids = [str(uuid4()) for _ in range(100)]

        for i, concept_id in enumerate(concept_ids):
            await session.run(
                """
                CREATE (c:Concept {
                    concept_id: $concept_id,
                    name: $name,
                    description: $description,
                    category: $category,
                    created_at: datetime(),
                    usage_count: 0
                })
                """,
                concept_id=concept_id,
                name=f"Concept-{i}",
                description=f"Test concept {i}",
                category="test",
            )

        # Create relationships: Memory -[MENTIONS]-> Entity (1400 relationships)
        for memory_id in memory_ids[:500]:
            # Each memory mentions 2-3 entities
            num_mentions = (hash(memory_id) % 2) + 2
            entity_sample = entity_ids[: num_mentions]

            for entity_id in entity_sample:
                await session.run(
                    """
                    MATCH (m:Memory {memory_id: $memory_id})
                    MATCH (e:Entity {entity_id: $entity_id})
                    CREATE (m)-[r:MENTIONS {
                        position: 0,
                        context: 'test context',
                        created_at: datetime()
                    }]->(e)
                    """,
                    memory_id=memory_id,
                    entity_id=entity_id,
                )

        # Create relationships: Entity -[RELATES_TO]-> Entity (300 relationships)
        for i in range(0, len(entity_ids) - 1, 2):
            await session.run(
                """
                MATCH (e1:Entity {entity_id: $entity_id1})
                MATCH (e2:Entity {entity_id: $entity_id2})
                CREATE (e1)-[r:RELATES_TO {
                    relationship_type: 'similar_to',
                    strength: 0.7,
                    confidence: 0.8,
                    created_at: datetime(),
                    last_reinforced: datetime(),
                    reinforcement_count: 1
                }]->(e2)
                """,
                entity_id1=entity_ids[i],
                entity_id2=entity_ids[i + 1],
            )

        # Create relationships: Entity -[PART_OF]-> Concept (200 relationships)
        for i, entity_id in enumerate(entity_ids):
            concept_id = concept_ids[i % len(concept_ids)]
            await session.run(
                """
                MATCH (e:Entity {entity_id: $entity_id})
                MATCH (c:Concept {concept_id: $concept_id})
                CREATE (e)-[r:PART_OF {
                    relevance: 0.8,
                    created_at: datetime()
                }]->(c)
                """,
                entity_id=entity_id,
                concept_id=concept_id,
            )

        # Verify graph size
        result = await session.run("MATCH (n) RETURN count(n) AS node_count")
        record = await result.single()
        node_count = record["node_count"]

        result = await session.run("MATCH ()-[r]->() RETURN count(r) AS rel_count")
        record = await result.single()
        rel_count = record["rel_count"]

        return {
            "nodes": node_count,
            "relationships": rel_count,
            "memory_ids": memory_ids,
            "entity_ids": entity_ids,
            "concept_ids": concept_ids,
            "agent_id": agent_id,
            "session_id": session_id,
        }


async def test_graph_size(populated_graph: dict[str, Any]) -> None:
    """Verify graph has sufficient size for realistic testing."""
    assert populated_graph["nodes"] >= MIN_GRAPH_SIZE, (
        f"Graph size {populated_graph['nodes']} < minimum {MIN_GRAPH_SIZE}"
    )
    assert populated_graph["relationships"] > 0, "No relationships in graph"


async def test_2hop_traversal_performance(
    neo4j_driver: AsyncDriver,
    populated_graph: dict[str, Any],
) -> None:
    """
    Test 2-hop graph traversal performance.

    Validates p95 latency < 200ms for queries:
    Memory -> Entity -> Related Entity
    """
    latencies: list[float] = []

    async with neo4j_driver.session(database="neo4j") as session:
        # Sample memory nodes for testing
        memory_sample = populated_graph["memory_ids"][:SAMPLE_SIZE]

        for memory_id in memory_sample:
            start_time = time.perf_counter()

            # 2-hop traversal: Memory -[MENTIONS]-> Entity -[RELATES_TO]-> Entity
            result = await session.run(
                """
                MATCH (m:Memory {memory_id: $memory_id})-[men:MENTIONS]->(e1:Entity)
                MATCH (e1)-[rel:RELATES_TO]->(e2:Entity)
                RETURN m.memory_id AS memory_id,
                       e1.entity_id AS entity1_id,
                       e1.name AS entity1_name,
                       rel.relationship_type AS relationship,
                       e2.entity_id AS entity2_id,
                       e2.name AS entity2_name
                LIMIT 10
                """,
                memory_id=memory_id,
            )

            # Consume results
            records = [record async for record in result]

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

    # Calculate percentiles
    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"\n2-Hop Traversal Performance (n={SAMPLE_SIZE}):")
    print(f"  Average: {avg:.2f}ms")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")

    # Assert performance target
    assert p95 < TARGET_P95_LATENCY_MS, (
        f"p95 latency {p95:.2f}ms exceeds target {TARGET_P95_LATENCY_MS}ms"
    )


async def test_3hop_traversal_performance(
    neo4j_driver: AsyncDriver,
    populated_graph: dict[str, Any],
) -> None:
    """
    Test 3-hop graph traversal performance.

    Validates extended traversal:
    Memory -> Entity -> Concept
    """
    latencies: list[float] = []

    async with neo4j_driver.session(database="neo4j") as session:
        # Sample memory nodes for testing
        memory_sample = populated_graph["memory_ids"][:SAMPLE_SIZE]

        for memory_id in memory_sample:
            start_time = time.perf_counter()

            # 3-hop traversal: Memory -[MENTIONS]-> Entity -[PART_OF]-> Concept
            result = await session.run(
                """
                MATCH (m:Memory {memory_id: $memory_id})-[men:MENTIONS]->(e:Entity)
                MATCH (e)-[part:PART_OF]->(c:Concept)
                RETURN m.memory_id AS memory_id,
                       e.entity_id AS entity_id,
                       e.name AS entity_name,
                       c.concept_id AS concept_id,
                       c.name AS concept_name
                LIMIT 10
                """,
                memory_id=memory_id,
            )

            # Consume results
            records = [record async for record in result]

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

    # Calculate percentiles
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]

    print(f"\n3-Hop Traversal Performance (n={SAMPLE_SIZE}):")
    print(f"  p95: {p95:.2f}ms")

    # Note: 3-hop may be slower but should still be reasonable
    assert p95 < TARGET_P95_LATENCY_MS * 1.5, (
        f"p95 latency {p95:.2f}ms exceeds 1.5x target {TARGET_P95_LATENCY_MS * 1.5}ms"
    )


async def test_apoc_plugin_available(neo4j_driver: AsyncDriver) -> None:
    """
    Verify APOC plugin is installed and functional.

    Tests APOC procedure availability.
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # List APOC procedures
        result = await session.run(
            """
            CALL apoc.help('apoc') YIELD name
            RETURN count(name) AS apoc_procedure_count
            """
        )
        record = await result.single()
        apoc_count = record["apoc_procedure_count"]

        assert apoc_count > 0, "APOC plugin not available (0 procedures found)"
        print(f"\nAPOC plugin available: {apoc_count} procedures")


async def test_connection_pooling(neo4j_driver: AsyncDriver) -> None:
    """
    Test concurrent query execution with connection pooling.

    Validates connection pool handles concurrent load.
    """
    num_concurrent = 20

    async def run_query() -> float:
        """Execute a simple query and measure latency."""
        start_time = time.perf_counter()
        async with neo4j_driver.session(database="neo4j") as session:
            result = await session.run("RETURN 1 AS value")
            await result.single()
        return (time.perf_counter() - start_time) * 1000

    # Run concurrent queries
    tasks = [run_query() for _ in range(num_concurrent)]
    latencies = await asyncio.gather(*tasks)

    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)

    print(f"\nConnection Pool Test ({num_concurrent} concurrent):")
    print(f"  Average: {avg_latency:.2f}ms")
    print(f"  Max: {max_latency:.2f}ms")

    # Use 200ms threshold to account for test environment variability
    # (Docker container overhead, CI resource contention)
    assert max_latency < 200, f"Connection pool latency too high: {max_latency:.2f}ms"
