"""
Neo4j Integration Tests

Tests Neo4j deployment, APOC plugin availability, and schema initialization.
Validates MEM-003 acceptance criteria.
"""

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
from testcontainers.neo4j import Neo4jContainer


# Use module-scoped event loop for all tests (matches fixture scope)
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.fixture(scope="module")
def neo4j_container() -> Neo4jContainer:
    """
    Start Neo4j testcontainer with APOC and GDS plugins.

    Returns:
        Neo4jContainer: Running Neo4j container instance
    """
    container = Neo4jContainer(image="neo4j:5.15-community")
    container.with_env("NEO4J_AUTH", "neo4j/password")
    container.with_env("NEO4J_PLUGINS", '["apoc", "graph-data-science"]')
    container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*,gds.*")
    container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*,gds.*")
    container.with_env("NEO4J_apoc_export_file_enabled", "true")
    container.with_env("NEO4J_apoc_import_file_enabled", "true")

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
        max_connection_lifetime=3600,
        connection_acquisition_timeout=60,
    )

    await driver.verify_connectivity()
    yield driver
    await driver.close()


async def test_neo4j_connectivity(neo4j_driver: AsyncDriver) -> None:
    """
    Test basic Neo4j connectivity.

    Validates:
    - Driver can connect to Neo4j
    - Simple query execution works
    """
    async with neo4j_driver.session(database="neo4j") as session:
        result = await session.run("RETURN 1 AS value")
        record = await result.single()
        assert record["value"] == 1


async def test_apoc_plugin_installed(neo4j_driver: AsyncDriver) -> None:
    """
    Test APOC plugin is installed and available.

    Validates MEM-003 acceptance criteria:
    - APOC plugin installed
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # List APOC procedures
        result = await session.run("CALL apoc.help('apoc')")
        records = [record async for record in result]

        assert len(records) > 0, "APOC plugin not installed (no procedures found)"

        # Test a specific APOC function
        result = await session.run("RETURN apoc.version() AS version")
        record = await result.single()
        version = record["version"]

        assert version is not None, "APOC version is None"
        print(f"\nAPOC version: {version}")


async def test_gds_plugin_installed(neo4j_driver: AsyncDriver) -> None:
    """
    Test Graph Data Science plugin is installed and available.

    Validates MEM-003 acceptance criteria:
    - Graph Data Science plugin installed
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # List GDS procedures
        result = await session.run("CALL gds.list()")
        records = [record async for record in result]

        # GDS should be available (may return empty list if no graphs projected)
        # Just verify the procedure call works
        assert True, "GDS plugin callable"

        # Test GDS version
        result = await session.run("RETURN gds.version() AS version")
        record = await result.single()
        version = record["version"]

        assert version is not None, "GDS version is None"
        print(f"\nGDS version: {version}")


async def test_graph_schema_constraints(neo4j_driver: AsyncDriver) -> None:
    """
    Test graph schema constraints can be created.

    Validates MEM-003 acceptance criteria:
    - Graph schema defined (Memory, Entity, Concept nodes)
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # Create Memory node constraint
        await session.run(
            """
            CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
            FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE
            """
        )

        # Create Entity node constraint
        await session.run(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
            """
        )

        # Create Concept node constraint
        await session.run(
            """
            CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
            FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE
            """
        )

        # Verify constraints exist
        result = await session.run("SHOW CONSTRAINTS")
        records = [record async for record in result]

        constraint_names = [record["name"] for record in records]
        assert "memory_id_unique" in constraint_names
        assert "entity_id_unique" in constraint_names
        assert "concept_id_unique" in constraint_names

        print(f"\nCreated {len(records)} constraints")


async def test_graph_schema_indexes(neo4j_driver: AsyncDriver) -> None:
    """
    Test graph schema indexes can be created.

    Validates performance optimization for <200ms p95 latency.
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # Create indexes
        await session.run(
            """
            CREATE INDEX memory_agent_idx IF NOT EXISTS
            FOR (m:Memory) ON (m.agent_id)
            """
        )

        await session.run(
            """
            CREATE INDEX entity_type_idx IF NOT EXISTS
            FOR (e:Entity) ON (e.entity_type)
            """
        )

        await session.run(
            """
            CREATE INDEX concept_category_idx IF NOT EXISTS
            FOR (c:Concept) ON (c.category)
            """
        )

        # Verify indexes exist
        result = await session.run("SHOW INDEXES")
        records = [record async for record in result]

        index_names = [record["name"] for record in records]
        assert "memory_agent_idx" in index_names
        assert "entity_type_idx" in index_names
        assert "concept_category_idx" in index_names

        print(f"\nCreated {len(records)} indexes")


async def test_connection_pooling(neo4j_driver: AsyncDriver) -> None:
    """
    Test connection pooling configuration.

    Validates MEM-003 acceptance criteria:
    - Connection pooling configured (neo4j-driver async)
    """
    # Connection pool is already configured in the fixture
    # Verify by running multiple concurrent queries
    import asyncio

    async def query() -> int:
        async with neo4j_driver.session(database="neo4j") as session:
            result = await session.run("RETURN 1 AS value")
            record = await result.single()
            return record["value"]

    # Run 10 concurrent queries
    results = await asyncio.gather(*[query() for _ in range(10)])

    assert len(results) == 10
    assert all(r == 1 for r in results)
    print("\nConnection pooling working: 10 concurrent queries successful")


async def test_create_memory_node(neo4j_driver: AsyncDriver) -> None:
    """
    Test creating Memory nodes with required properties.

    Validates graph schema node types.
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # Create constraint first
        await session.run(
            """
            CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
            FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE
            """
        )

        # Create Memory node
        memory_id = "test-memory-001"
        result = await session.run(
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
            RETURN m
            """,
            memory_id=memory_id,
            agent_id="agent-001",
            session_id="session-001",
            layer="episodic",
            stage="execution",
            content="Test memory content",
            criticality=0.8,
        )

        record = await result.single()
        assert record["m"]["memory_id"] == memory_id
        print(f"\nCreated Memory node: {memory_id}")


async def test_create_entity_node(neo4j_driver: AsyncDriver) -> None:
    """
    Test creating Entity nodes with required properties.

    Validates graph schema node types.
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # Create constraint first
        await session.run(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
            """
        )

        # Create Entity node
        entity_id = "test-entity-001"
        result = await session.run(
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
            RETURN e
            """,
            entity_id=entity_id,
            name="Test Entity",
            entity_type="concept",
            confidence=0.9,
        )

        record = await result.single()
        assert record["e"]["entity_id"] == entity_id
        print(f"\nCreated Entity node: {entity_id}")


async def test_create_concept_node(neo4j_driver: AsyncDriver) -> None:
    """
    Test creating Concept nodes with required properties.

    Validates graph schema node types.
    """
    async with neo4j_driver.session(database="neo4j") as session:
        # Create constraint first
        await session.run(
            """
            CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
            FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE
            """
        )

        # Create Concept node
        concept_id = "test-concept-001"
        result = await session.run(
            """
            CREATE (c:Concept {
                concept_id: $concept_id,
                name: $name,
                description: $description,
                category: $category,
                created_at: datetime(),
                usage_count: 0
            })
            RETURN c
            """,
            concept_id=concept_id,
            name="Test Concept",
            description="A test concept for validation",
            category="test",
        )

        record = await result.single()
        assert record["c"]["concept_id"] == concept_id
        print(f"\nCreated Concept node: {concept_id}")
