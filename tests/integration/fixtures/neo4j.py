"""Neo4j test fixtures for integration tests."""

import time
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase, AsyncDriver
from testcontainers.core.container import DockerContainer


@pytest.fixture(scope="session")
def neo4j_container() -> DockerContainer:
    """
    Create a Neo4j testcontainer for the session.

    Returns:
        DockerContainer: Running Neo4j container with APOC plugin
    """
    container = DockerContainer("neo4j:5.15-community")
    container.with_exposed_ports(7474, 7687)
    container.with_env("NEO4J_AUTH", "neo4j/testpassword")
    container.with_env("NEO4J_PLUGINS", '["apoc"]')
    # Enable APOC procedures
    container.with_env("NEO4J_dbms_security_procedures_unrestricted", "apoc.*")
    container.with_env("NEO4J_dbms_security_procedures_allowlist", "apoc.*")

    container.start()

    # Wait for Neo4j to be ready (needs more time than Qdrant)
    time.sleep(15)

    yield container
    container.stop()


@pytest.fixture(scope="session")
def neo4j_uri(neo4j_container: DockerContainer) -> str:
    """
    Get the Neo4j Bolt URI.

    Args:
        neo4j_container: Running Neo4j container

    Returns:
        str: Neo4j Bolt URI
    """
    host = neo4j_container.get_container_host_ip()
    port = neo4j_container.get_exposed_port(7687)
    return f"bolt://{host}:{port}"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def neo4j_driver(neo4j_uri: str) -> AsyncGenerator[AsyncDriver, None]:
    """
    Create an async Neo4j driver connected to the test container.

    Args:
        neo4j_uri: Neo4j Bolt URI

    Yields:
        AsyncDriver: Connected Neo4j driver
    """
    driver = AsyncGraphDatabase.driver(
        neo4j_uri,
        auth=("neo4j", "testpassword"),
        max_connection_pool_size=10,
    )

    try:
        # Verify connection
        await driver.verify_connectivity()
        yield driver
    finally:
        await driver.close()


@pytest_asyncio.fixture(scope="function")
async def clean_neo4j_db(neo4j_driver: AsyncDriver) -> AsyncGenerator[None, None]:
    """
    Clean Neo4j database before and after each test.

    Args:
        neo4j_driver: Connected Neo4j driver

    Yields:
        None
    """
    # Clean before test
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")

    yield

    # Clean after test
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")


@pytest_asyncio.fixture(scope="function")
async def neo4j_session_with_sample_graph(
    neo4j_driver: AsyncDriver, clean_neo4j_db
) -> AsyncGenerator[tuple[AsyncDriver, dict], None]:
    """
    Create Neo4j session with sample knowledge graph for testing.

    Args:
        neo4j_driver: Connected Neo4j driver
        clean_neo4j_db: Ensure clean database

    Yields:
        tuple: (driver, sample_data) containing driver and metadata about created nodes
    """
    async with neo4j_driver.session() as session:
        # Create sample entities and relationships
        result = await session.run(
            """
            // Create memory nodes
            CREATE (m1:Memory {
                memory_id: 'mem-1',
                content: 'User requested authentication implementation',
                memory_layer: 'episodic',
                timestamp: datetime()
            })
            CREATE (m2:Memory {
                memory_id: 'mem-2',
                content: 'JWT token implementation completed',
                memory_layer: 'episodic',
                timestamp: datetime()
            })

            // Create entity nodes
            CREATE (e1:Entity {
                entity_id: 'entity-auth',
                name: 'authentication',
                entity_type: 'concept',
                confidence: 0.95
            })
            CREATE (e2:Entity {
                entity_id: 'entity-jwt',
                name: 'JWT',
                entity_type: 'tool',
                confidence: 0.9
            })
            CREATE (e3:Entity {
                entity_id: 'entity-user',
                name: 'user',
                entity_type: 'person',
                confidence: 0.85
            })

            // Create relationships
            CREATE (m1)-[:MENTIONS {strength: 0.9}]->(e1)
            CREATE (m1)-[:MENTIONS {strength: 0.8}]->(e3)
            CREATE (m2)-[:MENTIONS {strength: 0.95}]->(e2)
            CREATE (e1)-[:RELATES_TO {strength: 0.9}]->(e2)
            CREATE (m2)-[:FOLLOWS {temporal_order: 1}]->(m1)

            RETURN m1.memory_id as mem1_id, m2.memory_id as mem2_id,
                   e1.entity_id as entity1_id, e2.entity_id as entity2_id, e3.entity_id as entity3_id
            """
        )

        # Get created IDs
        record = await result.single()
        sample_data = {
            "memory_ids": [record["mem1_id"], record["mem2_id"]],
            "entity_ids": [
                record["entity1_id"],
                record["entity2_id"],
                record["entity3_id"],
            ],
        }

    yield (neo4j_driver, sample_data)
