"""
Integration Tests for HybridSearchService

Tests end-to-end hybrid search with real Qdrant and Neo4j containers.
Measures performance metrics (p95 latency) and retrieval precision.
"""

import asyncio
import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

# Check Docker availability before importing testcontainers
# These tests require pulling Docker images which can fail with credential store issues
DOCKER_AVAILABLE = False
DOCKER_SKIP_REASON = "Docker integration tests disabled - credential store issues"

try:
    import docker
    from docker.credentials.errors import StoreError

    docker_client = docker.from_env()
    docker_client.ping()
    # Test that we can actually interact with registry (not just list local images)
    # Try to get info which tests credential store access
    info = docker_client.info()
    # Check if credential store is configured (potential source of issues)
    creds_store = info.get("RegistryConfig", {}).get("CredentialHelpers", {})
    if creds_store or "docker-credential-desktop" in str(info):
        # Desktop credential store can be problematic - test by actually trying to pull
        # Try pulling a small test image to trigger credential store
        try:
            # Use a very small image that should trigger credential store access
            docker_client.images.pull("hello-world", platform="linux/amd64")
            docker_client.images.list()
            DOCKER_AVAILABLE = True
            DOCKER_SKIP_REASON = ""
        except StoreError as store_err:
            DOCKER_SKIP_REASON = f"Docker credential store error: {store_err}"
        except Exception as pull_err:
            # Check if it's a nested StoreError
            if "StoreError" in str(type(pull_err).__name__) or "credential" in str(
                pull_err
            ).lower():
                DOCKER_SKIP_REASON = f"Docker credential store error: {pull_err}"
            else:
                DOCKER_SKIP_REASON = f"Docker pull error: {pull_err}"
    else:
        docker_client.images.list()
        DOCKER_AVAILABLE = True
        DOCKER_SKIP_REASON = ""
except ImportError:
    DOCKER_SKIP_REASON = "docker-py not installed"
except Exception as e:
    # Catch credential store errors that happen during client creation
    if "StoreError" in str(type(e).__name__) or "credential" in str(e).lower():
        DOCKER_SKIP_REASON = f"Docker credential store error: {e}"
    else:
        DOCKER_SKIP_REASON = f"Docker not available: {type(e).__name__}: {e}"

if DOCKER_AVAILABLE:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs
else:
    DockerContainer = None  # type: ignore[misc, assignment]
    wait_for_logs = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason=DOCKER_SKIP_REASON if not DOCKER_AVAILABLE else "",
)

from agentcore.a2a_protocol.models.memory import MemoryRecord, StageType
from agentcore.a2a_protocol.services.memory.graph_service import GraphMemoryService
from agentcore.a2a_protocol.services.memory.hybrid_search import (
    HybridSearchConfig,
    HybridSearchService,
)
from agentcore.a2a_protocol.services.memory.retrieval_service import (
    EnhancedRetrievalService,
)


def _check_docker_error(e: Exception) -> None:
    """Check if exception is a Docker credential store error and skip if so."""
    error_str = str(e).lower()
    type_str = str(type(e).__name__)
    if (
        "StoreError" in type_str
        or "credential" in error_str
        or "docker-credential-desktop" in error_str
    ):
        pytest.skip(f"Docker credential store error: {e}")


@pytest.fixture(scope="module")
def qdrant_container():
    """Start Qdrant container for testing."""
    container = None
    try:
        container = DockerContainer("qdrant/qdrant:latest")
        container.with_exposed_ports(6333)
        container.with_env("QDRANT__SERVICE__GRPC_PORT", "6334")

        container.start()
        wait_for_logs(container, "Qdrant gRPC listening", timeout=30)
    except Exception as e:
        _check_docker_error(e)
        raise

    yield container

    if container:
        container.stop()


@pytest.fixture(scope="module")
def neo4j_container():
    """Start Neo4j container for testing."""
    container = None
    try:
        container = DockerContainer("neo4j:5.15")
        container.with_exposed_ports(7687)
        container.with_env("NEO4J_AUTH", "neo4j/testpassword")
        container.with_env("NEO4J_PLUGINS", '["apoc"]')

        container.start()
        wait_for_logs(container, "Started", timeout=60)

        # Wait for Neo4j to be ready
        time.sleep(5)
    except Exception as e:
        _check_docker_error(e)
        raise

    yield container

    if container:
        container.stop()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def qdrant_client(qdrant_container):
    """Create Qdrant client connected to test container."""
    port = qdrant_container.get_exposed_port(6333)
    client = AsyncQdrantClient(host="localhost", port=port, timeout=30)

    # Create test collection
    await client.create_collection(
        collection_name="test_memories",
        vectors_config=qmodels.VectorParams(
            size=768, distance=qmodels.Distance.COSINE
        ),
    )

    yield client

    # Cleanup
    await client.delete_collection("test_memories")
    await client.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_driver(neo4j_container):
    """Create Neo4j driver connected to test container."""
    port = neo4j_container.get_exposed_port(7687)
    uri = f"bolt://localhost:{port}"
    driver = AsyncGraphDatabase.driver(uri, auth=("neo4j", "testpassword"))

    # Wait for driver to be ready
    await asyncio.sleep(2)

    yield driver

    await driver.close()


@pytest.fixture
async def graph_service(neo4j_driver):
    """Create GraphMemoryService with test Neo4j driver."""
    service = GraphMemoryService(driver=neo4j_driver)
    await service.initialize()

    yield service

    # Cleanup: delete all test data
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")


@pytest.fixture
def retrieval_service():
    """Create EnhancedRetrievalService."""
    return EnhancedRetrievalService()


@pytest.fixture
async def hybrid_search_service(qdrant_client, graph_service, retrieval_service):
    """Create HybridSearchService with test dependencies."""
    config = HybridSearchConfig(
        vector_weight=0.6,
        graph_weight=0.4,
        max_graph_depth=2,
        max_graph_seeds=10,
        enable_retrieval_scoring=True,
        retrieval_weight=0.3,
    )

    return HybridSearchService(
        qdrant_client=qdrant_client,
        graph_service=graph_service,
        collection_name="test_memories",
        retrieval_service=retrieval_service,
        config=config,
    )


def create_test_memory(
    memory_id: str | None = None,
    content: str = "test content",
    embedding: list[float] | None = None,
    task_id: str = "task-001",
    is_critical: bool = False,
) -> MemoryRecord:
    """Create test memory record."""
    return MemoryRecord(
        memory_id=memory_id or str(uuid4()),
        memory_layer="semantic",
        content=content,
        summary=content[:50],
        embedding=embedding or ([0.1] * 768),
        agent_id="agent-test",
        session_id="session-test",
        task_id=task_id,
        timestamp=datetime.now(UTC),
        entities=[],
        facts=[],
        keywords=[],
        stage_id="stage-planning",
        is_critical=is_critical,
        access_count=0,
    )


@pytest.mark.asyncio
@pytest.mark.integration
class TestHybridSearchIntegration:
    """Integration tests for HybridSearchService."""

    async def test_vector_search_integration(
        self, hybrid_search_service, qdrant_client
    ):
        """Test vector search with real Qdrant."""
        # Insert test memories
        memories = [
            create_test_memory(
                memory_id="mem-001",
                content="Python programming language",
                embedding=[0.9] + [0.1] * 767,
            ),
            create_test_memory(
                memory_id="mem-002",
                content="JavaScript web development",
                embedding=[0.1] + [0.9] * 767,
            ),
        ]

        points = []
        for memory in memories:
            points.append(
                qmodels.PointStruct(
                    id=memory.memory_id,
                    vector=memory.embedding,
                    payload=memory.model_dump(),
                )
            )

        await qdrant_client.upsert(collection_name="test_memories", points=points)

        # Wait for indexing
        await asyncio.sleep(0.5)

        # Search with query similar to first memory
        query_embedding = [0.95] + [0.05] * 767
        results = await hybrid_search_service.vector_search(
            query_embedding=query_embedding, limit=10
        )

        # Should find both, first one ranked higher
        assert len(results) >= 1
        assert results[0][0].memory_id == "mem-001"
        assert results[0][1] > 0.5  # High similarity

    async def test_hybrid_search_integration(
        self, hybrid_search_service, qdrant_client, graph_service
    ):
        """Test end-to-end hybrid search with vector + graph."""
        # Create test memories
        mem1 = create_test_memory(
            memory_id="mem-hybrid-001",
            content="Database design patterns",
            embedding=[0.8] + [0.2] * 767,
        )
        mem2 = create_test_memory(
            memory_id="mem-hybrid-002",
            content="SQL query optimization",
            embedding=[0.7] + [0.3] * 767,
        )

        # Insert into Qdrant
        points = [
            qmodels.PointStruct(
                id=mem1.memory_id, vector=mem1.embedding, payload=mem1.model_dump()
            ),
            qmodels.PointStruct(
                id=mem2.memory_id, vector=mem2.embedding, payload=mem2.model_dump()
            ),
        ]
        await qdrant_client.upsert(collection_name="test_memories", points=points)

        # Insert into Neo4j graph
        await graph_service.store_memory_node(mem1)
        await graph_service.store_memory_node(mem2)

        # Create relationship between memories
        await graph_service.create_relationship(
            from_id=mem1.memory_id,
            to_id=mem2.memory_id,
            relationship_type="RELATES_TO",
            from_label="Memory",
            to_label="Memory",
            strength=0.9,
        )

        # Wait for indexing
        await asyncio.sleep(0.5)

        # Perform hybrid search
        query_embedding = [0.85] + [0.15] * 767
        results = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding, limit=10, use_graph_expansion=True
        )

        # Should find both memories
        assert len(results) >= 1

        # Check metadata
        for memory, score, metadata in results:
            assert metadata.final_score > 0.0
            # At least one should be found in vector search
            if memory.memory_id == mem1.memory_id:
                assert metadata.found_in_vector is True

    async def test_hybrid_search_latency(
        self, hybrid_search_service, qdrant_client, graph_service
    ):
        """Test that hybrid search meets <300ms p95 latency target."""
        # Create 50 test memories
        memories = []
        points = []

        for i in range(50):
            memory = create_test_memory(
                memory_id=f"mem-perf-{i:03d}",
                content=f"Test memory {i}",
                embedding=[float(i % 10) / 10.0] * 768,
            )
            memories.append(memory)
            points.append(
                qmodels.PointStruct(
                    id=memory.memory_id,
                    vector=memory.embedding,
                    payload=memory.model_dump(),
                )
            )

        # Insert into Qdrant
        await qdrant_client.upsert(collection_name="test_memories", points=points)

        # Insert first 10 into Neo4j and link them
        for i in range(10):
            await graph_service.store_memory_node(memories[i])
            if i > 0:
                await graph_service.create_relationship(
                    from_id=memories[i - 1].memory_id,
                    to_id=memories[i].memory_id,
                    relationship_type="RELATES_TO",
                    from_label="Memory",
                    to_label="Memory",
                    strength=0.8,
                )

        await asyncio.sleep(1.0)  # Wait for indexing

        # Run 20 searches and measure latency
        latencies = []

        for i in range(20):
            query_embedding = [float(i % 10) / 10.0] * 768

            start_time = time.perf_counter()
            results = await hybrid_search_service.hybrid_search(
                query_embedding=query_embedding,
                limit=10,
                use_graph_expansion=True,
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            assert len(results) > 0  # Should find results

        # Calculate p95 latency
        latencies.sort()
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index]

        print(f"\nLatency stats:")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Median: {latencies[len(latencies) // 2]:.2f}ms")
        print(f"  p95: {p95_latency:.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")

        # Target: <300ms p95 latency
        assert p95_latency < 300.0, f"p95 latency {p95_latency:.2f}ms exceeds 300ms target"

    async def test_hybrid_search_precision(
        self, hybrid_search_service, qdrant_client, graph_service
    ):
        """Test retrieval precision with ground truth dataset."""
        # Create ground truth dataset
        # Query about "Python testing" should retrieve test-related memories

        relevant_memories = [
            create_test_memory(
                memory_id="mem-relevant-001",
                content="Python pytest framework for unit testing",
                embedding=[0.9, 0.8] + [0.1] * 766,
            ),
            create_test_memory(
                memory_id="mem-relevant-002",
                content="Writing test assertions in Python",
                embedding=[0.85, 0.75] + [0.15] * 766,
            ),
            create_test_memory(
                memory_id="mem-relevant-003",
                content="Test-driven development with Python",
                embedding=[0.88, 0.78] + [0.12] * 766,
            ),
        ]

        irrelevant_memories = [
            create_test_memory(
                memory_id="mem-irrelevant-001",
                content="JavaScript React components",
                embedding=[0.2, 0.3] + [0.8] * 766,
            ),
            create_test_memory(
                memory_id="mem-irrelevant-002",
                content="Database schema design",
                embedding=[0.3, 0.2] + [0.7] * 766,
            ),
        ]

        all_memories = relevant_memories + irrelevant_memories

        # Insert into Qdrant
        points = [
            qmodels.PointStruct(
                id=mem.memory_id, vector=mem.embedding, payload=mem.model_dump()
            )
            for mem in all_memories
        ]
        await qdrant_client.upsert(collection_name="test_memories", points=points)

        # Insert into Neo4j and link relevant memories
        for mem in relevant_memories:
            await graph_service.store_memory_node(mem)

        for i in range(len(relevant_memories) - 1):
            await graph_service.create_relationship(
                from_id=relevant_memories[i].memory_id,
                to_id=relevant_memories[i + 1].memory_id,
                relationship_type="RELATES_TO",
                from_label="Memory",
                to_label="Memory",
                strength=0.9,
            )

        await asyncio.sleep(0.5)

        # Search with query embedding similar to relevant memories
        query_embedding = [0.9, 0.8] + [0.1] * 766

        results = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding, limit=5, use_graph_expansion=True
        )

        # Calculate precision: relevant retrieved / total retrieved
        relevant_ids = {mem.memory_id for mem in relevant_memories}
        retrieved_ids = {mem.memory_id for mem, _, _ in results}

        relevant_retrieved = relevant_ids & retrieved_ids
        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0

        print(f"\nPrecision: {precision:.2%}")
        print(f"Relevant retrieved: {len(relevant_retrieved)}/{len(relevant_ids)}")
        print(f"Total retrieved: {len(retrieved_ids)}")

        # Target: 90%+ precision
        # With only 3 relevant memories, expect at least 2-3 in top 5
        assert precision >= 0.6, f"Precision {precision:.2%} below target"

    async def test_concurrent_hybrid_searches(
        self, hybrid_search_service, qdrant_client
    ):
        """Test concurrent hybrid searches for thread safety."""
        # Insert test memories
        memories = [
            create_test_memory(
                memory_id=f"mem-concurrent-{i:03d}",
                content=f"Test memory {i}",
                embedding=[float(i % 10) / 10.0] * 768,
            )
            for i in range(20)
        ]

        points = [
            qmodels.PointStruct(
                id=mem.memory_id, vector=mem.embedding, payload=mem.model_dump()
            )
            for mem in memories
        ]
        await qdrant_client.upsert(collection_name="test_memories", points=points)

        await asyncio.sleep(0.5)

        # Run 10 concurrent searches
        async def run_search(i: int):
            query_embedding = [float(i % 10) / 10.0] * 768
            results = await hybrid_search_service.hybrid_search(
                query_embedding=query_embedding, limit=5, use_graph_expansion=False
            )
            return results

        tasks = [run_search(i) for i in range(10)]
        results_list = await asyncio.gather(*tasks)

        # All searches should complete successfully
        assert len(results_list) == 10
        for results in results_list:
            assert len(results) > 0  # Each should find some results

    async def test_graph_expansion_benefits(
        self, hybrid_search_service, qdrant_client, graph_service
    ):
        """Test that graph expansion improves retrieval."""
        # Create memories: one highly relevant, others related via graph
        seed_memory = create_test_memory(
            memory_id="mem-graph-seed",
            content="Primary concept",
            embedding=[0.9] + [0.1] * 767,
        )

        related_memories = [
            create_test_memory(
                memory_id=f"mem-graph-related-{i}",
                content=f"Related concept {i}",
                embedding=[0.3] + [0.7] * 767,  # Low vector similarity
            )
            for i in range(3)
        ]

        # Insert into Qdrant
        all_mems = [seed_memory] + related_memories
        points = [
            qmodels.PointStruct(
                id=mem.memory_id, vector=mem.embedding, payload=mem.model_dump()
            )
            for mem in all_mems
        ]
        await qdrant_client.upsert(collection_name="test_memories", points=points)

        # Insert into Neo4j with strong relationships
        await graph_service.store_memory_node(seed_memory)
        for mem in related_memories:
            await graph_service.store_memory_node(mem)
            await graph_service.create_relationship(
                from_id=seed_memory.memory_id,
                to_id=mem.memory_id,
                relationship_type="RELATES_TO",
                from_label="Memory",
                to_label="Memory",
                strength=0.95,
            )

        await asyncio.sleep(0.5)

        # Search without graph expansion
        query_embedding = [0.95] + [0.05] * 767

        results_no_graph = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding, limit=10, use_graph_expansion=False
        )

        # Search with graph expansion
        results_with_graph = await hybrid_search_service.hybrid_search(
            query_embedding=query_embedding, limit=10, use_graph_expansion=True
        )

        # Graph expansion should find more memories
        assert len(results_with_graph) >= len(results_no_graph)

        # Check that related memories appear in graph-enabled search
        with_graph_ids = {mem.memory_id for mem, _, _ in results_with_graph}
        related_ids = {mem.memory_id for mem in related_memories}

        # At least one related memory should be found via graph
        assert len(with_graph_ids & related_ids) > 0
