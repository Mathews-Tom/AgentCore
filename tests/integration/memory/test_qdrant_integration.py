"""
Integration tests for Qdrant vector database.

Tests Qdrant deployment, collection management, vector search,
and performance benchmarks for MEM-002 acceptance criteria.
"""

import time
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# Use session-scoped event loop to match fixture scope
pytestmark = pytest.mark.asyncio(loop_scope="session")


class TestQdrantDeployment:
    """Test Qdrant deployment and connectivity."""

    async def test_qdrant_connection(self, qdrant_client: AsyncQdrantClient) -> None:
        """Test basic Qdrant connection."""
        collections = await qdrant_client.get_collections()
        assert collections is not None
        assert hasattr(collections, "collections")

    async def test_qdrant_health(self, qdrant_url: str) -> None:
        """Test Qdrant health endpoint."""
        import httpx

        async with httpx.AsyncClient() as client:
            # Qdrant uses /healthz endpoint (Kubernetes-style)
            response = await client.get(f"{qdrant_url}/healthz")
            assert response.status_code == 200


class TestQdrantCollections:
    """Test Qdrant collection management."""

    async def test_create_collection(
        self, qdrant_client: AsyncQdrantClient, qdrant_test_collection: str
    ) -> None:
        """Test collection creation with vector configuration."""
        # Collection should exist (created by fixture)
        collection_info = await qdrant_client.get_collection(
            collection_name=qdrant_test_collection
        )

        assert collection_info is not None
        assert collection_info.config is not None
        assert collection_info.config.params.vectors.size == 1536
        assert collection_info.config.params.vectors.distance == Distance.COSINE

    async def test_create_payload_indexes(
        self, qdrant_client: AsyncQdrantClient, qdrant_test_collection: str
    ) -> None:
        """Test payload index creation for efficient filtering."""
        collection_info = await qdrant_client.get_collection(
            collection_name=qdrant_test_collection
        )

        # Check that indexes were created
        assert collection_info.payload_schema is not None

        # Verify specific indexes exist
        schema = collection_info.payload_schema
        assert "memory_layer" in schema
        assert "agent_id" in schema
        assert "is_critical" in schema


class TestQdrantVectorSearch:
    """Test Qdrant vector similarity search."""

    async def test_insert_and_retrieve(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        qdrant_sample_points: list[dict[str, Any]],
    ) -> None:
        """Test inserting and retrieving points."""
        # Points inserted by fixture
        assert len(qdrant_sample_points) == 3

        # Retrieve collection info
        collection_info = await qdrant_client.get_collection(
            collection_name=qdrant_test_collection
        )
        assert collection_info.points_count == 3

    async def test_vector_similarity_search(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        qdrant_sample_points: list[dict[str, Any]],
    ) -> None:
        """Test vector similarity search."""
        # Search with a query vector matching first point's pattern (even indices)
        query_vector = [0.1 if i % 2 == 0 else 0.0 for i in range(1536)]
        response = await qdrant_client.query_points(
            collection_name=qdrant_test_collection,
            query=query_vector,
            limit=3,
        )
        results = response.points

        assert len(results) == 3
        # First result should be closest (highest score)
        assert results[0].id == 1
        assert results[0].score > 0.9  # High similarity

    async def test_filtered_search_by_memory_layer(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        qdrant_sample_points: list[dict[str, Any]],
    ) -> None:
        """Test filtered search by memory_layer."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Use vector pattern matching second point (every 3rd index)
        query_vector = [0.1 if i % 3 == 0 else 0.0 for i in range(1536)]
        response = await qdrant_client.query_points(
            collection_name=qdrant_test_collection,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="memory_layer",
                        match=MatchValue(value="semantic"),
                    )
                ]
            ),
            limit=10,
        )
        results = response.points

        assert len(results) == 1
        assert results[0].payload["memory_layer"] == "semantic"

    async def test_filtered_search_by_agent_id(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        qdrant_sample_points: list[dict[str, Any]],
    ) -> None:
        """Test filtered search by agent_id."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_vector = [0.15] * 1536
        response = await qdrant_client.query_points(
            collection_name=qdrant_test_collection,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="agent_id",
                        match=MatchValue(value="test-agent-1"),
                    )
                ]
            ),
            limit=10,
        )
        results = response.points

        assert len(results) == 3
        for result in results:
            assert result.payload["agent_id"] == "test-agent-1"

    async def test_filtered_search_by_criticality(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        qdrant_sample_points: list[dict[str, Any]],
    ) -> None:
        """Test filtered search by is_critical flag."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_vector = [0.15] * 1536
        response = await qdrant_client.query_points(
            collection_name=qdrant_test_collection,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="is_critical",
                        match=MatchValue(value=True),
                    )
                ]
            ),
            limit=10,
        )
        results = response.points

        assert len(results) == 2
        for result in results:
            assert result.payload["is_critical"] is True


class TestQdrantPerformance:
    """Test Qdrant performance benchmarks (MEM-002 acceptance criteria)."""

    async def test_vector_search_latency(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
    ) -> None:
        """
        Test vector search latency meets <100ms p95 requirement.

        Acceptance Criteria: <100ms vector search latency (p95)
        """
        # Insert 1000 test points for realistic benchmark
        points = [
            PointStruct(
                id=i,
                vector=[float(i % 100) / 100] * 1536,
                payload={
                    "memory_layer": ["episodic", "semantic", "procedural"][i % 3],
                    "agent_id": f"agent-{i % 10}",
                    "timestamp": i * 1000,
                    "is_critical": i % 5 == 0,
                },
            )
            for i in range(1000)
        ]

        await qdrant_client.upsert(
            collection_name=qdrant_test_collection,
            points=points,
        )

        # Run 100 search queries to measure latency
        latencies: list[float] = []
        query_vector = [0.5] * 1536

        for _ in range(100):
            start_time = time.perf_counter()
            await qdrant_client.query_points(
                collection_name=qdrant_test_collection,
                query=query_vector,
                limit=10,
            )
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print(f"\nVector Search Latency Benchmark (1000 vectors):")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")

        # Assert p95 < 100ms (acceptance criteria)
        assert p95 < 100, f"p95 latency ({p95:.2f}ms) exceeds 100ms requirement"

    async def test_filtered_search_latency(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
    ) -> None:
        """Test filtered search latency with payload filters."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Insert 1000 test points
        points = [
            PointStruct(
                id=i,
                vector=[float(i % 100) / 100] * 1536,
                payload={
                    "memory_layer": ["episodic", "semantic", "procedural"][i % 3],
                    "agent_id": f"agent-{i % 10}",
                    "is_critical": i % 5 == 0,
                },
            )
            for i in range(1000)
        ]

        await qdrant_client.upsert(
            collection_name=qdrant_test_collection,
            points=points,
        )

        # Run 50 filtered search queries
        latencies: list[float] = []
        query_vector = [0.5] * 1536

        for i in range(50):
            start_time = time.perf_counter()
            await qdrant_client.query_points(
                collection_name=qdrant_test_collection,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="agent_id",
                            match=MatchValue(value=f"agent-{i % 10}"),
                        )
                    ]
                ),
                limit=10,
            )
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\nFiltered Vector Search Latency (1000 vectors):")
        print(f"  p95: {p95:.2f}ms")

        # Should still be under 100ms with filters
        assert p95 < 100, f"p95 filtered latency ({p95:.2f}ms) exceeds 100ms"


class TestQdrantCollectionOperations:
    """Test Qdrant collection lifecycle operations."""

    async def test_update_point(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        qdrant_sample_points: list[dict[str, Any]],
    ) -> None:
        """Test updating a point's payload."""
        # Update first point
        await qdrant_client.set_payload(
            collection_name=qdrant_test_collection,
            points=[1],
            payload={"is_critical": False, "updated": True},
        )

        # Retrieve and verify
        points = await qdrant_client.retrieve(
            collection_name=qdrant_test_collection,
            ids=[1],
        )

        assert len(points) == 1
        assert points[0].payload["is_critical"] is False
        assert points[0].payload["updated"] is True

    async def test_delete_point(
        self,
        qdrant_client: AsyncQdrantClient,
        qdrant_test_collection: str,
        qdrant_sample_points: list[dict[str, Any]],
    ) -> None:
        """Test deleting points."""
        # Delete point with id=1
        await qdrant_client.delete(
            collection_name=qdrant_test_collection,
            points_selector=[1],
        )

        # Verify deletion
        collection_info = await qdrant_client.get_collection(
            collection_name=qdrant_test_collection
        )
        assert collection_info.points_count == 2
