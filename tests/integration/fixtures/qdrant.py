"""Qdrant test fixtures for integration tests."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from testcontainers.core.container import DockerContainer


@pytest.fixture(scope="session")
def qdrant_container() -> DockerContainer:
    """
    Create a Qdrant testcontainer for the session.

    Returns:
        DockerContainer: Running Qdrant container
    """
    container = DockerContainer("qdrant/qdrant:v1.7.0")
    container.with_exposed_ports(6333, 6334)
    container.with_env("QDRANT__SERVICE__HTTP_PORT", "6333")
    container.with_env("QDRANT__SERVICE__GRPC_PORT", "6334")

    container.start()

    # Wait for Qdrant to be ready (simple sleep)
    time.sleep(5)

    yield container
    container.stop()


@pytest.fixture(scope="session")
def qdrant_url(qdrant_container: DockerContainer) -> str:
    """
    Get the Qdrant HTTP API URL.

    Args:
        qdrant_container: Running Qdrant container

    Returns:
        str: Qdrant HTTP API URL
    """
    host = qdrant_container.get_container_host_ip()
    port = qdrant_container.get_exposed_port(6333)
    return f"http://{host}:{port}"


@pytest.fixture(scope="session")
async def qdrant_client(qdrant_url: str) -> AsyncGenerator[AsyncQdrantClient, None]:
    """
    Create an async Qdrant client connected to the test container.

    Args:
        qdrant_url: Qdrant HTTP API URL

    Yields:
        AsyncQdrantClient: Connected Qdrant client
    """
    client = AsyncQdrantClient(url=qdrant_url, timeout=30)

    try:
        # Verify connection
        await client.get_collections()
        yield client
    finally:
        await client.close()


@pytest.fixture(scope="function")
async def qdrant_test_collection(
    qdrant_client: AsyncQdrantClient,
) -> AsyncGenerator[str, None]:
    """
    Create a test collection for each test function.

    Args:
        qdrant_client: Connected Qdrant client

    Yields:
        str: Collection name
    """
    collection_name = f"test_collection_{id(asyncio.current_task())}"

    # Create collection with default vector configuration
    await qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Create payload indexes for common filters
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

    yield collection_name

    # Cleanup: Delete collection after test
    await qdrant_client.delete_collection(collection_name=collection_name)


@pytest.fixture(scope="function")
async def qdrant_sample_points(
    qdrant_client: AsyncQdrantClient, qdrant_test_collection: str
) -> list[dict[str, Any]]:
    """
    Insert sample points into test collection.

    Args:
        qdrant_client: Connected Qdrant client
        qdrant_test_collection: Test collection name

    Returns:
        list[dict]: Inserted sample points metadata
    """
    from qdrant_client.models import PointStruct

    # Create sample points
    points = [
        PointStruct(
            id=1,
            vector=[0.1] * 1536,
            payload={
                "memory_layer": "episodic",
                "content": "User asked about API authentication",
                "agent_id": "test-agent-1",
                "session_id": "test-session-1",
                "timestamp": 1000,
                "is_critical": True,
            },
        ),
        PointStruct(
            id=2,
            vector=[0.2] * 1536,
            payload={
                "memory_layer": "semantic",
                "content": "User prefers detailed technical explanations",
                "agent_id": "test-agent-1",
                "timestamp": 2000,
                "is_critical": True,
            },
        ),
        PointStruct(
            id=3,
            vector=[0.3] * 1536,
            payload={
                "memory_layer": "procedural",
                "content": "Action: /api/auth -> Outcome: 200 OK",
                "agent_id": "test-agent-1",
                "timestamp": 3000,
                "is_critical": False,
            },
        ),
    ]

    # Insert points
    await qdrant_client.upsert(
        collection_name=qdrant_test_collection,
        points=points,
    )

    # Return metadata for verification
    return [
        {
            "id": point.id,
            "payload": point.payload,
        }
        for point in points
    ]
