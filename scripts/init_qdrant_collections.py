#!/usr/bin/env python3
"""
Qdrant Collection Initialization Script

Creates collections for AgentCore memory system layers:
- Episodic memory (recent conversation episodes)
- Semantic memory (long-term facts and knowledge)
- Procedural memory (action-outcome patterns)

Usage:
    uv run python scripts/init_qdrant_collections.py

Environment Variables:
    QDRANT_URL: Qdrant server URL (default: http://localhost:6333)
    QDRANT_API_KEY: API key for cloud deployment (optional)
    QDRANT_COLLECTION_NAME: Base collection name (default: agentcore_memories)
"""

import asyncio
import sys
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Import config
sys.path.insert(0, "src")
from agentcore.a2a_protocol.config import settings


async def init_qdrant_collections() -> None:
    """Initialize Qdrant collections for memory layers."""
    print(f"Connecting to Qdrant at {settings.QDRANT_URL}...")

    # Create async client
    client = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=settings.QDRANT_TIMEOUT,
    )

    try:
        # Check connection
        collections = await client.get_collections()
        print(f"Connected successfully. Existing collections: {len(collections.collections)}")

        # Collection configuration
        collection_name = settings.QDRANT_COLLECTION_NAME
        vector_size = settings.QDRANT_VECTOR_SIZE
        distance = Distance.COSINE if settings.QDRANT_DISTANCE == "Cosine" else Distance.EUCLID

        # Check if collection exists
        collection_exists = False
        for collection in collections.collections:
            if collection.name == collection_name:
                collection_exists = True
                print(f"Collection '{collection_name}' already exists.")
                break

        if not collection_exists:
            # Create collection
            print(f"Creating collection '{collection_name}'...")
            print(f"  - Vector size: {vector_size}")
            print(f"  - Distance metric: {settings.QDRANT_DISTANCE}")

            await client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            print(f"Collection '{collection_name}' created successfully!")

            # Create payload indexes for efficient filtering
            print("Creating payload indexes...")

            # Index for memory_layer (episodic, semantic, procedural)
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="memory_layer",
                field_schema="keyword",
            )
            print("  - Indexed 'memory_layer' (keyword)")

            # Index for agent_id
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="agent_id",
                field_schema="keyword",
            )
            print("  - Indexed 'agent_id' (keyword)")

            # Index for session_id
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="session_id",
                field_schema="keyword",
            )
            print("  - Indexed 'session_id' (keyword)")

            # Index for task_id
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="task_id",
                field_schema="keyword",
            )
            print("  - Indexed 'task_id' (keyword)")

            # Index for is_critical
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="is_critical",
                field_schema="bool",
            )
            print("  - Indexed 'is_critical' (bool)")

            # Index for timestamp for time-based queries
            await client.create_payload_index(
                collection_name=collection_name,
                field_name="timestamp",
                field_schema="integer",
            )
            print("  - Indexed 'timestamp' (integer)")

            print("All indexes created successfully!")

            # Insert a test point to verify collection is working
            print("\nInserting test point...")
            test_point = PointStruct(
                id=1,
                vector=[0.0] * vector_size,
                payload={
                    "memory_layer": "test",
                    "content": "Test memory for collection initialization",
                    "agent_id": "test-agent",
                    "timestamp": 0,
                    "is_critical": False,
                },
            )
            await client.upsert(
                collection_name=collection_name,
                points=[test_point],
            )
            print("Test point inserted successfully!")

            # Verify with a simple search
            print("\nVerifying with test search...")
            search_results = await client.search(
                collection_name=collection_name,
                query_vector=[0.0] * vector_size,
                limit=1,
            )
            if search_results and len(search_results) > 0:
                print(f"Test search successful! Found {len(search_results)} result(s)")
            else:
                print("WARNING: Test search returned no results")

            # Clean up test point
            await client.delete(
                collection_name=collection_name,
                points_selector=[1],
            )
            print("Test point cleaned up.")

        # Display collection info
        print(f"\nCollection '{collection_name}' is ready!")
        collection_info = await client.get_collection(collection_name=collection_name)
        print(f"  - Points count: {collection_info.points_count}")
        print(f"  - Vectors count: {collection_info.vectors_count}")
        print(f"  - Status: {collection_info.status}")

        print("\nQdrant collection initialization complete!")

    except Exception as e:
        print(f"ERROR: Failed to initialize Qdrant collections: {e}")
        raise
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(init_qdrant_collections())
