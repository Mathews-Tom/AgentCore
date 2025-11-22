"""Pytest configuration for memory integration tests."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentcore.a2a_protocol.database.connection import Base

# Import fixtures from the fixtures module
from tests.integration.fixtures.qdrant import (
    qdrant_client,
    qdrant_container,
    qdrant_sample_points,
    qdrant_test_collection,
    qdrant_url,
)
from tests.integration.fixtures.neo4j import (
    clean_neo4j_db,
    neo4j_container,
    neo4j_driver,
    neo4j_session_with_sample_graph,
    neo4j_uri,
)


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create test database engine (in-memory SQLite for fast tests).

    Note: Memory models (memories, stage_memories, etc.) use PostgreSQL-specific
    ARRAY(Float) types which are incompatible with SQLite. These tables are excluded.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False)

    # Create tables excluding memory models (they use PostgreSQL-specific types)
    async with engine.begin() as conn:
        memory_tables = ["memories", "stage_memories", "task_contexts", "error_records", "compression_metrics"]
        tables_to_create = [
            table for table in Base.metadata.sorted_tables
            if table.name not in memory_tables
        ]
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables_to_create))

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture(scope="function")
async def init_test_db(test_db_engine):
    """Initialize test database for memory integration tests.

    Patches the global database connection to use the test engine.
    """
    from agentcore.a2a_protocol.database import connection

    # Override global engine and session factory with test versions
    original_engine = connection.engine
    original_session = connection.SessionLocal

    connection.engine = test_db_engine
    connection.SessionLocal = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    yield

    # Restore original connection (if any)
    connection.engine = original_engine
    connection.SessionLocal = original_session


# Re-export fixtures for pytest discovery
__all__ = [
    "qdrant_container",
    "qdrant_url",
    "qdrant_client",
    "qdrant_test_collection",
    "qdrant_sample_points",
    "neo4j_container",
    "neo4j_uri",
    "neo4j_driver",
    "neo4j_session_with_sample_graph",
    "clean_neo4j_db",
    "test_db_engine",
    "init_test_db",
]
