"""Real database fixtures using testcontainers."""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from testcontainers.postgres import PostgresContainer

from agentcore.a2a_protocol.database.connection import Base

# Import all ORM models to register them with Base.metadata
from agentcore.a2a_protocol.database import models as a2a_models  # noqa: F401
from agentcore.ace.database import ace_orm  # noqa: F401


@pytest.fixture(scope="module")
def postgres_container():
    """Start PostgreSQL container for integration tests.

    Uses pgvector/pgvector:pg15 image to match production environment.
    Container is scoped to module for performance (reused across tests).
    """
    with PostgresContainer(
        image="pgvector/pgvector:pg15",
        username="test_user",
        password="test_pass",
        dbname="test_db",
    ) as postgres:
        # Wait for container to be ready
        postgres.get_connection_url()
        yield postgres


@pytest.fixture(scope="function")
async def real_db_engine(postgres_container) -> AsyncEngine:
    """Create real PostgreSQL engine for integration tests.

    Converts sync PostgreSQL URL to async format (postgresql+asyncpg).
    Creates all database tables before yielding to tests.

    Note: Function-scoped to work with pytest-asyncio event loop.
    The postgres_container is still module-scoped, so connection overhead is minimal.
    """
    from sqlalchemy.engine import make_url

    # Get sync connection URL and convert to async
    sync_url = postgres_container.get_connection_url()
    url_obj = make_url(sync_url)

    # Create async URL with asyncpg driver
    async_url = url_obj.set(drivername="postgresql+asyncpg")

    # Create engine with connection pool
    engine = create_async_engine(
        async_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )

    # Enable pgvector extension and create all tables
    async with engine.begin() as conn:
        # Enable pgvector extension (required for VECTOR columns)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create all tables without indexes (avoid GIN/JSON operator class issues)
        # For integration tests, we care about foreign keys and constraints, not indexes
        def create_tables_no_indexes(connection):
            """Create tables but skip index creation to avoid PostgreSQL-specific index issues."""
            # Temporarily remove all indexes from metadata
            all_indexes = []
            for table in Base.metadata.sorted_tables:
                all_indexes.extend(list(table.indexes))
                table.indexes.clear()

            # Create tables without indexes
            Base.metadata.create_all(connection)

            # Restore indexes to metadata (for other tests)
            for table in Base.metadata.sorted_tables:
                for idx in all_indexes:
                    if idx.table == table:
                        table.indexes.add(idx)

        await conn.run_sync(create_tables_no_indexes)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture(scope="function")
async def init_real_db(real_db_engine):
    """Initialize real database connection for test.

    Overrides global database connection to use real PostgreSQL.
    Cleans up test data after each test function.
    """
    from agentcore.a2a_protocol.database import connection

    # Override global connection with real one
    original_engine = connection.engine
    original_session = connection.SessionLocal

    connection.engine = real_db_engine
    connection.SessionLocal = async_sessionmaker(
        real_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    yield

    # Clean up test data (truncate all tables, keep schema)
    async with real_db_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            await conn.execute(table.delete())

    # Restore original connection
    connection.engine = original_engine
    connection.SessionLocal = original_session
