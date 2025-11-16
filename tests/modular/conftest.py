"""
Pytest fixtures for modular agent core tests.
"""

from typing import AsyncGenerator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from agentcore.a2a_protocol.database.connection import Base
# Import models to ensure they're registered with Base metadata
from agentcore.a2a_protocol.database import models  # noqa: F401


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create test database engine with in-memory SQLite."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Create tables excluding memory models (they use PostgreSQL-specific types)
    # Memory models use ARRAY(Float) which is incompatible with SQLite
    async with engine.begin() as conn:
        # Get tables that are SQLite-compatible (exclude memory tables)
        memory_tables = ["memories", "stage_memories", "task_contexts", "error_records", "compression_metrics"]
        tables_to_create = [
            table for table in Base.metadata.sorted_tables
            if table.name not in memory_tables
        ]
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables_to_create))

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture
async def async_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for tests."""
    async_session_maker = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()
