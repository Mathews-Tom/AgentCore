"""Integration test fixtures for training module."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentcore.a2a_protocol.database.connection import Base


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create test database engine."""
    # Use in-memory SQLite for fast tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False)

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest.fixture(scope="function")
async def init_test_db(test_db_engine):
    """Initialize test database for training integration tests.

    Uses in-memory SQLite instead of PostgreSQL for fast, isolated tests.
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
