"""Integration test fixtures for training module.

Provides fast SQLite fixtures for default testing.
Real PostgreSQL fixtures are auto-discovered from tests/integration/fixtures/
when tests are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from agentcore.a2a_protocol.database.connection import Base

# Import testcontainer fixtures for integration tests
# These fixtures are only used when tests are marked with @pytest.mark.integration
from tests.integration.fixtures.database import (
    postgres_container,
    real_db_engine,
    init_real_db,
)
from tests.integration.fixtures.cache import (
    redis_container,
    real_redis_client,
    real_cache_service,
)


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create test database engine (fast mode with SQLite).

    This is the default fixture for fast integration tests.
    Uses in-memory SQLite for rapid feedback without Docker.
    """
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
async def init_test_db_fast(test_db_engine):
    """Initialize test database for training integration tests (fast mode).

    Uses in-memory SQLite for rapid feedback without Docker.
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


@pytest.fixture(scope="function")
async def init_test_db(init_test_db_fast):
    """Initialize test database for training integration tests (fast mode with SQLite).

    This is the default fixture using SQLite for fast feedback.

    Tests that need real PostgreSQL should:
    1. Mark with @pytest.mark.integration
    2. Use init_real_db fixture directly (auto-discovered from tests/integration/fixtures/database.py)
    """
    return init_test_db_fast
