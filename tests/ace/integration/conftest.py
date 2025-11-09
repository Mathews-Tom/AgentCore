"""
ACE Integration Test Fixtures

Pytest fixtures for ACE integration tests.
Uses in-memory SQLite for fast, isolated tests.
"""

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from agentcore.a2a_protocol.database.connection import Base
# Import all models to register them with Base.metadata
from agentcore.a2a_protocol.database import models as a2a_models  # noqa: F401
from agentcore.ace.database import ace_orm  # noqa: F401


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create test database engine with SQLite.

    Uses in-memory SQLite for fast tests with ACE tables.
    """
    # Use in-memory SQLite for fast tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )

    # Create all tables (including ACE tables)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture(scope="function", autouse=True)
async def setup_database(test_db_engine):
    """Initialize database for ACE integration tests.

    Overrides global database connection with test database.
    Tables are already created by test_db_engine fixture.
    """
    from agentcore.a2a_protocol.database import connection

    # Save original connection (if any)
    original_engine = connection.engine
    original_session = connection.SessionLocal

    # Override global engine and session factory with test versions
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
