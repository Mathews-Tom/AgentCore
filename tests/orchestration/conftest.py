"""
Pytest configuration for orchestration tests.

Provides database initialization and session fixtures.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

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
    """Initialize test database for orchestration tests."""
    from agentcore.a2a_protocol.database import connection

    # Use test engine instead of production database
    async_session = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False)

    # Patch the SessionLocal to use test database
    with patch.object(connection, "SessionLocal", async_session):
        yield

    # Cleanup - clear all tables
    async with test_db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
