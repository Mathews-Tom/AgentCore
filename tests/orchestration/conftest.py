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

    # Cleanup - clear all tables (excluding memory tables)
    async with test_db_engine.begin() as conn:
        memory_tables = ["memories", "stage_memories", "task_contexts", "error_records", "compression_metrics"]
        tables_to_create = [
            table for table in Base.metadata.sorted_tables
            if table.name not in memory_tables
        ]
        await conn.run_sync(lambda sync_conn: Base.metadata.drop_all(sync_conn, tables=tables_to_create))
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables_to_create))
