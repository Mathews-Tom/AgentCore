"""
Integration test fixtures for orchestration engine.

Provides database setup, Redis containers, and test data factories.
"""

from __future__ import annotations

import warnings
from typing import AsyncGenerator, Callable

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine)
from sqlalchemy.pool import StaticPool
from testcontainers.redis import RedisContainer

from agentcore.a2a_protocol.database.connection import Base
from agentcore.orchestration.state.models import (
    WorkflowExecutionDB,
    WorkflowStateDB,
    WorkflowStateVersion)
from agentcore.orchestration.streams.client import RedisStreamsClient
from agentcore.orchestration.streams.config import StreamConfig


@pytest.fixture
async def test_db_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create test database engine with in-memory SQLite.

    Uses StaticPool with shared cache to allow multiple connections,
    but drops/creates tables for each test to avoid conflicts.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False})

    # Drop and create tables excluding memory models (they use PostgreSQL-specific types)
    # Memory models use ARRAY(Float) which is incompatible with SQLite
    async with engine.begin() as conn:
        # Get tables that are SQLite-compatible (exclude memory tables)
        memory_tables = ["memories", "stage_memories", "task_contexts", "error_records", "compression_metrics"]
        tables_to_create = [
            table for table in Base.metadata.sorted_tables
            if table.name not in memory_tables
        ]
        await conn.run_sync(lambda sync_conn: Base.metadata.drop_all(sync_conn, tables=tables_to_create))
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables_to_create))

    yield engine

    # Clean up
    await engine.dispose()


@pytest.fixture
async def db_session_factory(
    test_db_engine: AsyncEngine) -> Callable[[], AsyncSession]:
    """
    Create database session factory for tests.

    Returns a callable that creates async context managers for database sessions.
    """
    session_factory = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False)
    return session_factory


@pytest.fixture
async def db_session(
    db_session_factory: Callable[[], AsyncSession]) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for individual tests."""
    async with db_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture(scope="module")
def redis_container() -> RedisContainer:
    """Provide Redis container for integration tests."""
    # Suppress testcontainers internal deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The @wait_container_is_ready decorator is deprecated",
            category=DeprecationWarning)
        container = RedisContainer("redis:7-alpine")
        with container:
            yield container


@pytest.fixture
async def redis_client(
    redis_container: RedisContainer) -> AsyncGenerator[RedisStreamsClient, None]:
    """Provide Redis Streams client connected to test container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    redis_url = f"redis://{host}:{port}/0"

    config = StreamConfig(
        stream_name="test:events",
        consumer_group_name="test-group",
        dead_letter_stream="test:events:dlq")

    client = RedisStreamsClient(redis_url=redis_url, config=config)
    await client.connect()
    yield client
    await client.disconnect()


@pytest.fixture
def test_stream_config() -> StreamConfig:
    """Create test stream configuration."""
    return StreamConfig(
        stream_name="test:orchestration:events",
        consumer_group_name="test-orchestrator-group",
        consumer_name="test-orchestrator-consumer",
        dead_letter_stream="test:orchestration:events:dlq",
        max_stream_length=1000,
        count=10,
        block_ms=100,
        max_retries=3,
        retry_backoff_ms=50,
        enable_auto_claim=True,
        auto_claim_idle_ms=5000)
