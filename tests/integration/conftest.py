"""
Integration Test Fixtures

Pytest fixtures for A2A protocol integration tests.
"""

import asyncio
from typing import AsyncGenerator

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from agentcore.a2a_protocol.main import create_app
from agentcore.a2a_protocol.database.connection import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


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


@pytest.fixture
async def test_db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_app():
    """Create test FastAPI application."""
    app = create_app()
    return app


@pytest.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_agent_card():
    """Sample AgentCard for testing."""
    return {
        "agent_id": "test-agent-001",
        "agent_name": "Test Agent",
        "agent_version": "1.0.0",
        "status": "active",
        "description": "Test agent for integration tests",
        "capabilities": [
            {
                "name": "text-generation",
                "version": "1.0.0",
                "description": "Generate text content"
            },
            {
                "name": "summarization",
                "version": "1.0.0",
                "description": "Summarize text content"
            }
        ],
        "endpoints": [
            {
                "url": "http://localhost:8080/api",
                "type": "https",
                "protocols": ["jsonrpc-2.0"]
            }
        ],
        "authentication": {
            "type": "jwt",
            "config": {
                "algorithm": "RS256",
                "public_key_url": "http://localhost:8080/.well-known/jwks.json"
            },
            "required": True
        }
    }


@pytest.fixture
def sample_task_definition():
    """Sample TaskDefinition for testing."""
    return {
        "task_id": "test-task-001",
        "task_type": "text.generation",
        "title": "Test Task",
        "description": "Test task for integration tests",
        "input_data": {
            "input": "Test input"
        },
        "parameters": {
            "max_tokens": 100
        },
        "requirements": {
            "required_capabilities": ["text-generation"]
        },
        "priority": "normal"
    }


@pytest.fixture
def jsonrpc_request_template():
    """Template for JSON-RPC 2.0 requests."""
    def make_request(method: str, params: dict = None, request_id: str = "1"):
        return {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": request_id
        }
    return make_request


@pytest.fixture(scope="function")
async def init_test_db(test_db_engine):
    """Initialize test database for session tests.

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