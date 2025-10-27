"""Integration test fixtures for training module."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="function")
async def init_test_db():
    """Initialize test database for training integration tests."""
    from agentcore.a2a_protocol.database import init_db, close_db

    # Initialize database connection
    await init_db()

    yield

    # Cleanup
    await close_db()
