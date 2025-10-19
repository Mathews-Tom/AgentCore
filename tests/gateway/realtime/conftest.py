"""Realtime tests configuration."""

from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio


@pytest.fixture(scope="module")
def realtime_redis_db(redis_container):
    """
    Configure Redis database for realtime tests.

    Uses database 1 to avoid conflicts with other test modules.
    The session-scoped redis_container fixture is provided by tests/gateway/conftest.py.
    """
    port = redis_container.get_exposed_port(6379)
    # Use database 1 for realtime tests
    redis_url = f"redis://localhost:{port}/1"

    # Temporarily override environment variables for this module
    old_rate_limit_url = os.environ.get("GATEWAY_RATE_LIMIT_REDIS_URL")
    old_session_url = os.environ.get("GATEWAY_SESSION_REDIS_URL")

    os.environ["GATEWAY_RATE_LIMIT_REDIS_URL"] = redis_url
    os.environ["GATEWAY_SESSION_REDIS_URL"] = redis_url

    yield redis_container

    # Restore original values
    if old_rate_limit_url:
        os.environ["GATEWAY_RATE_LIMIT_REDIS_URL"] = old_rate_limit_url
    if old_session_url:
        os.environ["GATEWAY_SESSION_REDIS_URL"] = old_session_url


@pytest.fixture
def authenticated_token(redis_container):
    """Create authenticated JWT token for testing (sync fixture)."""
    # Import AFTER Redis container is configured
    from gateway.auth.jwt import jwt_manager
    from gateway.auth.models import User, UserRole

    async def create_token():
        # Initialize JWT manager
        await jwt_manager.initialize()

        user = User(
            id="test-user-123",
            username="testuser",
            email="test@example.com",
            roles=[UserRole.USER],
        )

        token = jwt_manager.create_access_token(
            user=user,
            session_id="test-session",
        )

        return token

    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    try:
        token = loop.run_until_complete(create_token())
        return token
    finally:
        loop.close()
