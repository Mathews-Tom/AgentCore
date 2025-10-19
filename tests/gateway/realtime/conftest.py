"""Realtime tests configuration with Redis container setup."""

from __future__ import annotations

import asyncio
import os
import time

import pytest
import pytest_asyncio
from testcontainers.redis import RedisContainer


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for WebSocket testing."""
    container = RedisContainer("redis:7-alpine")

    try:
        container.start()

        # Wait for Redis to be ready
        max_retries = 30
        retry_count = 0

        while retry_count < max_retries:
            try:
                port = container.get_exposed_port(6379)
                # Set Redis URL environment variables BEFORE any gateway imports
                redis_url = f"redis://localhost:{port}/0"
                os.environ["GATEWAY_RATE_LIMIT_REDIS_URL"] = redis_url
                os.environ["GATEWAY_SESSION_REDIS_URL"] = redis_url

                # Test connection
                import redis
                client = redis.from_url(redis_url)
                client.ping()
                client.close()

                break
            except Exception:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                time.sleep(0.5)

        yield container

    finally:
        try:
            container.stop()
        except Exception:
            pass  # Ignore errors during cleanup


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
