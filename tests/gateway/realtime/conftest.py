"""Realtime tests configuration with Redis container setup."""

from __future__ import annotations

import asyncio
import os

import pytest
import pytest_asyncio
from testcontainers.redis import RedisContainer


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for WebSocket testing."""
    container = RedisContainer("redis:7-alpine")
    container.start()

    # Set Redis URL environment variables BEFORE any gateway imports
    redis_url = f"redis://localhost:{container.get_exposed_port(6379)}/0"
    os.environ["GATEWAY_RATE_LIMIT_REDIS_URL"] = redis_url
    os.environ["GATEWAY_SESSION_REDIS_URL"] = redis_url

    yield container

    container.stop()


@pytest_asyncio.fixture(scope="module")
def event_loop():
    """Create module-scoped event loop for async fixtures."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
