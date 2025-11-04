"""Real Redis fixtures using testcontainers."""

from __future__ import annotations

import pytest
import redis.asyncio as aioredis
from testcontainers.redis import RedisContainer


@pytest.fixture(scope="module")
def redis_container():
    """Start Redis container for integration tests.

    Uses redis:7-alpine image to match production environment.
    Container is scoped to module for performance (reused across tests).
    """
    with RedisContainer(image="redis:7-alpine") as redis:
        # Wait for container to be ready
        redis.get_connection_url()
        yield redis


@pytest.fixture(scope="module")
async def real_redis_client(redis_container):
    """Create real Redis client for integration tests.

    Provides a connected Redis client for testing Redis-specific behavior.
    Automatically flushes all data after module completion.
    """
    redis_url = redis_container.get_connection_url()

    client = await aioredis.from_url(
        redis_url,
        decode_responses=False,
        socket_connect_timeout=5,
    )

    yield client

    # Cleanup
    await client.flushall()
    await client.aclose()


@pytest.fixture(scope="function")
async def real_cache_service(real_redis_client):
    """Create cache service with real Redis for integration tests.

    Patches the cache service to use the real Redis client instead of fakeredis.
    Clears cache after each test function.
    """
    from unittest.mock import patch

    from agentcore.llm_gateway.cache_models import (
        CacheConfig,
        CacheMode,
        EvictionPolicy,
    )
    from agentcore.llm_gateway.cache_service import CacheService

    config = CacheConfig(
        enabled=True,
        l1_enabled=True,
        l1_max_size=100,
        l1_ttl_seconds=3600,
        l1_eviction_policy=EvictionPolicy.LRU,
        l2_enabled=True,
        l2_ttl_seconds=86400,
        mode=CacheMode.EXACT,
        stats_enabled=True,
    )

    # Patch to use our real Redis client
    with patch("redis.asyncio.from_url", return_value=real_redis_client):
        service = CacheService(config=config)
        await service.connect()

        yield service

        await service.clear()
        await service.close()
