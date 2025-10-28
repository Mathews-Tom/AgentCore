"""
Unit tests for Redis Streams Client.

Tests connection management, consumer group operations, and error handling
using mocks instead of real Redis connections.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis.asyncio as aioredis

from agentcore.orchestration.streams.client import RedisStreamsClient
from agentcore.orchestration.streams.config import StreamConfig


class TestRedisStreamsClient:
    """Test suite for RedisStreamsClient."""

    @pytest.fixture
    def config(self) -> StreamConfig:
        """Create test configuration."""
        return StreamConfig(
            stream_name="test:events",
            consumer_group_name="test-group")

    @pytest.fixture
    def mock_redis(self) -> AsyncMock:
        """Create mock Redis client."""
        mock = AsyncMock(spec=aioredis.Redis)
        mock.ping = AsyncMock(return_value=True)
        mock.aclose = AsyncMock()
        return mock

    @pytest.fixture
    def mock_pool(self) -> AsyncMock:
        """Create mock connection pool."""
        mock = AsyncMock()
        mock.aclose = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_connect_standalone(self, config: StreamConfig) -> None:
        """Test connecting to standalone Redis."""
        with patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ) as mock_pool_factory, patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory:
            mock_pool = AsyncMock()
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)

            mock_pool_factory.return_value = mock_pool
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(
                redis_url="redis://localhost:6379/0", config=config
            )
            await client.connect()

            # Verify connection pool was created
            mock_pool_factory.assert_called_once()
            mock_redis_factory.assert_called_once()
            mock_redis.ping.assert_called_once()

            await client.disconnect()

    @pytest.mark.asyncio
    async def test_connect_cluster(self, config: StreamConfig) -> None:
        """Test connecting to Redis Cluster."""
        with patch(
            "agentcore.orchestration.streams.client.RedisCluster"
        ) as mock_cluster:
            mock_instance = AsyncMock()
            mock_instance.ping = AsyncMock(return_value=True)
            mock_cluster.return_value = mock_instance

            client = RedisStreamsClient(
                cluster_urls=["redis://node1:6379", "redis://node2:6379"],
                config=config)
            await client.connect()

            # Verify cluster connection
            mock_cluster.assert_called_once()
            mock_instance.ping.assert_called_once()

            await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, config: StreamConfig, mock_redis: AsyncMock) -> None:
        """Test disconnecting from Redis."""
        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ) as mock_pool_factory:
            mock_pool = AsyncMock()
            mock_pool.aclose = AsyncMock()
            mock_redis.aclose = AsyncMock()

            mock_pool_factory.return_value = mock_pool
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()
            await client.disconnect()

            # Verify cleanup
            mock_redis.aclose.assert_called_once()
            mock_pool.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, config: StreamConfig) -> None:
        """Test async context manager."""
        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ) as mock_pool_factory:
            mock_redis = AsyncMock()
            mock_redis.ping = AsyncMock(return_value=True)
            mock_redis.aclose = AsyncMock()
            mock_redis_factory.return_value = mock_redis

            mock_pool = AsyncMock()
            mock_pool.aclose = AsyncMock()
            mock_pool_factory.return_value = mock_pool

            async with RedisStreamsClient(config=config) as client:
                assert client._client is not None

            # Verify disconnect was called
            mock_redis.aclose.assert_called_once()
            mock_pool.aclose.assert_called_once()

    def test_client_property_not_connected(self, config: StreamConfig) -> None:
        """Test accessing client before connection raises error."""
        client = RedisStreamsClient(config=config)

        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.client

    @pytest.mark.asyncio
    async def test_create_consumer_group(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test creating consumer group."""
        mock_redis.xgroup_create = AsyncMock()

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            result = await client.create_consumer_group("test:stream", "test-group")

            assert result is True
            mock_redis.xgroup_create.assert_called_once_with(
                name="test:stream", groupname="test-group", id="0", mkstream=True
            )

    @pytest.mark.asyncio
    async def test_create_consumer_group_already_exists(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test creating consumer group that already exists."""
        mock_redis.xgroup_create = AsyncMock(
            side_effect=aioredis.ResponseError("BUSYGROUP Consumer Group name already exists")
        )

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            result = await client.create_consumer_group("test:stream", "test-group")

            assert result is False

    @pytest.mark.asyncio
    async def test_create_consumer_group_error(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test creating consumer group with unexpected error."""
        mock_redis.xgroup_create = AsyncMock(
            side_effect=aioredis.ResponseError("UNEXPECTED")
        )

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            with pytest.raises(aioredis.ResponseError):
                await client.create_consumer_group("test:stream", "test-group")

    @pytest.mark.asyncio
    async def test_delete_consumer_group(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test deleting consumer group."""
        mock_redis.xgroup_destroy = AsyncMock(return_value=1)

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            result = await client.delete_consumer_group("test:stream", "test-group")

            assert result is True
            mock_redis.xgroup_destroy.assert_called_once_with(
                name="test:stream", groupname="test-group"
            )

    @pytest.mark.asyncio
    async def test_trim_stream(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test trimming stream."""
        mock_redis.xtrim = AsyncMock(return_value=50)

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            result = await client.trim_stream("test:stream", max_length=100)

            assert result == 50
            mock_redis.xtrim.assert_called_once_with(
                name="test:stream", maxlen=100, approximate=True
            )

    @pytest.mark.asyncio
    async def test_get_stream_length(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test getting stream length."""
        mock_redis.xlen = AsyncMock(return_value=150)

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            result = await client.get_stream_length("test:stream")

            assert result == 150
            mock_redis.xlen.assert_called_once_with(name="test:stream")

    @pytest.mark.asyncio
    async def test_get_pending_count(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test getting pending message count."""
        mock_redis.xpending = AsyncMock(return_value=[10, "msg-1", "msg-10", []])

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            result = await client.get_pending_count("test:stream", "test-group")

            assert result == 10
            mock_redis.xpending.assert_called_once_with(
                name="test:stream", groupname="test-group"
            )

    @pytest.mark.asyncio
    async def test_get_pending_count_empty(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test getting pending count when no pending messages."""
        mock_redis.xpending = AsyncMock(return_value=[])

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            result = await client.get_pending_count("test:stream", "test-group")

            assert result == 0

    @pytest.mark.asyncio
    async def test_health_check_healthy(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test health check when Redis is healthy."""
        mock_redis.info = AsyncMock(
            return_value={
                "connected_clients": 5,
                "used_memory_human": "1.5M",
                "redis_version": "7.0.0",
            }
        )

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            health = await client.health_check()

            assert health["status"] == "healthy"
            assert "latency_ms" in health
            assert health["connected_clients"] == 5
            assert health["used_memory_human"] == "1.5M"
            assert health["version"] == "7.0.0"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(
        self, config: StreamConfig, mock_redis: AsyncMock
    ) -> None:
        """Test health check when Redis is unhealthy."""
        # First allow connect to succeed
        mock_redis.ping = AsyncMock(return_value=True)

        with patch(
            "agentcore.orchestration.streams.client.aioredis.Redis"
        ) as mock_redis_factory, patch(
            "agentcore.orchestration.streams.client.ConnectionPool.from_url"
        ):
            mock_redis_factory.return_value = mock_redis

            client = RedisStreamsClient(config=config)
            await client.connect()

            # Now make ping fail for health check
            mock_redis.ping = AsyncMock(side_effect=Exception("Connection failed"))

            health = await client.health_check()

            assert health["status"] == "unhealthy"
            assert "error" in health
