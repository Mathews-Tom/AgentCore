"""
Redis Streams Client

Async Redis Streams client wrapper with connection pooling and cluster support.
"""

from __future__ import annotations

import asyncio
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.connection import ConnectionPool

from .config import StreamConfig


class RedisStreamsClient:
    """
    Redis Streams client with connection pooling and retry logic.

    Supports both standalone Redis and Redis Cluster configurations.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        cluster_urls: list[str] | None = None,
        config: StreamConfig | None = None,
    ) -> None:
        """
        Initialize Redis Streams client.

        Args:
            redis_url: Redis connection URL (standalone mode)
            cluster_urls: List of Redis cluster node URLs
            config: Stream configuration
        """
        self.redis_url = redis_url or "redis://localhost:6379/0"
        self.cluster_urls = cluster_urls
        self.config = config or StreamConfig()
        self._client: aioredis.Redis[bytes] | RedisCluster[bytes] | None = None
        self._pool: ConnectionPool | None = None

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self.cluster_urls:
            # Use Redis Cluster
            self._client = RedisCluster(
                startup_nodes=[
                    aioredis.connection.parse_url(url) for url in self.cluster_urls
                ],
                decode_responses=False,
                skip_full_coverage_check=True,
            )
        else:
            # Use standalone Redis with connection pooling
            self._pool = ConnectionPool.from_url(
                self.redis_url,
                decode_responses=False,
                max_connections=10,
            )
            self._client = aioredis.Redis(connection_pool=self._pool)

        # Verify connection
        await self._client.ping()

    async def disconnect(self) -> None:
        """Close Redis connection and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._pool:
            await self._pool.aclose()
            self._pool = None

    async def __aenter__(self) -> RedisStreamsClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.disconnect()

    @property
    def client(self) -> aioredis.Redis[bytes] | RedisCluster[bytes]:
        """Get Redis client instance."""
        if not self._client:
            raise RuntimeError(
                "Redis client not connected. Call connect() or use async context manager."
            )
        return self._client

    async def create_consumer_group(
        self, stream_name: str, group_name: str, start_id: str = "0"
    ) -> bool:
        """
        Create consumer group for a stream.

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            start_id: Starting message ID (default: "0" for beginning)

        Returns:
            True if created, False if already exists
        """
        try:
            await self.client.xgroup_create(
                name=stream_name, groupname=group_name, id=start_id, mkstream=True
            )
            return True
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists
                return False
            raise

    async def delete_consumer_group(
        self, stream_name: str, group_name: str
    ) -> bool:
        """
        Delete consumer group.

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group

        Returns:
            True if deleted
        """
        result = await self.client.xgroup_destroy(name=stream_name, groupname=group_name)
        return bool(result)

    async def trim_stream(
        self, stream_name: str, max_length: int | None = None
    ) -> int:
        """
        Trim stream to maximum length.

        Args:
            stream_name: Name of the stream
            max_length: Maximum stream length (uses config default if None)

        Returns:
            Number of entries removed
        """
        max_len = max_length or self.config.max_stream_length
        result = await self.client.xtrim(name=stream_name, maxlen=max_len, approximate=True)
        return int(result)

    async def get_stream_length(self, stream_name: str) -> int:
        """
        Get current stream length.

        Args:
            stream_name: Name of the stream

        Returns:
            Number of entries in stream
        """
        result = await self.client.xlen(name=stream_name)
        return int(result)

    async def get_pending_count(
        self, stream_name: str, group_name: str
    ) -> int:
        """
        Get count of pending messages in consumer group.

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group

        Returns:
            Number of pending messages
        """
        pending_info = await self.client.xpending(name=stream_name, groupname=group_name)
        if pending_info and len(pending_info) > 0:
            return int(pending_info[0])
        return 0

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on Redis connection.

        Returns:
            Dictionary with health status information
        """
        try:
            latency_start = asyncio.get_event_loop().time()
            await self.client.ping()
            latency_ms = (asyncio.get_event_loop().time() - latency_start) * 1000

            info = await self.client.info()

            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "version": info.get("redis_version", "unknown"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
