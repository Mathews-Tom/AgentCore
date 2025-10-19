"""
Connection Pool Manager

High-performance HTTP connection pooling with keep-alive for backend services.
Optimized for 60,000+ req/sec throughput.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration."""

    max_connections: int = 1000
    """Maximum total connections across all hosts"""

    max_keepalive_connections: int = 500
    """Maximum keep-alive connections"""

    keepalive_expiry: float = 30.0
    """Keep-alive timeout in seconds"""

    connect_timeout: float = 5.0
    """Connection timeout in seconds"""

    read_timeout: float = 30.0
    """Read timeout in seconds"""

    write_timeout: float = 30.0
    """Write timeout in seconds"""

    pool_timeout: float = 5.0
    """Pool timeout when waiting for available connection"""

    http2: bool = True
    """Enable HTTP/2 support"""

    retries: int = 3
    """Number of retry attempts"""


class ConnectionPool:
    """
    Base connection pool for managing connections.

    Provides connection reuse and lifecycle management.
    """

    def __init__(self, config: ConnectionPoolConfig | None = None):
        """
        Initialize connection pool.

        Args:
            config: Connection pool configuration
        """
        self.config = config or ConnectionPoolConfig()
        self._active_connections = 0
        self._total_requests = 0
        self._failed_requests = 0
        self._pool_lock = asyncio.Lock()

    @property
    def active_connections(self) -> int:
        """Get number of active connections."""
        return self._active_connections

    @property
    def total_requests(self) -> int:
        """Get total requests processed."""
        return self._total_requests

    @property
    def failed_requests(self) -> int:
        """Get total failed requests."""
        return self._failed_requests

    def get_stats(self) -> dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "active_connections": self._active_connections,
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                (self._total_requests - self._failed_requests) / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
            "config": {
                "max_connections": self.config.max_connections,
                "max_keepalive_connections": self.config.max_keepalive_connections,
                "keepalive_expiry": self.config.keepalive_expiry,
            },
        }


class HTTPConnectionPool(ConnectionPool):
    """
    HTTP/HTTPS connection pool using httpx with keep-alive.

    Optimized for high-throughput scenarios with connection reuse.
    """

    def __init__(self, config: ConnectionPoolConfig | None = None):
        """
        Initialize HTTP connection pool.

        Args:
            config: Connection pool configuration
        """
        super().__init__(config)

        # Create httpx limits
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
            keepalive_expiry=self.config.keepalive_expiry,
        )

        # Create timeout config
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.read_timeout,
            write=self.config.write_timeout,
            pool=self.config.pool_timeout,
        )

        # Create async client with pooling
        self._client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=self.config.http2,
            follow_redirects=False,  # Handle redirects manually
        )

        logger.info(
            "HTTP connection pool initialized",
            max_connections=self.config.max_connections,
            max_keepalive=self.config.max_keepalive_connections,
            http2=self.config.http2,
        )

    async def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make HTTP request using connection pool.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            content: Request body content
            **kwargs: Additional httpx request arguments

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: On request failure
        """
        self._total_requests += 1

        async with self._pool_lock:
            self._active_connections += 1

        try:
            response = await self._client.request(
                method=method,
                url=url,
                headers=headers,
                content=content,
                **kwargs,
            )

            logger.debug(
                "HTTP request completed",
                method=method,
                url=url,
                status_code=response.status_code,
            )

            return response

        except httpx.HTTPError as e:
            self._failed_requests += 1
            logger.warning(
                "HTTP request failed",
                method=method,
                url=url,
                error=str(e),
            )
            raise

        finally:
            async with self._pool_lock:
                self._active_connections -= 1

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """
        GET request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        return await self.request("GET", url, **kwargs)

    async def post(
        self,
        url: str,
        content: bytes | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        POST request.

        Args:
            url: Request URL
            content: Request body content
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        return await self.request("POST", url, content=content, **kwargs)

    async def put(
        self,
        url: str,
        content: bytes | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        PUT request.

        Args:
            url: Request URL
            content: Request body content
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        return await self.request("PUT", url, content=content, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """
        DELETE request.

        Args:
            url: Request URL
            **kwargs: Additional request arguments

        Returns:
            HTTP response
        """
        return await self.request("DELETE", url, **kwargs)

    async def close(self) -> None:
        """Close all connections in pool."""
        await self._client.aclose()
        logger.info("HTTP connection pool closed")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            asyncio.create_task(self.close())
        except RuntimeError:
            # Event loop may be closed
            pass
