"""PostgreSQL database connector implementation.

Provides PostgreSQL-specific implementation with asyncpg for high-performance
async database operations and connection pooling.
"""

from __future__ import annotations

import time
from typing import Any

import asyncpg
import structlog

from agentcore.integration.database.base import (
    DatabaseConfig,
    DatabaseConnector,
    QueryResult,
)

logger = structlog.get_logger(__name__)


class PostgreSQLConnector(DatabaseConnector):
    """PostgreSQL database connector with asyncpg.

    High-performance async PostgreSQL connector using asyncpg driver
    with connection pooling, query caching integration, and SSL support.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize PostgreSQL connector.

        Args:
            config: Database connection configuration
        """
        super().__init__(config)
        self._pool: asyncpg.Pool | None = None
        self._cache: dict[str, tuple[QueryResult, float]] = {}

        logger.info(
            "postgresql_connector_initialized",
            host=config.host,
            port=config.port,
            database=config.database,
            pool_size=config.pool_size,
        )

    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        if self._connected:
            return

        try:
            # Build connection string
            dsn = (
                f"postgresql://{self.config.username}:"
                f"{self.config.password.get_secret_value()}@"
                f"{self.config.host}:{self.config.port}/"
                f"{self.config.database}"
            )

            # SSL configuration
            ssl_context = None
            if self.config.ssl_enabled:
                import ssl

                ssl_context = ssl.create_default_context()
                if not self.config.ssl_verify:
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=self.config.pool_size,
                timeout=self.config.pool_timeout,
                command_timeout=self.config.query_timeout,
                ssl=ssl_context,
            )

            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            self._connected = True

            logger.info(
                "postgresql_connected",
                host=self.config.host,
                database=self.config.database,
                pool_size=self.config.pool_size,
            )

        except Exception as e:
            logger.error(
                "postgresql_connection_failed",
                error=str(e),
                host=self.config.host,
                database=self.config.database,
            )
            raise

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._connected = False
            self._cache.clear()

            logger.info(
                "postgresql_disconnected",
                host=self.config.host,
                database=self.config.database,
            )

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute SELECT query with caching support.

        Args:
            query: SQL SELECT query
            params: Query parameters

        Returns:
            Query result with rows and metadata
        """
        if not self._connected or self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Check cache
        if self.config.cache_enabled:
            cache_key = self._hash_query(query, params)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

        # Execute query
        start_time = time.time()

        try:
            async with self._pool.acquire() as conn:
                # Convert named parameters to positional for asyncpg
                if params:
                    # Simple parameter substitution (for demo purposes)
                    # In production, use proper parameter binding
                    query_with_params = query
                    for key, value in params.items():
                        placeholder = f"${key}"
                        if placeholder in query:
                            query_with_params = query_with_params.replace(
                                placeholder, f"${list(params.keys()).index(key) + 1}"
                            )
                    rows = await conn.fetch(query_with_params, *params.values())
                else:
                    rows = await conn.fetch(query)

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Convert to dict list
            result_rows = [dict(row) for row in rows]
            columns = list(rows[0].keys()) if rows else []

            result = QueryResult(
                rows=result_rows,
                row_count=len(result_rows),
                columns=columns,
                execution_time_ms=execution_time_ms,
                cached=False,
                query_hash=cache_key if self.config.cache_enabled else None,
            )

            # Store in cache
            if self.config.cache_enabled:
                self._store_in_cache(cache_key, result)

            logger.debug(
                "postgresql_query_executed",
                row_count=len(result_rows),
                execution_time_ms=execution_time_ms,
                cached=False,
            )

            return result

        except Exception as e:
            logger.error(
                "postgresql_query_failed",
                error=str(e),
                query=query[:100],  # Log first 100 chars
            )
            raise

    async def execute_update(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Execute INSERT/UPDATE/DELETE query.

        Args:
            query: SQL modification query
            params: Query parameters

        Returns:
            Number of rows affected
        """
        if not self._connected or self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        try:
            async with self._pool.acquire() as conn:
                if params:
                    query_with_params = query
                    for key, value in params.items():
                        placeholder = f"${key}"
                        if placeholder in query:
                            query_with_params = query_with_params.replace(
                                placeholder, f"${list(params.keys()).index(key) + 1}"
                            )
                    result = await conn.execute(query_with_params, *params.values())
                else:
                    result = await conn.execute(query)

            # Parse result (e.g., "UPDATE 5")
            affected_rows = int(result.split()[-1]) if result.split() else 0

            # Invalidate cache on updates
            if self.config.cache_enabled:
                self._cache.clear()

            logger.debug(
                "postgresql_update_executed",
                affected_rows=affected_rows,
                query_type=query.split()[0].upper(),
            )

            return affected_rows

        except Exception as e:
            logger.error(
                "postgresql_update_failed",
                error=str(e),
                query=query[:100],
            )
            raise

    async def execute_transaction(
        self,
        queries: list[tuple[str, dict[str, Any] | None]],
    ) -> list[int]:
        """Execute multiple queries in a transaction.

        Args:
            queries: List of (query, params) tuples

        Returns:
            List of affected row counts
        """
        if not self._connected or self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        results: list[int] = []

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for query, params in queries:
                        if params:
                            query_with_params = query
                            for key, value in params.items():
                                placeholder = f"${key}"
                                if placeholder in query:
                                    query_with_params = query_with_params.replace(
                                        placeholder,
                                        f"${list(params.keys()).index(key) + 1}",
                                    )
                            result = await conn.execute(
                                query_with_params, *params.values()
                            )
                        else:
                            result = await conn.execute(query)

                        affected = int(result.split()[-1]) if result.split() else 0
                        results.append(affected)

            # Invalidate cache after transaction
            if self.config.cache_enabled:
                self._cache.clear()

            logger.info(
                "postgresql_transaction_completed",
                query_count=len(queries),
                total_affected=sum(results),
            )

            return results

        except Exception as e:
            logger.error(
                "postgresql_transaction_failed",
                error=str(e),
                query_count=len(queries),
            )
            raise

    async def health_check(self) -> bool:
        """Check PostgreSQL connection health.

        Returns:
            True if connection is healthy
        """
        if not self._connected or self._pool is None:
            return False

        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1

        except Exception as e:
            logger.warning(
                "postgresql_health_check_failed",
                error=str(e),
            )
            return False

    def _get_from_cache(self, cache_key: str) -> QueryResult | None:
        """Retrieve query result from cache.

        Args:
            cache_key: Cache key hash

        Returns:
            Cached result if valid, None otherwise
        """
        if cache_key not in self._cache:
            return None

        result, timestamp = self._cache[cache_key]

        # Check if expired
        if time.time() - timestamp > self.config.cache_ttl:
            del self._cache[cache_key]
            return None

        # Mark as cached
        result.cached = True

        logger.debug(
            "postgresql_cache_hit",
            cache_key=cache_key[:16],
            age_seconds=int(time.time() - timestamp),
        )

        return result

    def _store_in_cache(self, cache_key: str, result: QueryResult) -> None:
        """Store query result in cache.

        Args:
            cache_key: Cache key hash
            result: Query result to cache
        """
        self._cache[cache_key] = (result, time.time())

        logger.debug(
            "postgresql_cache_stored",
            cache_key=cache_key[:16],
            row_count=result.row_count,
        )
