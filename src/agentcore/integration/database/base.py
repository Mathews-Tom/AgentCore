"""Base database connector interface and models.

Defines the abstract interface for all database connectors with
connection pooling, query execution, and security features.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, SecretStr


class DatabaseConfig(BaseModel):
    """Database connection configuration.

    Encapsulates connection parameters with secure credential handling.
    """

    type: str = Field(
        description="Database type (postgresql, mysql, oracle, etc.)",
    )
    host: str = Field(
        description="Database host address",
    )
    port: int = Field(
        description="Database port",
        ge=1,
        le=65535,
    )
    database: str = Field(
        description="Database name",
    )
    username: str = Field(
        description="Database username",
    )
    password: SecretStr = Field(
        description="Database password (encrypted in storage)",
    )
    pool_size: int = Field(
        default=10,
        description="Connection pool size",
        ge=1,
        le=100,
    )
    pool_timeout: int = Field(
        default=30,
        description="Connection pool timeout in seconds",
        ge=1,
        le=300,
    )
    ssl_enabled: bool = Field(
        default=True,
        description="Enable SSL/TLS encryption",
    )
    ssl_verify: bool = Field(
        default=True,
        description="Verify SSL certificate",
    )
    query_timeout: int = Field(
        default=30,
        description="Query execution timeout in seconds",
        ge=1,
        le=3600,
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable query result caching",
    )
    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds",
        ge=60,
        le=86400,
    )


class QueryResult(BaseModel):
    """Database query execution result.

    Encapsulates query results with metadata for monitoring and caching.
    """

    rows: list[dict[str, Any]] = Field(
        description="Result rows as dictionaries",
    )
    row_count: int = Field(
        description="Number of rows returned",
        ge=0,
    )
    columns: list[str] = Field(
        description="Column names in result set",
    )
    execution_time_ms: int = Field(
        description="Query execution time in milliseconds",
        ge=0,
    )
    cached: bool = Field(
        default=False,
        description="Whether result was served from cache",
    )
    query_hash: str | None = Field(
        default=None,
        description="Hash of executed query for caching",
    )


class DatabaseConnector(ABC):
    """Abstract base class for database connectors.

    Defines the interface that all database-specific connectors must implement.
    Provides connection pooling, query execution, and caching integration.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize database connector.

        Args:
            config: Database connection configuration
        """
        self.config = config
        self._pool: Any = None
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection pool.

        Must be called before executing queries.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection pool.

        Gracefully closes all connections in the pool.
        """
        ...

    @abstractmethod
    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)

        Returns:
            Query execution result with rows and metadata

        Raises:
            DatabaseError: If query execution fails
            TimeoutError: If query exceeds timeout
        """
        ...

    @abstractmethod
    async def execute_update(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> int:
        """Execute an INSERT/UPDATE/DELETE query.

        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)

        Returns:
            Number of rows affected

        Raises:
            DatabaseError: If query execution fails
            TimeoutError: If query exceeds timeout
        """
        ...

    @abstractmethod
    async def execute_transaction(
        self,
        queries: list[tuple[str, dict[str, Any] | None]],
    ) -> list[int]:
        """Execute multiple queries in a transaction.

        Args:
            queries: List of (query, params) tuples

        Returns:
            List of row counts for each query

        Raises:
            DatabaseError: If transaction fails (rolls back)
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if database connection is healthy.

        Returns:
            True if connection is active and responsive
        """
        ...

    def is_connected(self) -> bool:
        """Check if connector is connected.

        Returns:
            True if connection pool is active
        """
        return self._connected

    def _hash_query(self, query: str, params: dict[str, Any] | None) -> str:
        """Generate hash of query and parameters for caching.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            MD5 hash of normalized query
        """
        import hashlib
        import json

        # Normalize query (remove extra whitespace)
        normalized = " ".join(query.split())

        # Include parameters in hash
        params_str = json.dumps(params or {}, sort_keys=True)

        # Generate hash
        content = f"{normalized}:{params_str}"
        return hashlib.md5(content.encode()).hexdigest()
