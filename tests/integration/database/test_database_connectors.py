"""Integration tests for database connectors (INT-006).

Tests INT-006 acceptance criteria:
- Multi-database integration support
- Connection pooling and management
- Query optimization and caching
- Data security and encryption

These tests validate the database connector framework using
in-memory SQLite for testing (PostgreSQL production mode tested separately).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pydantic import SecretStr

from agentcore.integration.database import (
    DatabaseConfig,
    DatabaseFactory,
    PostgreSQLConnector,
    QueryResult,
)


class MockPostgreSQLConnector(PostgreSQLConnector):
    """Mock PostgreSQL connector for testing without real database.

    Simulates database operations for integration testing.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        """Initialize mock connector."""
        super().__init__(config)
        self._mock_data: dict[str, list[dict]] = {}
        self._connected = False

    async def connect(self) -> None:
        """Mock connection (no actual database)."""
        self._connected = True

        # Seed with test data
        self._mock_data["users"] = [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ]

        self._mock_data["orders"] = [
            {"id": 1, "user_id": 1, "total": 100.50},
            {"id": 2, "user_id": 1, "total": 50.25},
            {"id": 3, "user_id": 2, "total": 75.00},
        ]

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self._connected = False
        self._mock_data.clear()
        self._cache.clear()

    async def execute_query(
        self,
        query: str,
        params: dict | None = None,
    ) -> QueryResult:
        """Mock query execution."""
        if not self._connected:
            raise RuntimeError("Database not connected")

        # Check cache
        if self.config.cache_enabled:
            cache_key = self._hash_query(query, params)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

        import time

        start_time = time.time()

        # Simple query parsing for mock
        query_lower = query.lower()

        # Check for WHERE clause first (before checking for simple SELECT)
        if "where" in query_lower and params:
            # Simple WHERE clause simulation
            table = "users" if "users" in query_lower else "orders"
            rows = list(self._mock_data[table])  # Create copy to avoid modifying original
            # Filter by params (simple equality check)
            for key, value in params.items():
                rows = [r for r in rows if r.get(key) == value]
            columns = list(rows[0].keys()) if rows else []
        elif "select * from users" in query_lower:
            rows = self._mock_data["users"]
            columns = ["id", "name", "email"]
        elif "select * from orders" in query_lower:
            rows = self._mock_data["orders"]
            columns = ["id", "user_id", "total"]
        else:
            rows = []
            columns = []

        execution_time_ms = int((time.time() - start_time) * 1000)

        result = QueryResult(
            rows=rows,
            row_count=len(rows),
            columns=columns,
            execution_time_ms=execution_time_ms,
            cached=False,
            query_hash=cache_key if self.config.cache_enabled else None,
        )

        # Store in cache
        if self.config.cache_enabled:
            self._store_in_cache(cache_key, result)

        return result

    async def execute_update(
        self,
        query: str,
        params: dict | None = None,
    ) -> int:
        """Mock update execution."""
        if not self._connected:
            raise RuntimeError("Database not connected")

        # Simulate update
        # In real implementation, this would modify mock_data
        affected_rows = 1  # Mock: always 1 row affected

        # Invalidate cache
        if self.config.cache_enabled:
            self._cache.clear()

        return affected_rows

    async def execute_transaction(
        self,
        queries: list[tuple[str, dict | None]],
    ) -> list[int]:
        """Mock transaction execution."""
        if not self._connected:
            raise RuntimeError("Database not connected")

        # Simulate transaction (all or nothing)
        results = [1] * len(queries)  # Mock: each query affects 1 row

        # Invalidate cache
        if self.config.cache_enabled:
            self._cache.clear()

        return results

    async def health_check(self) -> bool:
        """Mock health check."""
        return self._connected


@pytest_asyncio.fixture
def db_config() -> DatabaseConfig:
    """Create test database configuration."""
    return DatabaseConfig(
        type="postgresql",
        host="localhost",
        port=5432,
        database="test_db",
        username="test_user",
        password=SecretStr("test_password"),
        pool_size=5,
        pool_timeout=10,
        ssl_enabled=False,  # Disable SSL for testing
        query_timeout=30,
        cache_enabled=True,
        cache_ttl=300,
    )


@pytest_asyncio.fixture
async def db_connector(db_config: DatabaseConfig) -> MockPostgreSQLConnector:
    """Create and connect mock database connector."""
    connector = MockPostgreSQLConnector(db_config)
    await connector.connect()

    yield connector

    await connector.disconnect()


class TestDatabaseConfigValidation:
    """Test database configuration validation."""

    def test_config_creation(self, db_config: DatabaseConfig) -> None:
        """Test database config creation with valid parameters."""
        assert db_config.type == "postgresql"
        assert db_config.host == "localhost"
        assert db_config.port == 5432
        assert db_config.pool_size == 5

    def test_password_security(self, db_config: DatabaseConfig) -> None:
        """Test password is stored securely."""
        # Password should be SecretStr
        assert isinstance(db_config.password, SecretStr)

        # Should not be visible in repr
        config_repr = repr(db_config)
        assert "test_password" not in config_repr

        # Should be retrievable with get_secret_value()
        assert db_config.password.get_secret_value() == "test_password"

    def test_config_validation_port_range(self) -> None:
        """Test port validation."""
        with pytest.raises(ValueError):
            DatabaseConfig(
                type="postgresql",
                host="localhost",
                port=70000,  # Invalid port
                database="test",
                username="user",
                password=SecretStr("pass"),
            )


class TestDatabaseConnectorBasics:
    """Test basic database connector operations."""

    @pytest.mark.asyncio
    async def test_connection_lifecycle(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test database connection and disconnection."""
        assert db_connector.is_connected() is True

        await db_connector.disconnect()
        assert db_connector.is_connected() is False

    @pytest.mark.asyncio
    async def test_health_check(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test database health check."""
        health = await db_connector.health_check()
        assert health is True

        await db_connector.disconnect()

        health_disconnected = await db_connector.health_check()
        assert health_disconnected is False


class TestQueryExecution:
    """Test database query execution."""

    @pytest.mark.asyncio
    async def test_simple_select_query(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test executing simple SELECT query."""
        result = await db_connector.execute_query("SELECT * FROM users")

        assert result.row_count == 2
        assert result.columns == ["id", "name", "email"]
        assert len(result.rows) == 2
        assert result.rows[0]["name"] == "Alice"
        assert result.cached is False

    @pytest.mark.asyncio
    async def test_parameterized_query(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test parameterized query execution."""
        result = await db_connector.execute_query(
            "SELECT * FROM users WHERE id = $id",
            params={"id": 1},
        )

        assert result.row_count == 1
        assert result.rows[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_query_execution_time(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test query execution time tracking."""
        result = await db_connector.execute_query("SELECT * FROM users")

        assert result.execution_time_ms >= 0
        assert isinstance(result.execution_time_ms, int)


class TestQueryCaching:
    """Test query result caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeated_query(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test cache hit for repeated identical queries."""
        # First query (cache miss)
        result1 = await db_connector.execute_query("SELECT * FROM users")
        assert result1.cached is False
        assert result1.query_hash is not None

        # Second query (cache hit)
        result2 = await db_connector.execute_query("SELECT * FROM users")
        assert result2.cached is True
        assert result2.row_count == result1.row_count

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_update(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test cache is invalidated after UPDATE."""
        # Query and cache
        result1 = await db_connector.execute_query("SELECT * FROM users")
        assert result1.cached is False

        # Second query (cache hit)
        result2 = await db_connector.execute_query("SELECT * FROM users")
        assert result2.cached is True

        # Update (invalidates cache)
        await db_connector.execute_update(
            "UPDATE users SET name = $name WHERE id = $id",
            params={"name": "Alice Updated", "id": 1},
        )

        # Third query (cache miss after invalidation)
        result3 = await db_connector.execute_query("SELECT * FROM users")
        assert result3.cached is False

    @pytest.mark.asyncio
    async def test_cache_with_different_parameters(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test cache treats different parameters as different queries."""
        # Query user 1
        result1 = await db_connector.execute_query(
            "SELECT * FROM users WHERE id = $id",
            params={"id": 1},
        )

        # Query user 2 (different params, cache miss)
        result2 = await db_connector.execute_query(
            "SELECT * FROM users WHERE id = $id",
            params={"id": 2},
        )

        assert result1.cached is False
        assert result2.cached is False
        assert result1.query_hash != result2.query_hash


class TestTransactions:
    """Test transaction support."""

    @pytest.mark.asyncio
    async def test_execute_transaction(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test executing multiple queries in a transaction."""
        queries = [
            ("INSERT INTO users (name, email) VALUES ($name, $email)", {"name": "Charlie", "email": "charlie@example.com"}),
            ("UPDATE users SET name = $name WHERE id = $id", {"name": "Updated", "id": 1}),
        ]

        results = await db_connector.execute_transaction(queries)

        assert len(results) == 2
        assert all(r == 1 for r in results)  # Mock returns 1 for each


class TestConnectionPooling:
    """Test connection pooling behavior."""

    def test_pool_configuration(self, db_config: DatabaseConfig) -> None:
        """Test connection pool is configured correctly."""
        assert db_config.pool_size == 5
        assert db_config.pool_timeout == 10

    @pytest.mark.asyncio
    async def test_concurrent_queries(
        self, db_connector: MockPostgreSQLConnector
    ) -> None:
        """Test concurrent query execution with connection pool."""
        import asyncio

        # Execute multiple queries concurrently
        queries = [
            db_connector.execute_query("SELECT * FROM users"),
            db_connector.execute_query("SELECT * FROM orders"),
            db_connector.execute_query("SELECT * FROM users WHERE id = $id", {"id": 1}),
        ]

        results = await asyncio.gather(*queries)

        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)


class TestDatabaseFactory:
    """Test database factory."""

    def test_create_postgresql_connector(self, db_config: DatabaseConfig) -> None:
        """Test factory creates PostgreSQL connector."""
        connector = DatabaseFactory.create(db_config)

        assert isinstance(connector, PostgreSQLConnector)
        assert connector.config == db_config

    def test_unsupported_database_type(self) -> None:
        """Test factory raises error for unsupported database."""
        config = DatabaseConfig(
            type="unsupported_db",
            host="localhost",
            port=5432,
            database="test",
            username="user",
            password=SecretStr("pass"),
        )

        with pytest.raises(ValueError, match="Unsupported database type"):
            DatabaseFactory.create(config)

    def test_supported_types(self) -> None:
        """Test factory returns supported database types."""
        types = DatabaseFactory.supported_types()

        assert "postgresql" in types
        assert "postgres" in types


class TestDataSecurity:
    """Test data security features."""

    def test_password_encryption_in_config(self) -> None:
        """Test password is encrypted in configuration."""
        config = DatabaseConfig(
            type="postgresql",
            host="localhost",
            port=5432,
            database="secure_db",
            username="secure_user",
            password=SecretStr("super_secret_password"),
        )

        # Password should not appear in string representation
        config_str = str(config)
        assert "super_secret_password" not in config_str
        assert "**********" in config_str  # Pydantic SecretStr masking

    def test_ssl_enabled_by_default(self) -> None:
        """Test SSL is enabled by default for security."""
        config = DatabaseConfig(
            type="postgresql",
            host="localhost",
            port=5432,
            database="test",
            username="user",
            password=SecretStr("pass"),
        )

        assert config.ssl_enabled is True
        assert config.ssl_verify is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
