"""
Neo4j Connection Management

Provides async Neo4j driver initialization, connection pooling,
and schema setup for the memory graph database.
"""

import asyncio
from typing import AsyncGenerator

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
import structlog

from agentcore.a2a_protocol.config import settings
from agentcore.memory.graph import SCHEMA_FILE

logger = structlog.get_logger(__name__)

# Global Neo4j driver instance
_driver: AsyncDriver | None = None


async def init_neo4j() -> AsyncDriver:
    """
    Initialize Neo4j async driver with connection pooling.

    Returns:
        AsyncDriver: Configured Neo4j async driver

    Raises:
        RuntimeError: If driver initialization fails
    """
    global _driver

    if _driver is not None:
        logger.warning("Neo4j driver already initialized, returning existing instance")
        return _driver

    try:
        logger.info(
            "Initializing Neo4j driver",
            uri=settings.NEO4J_URI,
            database=settings.NEO4J_DATABASE,
            pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
        )

        _driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_lifetime=settings.NEO4J_MAX_CONNECTION_LIFETIME,
            max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
            connection_acquisition_timeout=settings.NEO4J_CONNECTION_ACQUISITION_TIMEOUT,
            encrypted=settings.NEO4J_ENCRYPTED,
        )

        # Verify connectivity
        await _driver.verify_connectivity()

        logger.info("Neo4j driver initialized successfully")

        # Initialize schema
        await _init_schema(_driver)

        return _driver

    except Exception as e:
        logger.error("Failed to initialize Neo4j driver", error=str(e))
        raise RuntimeError(f"Neo4j initialization failed: {e}") from e


async def close_neo4j() -> None:
    """Close Neo4j driver and cleanup connections."""
    global _driver

    if _driver is None:
        logger.warning("Neo4j driver not initialized, nothing to close")
        return

    try:
        logger.info("Closing Neo4j driver")
        await _driver.close()
        _driver = None
        logger.info("Neo4j driver closed successfully")
    except Exception as e:
        logger.error("Error closing Neo4j driver", error=str(e))
        raise


async def get_driver() -> AsyncDriver:
    """
    Get the global Neo4j driver instance.

    Returns:
        AsyncDriver: Neo4j async driver

    Raises:
        RuntimeError: If driver not initialized
    """
    if _driver is None:
        raise RuntimeError("Neo4j driver not initialized. Call init_neo4j() first.")
    return _driver


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a Neo4j async session as a context manager.

    Yields:
        AsyncSession: Neo4j async session

    Example:
        async with get_session() as session:
            result = await session.run("MATCH (n) RETURN count(n)")
            count = await result.single()
    """
    driver = await get_driver()
    async with driver.session(database=settings.NEO4J_DATABASE) as session:
        yield session


async def _init_schema(driver: AsyncDriver) -> None:
    """
    Initialize Neo4j graph schema from Cypher file.

    Args:
        driver: Neo4j async driver

    Raises:
        RuntimeError: If schema initialization fails
    """
    try:
        logger.info("Initializing Neo4j graph schema", schema_file=str(SCHEMA_FILE))

        if not SCHEMA_FILE.exists():
            raise FileNotFoundError(f"Schema file not found: {SCHEMA_FILE}")

        # Read schema file
        schema_cypher = SCHEMA_FILE.read_text()

        # Split into individual statements (separated by semicolons or blank lines)
        statements = [
            stmt.strip()
            for stmt in schema_cypher.split(";")
            if stmt.strip() and not stmt.strip().startswith("//")
        ]

        # Execute each statement
        async with driver.session(database=settings.NEO4J_DATABASE) as session:
            for i, statement in enumerate(statements, 1):
                # Skip comments and empty statements
                if not statement or statement.startswith("//"):
                    continue

                try:
                    await session.run(statement)
                    logger.debug(
                        "Executed schema statement",
                        statement_num=i,
                        total=len(statements),
                    )
                except Exception as e:
                    # Log warning for errors but continue (some statements may be idempotent)
                    logger.warning(
                        "Schema statement failed (may be idempotent)",
                        statement_num=i,
                        error=str(e),
                    )

        logger.info(
            "Neo4j graph schema initialized successfully",
            statements_executed=len(statements),
        )

    except Exception as e:
        logger.error("Failed to initialize Neo4j schema", error=str(e))
        raise RuntimeError(f"Neo4j schema initialization failed: {e}") from e


async def verify_neo4j_health() -> bool:
    """
    Verify Neo4j connection health.

    Returns:
        bool: True if healthy, False otherwise
    """
    try:
        driver = await get_driver()
        await driver.verify_connectivity()

        # Test query
        async with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = await session.run("RETURN 1 AS health")
            record = await result.single()
            if record and record["health"] == 1:
                logger.info("Neo4j health check passed")
                return True

        return False

    except Exception as e:
        logger.error("Neo4j health check failed", error=str(e))
        return False
