"""
Database Connection Management

SQLAlchemy engine, session factory, and connection pooling for PostgreSQL.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from agentcore.a2a_protocol.config import settings

logger = structlog.get_logger()

# SQLAlchemy declarative base
Base = declarative_base()

# Global engine instance
engine: AsyncEngine | None = None
SessionLocal: async_sessionmaker[AsyncSession] | None = None


def get_database_url() -> str:
    """Construct async PostgreSQL database URL."""
    if settings.DATABASE_URL:
        # Replace postgresql:// with postgresql+asyncpg://
        url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
        return url

    # Construct from components
    return (
        f"postgresql+asyncpg://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


async def init_db() -> None:
    """Initialize database engine and session factory."""
    global engine, SessionLocal

    if engine is not None:
        logger.warning("Database already initialized")
        return

    database_url = get_database_url()
    logger.info("Initializing database connection", url=database_url.split("@")[-1])  # Hide credentials

    # Create async engine with connection pooling
    engine = create_async_engine(
        database_url,
        echo=settings.DEBUG,
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        pool_recycle=settings.DATABASE_POOL_RECYCLE,
        pool_pre_ping=True,  # Verify connections before using
    )

    # Add connection event listeners
    @event.listens_for(engine.sync_engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")

    @event.listens_for(engine.sync_engine, "close")
    def receive_close(dbapi_conn, connection_record):
        logger.debug("Database connection closed")

    # Create session factory
    SessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    logger.info("Database initialized successfully",
               pool_size=settings.DATABASE_POOL_SIZE,
               max_overflow=settings.DATABASE_MAX_OVERFLOW)


async def close_db() -> None:
    """Close database engine and cleanup connections."""
    global engine, SessionLocal

    if engine is None:
        return

    logger.info("Closing database connections")
    await engine.dispose()
    engine = None
    SessionLocal = None
    logger.info("Database connections closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get async database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(query)

    Raises:
        RuntimeError: If database not initialized
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_health() -> bool:
    """
    Check database connectivity.

    Returns:
        True if database is healthy, False otherwise
    """
    if engine is None:
        return False

    try:
        async with get_session() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return False