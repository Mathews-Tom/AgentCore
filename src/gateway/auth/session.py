"""
Session Management

Redis-based session management with automatic cleanup and tracking.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis
import structlog
from redis.asyncio.connection import ConnectionPool

from gateway.auth.models import Session, User, UserRole
from gateway.config import settings

logger = structlog.get_logger()


class SessionManager:
    """
    Redis-based session manager.

    Handles user session creation, validation, and automatic cleanup.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        """
        Initialize session manager.

        Args:
            redis_url: Redis connection URL (uses settings default if None)
        """
        self.redis_url = redis_url or settings.SESSION_REDIS_URL
        self.max_age_hours = settings.SESSION_MAX_AGE_HOURS
        self.cleanup_interval = settings.SESSION_CLEANUP_INTERVAL_MINUTES

        self._client: aioredis.Redis[bytes] | None = None
        self._pool: ConnectionPool | None = None
        self._cleanup_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection and start cleanup task."""
        logger.info("Initializing session manager", redis_url=self.redis_url)

        # Create connection pool
        self._pool = ConnectionPool.from_url(
            self.redis_url,
            decode_responses=False,
            max_connections=20,
        )

        # Create Redis client
        self._client = aioredis.Redis(connection_pool=self._pool)

        # Verify connection
        await self._client.ping()

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_sessions())

        logger.info("Session manager initialized successfully")

    async def close(self) -> None:
        """Close Redis connection and cleanup resources."""
        logger.info("Closing session manager")

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close Redis connection
        if self._client:
            await self._client.aclose()
            self._client = None

        if self._pool:
            await self._pool.aclose()
            self._pool = None

        logger.info("Session manager closed")

    @property
    def client(self) -> aioredis.Redis[bytes]:
        """Get Redis client instance."""
        if not self._client:
            raise RuntimeError(
                "Session manager not initialized. Call initialize() first."
            )
        return self._client

    def _get_session_key(self, session_id: str) -> str:
        """
        Get Redis key for session.

        Args:
            session_id: Session identifier

        Returns:
            Redis key string
        """
        return f"session:{session_id}"

    def _get_user_sessions_key(self, user_id: str) -> str:
        """
        Get Redis key for user's active sessions.

        Args:
            user_id: User identifier

        Returns:
            Redis key string
        """
        return f"user_sessions:{user_id}"

    async def create_session(
        self,
        user: User,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """
        Create new user session.

        Args:
            user: User information
            ip_address: Client IP address (optional)
            user_agent: Client user agent (optional)
            metadata: Additional session metadata (optional)

        Returns:
            Created session
        """
        session_id = str(uuid4())
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.max_age_hours)

        session = Session(
            session_id=session_id,
            user_id=str(user.id),
            username=user.username,
            roles=user.roles,
            created_at=now,
            expires_at=expires_at,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        # Store session in Redis
        session_key = self._get_session_key(session_id)
        session_data = session.model_dump_json()
        ttl_seconds = int(timedelta(hours=self.max_age_hours).total_seconds())

        await self.client.setex(session_key, ttl_seconds, session_data)

        # Add to user's active sessions set
        user_sessions_key = self._get_user_sessions_key(str(user.id))
        await self.client.sadd(user_sessions_key, session_id)
        await self.client.expire(user_sessions_key, ttl_seconds)

        logger.info(
            "Session created",
            session_id=session_id,
            user_id=str(user.id),
            username=user.username,
            expires_at=expires_at.isoformat(),
        )

        return session

    async def get_session(self, session_id: str) -> Session | None:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found and valid, None otherwise
        """
        session_key = self._get_session_key(session_id)
        session_data = await self.client.get(session_key)

        if not session_data:
            return None

        session = Session.model_validate_json(session_data)

        # Check if session is expired
        if datetime.utcnow() >= session.expires_at:
            await self.delete_session(session_id)
            return None

        return session

    async def update_session_activity(self, session_id: str) -> bool:
        """
        Update session last activity timestamp.

        Args:
            session_id: Session identifier

        Returns:
            True if session was updated, False if not found
        """
        session = await self.get_session(session_id)

        if not session:
            return False

        # Update last activity
        session.last_activity = datetime.utcnow()

        # Save updated session
        session_key = self._get_session_key(session_id)
        session_data = session.model_dump_json()
        ttl_seconds = int((session.expires_at - datetime.utcnow()).total_seconds())

        if ttl_seconds > 0:
            await self.client.setex(session_key, ttl_seconds, session_data)
            return True

        return False

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session.

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        # Get session to find user_id
        session = await self.get_session(session_id)

        if not session:
            return False

        # Delete session
        session_key = self._get_session_key(session_id)
        await self.client.delete(session_key)

        # Remove from user's active sessions
        user_sessions_key = self._get_user_sessions_key(session.user_id)
        await self.client.srem(user_sessions_key, session_id)

        logger.info(
            "Session deleted",
            session_id=session_id,
            user_id=session.user_id,
        )

        return True

    async def delete_user_sessions(self, user_id: str) -> int:
        """
        Delete all sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of sessions deleted
        """
        user_sessions_key = self._get_user_sessions_key(user_id)
        session_ids = await self.client.smembers(user_sessions_key)

        count = 0
        for session_id_bytes in session_ids:
            session_id = session_id_bytes.decode("utf-8")
            if await self.delete_session(session_id):
                count += 1

        logger.info(
            "User sessions deleted",
            user_id=user_id,
            count=count,
        )

        return count

    async def get_user_sessions(self, user_id: str) -> list[Session]:
        """
        Get all active sessions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of active sessions
        """
        user_sessions_key = self._get_user_sessions_key(user_id)
        session_ids = await self.client.smembers(user_sessions_key)

        sessions = []
        for session_id_bytes in session_ids:
            session_id = session_id_bytes.decode("utf-8")
            session = await self.get_session(session_id)
            if session:
                sessions.append(session)

        return sessions

    async def validate_session(self, session_id: str) -> bool:
        """
        Validate session existence and expiry.

        Args:
            session_id: Session identifier

        Returns:
            True if session is valid, False otherwise
        """
        session = await self.get_session(session_id)
        return session is not None

    async def extend_session(self, session_id: str, hours: int | None = None) -> bool:
        """
        Extend session expiration time.

        Args:
            session_id: Session identifier
            hours: Additional hours to extend (uses default max_age if None)

        Returns:
            True if session was extended, False if not found
        """
        session = await self.get_session(session_id)

        if not session:
            return False

        # Extend expiration
        extension_hours = hours or self.max_age_hours
        session.expires_at = datetime.utcnow() + timedelta(hours=extension_hours)

        # Save updated session
        session_key = self._get_session_key(session_id)
        session_data = session.model_dump_json()
        ttl_seconds = int(timedelta(hours=extension_hours).total_seconds())

        await self.client.setex(session_key, ttl_seconds, session_data)

        logger.info(
            "Session extended",
            session_id=session_id,
            new_expires_at=session.expires_at.isoformat(),
        )

        return True

    async def get_session_count(self) -> int:
        """
        Get total number of active sessions.

        Returns:
            Number of active sessions
        """
        keys = await self.client.keys("session:*")
        return len(keys)

    async def _cleanup_sessions(self) -> None:
        """Background task to cleanup expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval * 60)

                # Redis TTL handles expiration automatically
                # This task is for monitoring and logging
                count = await self.get_session_count()

                logger.debug(
                    "Session cleanup check",
                    active_sessions=count,
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Session cleanup error",
                    error=str(e),
                )

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on session manager.

        Returns:
            Dictionary with health status information
        """
        try:
            # Ping Redis
            await self.client.ping()

            # Get session count
            session_count = await self.get_session_count()

            return {
                "status": "healthy",
                "active_sessions": session_count,
                "redis_url": self.redis_url,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Global session manager instance
session_manager = SessionManager()
