"""
OAuth State Management

Redis-based OAuth state storage for CSRF protection and PKCE verifier storage.
"""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as aioredis
import structlog
from redis.asyncio.connection import ConnectionPool

from gateway.auth.oauth.models import OAuthProvider, OAuthState
from gateway.config import settings

logger = structlog.get_logger()


class OAuthStateManager:
    """
    OAuth state manager with Redis backend.

    Handles secure state generation, storage, and validation for OAuth flows.
    Stores PKCE verifiers and prevents CSRF attacks.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        """
        Initialize OAuth state manager.

        Args:
            redis_url: Redis connection URL (uses settings default if None)
        """
        self.redis_url = redis_url or settings.SESSION_REDIS_URL
        self.state_ttl_minutes = 10  # OAuth state valid for 10 minutes

        self._client: aioredis.Redis[bytes] | None = None
        self._pool: ConnectionPool | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        logger.info("Initializing OAuth state manager", redis_url=self.redis_url)

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

        logger.info("OAuth state manager initialized")

    async def close(self) -> None:
        """Close Redis connection."""
        logger.info("Closing OAuth state manager")

        if self._client:
            await self._client.aclose()
            self._client = None

        if self._pool:
            await self._pool.aclose()
            self._pool = None

        logger.info("OAuth state manager closed")

    @property
    def client(self) -> aioredis.Redis[bytes]:
        """Get Redis client instance."""
        if not self._client:
            raise RuntimeError(
                "OAuth state manager not initialized. Call initialize() first."
            )
        return self._client

    def _get_state_key(self, state: str) -> str:
        """
        Get Redis key for OAuth state.

        Args:
            state: State parameter

        Returns:
            Redis key string
        """
        return f"oauth_state:{state}"

    def generate_state(self) -> str:
        """
        Generate cryptographically secure state parameter.

        Returns:
            Random state string (32 bytes, URL-safe base64)
        """
        return secrets.token_urlsafe(32)

    async def save_state(
        self,
        state: str,
        provider: OAuthProvider,
        redirect_uri: str,
        code_verifier: str | None = None,
        requested_scopes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OAuthState:
        """
        Save OAuth state to Redis.

        Args:
            state: State parameter
            provider: OAuth provider
            redirect_uri: Callback redirect URI
            code_verifier: PKCE code verifier (optional)
            requested_scopes: Requested scopes
            metadata: Additional metadata

        Returns:
            OAuth state object
        """
        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(minutes=self.state_ttl_minutes)

        oauth_state = OAuthState(
            state=state,
            provider=provider,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            requested_scopes=requested_scopes,
            created_at=created_at,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        # Store in Redis
        state_key = self._get_state_key(state)
        state_data = oauth_state.model_dump_json()
        ttl_seconds = int(timedelta(minutes=self.state_ttl_minutes).total_seconds())

        await self.client.setex(state_key, ttl_seconds, state_data)

        logger.debug(
            "OAuth state saved",
            state=state,
            provider=provider.value,
            expires_at=expires_at.isoformat(),
        )

        return oauth_state

    async def get_state(self, state: str) -> OAuthState | None:
        """
        Get OAuth state from Redis.

        Args:
            state: State parameter

        Returns:
            OAuth state if found and valid, None otherwise
        """
        state_key = self._get_state_key(state)
        state_data = await self.client.get(state_key)

        if not state_data:
            logger.warning("OAuth state not found", state=state)
            return None

        oauth_state = OAuthState.model_validate_json(state_data)

        # Check if state is expired
        if datetime.now(UTC) >= oauth_state.expires_at:
            logger.warning("OAuth state expired", state=state)
            await self.delete_state(state)
            return None

        return oauth_state

    async def validate_and_consume_state(
        self,
        state: str,
        expected_provider: OAuthProvider | None = None,
    ) -> OAuthState | None:
        """
        Validate OAuth state and delete it (one-time use).

        Args:
            state: State parameter to validate
            expected_provider: Expected OAuth provider (optional)

        Returns:
            OAuth state if valid, None otherwise
        """
        oauth_state = await self.get_state(state)

        if not oauth_state:
            return None

        # Verify provider if specified
        if expected_provider and oauth_state.provider != expected_provider:
            logger.warning(
                "OAuth state provider mismatch",
                state=state,
                expected=expected_provider.value,
                actual=oauth_state.provider.value,
            )
            await self.delete_state(state)
            return None

        # Delete state (one-time use)
        await self.delete_state(state)

        logger.info(
            "OAuth state validated and consumed",
            state=state,
            provider=oauth_state.provider.value,
        )

        return oauth_state

    async def delete_state(self, state: str) -> bool:
        """
        Delete OAuth state from Redis.

        Args:
            state: State parameter

        Returns:
            True if state was deleted, False if not found
        """
        state_key = self._get_state_key(state)
        result = await self.client.delete(state_key)
        return result > 0

    async def cleanup_expired_states(self) -> int:
        """
        Cleanup expired OAuth states.

        Redis TTL handles automatic expiration, so this is mainly
        for monitoring and manual cleanup.

        Returns:
            Number of states cleaned up
        """
        # Get all OAuth state keys
        pattern = "oauth_state:*"
        keys = await self.client.keys(pattern)

        cleaned = 0
        for key_bytes in keys:
            key = key_bytes.decode("utf-8")
            state_data = await self.client.get(key)

            if not state_data:
                continue

            try:
                oauth_state = OAuthState.model_validate_json(state_data)

                if datetime.now(UTC) >= oauth_state.expires_at:
                    await self.client.delete(key)
                    cleaned += 1
            except Exception as e:
                logger.error("Error cleaning up OAuth state", key=key, error=str(e))

        if cleaned > 0:
            logger.info("OAuth state cleanup complete", cleaned=cleaned)

        return cleaned

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on OAuth state manager.

        Returns:
            Dictionary with health status information
        """
        try:
            # Ping Redis
            await self.client.ping()

            # Count active states
            state_keys = await self.client.keys("oauth_state:*")
            state_count = len(state_keys)

            return {
                "status": "healthy",
                "active_states": state_count,
                "redis_url": self.redis_url,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }


# Global OAuth state manager instance
oauth_state_manager = OAuthStateManager()
