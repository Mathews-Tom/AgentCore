"""
Unit tests for session management.

Tests Redis-based session creation, validation, and cleanup.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from testcontainers.redis import RedisContainer

from gateway.auth.models import User, UserRole
from gateway.auth.session import SessionManager


@pytest.fixture
async def redis_container():
    """Start Redis container for testing."""
    container = RedisContainer("redis:7-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture
async def session_manager_test(redis_container) -> SessionManager:
    """Create session manager with test Redis instance."""
    redis_url = f"redis://localhost:{redis_container.get_exposed_port(6379)}/0"
    manager = SessionManager(redis_url=redis_url)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def test_user() -> User:
    """Create test user."""
    return User(
        id=uuid4(),
        username="testuser",
        email="test@example.com",
        roles=[UserRole.USER],
        is_active=True)


class TestSessionManager:
    """Test session manager functionality."""

    @pytest.mark.asyncio
    async def test_initialize_and_close(self, redis_container) -> None:
        """Test session manager initialization and cleanup."""
        redis_url = f"redis://localhost:{redis_container.get_exposed_port(6379)}/0"
        manager = SessionManager(redis_url=redis_url)

        await manager.initialize()
        assert manager._client is not None
        assert manager._pool is not None

        await manager.close()
        assert manager._client is None
        assert manager._pool is None

    @pytest.mark.asyncio
    async def test_create_session(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test session creation."""
        session = await session_manager_test.create_session(
            user=test_user,
            ip_address="192.168.1.1",
            user_agent="Test Agent")

        assert session is not None
        assert session.session_id is not None
        assert session.user_id == str(test_user.id)
        assert session.username == test_user.username
        assert session.roles == test_user.roles
        assert session.ip_address == "192.168.1.1"
        assert session.user_agent == "Test Agent"
        assert session.created_at is not None
        assert session.expires_at > session.created_at

    @pytest.mark.asyncio
    async def test_get_session(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test getting session by ID."""
        # Create session
        created = await session_manager_test.create_session(user=test_user)

        # Retrieve session
        session = await session_manager_test.get_session(created.session_id)

        assert session is not None
        assert session.session_id == created.session_id
        assert session.user_id == str(test_user.id)

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(
        self, session_manager_test: SessionManager
    ) -> None:
        """Test getting nonexistent session returns None."""
        session = await session_manager_test.get_session("nonexistent")
        assert session is None

    @pytest.mark.asyncio
    async def test_update_session_activity(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test updating session last activity."""
        session = await session_manager_test.create_session(user=test_user)
        original_activity = session.last_activity

        # Wait a moment
        await asyncio.sleep(0.1)

        # Update activity
        updated = await session_manager_test.update_session_activity(session.session_id)
        assert updated is True

        # Get updated session
        session = await session_manager_test.get_session(session.session_id)
        assert session is not None
        assert session.last_activity > original_activity

    @pytest.mark.asyncio
    async def test_delete_session(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test session deletion."""
        session = await session_manager_test.create_session(user=test_user)

        # Delete session
        deleted = await session_manager_test.delete_session(session.session_id)
        assert deleted is True

        # Verify session is gone
        session = await session_manager_test.get_session(session.session_id)
        assert session is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(
        self, session_manager_test: SessionManager
    ) -> None:
        """Test deleting nonexistent session returns False."""
        deleted = await session_manager_test.delete_session("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_delete_user_sessions(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test deleting all sessions for a user."""
        # Create multiple sessions
        session1 = await session_manager_test.create_session(user=test_user)
        session2 = await session_manager_test.create_session(user=test_user)
        session3 = await session_manager_test.create_session(user=test_user)

        # Delete all user sessions
        count = await session_manager_test.delete_user_sessions(str(test_user.id))
        assert count == 3

        # Verify all sessions are gone
        assert await session_manager_test.get_session(session1.session_id) is None
        assert await session_manager_test.get_session(session2.session_id) is None
        assert await session_manager_test.get_session(session3.session_id) is None

    @pytest.mark.asyncio
    async def test_get_user_sessions(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test getting all sessions for a user."""
        # Create sessions
        session1 = await session_manager_test.create_session(user=test_user)
        session2 = await session_manager_test.create_session(user=test_user)

        # Get user sessions
        sessions = await session_manager_test.get_user_sessions(str(test_user.id))
        assert len(sessions) == 2

        session_ids = {s.session_id for s in sessions}
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids

    @pytest.mark.asyncio
    async def test_validate_session(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test session validation."""
        session = await session_manager_test.create_session(user=test_user)

        # Valid session
        valid = await session_manager_test.validate_session(session.session_id)
        assert valid is True

        # Invalid session
        valid = await session_manager_test.validate_session("nonexistent")
        assert valid is False

    @pytest.mark.asyncio
    async def test_extend_session(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test extending session expiration."""
        session = await session_manager_test.create_session(user=test_user)
        original_expires = session.expires_at

        # Wait a moment
        await asyncio.sleep(0.1)

        # Extend session
        extended = await session_manager_test.extend_session(session.session_id, hours=48)
        assert extended is True

        # Get updated session
        session = await session_manager_test.get_session(session.session_id)
        assert session is not None
        assert session.expires_at > original_expires

    @pytest.mark.asyncio
    async def test_get_session_count(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test getting total session count."""
        initial_count = await session_manager_test.get_session_count()

        # Create sessions
        await session_manager_test.create_session(user=test_user)
        await session_manager_test.create_session(user=test_user)

        count = await session_manager_test.get_session_count()
        assert count == initial_count + 2

    @pytest.mark.asyncio
    async def test_session_with_metadata(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test session creation with metadata."""
        metadata = {"client": "web", "version": "1.0"}
        session = await session_manager_test.create_session(
            user=test_user,
            metadata=metadata)

        assert session.metadata == metadata

        # Verify metadata persists
        retrieved = await session_manager_test.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.metadata == metadata

    @pytest.mark.asyncio
    async def test_health_check(self, session_manager_test: SessionManager) -> None:
        """Test session manager health check."""
        health = await session_manager_test.health_check()

        assert health["status"] == "healthy"
        assert "active_sessions" in health
        assert "redis_url" in health

    @pytest.mark.asyncio
    async def test_session_expiration(
        self, session_manager_test: SessionManager, test_user: User
    ) -> None:
        """Test expired sessions are not returned."""
        # Override max_age for this test
        original_max_age = session_manager_test.max_age_hours
        # Use 1 second (Redis doesn't accept 0 TTL)
        session_manager_test.max_age_hours = 1 / 3600  # 1 second in hours

        session = await session_manager_test.create_session(user=test_user)

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Session should be expired
        retrieved = await session_manager_test.get_session(session.session_id)
        assert retrieved is None

        # Restore original
        session_manager_test.max_age_hours = original_max_age
