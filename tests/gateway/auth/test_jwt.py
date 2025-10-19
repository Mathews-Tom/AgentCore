"""
Unit tests for JWT token management.

Tests JWT token generation, validation, RSA key management, and token expiration.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from jose import JWTError, jwt

from gateway.auth.jwt import JWTManager
from gateway.auth.models import User, UserRole
from gateway.config import settings


@pytest.fixture
async def jwt_manager_test() -> JWTManager:
    """Create JWT manager instance for testing."""
    # Use test-specific key paths
    manager = JWTManager()
    manager.private_key_path = Path("/tmp/test_rsa_private.pem")
    manager.public_key_path = Path("/tmp/test_rsa_public.pem")
    await manager.initialize()
    yield manager

    # Cleanup test keys
    if manager.private_key_path.exists():
        manager.private_key_path.unlink()
    if manager.public_key_path.exists():
        manager.public_key_path.unlink()


@pytest.fixture
def test_user() -> User:
    """Create test user."""
    return User(
        id=uuid4(),
        username="testuser",
        email="test@example.com",
        roles=[UserRole.USER],
        is_active=True,
    )


class TestJWTManager:
    """Test JWT manager functionality."""

    @pytest.mark.asyncio
    async def test_initialize_generates_keys(self, jwt_manager_test: JWTManager) -> None:
        """Test JWT manager initialization generates RSA keys."""
        assert jwt_manager_test._private_key is not None
        assert jwt_manager_test._public_key is not None
        assert jwt_manager_test._key_created_at is not None
        assert jwt_manager_test.private_key_path.exists()
        assert jwt_manager_test.public_key_path.exists()

    @pytest.mark.asyncio
    async def test_load_existing_keys(self, jwt_manager_test: JWTManager) -> None:
        """Test loading existing RSA keys."""
        # Create new manager to load existing keys
        manager2 = JWTManager()
        manager2.private_key_path = jwt_manager_test.private_key_path
        manager2.public_key_path = jwt_manager_test.public_key_path
        await manager2.initialize()

        # Should load same keys
        assert manager2._private_key == jwt_manager_test._private_key
        assert manager2._public_key == jwt_manager_test._public_key

    @pytest.mark.asyncio
    async def test_create_access_token(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test access token creation."""
        session_id = str(uuid4())
        token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify token
        payload = jwt.decode(
            token,
            jwt_manager_test._public_key,
            algorithms=[jwt_manager_test.algorithm],
            audience=jwt_manager_test.audience,
            issuer=jwt_manager_test.issuer,
        )

        assert payload["sub"] == str(test_user.id)
        assert payload["username"] == test_user.username
        assert payload["session_id"] == session_id
        assert payload["iss"] == jwt_manager_test.issuer
        assert payload["aud"] == jwt_manager_test.audience
        assert "jti" in payload
        assert "iat" in payload
        assert "exp" in payload

    @pytest.mark.asyncio
    async def test_create_access_token_with_scope(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test access token creation with custom scope."""
        session_id = str(uuid4())
        scope = "read:agents write:tasks"
        token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
            scope=scope,
        )

        payload = jwt_manager_test.decode_token(token)
        assert payload["scope"] == scope

    @pytest.mark.asyncio
    async def test_create_access_token_custom_expiration(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test access token with custom expiration."""
        session_id = str(uuid4())
        expires_delta = timedelta(minutes=5)
        token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
            expires_delta=expires_delta,
        )

        payload = jwt_manager_test.decode_token(token)
        exp_time = datetime.fromtimestamp(payload["exp"])
        iat_time = datetime.fromtimestamp(payload["iat"])

        # Check expiration is approximately 5 minutes
        diff = (exp_time - iat_time).total_seconds()
        assert 290 <= diff <= 310  # Allow 10 second variance

    @pytest.mark.asyncio
    async def test_create_refresh_token(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test refresh token creation."""
        session_id = str(uuid4())
        token = jwt_manager_test.create_refresh_token(
            user_id=str(test_user.id),
            session_id=session_id,
        )

        assert token is not None
        assert isinstance(token, str)

        payload = jwt_manager_test.decode_token(token)
        assert payload["sub"] == str(test_user.id)
        assert payload["session_id"] == session_id
        assert payload["token_type"] == "refresh"

    @pytest.mark.asyncio
    async def test_validate_access_token(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test access token validation."""
        session_id = str(uuid4())
        token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
        )

        # Validate token
        payload = jwt_manager_test.validate_access_token(token)
        assert payload.sub == str(test_user.id)
        assert payload.username == test_user.username
        assert payload.session_id == session_id
        assert payload.roles == test_user.roles

    @pytest.mark.asyncio
    async def test_validate_refresh_token(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test refresh token validation."""
        session_id = str(uuid4())
        token = jwt_manager_test.create_refresh_token(
            user_id=str(test_user.id),
            session_id=session_id,
        )

        payload = jwt_manager_test.validate_refresh_token(token)
        assert payload.sub == str(test_user.id)
        assert payload.session_id == session_id
        assert payload.token_type == "refresh"

    @pytest.mark.asyncio
    async def test_validate_wrong_token_type(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test validation fails with wrong token type."""
        session_id = str(uuid4())

        # Create refresh token
        refresh_token = jwt_manager_test.create_refresh_token(
            user_id=str(test_user.id),
            session_id=session_id,
        )

        # Try to validate as access token - should fail
        with pytest.raises(JWTError, match="Invalid token type"):
            jwt_manager_test.validate_access_token(refresh_token)

        # Create access token
        access_token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
        )

        # Try to validate as refresh token - should fail
        with pytest.raises(JWTError, match="Invalid token type"):
            jwt_manager_test.validate_refresh_token(access_token)

    @pytest.mark.asyncio
    async def test_decode_invalid_token(self, jwt_manager_test: JWTManager) -> None:
        """Test decoding invalid token raises error."""
        with pytest.raises(JWTError):
            jwt_manager_test.decode_token("invalid.token.here")

    @pytest.mark.asyncio
    async def test_decode_expired_token(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test decoding expired token raises error."""
        session_id = str(uuid4())

        # Create token with very short expiration
        token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
            expires_delta=timedelta(seconds=1),
        )

        # Wait for expiration
        time.sleep(2)

        # Should raise error
        with pytest.raises(JWTError):
            jwt_manager_test.validate_access_token(token)

    @pytest.mark.asyncio
    async def test_get_token_expiry(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test getting token expiration time."""
        session_id = str(uuid4())
        before = time.time()
        token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
        )
        after = time.time()

        expiry = jwt_manager_test.get_token_expiry(token)

        # Expiry should be approximately now + default expiration
        expected_timestamp = before + (settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        diff = abs(expiry.timestamp() - expected_timestamp)
        assert diff < 10  # Within 10 seconds

    @pytest.mark.asyncio
    async def test_is_token_expired(
        self, jwt_manager_test: JWTManager, test_user: User
    ) -> None:
        """Test checking if token is expired."""
        session_id = str(uuid4())

        # Create valid token
        valid_token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
        )
        assert not jwt_manager_test.is_token_expired(valid_token)

        # Create expired token
        expired_token = jwt_manager_test.create_access_token(
            user=test_user,
            session_id=session_id,
            expires_delta=timedelta(seconds=1),
        )
        time.sleep(2)
        assert jwt_manager_test.is_token_expired(expired_token)

    @pytest.mark.asyncio
    async def test_public_key_property(self, jwt_manager_test: JWTManager) -> None:
        """Test public key property."""
        public_key = jwt_manager_test.public_key
        assert public_key is not None
        assert "BEGIN PUBLIC KEY" in public_key
        assert "END PUBLIC KEY" in public_key

    @pytest.mark.asyncio
    async def test_key_metadata(self, jwt_manager_test: JWTManager) -> None:
        """Test key metadata property."""
        metadata = jwt_manager_test.key_metadata
        assert metadata["algorithm"] == jwt_manager_test.algorithm
        assert metadata["key_size"] == jwt_manager_test.key_size
        assert metadata["rotation_days"] == jwt_manager_test.rotation_days
        assert metadata["created_at"] is not None

    @pytest.mark.asyncio
    async def test_token_includes_roles(
        self, jwt_manager_test: JWTManager
    ) -> None:
        """Test token includes user roles."""
        admin_user = User(
            id=uuid4(),
            username="admin",
            roles=[UserRole.ADMIN, UserRole.USER],
        )

        session_id = str(uuid4())
        token = jwt_manager_test.create_access_token(
            user=admin_user,
            session_id=session_id,
        )

        payload = jwt_manager_test.validate_access_token(token)
        assert UserRole.ADMIN in payload.roles
        assert UserRole.USER in payload.roles
