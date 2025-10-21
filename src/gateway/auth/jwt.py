"""
JWT Token Management

JWT token generation and validation with RSA-256 signing and key rotation support.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import JWTError, jwt

from gateway.auth.models import RefreshTokenPayload, TokenPayload, User, UserRole
from gateway.config import settings

logger = structlog.get_logger()


class JWTManager:
    """
    JWT token manager with RSA-256 signing and key rotation.

    Handles token generation, validation, and RSA key pair management.
    """

    def __init__(self) -> None:
        """Initialize JWT manager."""
        self.algorithm = settings.JWT_ALGORITHM
        self.issuer = settings.JWT_ISSUER
        self.audience = settings.JWT_AUDIENCE
        self.private_key_path = Path(settings.RSA_PRIVATE_KEY_PATH)
        self.public_key_path = Path(settings.RSA_PUBLIC_KEY_PATH)
        self.key_size = settings.RSA_KEY_SIZE
        self.rotation_days = settings.RSA_KEY_ROTATION_DAYS

        self._private_key: str | None = None
        self._public_key: str | None = None
        self._key_created_at: datetime | None = None

    async def initialize(self) -> None:
        """
        Initialize RSA keys.

        Loads existing keys or generates new ones if not found.
        """
        if self._should_rotate_keys():
            logger.info("Rotating RSA keys", rotation_days=self.rotation_days)
            await self._generate_keys()
        else:
            await self._load_keys()

        logger.info("JWT Manager initialized", algorithm=self.algorithm)

    def _should_rotate_keys(self) -> bool:
        """
        Check if RSA keys should be rotated.

        Returns:
            True if keys should be rotated, False otherwise
        """
        if not self.private_key_path.exists() or not self.public_key_path.exists():
            return True

        # Check key age
        key_age_seconds = time.time() - self.private_key_path.stat().st_mtime
        key_age_days = key_age_seconds / (24 * 3600)
        return key_age_days >= self.rotation_days

    async def _generate_keys(self) -> None:
        """Generate new RSA key pair."""
        logger.info("Generating RSA key pair", key_size=self.key_size)

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend(),
        )

        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Get public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        # Create directories if needed
        self.private_key_path.parent.mkdir(parents=True, exist_ok=True)
        self.public_key_path.parent.mkdir(parents=True, exist_ok=True)

        # Write keys to files
        self.private_key_path.write_bytes(private_pem)
        self.public_key_path.write_bytes(public_pem)

        # Set restrictive permissions on private key
        os.chmod(self.private_key_path, 0o600)

        self._private_key = private_pem.decode("utf-8")
        self._public_key = public_pem.decode("utf-8")
        self._key_created_at = datetime.fromtimestamp(time.time())

        logger.info("RSA key pair generated successfully")

    async def _load_keys(self) -> None:
        """Load existing RSA keys from files."""
        logger.info("Loading RSA keys from files")

        self._private_key = self.private_key_path.read_text()
        self._public_key = self.public_key_path.read_text()
        self._key_created_at = datetime.fromtimestamp(
            self.private_key_path.stat().st_mtime
        )

        logger.info("RSA keys loaded successfully")

    def create_access_token(
        self,
        user: User,
        session_id: str,
        scope: str | None = None,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create JWT access token.

        Args:
            user: User information
            session_id: Session identifier
            scope: Token scope (optional)
            expires_delta: Custom expiration time (optional)

        Returns:
            Encoded JWT access token
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

        now = int(time.time())
        expire = now + int(expires_delta.total_seconds())

        payload = TokenPayload(
            sub=str(user.id),
            username=user.username,
            roles=user.roles,
            session_id=session_id,
            iat=now,
            exp=expire,
            scope=scope,
            jti=str(uuid4()),
        )

        # Add issuer and audience claims
        token_data = payload.model_dump()
        token_data["iss"] = self.issuer
        token_data["aud"] = self.audience

        token = jwt.encode(token_data, self._private_key, algorithm=self.algorithm)

        logger.debug(
            "Access token created",
            user_id=str(user.id),
            username=user.username,
            session_id=session_id,
            expires_at=datetime.fromtimestamp(expire).isoformat(),
        )

        return token

    def create_refresh_token(
        self,
        user_id: str,
        session_id: str,
        expires_delta: timedelta | None = None,
    ) -> str:
        """
        Create JWT refresh token.

        Args:
            user_id: User identifier
            session_id: Session identifier
            expires_delta: Custom expiration time (optional)

        Returns:
            Encoded JWT refresh token
        """
        if expires_delta is None:
            expires_delta = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

        now = int(time.time())
        expire = now + int(expires_delta.total_seconds())

        payload = RefreshTokenPayload(
            sub=user_id,
            session_id=session_id,
            iat=now,
            exp=expire,
            jti=str(uuid4()),
            token_type="refresh",
        )

        # Add issuer and audience claims
        token_data = payload.model_dump()
        token_data["iss"] = self.issuer
        token_data["aud"] = self.audience

        token = jwt.encode(token_data, self._private_key, algorithm=self.algorithm)

        logger.debug(
            "Refresh token created",
            user_id=user_id,
            session_id=session_id,
            expires_at=datetime.fromtimestamp(expire).isoformat(),
        )

        return token

    def decode_token(self, token: str) -> dict[str, Any]:
        """
        Decode and validate JWT token.

        Args:
            token: Encoded JWT token

        Returns:
            Token payload dictionary

        Raises:
            JWTError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
            )
            return payload
        except JWTError as e:
            logger.warning("Token validation failed", error=str(e))
            raise

    def validate_access_token(self, token: str) -> TokenPayload:
        """
        Validate access token and extract payload.

        Args:
            token: Encoded JWT access token

        Returns:
            Token payload model

        Raises:
            JWTError: If token is invalid or expired
        """
        payload = self.decode_token(token)

        # Ensure token is not a refresh token
        if payload.get("token_type") == "refresh":
            raise JWTError("Invalid token type: expected access token")

        return TokenPayload(**payload)

    def validate_refresh_token(self, token: str) -> RefreshTokenPayload:
        """
        Validate refresh token and extract payload.

        Args:
            token: Encoded JWT refresh token

        Returns:
            Refresh token payload model

        Raises:
            JWTError: If token is invalid or expired
        """
        payload = self.decode_token(token)

        # Ensure token is a refresh token
        if payload.get("token_type") != "refresh":
            raise JWTError("Invalid token type: expected refresh token")

        return RefreshTokenPayload(**payload)

    def get_token_expiry(self, token: str) -> datetime:
        """
        Get token expiration time.

        Args:
            token: Encoded JWT token

        Returns:
            Token expiration datetime

        Raises:
            JWTError: If token is invalid
        """
        payload = self.decode_token(token)
        exp_timestamp = payload.get("exp")

        if not exp_timestamp:
            raise JWTError("Token missing expiration claim")

        return datetime.fromtimestamp(exp_timestamp, tz=UTC)

    def is_token_expired(self, token: str) -> bool:
        """
        Check if token is expired.

        Args:
            token: Encoded JWT token

        Returns:
            True if token is expired, False otherwise
        """
        try:
            expiry = self.get_token_expiry(token)
            return datetime.now(UTC) >= expiry
        except JWTError:
            return True

    @property
    def public_key(self) -> str:
        """Get public key for external verification."""
        if not self._public_key:
            raise RuntimeError("JWT Manager not initialized. Call initialize() first.")
        return self._public_key

    @property
    def key_metadata(self) -> dict[str, Any]:
        """Get RSA key metadata."""
        return {
            "algorithm": self.algorithm,
            "key_size": self.key_size,
            "created_at": self._key_created_at.isoformat()
            if self._key_created_at
            else None,
            "rotation_days": self.rotation_days,
        }


# Global JWT manager instance
jwt_manager = JWTManager()
jwt_manager = JWTManager()
jwt_manager = JWTManager()
