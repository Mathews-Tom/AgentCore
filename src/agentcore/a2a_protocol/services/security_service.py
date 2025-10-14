"""
Security Service

Comprehensive security service providing JWT authentication, RSA request signing,
rate limiting, and input validation.
"""

import base64
import hashlib
import hmac
import json
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import structlog
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from jose import JWTError, jwt

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.security import (
    AuthenticationRequest,
    AuthenticationResponse,
    Permission,
    RateLimitInfo,
    Role,
    SignedRequest,
    TokenPayload,
    TokenType,
)

logger = structlog.get_logger()


class SecurityService:
    """
    Comprehensive security service.

    Provides JWT token management, RSA request signing/verification,
    rate limiting, and input validation.
    """

    def __init__(self):
        self.logger = structlog.get_logger()

        # RSA key pairs (agent_id -> keys)
        self._agent_public_keys: dict[str, Any] = {}
        self._agent_private_keys: dict[str, Any] = {}

        # Rate limiting
        self._rate_limits: dict[str, RateLimitInfo] = {}
        self._default_rate_limit = 1000  # requests per minute

        # Nonce tracking for replay attack prevention
        self._used_nonces: dict[str, set[str]] = defaultdict(set)
        self._nonce_expiry = 300  # 5 minutes

        # Security statistics
        self._security_stats = {
            "tokens_generated": 0,
            "tokens_validated": 0,
            "tokens_expired": 0,
            "rate_limited_requests": 0,
            "invalid_signatures": 0,
            "replay_attacks_prevented": 0,
        }

    # ==================== JWT Token Management ====================

    def generate_token(
        self,
        subject: str,
        role: Role,
        token_type: TokenType = TokenType.ACCESS,
        agent_id: str | None = None,
        additional_permissions: list[Permission] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate JWT token.

        Args:
            subject: Token subject (agent_id or user_id)
            role: User role
            token_type: Token type
            agent_id: Optional agent identifier
            additional_permissions: Additional permissions
            metadata: Additional metadata

        Returns:
            Encoded JWT token
        """
        # Create token payload
        payload = TokenPayload.create(
            subject=subject,
            role=role,
            token_type=token_type,
            expiration_hours=settings.JWT_EXPIRATION_HOURS,
            agent_id=agent_id,
            additional_permissions=additional_permissions,
            metadata=metadata,
        )

        # Encode JWT - convert datetime to timestamps for jose library
        payload_dict = payload.model_dump(mode="json")
        # jose library expects Unix timestamps (integers) for exp and iat
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())

        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        self._security_stats["tokens_generated"] += 1

        self.logger.info(
            "JWT token generated",
            subject=subject,
            role=role.value,
            token_type=token_type.value,
            agent_id=agent_id,
        )

        return token

    def validate_token(self, token: str) -> TokenPayload | None:
        """
        Validate and decode JWT token.

        Args:
            token: Encoded JWT token

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            # Decode JWT
            payload_dict = jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )

            # Parse payload
            payload = TokenPayload.model_validate(payload_dict)

            # Check expiration
            if payload.is_expired():
                self._security_stats["tokens_expired"] += 1
                self.logger.warning(
                    "Token expired",
                    subject=payload.sub,
                    expired_at=payload.exp.isoformat(),
                )
                return None

            self._security_stats["tokens_validated"] += 1
            return payload

        except JWTError as e:
            self.logger.error("JWT validation failed", error=str(e))
            return None
        except Exception as e:
            self.logger.error("Token validation error", error=str(e))
            return None

    def check_permission(self, token: str, required_permission: Permission) -> bool:
        """
        Check if token has required permission.

        Args:
            token: JWT token
            required_permission: Required permission

        Returns:
            True if token has permission, False otherwise
        """
        payload = self.validate_token(token)
        if not payload:
            return False

        return payload.has_permission(required_permission)

    # ==================== RSA Request Signing ====================

    def generate_rsa_keypair(self, agent_id: str) -> dict[str, str]:
        """
        Generate RSA key pair for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary with public and private keys (PEM format)
        """
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        # Get public key
        public_key = private_key.public_key()

        # Serialize keys to PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        # Store keys
        self._agent_public_keys[agent_id] = public_key
        self._agent_private_keys[agent_id] = private_key

        self.logger.info("RSA keypair generated", agent_id=agent_id)

        return {"public_key": public_pem, "private_key": private_pem}

    def register_public_key(self, agent_id: str, public_key_pem: str) -> bool:
        """
        Register agent's public key.

        Args:
            agent_id: Agent identifier
            public_key_pem: Public key in PEM format

        Returns:
            True if registered successfully
        """
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )

            self._agent_public_keys[agent_id] = public_key

            self.logger.info("Public key registered", agent_id=agent_id)
            return True

        except Exception as e:
            self.logger.error(
                "Failed to register public key", agent_id=agent_id, error=str(e)
            )
            return False

    def sign_request(
        self, agent_id: str, payload: dict[str, Any]
    ) -> SignedRequest | None:
        """
        Sign request with agent's private key.

        Args:
            agent_id: Agent identifier
            payload: Request payload

        Returns:
            Signed request, or None if agent has no private key
        """
        private_key = self._agent_private_keys.get(agent_id)
        if not private_key:
            self.logger.error("No private key for agent", agent_id=agent_id)
            return None

        # Create request with nonce and timestamp
        signed_request = SignedRequest(
            agent_id=agent_id,
            timestamp=datetime.now(UTC),
            payload=payload,
            signature="",  # Placeholder
        )

        # Create message to sign (canonical representation)
        message = json.dumps(
            {
                "agent_id": agent_id,
                "timestamp": signed_request.timestamp.isoformat(),
                "nonce": signed_request.nonce,
                "payload": payload,
            },
            sort_keys=True,
        ).encode("utf-8")

        # Sign message
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        # Encode signature to base64
        signed_request.signature = base64.b64encode(signature).decode("utf-8")

        return signed_request

    def verify_signature(self, signed_request: SignedRequest) -> bool:
        """
        Verify request signature.

        Args:
            signed_request: Signed request to verify

        Returns:
            True if signature is valid, False otherwise
        """
        # Get agent's public key
        public_key = self._agent_public_keys.get(signed_request.agent_id)
        if not public_key:
            self.logger.warning(
                "No public key for agent", agent_id=signed_request.agent_id
            )
            self._security_stats["invalid_signatures"] += 1
            return False

        try:
            # Check if request is expired
            if signed_request.is_expired():
                self.logger.warning(
                    "Signed request expired", agent_id=signed_request.agent_id
                )
                return False

            # Check for replay attack (nonce reuse)
            if signed_request.nonce in self._used_nonces[signed_request.agent_id]:
                self.logger.warning(
                    "Replay attack detected",
                    agent_id=signed_request.agent_id,
                    nonce=signed_request.nonce,
                )
                self._security_stats["replay_attacks_prevented"] += 1
                return False

            # Reconstruct message
            message = json.dumps(
                {
                    "agent_id": signed_request.agent_id,
                    "timestamp": signed_request.timestamp.isoformat(),
                    "nonce": signed_request.nonce,
                    "payload": signed_request.payload,
                },
                sort_keys=True,
            ).encode("utf-8")

            # Decode signature
            signature = base64.b64decode(signed_request.signature)

            # Verify signature
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            # Record nonce to prevent replay
            self._used_nonces[signed_request.agent_id].add(signed_request.nonce)

            self.logger.debug("Signature verified", agent_id=signed_request.agent_id)
            return True

        except Exception as e:
            self.logger.error(
                "Signature verification failed",
                agent_id=signed_request.agent_id,
                error=str(e),
            )
            self._security_stats["invalid_signatures"] += 1
            return False

    # ==================== Rate Limiting ====================

    def check_rate_limit(
        self,
        agent_id: str,
        max_requests: int | None = None,
        window_seconds: int = 60,
    ) -> bool:
        """
        Check if agent is within rate limit.

        Args:
            agent_id: Agent identifier
            max_requests: Maximum requests per window (default from config)
            window_seconds: Rate limit window in seconds

        Returns:
            True if within limit, False if rate limited
        """
        # Get or create rate limit info
        if agent_id not in self._rate_limits:
            self._rate_limits[agent_id] = RateLimitInfo(
                agent_id=agent_id,
                max_requests=max_requests or self._default_rate_limit,
                window_duration_seconds=window_seconds,
            )

        rate_limit = self._rate_limits[agent_id]

        # Increment and check
        if not rate_limit.increment():
            self._security_stats["rate_limited_requests"] += 1
            self.logger.warning(
                "Agent rate limited",
                agent_id=agent_id,
                requests=rate_limit.requests_count,
                max_requests=rate_limit.max_requests,
            )
            return False

        return True

    def get_rate_limit_info(self, agent_id: str) -> RateLimitInfo | None:
        """Get rate limit info for agent."""
        return self._rate_limits.get(agent_id)

    def reset_rate_limit(self, agent_id: str) -> None:
        """Reset rate limit for agent."""
        if agent_id in self._rate_limits:
            self._rate_limits[agent_id].requests_count = 0
            self._rate_limits[agent_id].window_start = datetime.now(UTC)
            self.logger.info("Rate limit reset", agent_id=agent_id)

    # ==================== Input Validation ====================

    @staticmethod
    def sanitize_input(data: str, max_length: int = 10000) -> str:
        """
        Sanitize input string.

        Args:
            data: Input data
            max_length: Maximum allowed length

        Returns:
            Sanitized data

        Raises:
            ValueError: If input is invalid
        """
        if not data:
            return data

        # Check length
        if len(data) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")

        # Remove null bytes
        data = data.replace("\x00", "")

        # Strip leading/trailing whitespace
        data = data.strip()

        return data

    @staticmethod
    def validate_agent_id(agent_id: str) -> bool:
        """
        Validate agent ID format.

        Args:
            agent_id: Agent identifier

        Returns:
            True if valid, False otherwise
        """
        if not agent_id or not isinstance(agent_id, str):
            return False

        # Check length (reasonable bounds)
        if len(agent_id) < 3 or len(agent_id) > 256:
            return False

        # Check for null bytes or control characters
        if "\x00" in agent_id or any(ord(c) < 32 for c in agent_id):
            return False

        return True

    # ==================== Authentication ====================

    def authenticate_agent(
        self, request: AuthenticationRequest
    ) -> AuthenticationResponse:
        """
        Authenticate agent and issue tokens.

        Args:
            request: Authentication request

        Returns:
            Authentication response with tokens
        """
        try:
            # Validate agent_id format
            if not self.validate_agent_id(request.agent_id):
                return AuthenticationResponse(
                    success=False, error_message="Invalid agent ID format"
                )

            # TODO: Implement actual credential verification
            # For now, accept any request with credentials
            if not request.credentials:
                return AuthenticationResponse(
                    success=False, error_message="Missing credentials"
                )

            # Generate tokens
            access_token = self.generate_token(
                subject=request.agent_id,
                role=Role.AGENT,
                token_type=TokenType.ACCESS,
                agent_id=request.agent_id,
                additional_permissions=request.requested_permissions,
            )

            refresh_token = self.generate_token(
                subject=request.agent_id,
                role=Role.AGENT,
                token_type=TokenType.REFRESH,
                agent_id=request.agent_id,
            )

            self.logger.info("Agent authenticated", agent_id=request.agent_id)

            return AuthenticationResponse(
                success=True,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=settings.JWT_EXPIRATION_HOURS * 3600,
            )

        except Exception as e:
            self.logger.error("Authentication failed", error=str(e))
            return AuthenticationResponse(success=False, error_message=str(e))

    # ==================== Cleanup & Statistics ====================

    def cleanup_expired_nonces(self) -> int:
        """Remove expired nonces."""
        # This is a simplified version - in production, use timestamps
        removed = 0
        for agent_id in list(self._used_nonces.keys()):
            # Clear all nonces older than expiry time
            # In production, store nonces with timestamps
            self._used_nonces[agent_id].clear()
            removed += len(self._used_nonces[agent_id])

        return removed

    def get_security_stats(self) -> dict[str, Any]:
        """Get security statistics."""
        return {
            **self._security_stats,
            "registered_public_keys": len(self._agent_public_keys),
            "active_rate_limits": len(self._rate_limits),
            "tracked_nonces": sum(len(nonces) for nonces in self._used_nonces.values()),
        }


# Global security service instance
security_service = SecurityService()
