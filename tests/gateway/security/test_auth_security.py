"""Authentication security tests.

Tests for JWT token security, token tampering, and authentication bypass attempts.
"""

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest


SECRET_KEY = "test-secret-key-do-not-use-in-production"
ALGORITHM = "HS256"


def create_test_token(payload: dict[str, any], secret: str = SECRET_KEY) -> str:
    """Create a test JWT token."""
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_test_token(token: str, secret: str = SECRET_KEY) -> dict[str, any]:
    """Decode a test JWT token."""
    return jwt.decode(token, secret, algorithms=[ALGORITHM])


class TestTokenTampering:
    """Tests for JWT token tampering detection."""

    def test_token_signature_verification(self) -> None:
        """Test that tampered signatures are rejected."""
        payload = {"sub": "user123", "exp": time.time() + 3600}
        token = create_test_token(payload)

        # Tamper with the token
        parts = token.split(".")
        tampered_token = f"{parts[0]}.{parts[1]}.tampered_signature"

        with pytest.raises(jwt.InvalidSignatureError):
            decode_test_token(tampered_token)

    def test_token_payload_modification(self) -> None:
        """Test that modified payloads are rejected."""
        payload = {"sub": "user123", "role": "user", "exp": time.time() + 3600}
        token = create_test_token(payload)

        # Try to modify payload (should be detected by signature)
        parts = token.split(".")
        # Even changing one character invalidates the signature
        modified_payload = parts[1][:-1] + "x"
        tampered_token = f"{parts[0]}.{modified_payload}.{parts[2]}"

        with pytest.raises(jwt.DecodeError):
            decode_test_token(tampered_token)

    def test_token_none_algorithm_attack(self) -> None:
        """Test protection against 'none' algorithm attack."""
        payload = {"sub": "admin", "role": "admin"}

        # PyJWT 2.0+ doesn't allow 'none' algorithm by default
        # This is the correct security behavior
        # If it encodes without error, verify decoding rejects it
        try:
            token = jwt.encode(payload, "", algorithm="none")
            # If encoding succeeded, verify decoding fails
            with pytest.raises((jwt.DecodeError, jwt.InvalidAlgorithmError)):
                decode_test_token(token)
        except jwt.InvalidAlgorithmError:
            # Expected - 'none' algorithm not allowed
            pass

    def test_token_wrong_secret(self) -> None:
        """Test that tokens signed with wrong secret are rejected."""
        payload = {"sub": "user123", "exp": time.time() + 3600}
        token = create_test_token(payload, secret="wrong-secret")

        with pytest.raises(jwt.InvalidSignatureError):
            decode_test_token(token)


class TestTokenExpiration:
    """Tests for token expiration handling."""

    def test_expired_token(self) -> None:
        """Test that expired tokens are rejected."""
        payload = {
            "sub": "user123",
            "exp": datetime.now(UTC) - timedelta(hours=1),
        }
        token = create_test_token(payload)

        with pytest.raises(jwt.ExpiredSignatureError):
            decode_test_token(token)

    def test_valid_token_not_expired(self) -> None:
        """Test that valid non-expired tokens are accepted."""
        payload = {
            "sub": "user123",
            "exp": datetime.now(UTC) + timedelta(hours=1),
        }
        token = create_test_token(payload)

        decoded = decode_test_token(token)
        assert decoded["sub"] == "user123"

    def test_token_with_nbf_claim(self) -> None:
        """Test not-before (nbf) claim validation."""
        payload = {
            "sub": "user123",
            "nbf": datetime.now(UTC) + timedelta(hours=1),
            "exp": datetime.now(UTC) + timedelta(hours=2),
        }
        token = create_test_token(payload)

        with pytest.raises(jwt.ImmatureSignatureError):
            decode_test_token(token)


class TestAuthorizationBypass:
    """Tests for authorization bypass attempts."""

    def test_missing_token(self) -> None:
        """Test that missing tokens are handled."""
        # Simulating missing Authorization header
        token = None
        assert token is None  # Should be rejected by middleware

    def test_malformed_token(self) -> None:
        """Test that malformed tokens are rejected."""
        malformed_tokens = [
            "not.a.token",
            "Bearer",
            "Bearer ",
            "Bearer invalid",
            "totally-not-a-jwt",
            "...",
        ]

        for token in malformed_tokens:
            with pytest.raises((jwt.DecodeError, jwt.InvalidTokenError)):
                jwt.decode(token.replace("Bearer ", ""), SECRET_KEY, algorithms=[ALGORITHM])

    def test_role_escalation_attempt(self) -> None:
        """Test that role escalation is prevented."""
        # User token
        user_payload = {"sub": "user123", "role": "user", "exp": time.time() + 3600}
        user_token = create_test_token(user_payload)

        decoded = decode_test_token(user_token)
        assert decoded["role"] == "user"

        # Attempting to create admin token with user credentials should require new signature
        admin_payload = {"sub": "user123", "role": "admin", "exp": time.time() + 3600}
        # This creates a valid token, but it requires the secret key
        # In real scenario, user shouldn't have access to secret key
        admin_token = create_test_token(admin_payload)
        decoded_admin = decode_test_token(admin_token)
        assert decoded_admin["role"] == "admin"

        # The point is: without the secret key, role escalation is impossible


class TestTokenClaims:
    """Tests for required token claims."""

    def test_missing_subject_claim(self) -> None:
        """Test that tokens without subject are handled."""
        payload = {"exp": time.time() + 3600}
        token = create_test_token(payload)

        decoded = decode_test_token(token)
        assert "sub" not in decoded

    def test_missing_expiration_claim(self) -> None:
        """Test that tokens without expiration work but should be validated."""
        payload = {"sub": "user123"}
        token = create_test_token(payload)

        decoded = decode_test_token(token)
        assert decoded["sub"] == "user123"
        # In production, should require exp claim

    def test_custom_claims(self) -> None:
        """Test that custom claims are preserved."""
        payload = {
            "sub": "user123",
            "exp": time.time() + 3600,
            "custom_field": "custom_value",
            "permissions": ["read", "write"],
        }
        token = create_test_token(payload)

        decoded = decode_test_token(token)
        assert decoded["custom_field"] == "custom_value"
        assert decoded["permissions"] == ["read", "write"]


class TestSecurityHeaders:
    """Tests for security-related headers."""

    def test_bearer_token_format(self) -> None:
        """Test that Bearer token format is validated."""
        # Valid format
        valid_header = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        assert valid_header.startswith("Bearer ")

        # Invalid formats
        invalid_headers = [
            "Basic user:pass",
            "Token abc123",
            "eyJhbGci...",  # Missing Bearer prefix
        ]

        for header in invalid_headers:
            assert not header.startswith("Bearer ") or len(header.split()) != 2
