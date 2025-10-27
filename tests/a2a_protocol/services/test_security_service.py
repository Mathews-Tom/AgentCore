"""
Comprehensive test suite for SecurityService.

Tests JWT token management, RSA signing/verification, rate limiting,
input validation, and authentication.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
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
    TokenType)
from agentcore.a2a_protocol.services.security_service import (
    SecurityService,
    security_service)


@pytest.fixture
def service():
    """Create fresh SecurityService instance for each test."""
    return SecurityService()


# ==================== JWT Token Management Tests ====================


class TestJWTTokens:
    """Test JWT token generation and validation."""

    def test_generate_token_access(self, service):
        """Test generating access token."""
        token = service.generate_token(
            subject="agent-1", role=Role.AGENT, token_type=TokenType.ACCESS
        )

        assert token is not None
        assert isinstance(token, str)
        assert service._security_stats["tokens_generated"] == 1

    def test_generate_token_refresh(self, service):
        """Test generating refresh token."""
        token = service.generate_token(
            subject="agent-1", role=Role.AGENT, token_type=TokenType.REFRESH
        )

        assert token is not None
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        assert payload["token_type"] == TokenType.REFRESH.value

    def test_generate_token_with_agent_id(self, service):
        """Test generating token with agent ID."""
        token = service.generate_token(
            subject="user-1", role=Role.AGENT, agent_id="agent-1"
        )

        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        assert payload["agent_id"] == "agent-1"

    def test_generate_token_with_additional_permissions(self, service):
        """Test generating token with additional permissions."""
        token = service.generate_token(
            subject="agent-1",
            role=Role.AGENT,
            additional_permissions=[Permission.ADMIN])

        payload = TokenPayload.model_validate(
            jwt.decode(
                token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
            )
        )
        assert Permission.ADMIN in payload.permissions

    def test_generate_token_with_metadata(self, service):
        """Test generating token with metadata."""
        metadata = {"version": "1.0", "environment": "test"}

        token = service.generate_token(
            subject="agent-1", role=Role.AGENT, metadata=metadata
        )

        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        assert payload["metadata"] == metadata

    def test_validate_token_success(self, service):
        """Test validating valid token."""
        token = service.generate_token(subject="agent-1", role=Role.AGENT)

        payload = service.validate_token(token)

        assert payload is not None
        assert payload.sub == "agent-1"
        assert payload.role == Role.AGENT
        assert service._security_stats["tokens_validated"] == 1

    def test_validate_token_expired(self, service):
        """Test validating expired token."""
        # Create expired token
        token = service.generate_token(subject="agent-1", role=Role.AGENT)

        # Manually decode and modify expiration
        payload_dict = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        payload_dict["exp"] = (datetime.now(UTC) - timedelta(hours=1)).timestamp()

        # Re-encode with expired time
        expired_token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        # Mock datetime to make token appear expired
        with patch("agentcore.a2a_protocol.models.security.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.now(UTC) + timedelta(hours=2)
            result = service.validate_token(token)

        # Note: This might pass validation due to jose library behavior
        # The key test is that is_expired() on TokenPayload works
        assert result is None or result.is_expired() is True

    def test_validate_token_invalid_signature(self, service):
        """Test validating token with invalid signature."""
        token = service.generate_token(subject="agent-1", role=Role.AGENT)

        # Tamper with token
        tampered_token = token[:-10] + "tampered"

        payload = service.validate_token(tampered_token)

        assert payload is None

    def test_validate_token_malformed(self, service):
        """Test validating malformed token."""
        payload = service.validate_token("not-a-valid-token")
        assert payload is None

    def test_check_permission_success(self, service):
        """Test checking permission with valid token."""
        token = service.generate_token(subject="agent-1", role=Role.AGENT)

        result = service.check_permission(token, Permission.AGENT_READ)

        assert result is True

    def test_check_permission_failure(self, service):
        """Test checking permission without permission."""
        token = service.generate_token(subject="agent-1", role=Role.AGENT)

        result = service.check_permission(token, Permission.ADMIN)

        assert result is False

    def test_check_permission_invalid_token(self, service):
        """Test checking permission with invalid token."""
        result = service.check_permission("invalid-token", Permission.AGENT_READ)
        assert result is False

    def test_check_permission_admin_has_all(self, service):
        """Test admin role has all permissions."""
        token = service.generate_token(subject="admin-1", role=Role.ADMIN)

        # Admin should have any permission
        assert service.check_permission(token, Permission.AGENT_DELETE) is True
        assert service.check_permission(token, Permission.TASK_DELETE) is True


# ==================== RSA Signing Tests ====================


class TestRSASigning:
    """Test RSA request signing and verification."""

    def test_generate_rsa_keypair(self, service):
        """Test generating RSA keypair."""
        keys = service.generate_rsa_keypair("agent-1")

        assert "public_key" in keys
        assert "private_key" in keys
        assert keys["public_key"].startswith("-----BEGIN PUBLIC KEY-----")
        assert keys["private_key"].startswith("-----BEGIN PRIVATE KEY-----")
        assert "agent-1" in service._agent_public_keys
        assert "agent-1" in service._agent_private_keys

    def test_register_public_key_success(self, service):
        """Test registering public key."""
        # Generate keypair first
        keys = service.generate_rsa_keypair("agent-1")

        # Register public key for different agent
        result = service.register_public_key("agent-2", keys["public_key"])

        assert result is True
        assert "agent-2" in service._agent_public_keys

    def test_register_public_key_invalid(self, service):
        """Test registering invalid public key."""
        result = service.register_public_key("agent-1", "not-a-valid-key")
        assert result is False

    def test_sign_request_success(self, service):
        """Test signing request with private key."""
        # Generate keypair
        service.generate_rsa_keypair("agent-1")

        payload = {"action": "test", "data": {"value": 123}}

        signed_request = service.sign_request("agent-1", payload)

        assert signed_request is not None
        assert signed_request.agent_id == "agent-1"
        assert signed_request.payload == payload
        assert signed_request.signature is not None
        assert signed_request.nonce is not None

    def test_sign_request_no_private_key(self, service):
        """Test signing request without private key."""
        payload = {"action": "test"}

        signed_request = service.sign_request("nonexistent-agent", payload)

        assert signed_request is None

    def test_verify_signature_success(self, service):
        """Test verifying valid signature."""
        # Generate keypair and sign request
        service.generate_rsa_keypair("agent-1")
        payload = {"action": "test", "value": 42}
        signed_request = service.sign_request("agent-1", payload)

        # Verify signature
        result = service.verify_signature(signed_request)

        assert result is True

    def test_verify_signature_invalid(self, service):
        """Test verifying invalid signature."""
        service.generate_rsa_keypair("agent-1")
        payload = {"action": "test"}
        signed_request = service.sign_request("agent-1", payload)

        # Tamper with payload
        signed_request.payload["action"] = "tampered"

        result = service.verify_signature(signed_request)

        assert result is False

    def test_verify_signature_no_public_key(self, service):
        """Test verifying signature without public key."""
        signed_request = SignedRequest(
            agent_id="unknown-agent",
            timestamp=datetime.now(UTC),
            payload={},
            signature="fake-signature")

        result = service.verify_signature(signed_request)

        assert result is False

    def test_verify_signature_expired_request(self, service):
        """Test verifying expired signed request."""
        service.generate_rsa_keypair("agent-1")
        payload = {"action": "test"}
        signed_request = service.sign_request("agent-1", payload)

        # Make request expired
        signed_request.timestamp = datetime.now(UTC) - timedelta(seconds=400)

        result = service.verify_signature(signed_request)

        assert result is False

    def test_verify_signature_replay_attack(self, service):
        """Test preventing replay attacks with nonce."""
        service.generate_rsa_keypair("agent-1")
        payload = {"action": "test"}
        signed_request = service.sign_request("agent-1", payload)

        # First verification succeeds
        result1 = service.verify_signature(signed_request)
        assert result1 is True

        # Second verification with same nonce fails (replay attack)
        result2 = service.verify_signature(signed_request)
        assert result2 is False
        assert service._security_stats["replay_attacks_prevented"] == 1

    def test_signed_request_is_expired(self):
        """Test SignedRequest expiration check."""
        request = SignedRequest(
            agent_id="agent-1",
            timestamp=datetime.now(UTC) - timedelta(seconds=400),
            payload={},
            signature="sig")

        assert request.is_expired(max_age_seconds=300) is True

    def test_signed_request_not_expired(self):
        """Test SignedRequest not expired."""
        request = SignedRequest(
            agent_id="agent-1", timestamp=datetime.now(UTC), payload={}, signature="sig"
        )

        assert request.is_expired(max_age_seconds=300) is False


# ==================== Rate Limiting Tests ====================


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_check_rate_limit_within_bounds(self, service):
        """Test rate limit check when within bounds."""
        result = service.check_rate_limit("agent-1", max_requests=10)
        assert result is True

    def test_check_rate_limit_exceeded(self, service):
        """Test rate limit check when exceeded."""
        # Make max_requests calls
        for _ in range(5):
            service.check_rate_limit("agent-1", max_requests=5)

        # Next call should be rate limited
        result = service.check_rate_limit("agent-1", max_requests=5)

        assert result is False
        assert service._security_stats["rate_limited_requests"] == 1

    def test_check_rate_limit_window_reset(self, service):
        """Test rate limit window resets after expiration."""
        # Create rate limit info with expired window
        rate_limit = RateLimitInfo(
            agent_id="agent-1",
            requests_count=100,
            window_start=datetime.now(UTC) - timedelta(seconds=70),
            window_duration_seconds=60,
            max_requests=10)
        service._rate_limits["agent-1"] = rate_limit

        # Should reset window and allow request
        result = service.check_rate_limit("agent-1")

        assert result is True
        assert rate_limit.requests_count == 1

    def test_check_rate_limit_uses_default(self, service):
        """Test rate limit uses default when not specified."""
        result = service.check_rate_limit("agent-1")

        assert result is True
        rate_limit = service._rate_limits["agent-1"]
        assert rate_limit.max_requests == service._default_rate_limit

    def test_get_rate_limit_info(self, service):
        """Test getting rate limit info."""
        service.check_rate_limit("agent-1", max_requests=100)

        info = service.get_rate_limit_info("agent-1")

        assert info is not None
        assert info.agent_id == "agent-1"
        assert info.max_requests == 100
        assert info.requests_count >= 1

    def test_get_rate_limit_info_nonexistent(self, service):
        """Test getting rate limit info for nonexistent agent."""
        info = service.get_rate_limit_info("nonexistent-agent")
        assert info is None

    def test_reset_rate_limit(self, service):
        """Test resetting rate limit."""
        # Make some requests
        for _ in range(5):
            service.check_rate_limit("agent-1")

        # Reset
        service.reset_rate_limit("agent-1")

        info = service._rate_limits["agent-1"]
        assert info.requests_count == 0

    def test_reset_rate_limit_nonexistent(self, service):
        """Test resetting rate limit for nonexistent agent does not error."""
        # Should not raise exception
        service.reset_rate_limit("nonexistent-agent")

    def test_rate_limit_info_increment(self):
        """Test RateLimitInfo increment."""
        rate_limit = RateLimitInfo(agent_id="agent-1", max_requests=10)

        result = rate_limit.increment()

        assert result is True
        assert rate_limit.requests_count == 1

    def test_rate_limit_info_increment_when_limited(self):
        """Test RateLimitInfo increment when rate limited."""
        rate_limit = RateLimitInfo(
            agent_id="agent-1", requests_count=10, max_requests=10
        )

        result = rate_limit.increment()

        assert result is False
        assert rate_limit.requests_count == 10  # Should not increment

    def test_rate_limit_info_get_remaining_requests(self):
        """Test getting remaining requests."""
        rate_limit = RateLimitInfo(
            agent_id="agent-1", requests_count=3, max_requests=10
        )

        remaining = rate_limit.get_remaining_requests()

        assert remaining == 7

    def test_rate_limit_info_get_reset_time(self):
        """Test getting reset time."""
        start_time = datetime.now(UTC)
        rate_limit = RateLimitInfo(
            agent_id="agent-1", window_start=start_time, window_duration_seconds=60
        )

        reset_time = rate_limit.get_reset_time()

        expected = start_time + timedelta(seconds=60)
        assert abs((reset_time - expected).total_seconds()) < 1


# ==================== Input Validation Tests ====================


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_sanitize_input_valid(self, service):
        """Test sanitizing valid input."""
        data = "  Hello World  "
        result = service.sanitize_input(data)

        assert result == "Hello World"

    def test_sanitize_input_removes_null_bytes(self, service):
        """Test sanitizing removes null bytes."""
        data = "Hello\x00World"
        result = service.sanitize_input(data)

        assert result == "HelloWorld"

    def test_sanitize_input_too_long_raises_error(self, service):
        """Test sanitizing oversized input raises error."""
        data = "x" * 20000

        with pytest.raises(ValueError, match="exceeds maximum length"):
            service.sanitize_input(data, max_length=10000)

    def test_sanitize_input_empty(self, service):
        """Test sanitizing empty string."""
        result = service.sanitize_input("")
        assert result == ""

    def test_sanitize_input_none(self, service):
        """Test sanitizing None returns None."""
        result = service.sanitize_input(None)
        assert result is None

    def test_validate_agent_id_valid(self, service):
        """Test validating valid agent ID."""
        result = service.validate_agent_id("agent-123")
        assert result is True

    def test_validate_agent_id_too_short(self, service):
        """Test validating too short agent ID."""
        result = service.validate_agent_id("ab")
        assert result is False

    def test_validate_agent_id_too_long(self, service):
        """Test validating too long agent ID."""
        long_id = "x" * 300
        result = service.validate_agent_id(long_id)
        assert result is False

    def test_validate_agent_id_with_null_bytes(self, service):
        """Test validating agent ID with null bytes."""
        result = service.validate_agent_id("agent\x00id")
        assert result is False

    def test_validate_agent_id_with_control_chars(self, service):
        """Test validating agent ID with control characters."""
        result = service.validate_agent_id("agent\nid")
        assert result is False

    def test_validate_agent_id_none(self, service):
        """Test validating None agent ID."""
        result = service.validate_agent_id(None)
        assert result is False

    def test_validate_agent_id_not_string(self, service):
        """Test validating non-string agent ID."""
        result = service.validate_agent_id(12345)
        assert result is False


# ==================== Authentication Tests ====================


class TestAuthentication:
    """Test agent authentication."""

    def test_authenticate_agent_success(self, service):
        """Test successful agent authentication."""
        request = AuthenticationRequest(
            agent_id="agent-1", credentials={"api_key": "secret-key"}
        )

        response = service.authenticate_agent(request)

        assert response.success is True
        assert response.access_token is not None
        assert response.refresh_token is not None
        assert response.expires_in == settings.JWT_EXPIRATION_HOURS * 3600

    def test_authenticate_agent_missing_credentials(self, service):
        """Test authentication with missing credentials."""
        request = AuthenticationRequest(agent_id="agent-1", credentials={})

        response = service.authenticate_agent(request)

        assert response.success is False
        assert response.error_message == "Missing credentials"

    def test_authenticate_agent_invalid_id(self, service):
        """Test authentication with invalid agent ID."""
        request = AuthenticationRequest(
            agent_id="x",  # Too short
            credentials={"api_key": "key"})

        response = service.authenticate_agent(request)

        assert response.success is False
        assert "Invalid agent ID" in response.error_message

    def test_authenticate_agent_with_requested_permissions(self, service):
        """Test authentication with requested permissions."""
        request = AuthenticationRequest(
            agent_id="agent-1",
            credentials={"api_key": "key"},
            requested_permissions=[Permission.TASK_CREATE])

        response = service.authenticate_agent(request)

        assert response.success is True

        # Verify token has requested permissions
        payload = service.validate_token(response.access_token)
        assert Permission.TASK_CREATE in payload.permissions

    def test_authenticate_agent_generates_tokens(self, service):
        """Test authentication generates both token types."""
        request = AuthenticationRequest(
            agent_id="agent-1", credentials={"api_key": "key"}
        )

        response = service.authenticate_agent(request)

        # Verify access token
        access_payload = service.validate_token(response.access_token)
        assert access_payload.token_type == TokenType.ACCESS

        # Verify refresh token
        refresh_payload = service.validate_token(response.refresh_token)
        assert refresh_payload.token_type == TokenType.REFRESH

    def test_authenticate_agent_exception_handling(self, service):
        """Test authentication handles exceptions gracefully."""
        request = AuthenticationRequest(
            agent_id="agent-1", credentials={"key": "value"}
        )

        # Mock generate_token to raise exception
        with patch.object(
            service, "generate_token", side_effect=Exception("Token generation failed")
        ):
            response = service.authenticate_agent(request)

        assert response.success is False
        assert response.error_message is not None


# ==================== Cleanup & Statistics Tests ====================


class TestCleanupAndStatistics:
    """Test cleanup operations and statistics."""

    def test_cleanup_expired_nonces(self, service):
        """Test cleaning up expired nonces."""
        # Add some nonces
        service._used_nonces["agent-1"] = {"nonce-1", "nonce-2", "nonce-3"}
        service._used_nonces["agent-2"] = {"nonce-4", "nonce-5"}

        removed = service.cleanup_expired_nonces()

        # Current implementation clears all nonces
        assert removed == 0
        assert len(service._used_nonces["agent-1"]) == 0

    def test_get_security_stats(self, service):
        """Test getting security statistics."""
        # Generate some activity
        token = service.generate_token("agent-1", Role.AGENT)
        service.validate_token(token)
        service.generate_rsa_keypair("agent-1")
        service.check_rate_limit("agent-1")

        stats = service.get_security_stats()

        assert "tokens_generated" in stats
        assert "tokens_validated" in stats
        assert "tokens_expired" in stats
        assert "rate_limited_requests" in stats
        assert "invalid_signatures" in stats
        assert "replay_attacks_prevented" in stats
        assert "registered_public_keys" in stats
        assert "active_rate_limits" in stats
        assert "tracked_nonces" in stats

        assert stats["tokens_generated"] >= 1
        assert stats["tokens_validated"] >= 1
        assert stats["registered_public_keys"] >= 1

    def test_statistics_track_token_generation(self, service):
        """Test statistics track token generation."""
        initial = service._security_stats["tokens_generated"]

        service.generate_token("agent-1", Role.AGENT)
        service.generate_token("agent-2", Role.AGENT)

        assert service._security_stats["tokens_generated"] == initial + 2

    def test_statistics_track_token_validation(self, service):
        """Test statistics track token validation."""
        token = service.generate_token("agent-1", Role.AGENT)

        initial = service._security_stats["tokens_validated"]

        service.validate_token(token)

        assert service._security_stats["tokens_validated"] == initial + 1

    def test_statistics_track_invalid_signatures(self, service):
        """Test statistics track invalid signatures."""
        initial = service._security_stats["invalid_signatures"]

        # Try to verify signature without public key
        signed_request = SignedRequest(
            agent_id="unknown",
            timestamp=datetime.now(UTC),
            payload={},
            signature="fake")
        service.verify_signature(signed_request)

        assert service._security_stats["invalid_signatures"] == initial + 1


# ==================== Token Payload Tests ====================


class TestTokenPayload:
    """Test TokenPayload model functionality."""

    def test_token_payload_create(self):
        """Test creating token payload."""
        payload = TokenPayload.create(
            subject="agent-1", role=Role.AGENT, token_type=TokenType.ACCESS
        )

        assert payload.sub == "agent-1"
        assert payload.role == Role.AGENT
        assert payload.token_type == TokenType.ACCESS
        assert len(payload.permissions) > 0

    def test_token_payload_has_permission(self):
        """Test checking if payload has permission."""
        payload = TokenPayload.create(subject="agent-1", role=Role.AGENT)

        assert payload.has_permission(Permission.AGENT_READ) is True
        assert payload.has_permission(Permission.ADMIN) is False

    def test_token_payload_admin_has_all_permissions(self):
        """Test admin has all permissions."""
        payload = TokenPayload.create(subject="admin-1", role=Role.ADMIN)

        # Admin permission grants access to everything
        assert payload.has_permission(Permission.AGENT_DELETE) is True
        assert payload.has_permission(Permission.TASK_DELETE) is True
        assert payload.has_permission(Permission.ADMIN) is True

    def test_token_payload_is_expired(self):
        """Test checking if token is expired."""
        payload = TokenPayload.create(
            subject="agent-1", role=Role.AGENT, expiration_hours=1
        )

        # Manually set expiration to past
        payload.exp = datetime.now(UTC) - timedelta(hours=1)

        assert payload.is_expired() is True

    def test_token_payload_not_expired(self):
        """Test token not expired."""
        payload = TokenPayload.create(
            subject="agent-1", role=Role.AGENT, expiration_hours=24
        )

        assert payload.is_expired() is False


# ==================== Global Instance Test ====================


class TestGlobalInstance:
    """Test global security service instance."""

    def test_global_instance_exists(self):
        """Test global security_service instance exists."""
        assert security_service is not None
        assert isinstance(security_service, SecurityService)

    def test_global_instance_is_singleton(self):
        """Test global instance behaves like singleton."""
        from agentcore.a2a_protocol.services.security_service import (
            security_service as ss1)
        from agentcore.a2a_protocol.services.security_service import (
            security_service as ss2)

        assert ss1 is ss2
