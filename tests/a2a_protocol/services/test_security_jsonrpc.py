"""
Unit tests for Security JSON-RPC Service.

Tests for security JSON-RPC method handlers covering authentication, token management,
RSA key management, rate limiting, and security statistics.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.security import (
    AuthenticationResponse,
    Permission,
    RateLimitInfo,
    Role,
    SignedRequest,
    TokenPayload,
    TokenType)


class TestAuthentication:
    """Test auth.authenticate JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_authenticate_success(self, mock_service):
        """Test successful authentication."""
        from agentcore.a2a_protocol.services.security_jsonrpc import handle_authenticate

        # Setup mock response
        auth_response = AuthenticationResponse(
            success=True,
            access_token="test-access-token",
            refresh_token="test-refresh-token",
            expires_in=3600,
            token_type="Bearer")
        mock_service.authenticate_agent = Mock(return_value=auth_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.authenticate",
            params={
                "agent_id": "test-agent",
                "credentials": {"api_key": "secret-key"},
            },
            id="1")

        result = await handle_authenticate(request)

        assert result["success"] is True
        assert result["access_token"] == "test-access-token"
        assert result["expires_in"] == 3600
        mock_service.authenticate_agent.assert_called_once()

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_authenticate_with_permissions(self, mock_service):
        """Test authentication with requested permissions."""
        from agentcore.a2a_protocol.services.security_jsonrpc import handle_authenticate

        auth_response = AuthenticationResponse(
            success=True,
            access_token="token",
            refresh_token="refresh",
            expires_in=3600,
            token_type="Bearer")
        mock_service.authenticate_agent = Mock(return_value=auth_response)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.authenticate",
            params={
                "agent_id": "test-agent",
                "credentials": {"api_key": "key"},
                "requested_permissions": ["task:read"],
            },
            id="1")

        result = await handle_authenticate(request)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_authenticate_missing_params(self):
        """Test authentication with missing parameters."""
        from agentcore.a2a_protocol.services.security_jsonrpc import handle_authenticate

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.authenticate",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_authenticate(request)

    @pytest.mark.asyncio
    async def test_authenticate_missing_agent_id(self):
        """Test authentication with missing agent_id."""
        from agentcore.a2a_protocol.services.security_jsonrpc import handle_authenticate

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.authenticate",
            params={"credentials": {"api_key": "key"}},
            id="1")

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_authenticate(request)

    @pytest.mark.asyncio
    async def test_authenticate_missing_credentials(self):
        """Test authentication with missing credentials."""
        from agentcore.a2a_protocol.services.security_jsonrpc import handle_authenticate

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.authenticate",
            params={"agent_id": "test-agent"},
            id="1")

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_authenticate(request)


class TestTokenValidation:
    """Test auth.validate_token JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_validate_token_success(self, mock_service):
        """Test successful token validation."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_validate_token)

        token_payload = TokenPayload(
            sub="test-agent",
            token_type=TokenType.ACCESS,
            role=Role.AGENT,
            exp=datetime.now(UTC) + timedelta(hours=1),
            iat=datetime.now(UTC),
            permissions=[Permission.AGENT_READ])
        mock_service.validate_token = Mock(return_value=token_payload)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.validate_token",
            params={"token": "valid-token"},
            id="1")

        result = await handle_validate_token(request)

        assert result["valid"] is True
        assert result["payload"]["sub"] == "test-agent"
        mock_service.validate_token.assert_called_once_with("valid-token")

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_validate_token_invalid(self, mock_service):
        """Test invalid token validation."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_validate_token)

        mock_service.validate_token = Mock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.validate_token",
            params={"token": "invalid-token"},
            id="1")

        with pytest.raises(ValueError, match="Invalid or expired token"):
            await handle_validate_token(request)

    @pytest.mark.asyncio
    async def test_validate_token_missing_params(self):
        """Test token validation with missing parameters."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_validate_token)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.validate_token",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameter required: token"):
            await handle_validate_token(request)

    @pytest.mark.asyncio
    async def test_validate_token_missing_token(self):
        """Test token validation with missing token field."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_validate_token)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.validate_token",
            params={},
            id="1")

        with pytest.raises(ValueError, match="Parameter required: token"):
            await handle_validate_token(request)


class TestPermissionCheck:
    """Test auth.check_permission JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_check_permission_has_permission(self, mock_service):
        """Test permission check when user has permission."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_check_permission)

        mock_service.check_permission = Mock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.check_permission",
            params={"token": "valid-token", "permission": "agent:read"},
            id="1")

        result = await handle_check_permission(request)

        assert result["has_permission"] is True
        assert result["permission"] == "agent:read"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_check_permission_no_permission(self, mock_service):
        """Test permission check when user lacks permission."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_check_permission)

        mock_service.check_permission = Mock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.check_permission",
            params={"token": "valid-token", "permission": "agent:delete"},
            id="1")

        result = await handle_check_permission(request)

        assert result["has_permission"] is False
        assert result["permission"] == "agent:delete"

    @pytest.mark.asyncio
    async def test_check_permission_missing_params(self):
        """Test permission check with missing parameters."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_check_permission)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.check_permission",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameters required"):
            await handle_check_permission(request)

    @pytest.mark.asyncio
    async def test_check_permission_missing_token(self):
        """Test permission check with missing token."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_check_permission)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="auth.check_permission",
            params={"permission": "agent:read"},
            id="1")

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_check_permission(request)


class TestRSAKeypair:
    """Test security.generate_keypair JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_generate_keypair_success(self, mock_service):
        """Test successful keypair generation."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_generate_keypair)

        mock_service.generate_rsa_keypair = Mock(
            return_value={
                "public_key": "-----BEGIN PUBLIC KEY-----\ntest",
                "private_key": "-----BEGIN PRIVATE KEY-----\ntest",
            }
        )

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.generate_keypair",
            params={"agent_id": "test-agent"},
            id="1")

        result = await handle_generate_keypair(request)

        assert result["success"] is True
        assert result["agent_id"] == "test-agent"
        assert "public_key" in result
        assert "private_key" in result

    @pytest.mark.asyncio
    async def test_generate_keypair_missing_params(self):
        """Test keypair generation with missing parameters."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_generate_keypair)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.generate_keypair",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_generate_keypair(request)


class TestPublicKeyRegistration:
    """Test security.register_public_key JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_register_public_key_success(self, mock_service):
        """Test successful public key registration."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_register_public_key)

        mock_service.register_public_key = Mock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.register_public_key",
            params={
                "agent_id": "test-agent",
                "public_key": "-----BEGIN PUBLIC KEY-----\ntest",
            },
            id="1")

        result = await handle_register_public_key(request)

        assert result["success"] is True
        assert result["agent_id"] == "test-agent"
        assert "successfully" in result["message"].lower()

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_register_public_key_failure(self, mock_service):
        """Test failed public key registration."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_register_public_key)

        mock_service.register_public_key = Mock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.register_public_key",
            params={
                "agent_id": "test-agent",
                "public_key": "invalid-key",
            },
            id="1")

        with pytest.raises(ValueError, match="Failed to register public key"):
            await handle_register_public_key(request)

    @pytest.mark.asyncio
    async def test_register_public_key_missing_params(self):
        """Test public key registration with missing parameters."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_register_public_key)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.register_public_key",
            params={"agent_id": "test-agent"},
            id="1")

        with pytest.raises(ValueError, match="Missing required parameters"):
            await handle_register_public_key(request)


class TestSignatureVerification:
    """Test security.verify_signature JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_verify_signature_valid(self, mock_service):
        """Test valid signature verification."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_verify_signature)

        mock_service.verify_signature = Mock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.verify_signature",
            params={
                "signed_request": {
                    "agent_id": "test-agent",
                    "payload": {"test": "data"},
                    "signature": "test-signature",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            },
            id="1")

        result = await handle_verify_signature(request)

        assert result["valid"] is True
        assert result["agent_id"] == "test-agent"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_verify_signature_invalid(self, mock_service):
        """Test invalid signature verification."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_verify_signature)

        mock_service.verify_signature = Mock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.verify_signature",
            params={
                "signed_request": {
                    "agent_id": "test-agent",
                    "payload": {"test": "data"},
                    "signature": "bad-signature",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            },
            id="1")

        result = await handle_verify_signature(request)

        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_verify_signature_missing_params(self):
        """Test signature verification with missing parameters."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_verify_signature)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.verify_signature",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameter required"):
            await handle_verify_signature(request)


class TestRateLimiting:
    """Test rate limiting JSON-RPC methods."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_check_rate_limit_within_limit(self, mock_service):
        """Test rate limit check when within limit."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_check_rate_limit)

        rate_limit_info = RateLimitInfo(
            agent_id="test-agent",
            requests_count=5,
            max_requests=100,
            window_seconds=60,
            window_start=datetime.now(UTC))
        mock_service.check_rate_limit = Mock(return_value=True)
        mock_service.get_rate_limit_info = Mock(return_value=rate_limit_info)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.check_rate_limit",
            params={"agent_id": "test-agent"},
            id="1")

        result = await handle_check_rate_limit(request)

        assert result["within_limit"] is True
        assert result["requests_count"] == 5
        assert result["max_requests"] == 100

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_check_rate_limit_exceeded(self, mock_service):
        """Test rate limit check when limit exceeded."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_check_rate_limit)

        rate_limit_info = RateLimitInfo(
            agent_id="test-agent",
            requests_count=101,
            max_requests=100,
            window_seconds=60,
            window_start=datetime.now(UTC))
        mock_service.check_rate_limit = Mock(return_value=False)
        mock_service.get_rate_limit_info = Mock(return_value=rate_limit_info)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.check_rate_limit",
            params={"agent_id": "test-agent"},
            id="1")

        result = await handle_check_rate_limit(request)

        assert result["within_limit"] is False
        assert result["requests_count"] == 101

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_check_rate_limit_no_info(self, mock_service):
        """Test rate limit check when no info exists."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_check_rate_limit)

        mock_service.check_rate_limit = Mock(return_value=True)
        mock_service.get_rate_limit_info = Mock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.check_rate_limit",
            params={"agent_id": "test-agent"},
            id="1")

        result = await handle_check_rate_limit(request)

        assert result["within_limit"] is True
        assert result["requests_count"] == 0

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_get_rate_limit_info_exists(self, mock_service):
        """Test getting rate limit info when it exists."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_get_rate_limit_info)

        rate_limit_info = RateLimitInfo(
            agent_id="test-agent",
            requests_count=10,
            max_requests=100,
            window_seconds=60,
            window_start=datetime.now(UTC))
        mock_service.get_rate_limit_info = Mock(return_value=rate_limit_info)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.get_rate_limit_info",
            params={"agent_id": "test-agent"},
            id="1")

        result = await handle_get_rate_limit_info(request)

        assert result["has_limit"] is True
        assert result["agent_id"] == "test-agent"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_get_rate_limit_info_not_exists(self, mock_service):
        """Test getting rate limit info when it does not exist."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_get_rate_limit_info)

        mock_service.get_rate_limit_info = Mock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.get_rate_limit_info",
            params={"agent_id": "test-agent"},
            id="1")

        result = await handle_get_rate_limit_info(request)

        assert result["has_limit"] is False
        assert result["agent_id"] == "test-agent"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_reset_rate_limit_success(self, mock_service):
        """Test successful rate limit reset."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_reset_rate_limit)

        mock_service.reset_rate_limit = Mock(return_value=None)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.reset_rate_limit",
            params={"agent_id": "test-agent"},
            id="1")

        result = await handle_reset_rate_limit(request)

        assert result["success"] is True
        assert result["agent_id"] == "test-agent"
        mock_service.reset_rate_limit.assert_called_once_with("test-agent")


class TestSecurityStats:
    """Test security.get_stats JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_get_security_stats(self, mock_service):
        """Test getting security statistics."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_get_security_stats)

        stats = {
            "total_authentications": 100,
            "active_tokens": 50,
            "failed_attempts": 5,
        }
        mock_service.get_security_stats = Mock(return_value=stats)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.get_stats",
            params={},
            id="1")

        result = await handle_get_security_stats(request)

        assert result["success"] is True
        assert result["stats"] == stats
        assert "timestamp" in result


class TestAgentIdValidation:
    """Test security.validate_agent_id JSON-RPC method."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_validate_agent_id_valid(self, mock_service):
        """Test valid agent ID validation."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_validate_agent_id)

        mock_service.validate_agent_id = Mock(return_value=True)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.validate_agent_id",
            params={"agent_id": "valid-agent-123"},
            id="1")

        result = await handle_validate_agent_id(request)

        assert result["valid"] is True
        assert result["agent_id"] == "valid-agent-123"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.security_jsonrpc.security_service")
    async def test_validate_agent_id_invalid(self, mock_service):
        """Test invalid agent ID validation."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_validate_agent_id)

        mock_service.validate_agent_id = Mock(return_value=False)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.validate_agent_id",
            params={"agent_id": "invalid@agent"},
            id="1")

        result = await handle_validate_agent_id(request)

        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_agent_id_missing_params(self):
        """Test agent ID validation with missing parameters."""
        from agentcore.a2a_protocol.services.security_jsonrpc import (
            handle_validate_agent_id)

        request = JsonRpcRequest(
            jsonrpc="2.0",
            method="security.validate_agent_id",
            params=None,
            id="1")

        with pytest.raises(ValueError, match="Parameter required: agent_id"):
            await handle_validate_agent_id(request)
