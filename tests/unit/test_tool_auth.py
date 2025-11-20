"""Unit tests for tool authentication utilities (TOOL-019).

Tests JWT extraction, claim validation, and RBAC policy enforcement.
"""

from datetime import UTC, datetime, timedelta

import pytest
from jose import jwt

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.security import Permission, Role, TokenPayload
from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
)
from agentcore.agent_runtime.tools.auth import (
    AuthenticationError,
    check_tool_access_permission,
    create_execution_context_from_jwt,
    extract_jwt_claims,
    get_agent_id_from_context,
    get_user_id_from_context,
    validate_authentication,
)
from agentcore.agent_runtime.tools.base import ExecutionContext


class TestJWTExtraction:
    """Test JWT token extraction and validation."""

    def test_extract_jwt_claims_valid_token(self):
        """Test extracting claims from valid JWT token."""
        # Create a valid token
        payload = TokenPayload.create(
            subject="user123",
            role=Role.AGENT,
            agent_id="agent456",
            expiration_hours=1,
        )

        # Encode JWT
        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        # Extract claims
        extracted = extract_jwt_claims(token)

        assert extracted.sub == "user123"
        assert extracted.role == Role.AGENT
        assert extracted.agent_id == "agent456"
        assert not extracted.is_expired()

    def test_extract_jwt_claims_expired_token(self):
        """Test that expired token raises AuthenticationError."""
        # Create an expired token
        payload = TokenPayload.create(
            subject="user123",
            role=Role.AGENT,
            expiration_hours=-1,  # Expired 1 hour ago
        )

        # Encode JWT
        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        # Extract claims should fail
        with pytest.raises(AuthenticationError, match="Token expired"):
            extract_jwt_claims(token)

    def test_extract_jwt_claims_invalid_signature(self):
        """Test that token with invalid signature raises AuthenticationError."""
        # Create token with wrong secret
        payload = TokenPayload.create(
            subject="user123",
            role=Role.AGENT,
        )

        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(payload_dict, "wrong_secret", algorithm=settings.JWT_ALGORITHM)

        with pytest.raises(AuthenticationError, match="Invalid JWT token"):
            extract_jwt_claims(token)

    def test_extract_jwt_claims_malformed_token(self):
        """Test that malformed token raises AuthenticationError."""
        with pytest.raises(AuthenticationError, match="Invalid JWT token"):
            extract_jwt_claims("not.a.valid.jwt.token")

    def test_extract_jwt_claims_with_permissions(self):
        """Test extracting claims from token with custom permissions."""
        payload = TokenPayload.create(
            subject="admin",
            role=Role.ADMIN,
            additional_permissions=[Permission.ADMIN],
        )

        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        extracted = extract_jwt_claims(token)

        assert extracted.sub == "admin"
        assert extracted.role == Role.ADMIN
        assert Permission.ADMIN in extracted.permissions


class TestExecutionContextCreation:
    """Test creating ExecutionContext from JWT."""

    def test_create_execution_context_from_jwt(self):
        """Test creating ExecutionContext with user_id and agent_id from JWT."""
        payload = TokenPayload.create(
            subject="user123",
            role=Role.AGENT,
            agent_id="agent456",
        )

        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        context = create_execution_context_from_jwt(
            token=token,
            trace_id="trace789",
            session_id="session999",
        )

        assert context.user_id == "user123"
        assert context.agent_id == "agent456"
        assert context.trace_id == "trace789"
        assert context.session_id == "session999"
        assert "jwt_payload" in context.metadata
        assert isinstance(context.metadata["jwt_payload"], TokenPayload)

    def test_create_execution_context_agent_id_fallback(self):
        """Test that agent_id falls back to subject if not provided."""
        payload = TokenPayload.create(
            subject="agent123",
            role=Role.AGENT,
            agent_id=None,  # No explicit agent_id
        )

        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        context = create_execution_context_from_jwt(token)

        assert context.user_id == "agent123"
        assert context.agent_id == "agent123"  # Falls back to subject

    def test_create_execution_context_with_metadata(self):
        """Test creating ExecutionContext with additional metadata."""
        payload = TokenPayload.create(subject="user123", role=Role.AGENT)

        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        context = create_execution_context_from_jwt(
            token,
            metadata={"custom_key": "custom_value"},
        )

        assert context.metadata["custom_key"] == "custom_value"
        assert "jwt_payload" in context.metadata


class TestRBACEnforcement:
    """Test RBAC policy enforcement for tool access."""

    def test_check_tool_access_admin_role(self):
        """Test that admin role has access to all tools."""
        payload = TokenPayload.create(subject="admin", role=Role.ADMIN)
        context = ExecutionContext(
            user_id="admin",
            agent_id="admin",
            metadata={"jwt_payload": payload},
        )

        tool = ToolDefinition(
            tool_id="test_tool",
            name="Test Tool",
            description="Test",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
        )

        is_authorized, error = check_tool_access_permission(context, tool)

        assert is_authorized is True
        assert error is None

    def test_check_tool_access_agent_role_allowed(self):
        """Test that agent role can access tools without specific permissions."""
        payload = TokenPayload.create(subject="agent123", role=Role.AGENT)
        context = ExecutionContext(
            user_id="agent123",
            agent_id="agent123",
            metadata={"jwt_payload": payload},
        )

        tool = ToolDefinition(
            tool_id="test_tool",
            name="Test Tool",
            description="Test",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
        )

        is_authorized, error = check_tool_access_permission(context, tool)

        assert is_authorized is True
        assert error is None

    def test_check_tool_access_required_permission_granted(self):
        """Test that user with required permission can access tool."""
        payload = TokenPayload.create(
            subject="agent123",
            role=Role.AGENT,
            additional_permissions=[Permission.TASK_CREATE],
        )
        context = ExecutionContext(
            user_id="agent123",
            agent_id="agent123",
            metadata={"jwt_payload": payload},
        )

        tool = ToolDefinition(
            tool_id="task_creator",
            name="Task Creator",
            description="Creates tasks",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
            required_permissions=[Permission.TASK_CREATE],
        )

        is_authorized, error = check_tool_access_permission(context, tool)

        assert is_authorized is True
        assert error is None

    def test_check_tool_access_required_permission_denied(self):
        """Test that user without required permission is denied access."""
        payload = TokenPayload.create(
            subject="agent123",
            role=Role.AGENT,
            # AGENT role doesn't have AGENT_DELETE permission
        )
        context = ExecutionContext(
            user_id="agent123",
            agent_id="agent123",
            metadata={"jwt_payload": payload},
        )

        tool = ToolDefinition(
            tool_id="agent_deleter",
            name="Agent Deleter",
            description="Deletes agents",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
            required_permissions=[Permission.AGENT_DELETE],
        )

        is_authorized, error = check_tool_access_permission(context, tool)

        assert is_authorized is False
        assert error is not None
        assert "Insufficient permissions" in error
        assert "agent:delete" in error

    def test_check_tool_access_missing_jwt_payload(self):
        """Test that missing JWT payload results in denied access."""
        context = ExecutionContext(
            user_id="agent123",
            agent_id="agent123",
            # No jwt_payload in metadata
        )

        tool = ToolDefinition(
            tool_id="test_tool",
            name="Test Tool",
            description="Test",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
        )

        is_authorized, error = check_tool_access_permission(context, tool)

        assert is_authorized is False
        assert error is not None
        assert "Missing JWT payload" in error

    def test_check_tool_access_multiple_required_permissions(self):
        """Test that user must have all required permissions."""
        payload = TokenPayload.create(
            subject="agent123",
            role=Role.AGENT,
            # AGENT role has TASK_CREATE but not AGENT_DELETE
        )
        context = ExecutionContext(
            user_id="agent123",
            agent_id="agent123",
            metadata={"jwt_payload": payload},
        )

        tool = ToolDefinition(
            tool_id="admin_tool",
            name="Admin Tool",
            description="Requires multiple permissions",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
            required_permissions=[Permission.TASK_CREATE, Permission.AGENT_DELETE],
        )

        is_authorized, error = check_tool_access_permission(context, tool)

        assert is_authorized is False
        assert error is not None
        assert "Insufficient permissions" in error


class TestAuthenticationValidation:
    """Test authentication validation utilities."""

    def test_validate_authentication_valid_token(self):
        """Test validating a valid authentication token."""
        payload = TokenPayload.create(subject="user123", role=Role.AGENT)
        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        is_valid, error, extracted_payload = validate_authentication(token)

        assert is_valid is True
        assert error is None
        assert extracted_payload is not None
        assert extracted_payload.sub == "user123"

    def test_validate_authentication_missing_token_required(self):
        """Test that missing token fails when required."""
        is_valid, error, payload = validate_authentication(None, required=True)

        assert is_valid is False
        assert error == "Authentication required: JWT token missing"
        assert payload is None

    def test_validate_authentication_missing_token_not_required(self):
        """Test that missing token succeeds when not required."""
        is_valid, error, payload = validate_authentication(None, required=False)

        assert is_valid is True
        assert error is None
        assert payload is None

    def test_validate_authentication_invalid_token(self):
        """Test that invalid token fails validation."""
        is_valid, error, payload = validate_authentication("invalid.token.here")

        assert is_valid is False
        assert error is not None
        assert "Invalid JWT token" in error
        assert payload is None


class TestContextUtilities:
    """Test context utility functions."""

    def test_get_user_id_from_context(self):
        """Test extracting user_id from context."""
        context = ExecutionContext(user_id="user123", agent_id="agent456")

        user_id = get_user_id_from_context(context)

        assert user_id == "user123"

    def test_get_user_id_from_context_none(self):
        """Test extracting user_id when None."""
        context = ExecutionContext(agent_id="agent456")

        user_id = get_user_id_from_context(context)

        assert user_id is None

    def test_get_agent_id_from_context(self):
        """Test extracting agent_id from context."""
        context = ExecutionContext(user_id="user123", agent_id="agent456")

        agent_id = get_agent_id_from_context(context)

        assert agent_id == "agent456"

    def test_get_agent_id_from_context_none(self):
        """Test extracting agent_id when None."""
        context = ExecutionContext(user_id="user123")

        agent_id = get_agent_id_from_context(context)

        assert agent_id is None
