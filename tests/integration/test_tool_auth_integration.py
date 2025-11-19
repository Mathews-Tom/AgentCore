"""Integration tests for A2A authentication with tool execution (TOOL-019).

Tests end-to-end authentication flow with valid/invalid JWTs, RBAC enforcement,
and error handling.
"""

import pytest
from jose import jwt

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.security import Permission, Role, TokenPayload
from agentcore.agent_runtime.models.tool_integration import (
    AuthMethod,
    ToolCategory,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
)
from agentcore.agent_runtime.tools.auth import create_execution_context_from_jwt
from agentcore.agent_runtime.tools.base import ExecutionContext, Tool
from agentcore.agent_runtime.tools.executor import ToolExecutor
from agentcore.agent_runtime.tools.registry import ToolRegistry


class MockTool(Tool):
    """Mock tool for testing authentication."""

    async def execute(self, parameters: dict, context: ExecutionContext):
        """Execute mock tool."""
        from agentcore.agent_runtime.models.tool_integration import ToolResult

        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result={"message": "Success", "user_id": context.user_id},
            execution_time_ms=10,
        )


class MockAuthRequiredTool(Tool):
    """Mock tool requiring specific permissions."""

    async def execute(self, parameters: dict, context: ExecutionContext):
        """Execute mock tool."""
        from agentcore.agent_runtime.models.tool_integration import ToolResult

        return ToolResult(
            request_id=context.request_id,
            tool_id=self.metadata.tool_id,
            status=ToolExecutionStatus.SUCCESS,
            result={"message": "Authorized execution", "user_id": context.user_id},
            execution_time_ms=10,
        )


@pytest.fixture
def tool_registry():
    """Create tool registry with mock tools."""
    registry = ToolRegistry()

    # Tool with no authentication
    no_auth_tool = MockTool(
        ToolDefinition(
            tool_id="no_auth_tool",
            name="No Auth Tool",
            description="Tool without authentication",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.NONE,
        )
    )
    registry.register(no_auth_tool)

    # Tool requiring JWT authentication
    jwt_tool = MockTool(
        ToolDefinition(
            tool_id="jwt_tool",
            name="JWT Tool",
            description="Tool requiring JWT",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
        )
    )
    registry.register(jwt_tool)

    # Tool requiring specific permission (AGENT role doesn't have AGENT_DELETE)
    permission_tool = MockAuthRequiredTool(
        ToolDefinition(
            tool_id="permission_tool",
            name="Permission Tool",
            description="Tool requiring AGENT_DELETE permission",
            category=ToolCategory.UTILITY,
            parameters={},
            auth_method=AuthMethod.JWT,
            required_permissions=[Permission.AGENT_DELETE],
        )
    )
    registry.register(permission_tool)

    return registry


@pytest.fixture
def tool_executor(tool_registry):
    """Create tool executor with registry."""
    return ToolExecutor(registry=tool_registry)


def create_valid_jwt(
    subject: str = "user123",
    role: Role = Role.AGENT,
    agent_id: str | None = "agent456",
    additional_permissions: list[Permission] | None = None,
) -> str:
    """Create a valid JWT token for testing."""
    payload = TokenPayload.create(
        subject=subject,
        role=role,
        agent_id=agent_id,
        additional_permissions=additional_permissions,
    )

    payload_dict = payload.model_dump(mode="json")
    payload_dict["exp"] = int(payload.exp.timestamp())
    payload_dict["iat"] = int(payload.iat.timestamp())

    return jwt.encode(
        payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )


class TestToolExecutionWithValidJWT:
    """Test tool execution with valid JWT authentication."""

    @pytest.mark.asyncio
    async def test_execute_tool_with_valid_jwt(self, tool_executor):
        """Test executing tool with valid JWT creates proper context."""
        token = create_valid_jwt()
        context = create_execution_context_from_jwt(token)

        result = await tool_executor.execute_tool(
            tool_id="jwt_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_execute_tool_with_admin_role(self, tool_executor):
        """Test that admin role can execute any tool."""
        token = create_valid_jwt(subject="admin", role=Role.ADMIN)
        context = create_execution_context_from_jwt(token)

        result = await tool_executor.execute_tool(
            tool_id="permission_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_tool_with_required_permission(self, tool_executor):
        """Test executing tool when user has required permission."""
        token = create_valid_jwt(
            additional_permissions=[Permission.AGENT_DELETE],
        )
        context = create_execution_context_from_jwt(token)

        result = await tool_executor.execute_tool(
            tool_id="permission_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["message"] == "Authorized execution"


class TestToolExecutionWithInvalidJWT:
    """Test tool execution with invalid JWT authentication."""

    @pytest.mark.asyncio
    async def test_execute_tool_missing_user_id(self, tool_executor):
        """Test that missing user_id fails authentication."""
        context = ExecutionContext(
            user_id=None,  # Missing user_id
            agent_id="agent456",
        )

        result = await tool_executor.execute_tool(
            tool_id="jwt_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolAuthenticationError"
        assert "user_id missing" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_missing_agent_id(self, tool_executor):
        """Test that missing agent_id fails authentication."""
        context = ExecutionContext(
            user_id="user123",
            agent_id=None,  # Missing agent_id
        )

        result = await tool_executor.execute_tool(
            tool_id="jwt_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolAuthenticationError"
        assert "agent_id missing" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_missing_jwt_payload(self, tool_executor):
        """Test that missing JWT payload in context fails RBAC check."""
        context = ExecutionContext(
            user_id="user123",
            agent_id="agent456",
            # No jwt_payload in metadata
        )

        result = await tool_executor.execute_tool(
            tool_id="jwt_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolAuthenticationError"
        assert "Missing JWT payload" in result.error


class TestRBACEnforcement:
    """Test RBAC enforcement during tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_insufficient_permissions(self, tool_executor):
        """Test that user without required permission is denied access."""
        # Create token without AGENT_DELETE permission
        token = create_valid_jwt(
            subject="agent123",
            role=Role.AGENT,
            # No additional permissions - AGENT role doesn't have AGENT_DELETE
        )
        context = create_execution_context_from_jwt(token)

        result = await tool_executor.execute_tool(
            tool_id="permission_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolAuthenticationError"
        assert "Insufficient permissions" in result.error
        assert "agent:delete" in result.error

    @pytest.mark.asyncio
    async def test_execute_tool_service_role_allowed(self, tool_executor):
        """Test that service role has broader permissions."""
        token = create_valid_jwt(
            subject="service",
            role=Role.SERVICE,
        )
        context = create_execution_context_from_jwt(token)

        # Service role has AGENT_DELETE by default
        result = await tool_executor.execute_tool(
            tool_id="permission_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.SUCCESS


class TestNoAuthTools:
    """Test tools that don't require authentication."""

    @pytest.mark.asyncio
    async def test_execute_no_auth_tool_without_jwt(self, tool_executor):
        """Test that tools with auth_method=NONE don't require JWT."""
        context = ExecutionContext(
            user_id="anonymous",
            agent_id="anonymous",
            # No jwt_payload
        )

        result = await tool_executor.execute_tool(
            tool_id="no_auth_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.SUCCESS


class TestAuthenticationErrors:
    """Test authentication error handling."""

    @pytest.mark.asyncio
    async def test_execute_tool_returns_401_error_metadata(self, tool_executor):
        """Test that authentication failures return proper error metadata."""
        context = ExecutionContext(
            user_id=None,  # Missing - will fail auth
            agent_id="agent456",
        )

        result = await tool_executor.execute_tool(
            tool_id="jwt_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.FAILED
        assert result.error_type == "ToolAuthenticationError"
        assert result.error is not None

        # Check error categorization metadata
        assert result.metadata is not None
        assert "error_category" in result.metadata
        assert "error_code" in result.metadata
        assert "user_message" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_tool_authorization_error_metadata(self, tool_executor):
        """Test that authorization failures return proper error metadata."""
        # Create token without required permission
        token = create_valid_jwt()
        context = create_execution_context_from_jwt(token)

        result = await tool_executor.execute_tool(
            tool_id="permission_tool",
            parameters={},
            context=context,
        )

        assert result.status == ToolExecutionStatus.FAILED
        assert "Insufficient permissions" in result.error

        # Check error metadata
        assert result.metadata is not None


class TestEndToEndAuthentication:
    """Test end-to-end authentication scenarios."""

    @pytest.mark.asyncio
    async def test_complete_auth_flow_valid_jwt(self, tool_executor):
        """Test complete authentication flow with valid JWT."""
        # Create valid JWT
        token = create_valid_jwt(
            subject="user123",
            role=Role.AGENT,
            agent_id="agent456",
        )

        # Create execution context from JWT
        context = create_execution_context_from_jwt(
            token=token,
            trace_id="trace789",
            session_id="session999",
        )

        # Execute tool
        result = await tool_executor.execute_tool(
            tool_id="jwt_tool",
            parameters={},
            context=context,
        )

        # Verify success
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_complete_auth_flow_expired_jwt(self, tool_executor):
        """Test that expired JWT fails during context creation."""
        from agentcore.agent_runtime.tools.auth import AuthenticationError

        # Create expired token
        payload = TokenPayload.create(
            subject="user123",
            role=Role.AGENT,
            expiration_hours=-1,  # Expired
        )

        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(
            payload_dict, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
        )

        # Attempt to create context should fail
        with pytest.raises(AuthenticationError, match="Token expired"):
            create_execution_context_from_jwt(token)

    @pytest.mark.asyncio
    async def test_complete_auth_flow_invalid_signature(self, tool_executor):
        """Test that JWT with invalid signature fails."""
        from agentcore.agent_runtime.tools.auth import AuthenticationError

        # Create token with wrong secret
        payload = TokenPayload.create(subject="user123", role=Role.AGENT)

        payload_dict = payload.model_dump(mode="json")
        payload_dict["exp"] = int(payload.exp.timestamp())
        payload_dict["iat"] = int(payload.iat.timestamp())
        token = jwt.encode(payload_dict, "wrong_secret", algorithm=settings.JWT_ALGORITHM)

        # Attempt to create context should fail
        with pytest.raises(AuthenticationError, match="Invalid JWT token"):
            create_execution_context_from_jwt(token)
