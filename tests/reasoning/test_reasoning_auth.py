"""
Integration tests for Reasoning JSON-RPC authentication and authorization.

Tests JWT token validation, permission checking, and auth error scenarios.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest
from agentcore.a2a_protocol.models.security import Permission, Role, TokenPayload
from src.agentcore.reasoning.models.reasoning_models import (
    BoundedContextIterationResult,
    BoundedContextResult,
    IterationMetrics,
)
from src.agentcore.reasoning.services.reasoning_jsonrpc import (
    _extract_jwt_token,
    _validate_authentication,
    handle_bounded_reasoning,
)


@pytest.fixture
def valid_token_payload() -> TokenPayload:
    """Create valid token payload with reasoning:execute permission."""
    return TokenPayload.create(
        subject="agent-123",
        role=Role.AGENT,
        token_type="access",
        expiration_hours=24,
        agent_id="agent-123",
    )


@pytest.fixture
def mock_reasoning_result() -> BoundedContextResult:
    """Create mock reasoning result for testing."""
    iteration = BoundedContextIterationResult(
        content="Reasoning content <answer>42</answer>",
        has_answer=True,
        answer="42",
        carryover=None,
        metrics=IterationMetrics(
            iteration=0,
            tokens=500,
            has_answer=True,
            carryover_generated=False,
            execution_time_ms=1250,
        ),
    )

    return BoundedContextResult(
        answer="42",
        iterations=[iteration],
        total_tokens=500,
        total_iterations=1,
        compute_savings_pct=15.5,
        carryover_compressions=0,
        execution_time_ms=1250,
    )


@pytest.fixture
def valid_request_params() -> dict:
    """Valid request parameters for testing."""
    return {
        "query": "What is 2+2?",
        "temperature": 0.7,
        "chunk_size": 8192,
        "carryover_size": 4096,
        "max_iterations": 5,
    }


def test_extract_jwt_token_from_params():
    """Test JWT token extraction from request params."""
    # Token in params
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={"auth_token": "test-token-123", "query": "test"},
        id="test-123",
    )

    token = _extract_jwt_token(request)
    assert token == "test-token-123"


def test_extract_jwt_token_missing():
    """Test JWT token extraction when missing."""
    # No params
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        id="test-123",
    )

    token = _extract_jwt_token(request)
    assert token is None

    # Params without auth_token
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={"query": "test"},
        id="test-123",
    )

    token = _extract_jwt_token(request)
    assert token is None


def test_extract_jwt_token_invalid_type():
    """Test JWT token extraction with invalid type."""
    # auth_token not a string
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={"auth_token": 12345, "query": "test"},
        id="test-123",
    )

    token = _extract_jwt_token(request)
    assert token is None


@pytest.mark.asyncio
async def test_validate_authentication_missing_token():
    """Test authentication validation with missing token."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={"query": "test"},
        id="test-123",
    )

    with pytest.raises(ValueError, match="Authentication required: Missing JWT token"):
        _validate_authentication(request)


@pytest.mark.asyncio
async def test_validate_authentication_invalid_token():
    """Test authentication validation with invalid token."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={"auth_token": "invalid-token", "query": "test"},
        id="test-123",
    )

    with patch(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
    ) as mock_security:
        mock_security.validate_token.return_value = None

        with pytest.raises(ValueError, match="Authentication failed: Invalid or expired JWT token"):
            _validate_authentication(request)

        mock_security.validate_token.assert_called_once_with("invalid-token")


@pytest.mark.asyncio
async def test_validate_authentication_insufficient_permissions(valid_token_payload):
    """Test authentication validation with insufficient permissions."""
    # Create token without reasoning:execute permission
    token_payload = TokenPayload.create(
        subject="agent-123",
        role=Role.AGENT,
        token_type="access",
        expiration_hours=24,
        agent_id="agent-123",
        additional_permissions=[],  # No additional permissions
    )
    # Remove reasoning:execute from permissions
    token_payload.permissions = [p for p in token_payload.permissions if p != Permission.REASONING_EXECUTE]

    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={"auth_token": "valid-token", "query": "test"},
        id="test-123",
    )

    with patch(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
    ) as mock_security:
        mock_security.validate_token.return_value = token_payload

        with pytest.raises(PermissionError, match="Authorization failed: Missing required permission"):
            _validate_authentication(request)


@pytest.mark.asyncio
async def test_validate_authentication_success(valid_token_payload):
    """Test successful authentication validation."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={"auth_token": "valid-token", "query": "test"},
        id="test-123",
    )

    with patch(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
    ) as mock_security:
        mock_security.validate_token.return_value = valid_token_payload

        # Should not raise
        _validate_authentication(request)

        mock_security.validate_token.assert_called_once_with("valid-token")


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_auth_success(
    valid_request_params: dict,
    valid_token_payload: TokenPayload,
    mock_reasoning_result: BoundedContextResult,
):
    """Test successful bounded reasoning with valid authentication."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={**valid_request_params, "auth_token": "valid-token"},
        id="test-123",
    )

    with (
        patch(
            "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
        ) as mock_security,
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch(
            "src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine"
        ) as mock_engine_class,
    ):
        mock_security.validate_token.return_value = valid_token_payload
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(return_value=mock_reasoning_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        result = await handle_bounded_reasoning(request)

        # Assertions
        assert result["success"] is True
        assert result["answer"] == "42"
        mock_security.validate_token.assert_called_once_with("valid-token")


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_missing_token(valid_request_params: dict):
    """Test bounded reasoning fails with missing token."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params=valid_request_params,  # No auth_token
        id="test-123",
    )

    with pytest.raises(ValueError, match="Authentication required: Missing JWT token"):
        await handle_bounded_reasoning(request)


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_invalid_token(valid_request_params: dict):
    """Test bounded reasoning fails with invalid token."""
    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={**valid_request_params, "auth_token": "invalid-token"},
        id="test-123",
    )

    with patch(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
    ) as mock_security:
        mock_security.validate_token.return_value = None

        with pytest.raises(ValueError, match="Authentication failed: Invalid or expired JWT token"):
            await handle_bounded_reasoning(request)

        mock_security.validate_token.assert_called_once_with("invalid-token")


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_insufficient_permissions(valid_request_params: dict):
    """Test bounded reasoning fails with insufficient permissions."""
    # Create token without reasoning:execute permission
    token_payload = TokenPayload.create(
        subject="agent-123",
        role=Role.AGENT,
        token_type="access",
        expiration_hours=24,
        agent_id="agent-123",
        additional_permissions=[],
    )
    token_payload.permissions = [p for p in token_payload.permissions if p != Permission.REASONING_EXECUTE]

    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={**valid_request_params, "auth_token": "valid-token"},
        id="test-123",
    )

    with patch(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
    ) as mock_security:
        mock_security.validate_token.return_value = token_payload

        with pytest.raises(ValueError, match="Authorization failed: Missing required permission"):
            await handle_bounded_reasoning(request)


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_admin_permission(
    valid_request_params: dict,
    mock_reasoning_result: BoundedContextResult,
):
    """Test bounded reasoning succeeds with admin permission."""
    # Create admin token
    admin_token_payload = TokenPayload.create(
        subject="admin-user",
        role=Role.ADMIN,
        token_type="access",
        expiration_hours=24,
    )

    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={**valid_request_params, "auth_token": "admin-token"},
        id="test-123",
    )

    with (
        patch(
            "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
        ) as mock_security,
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch(
            "src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine"
        ) as mock_engine_class,
    ):
        mock_security.validate_token.return_value = admin_token_payload
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(return_value=mock_reasoning_result)
        mock_engine_class.return_value = mock_engine

        # Execute - admin has all permissions
        result = await handle_bounded_reasoning(request)

        assert result["success"] is True
        assert result["answer"] == "42"


@pytest.mark.asyncio
async def test_handle_bounded_reasoning_with_a2a_context_auth(
    valid_request_params: dict,
    valid_token_payload: TokenPayload,
    mock_reasoning_result: BoundedContextResult,
):
    """Test bounded reasoning with A2A context and authentication."""
    a2a_context = A2AContext(
        source_agent="agent-123",
        target_agent="agent-456",
        trace_id="trace-abc",
        timestamp="2025-01-01T00:00:00Z",
    )

    request = JsonRpcRequest(
        method="reasoning.bounded_context",
        params={**valid_request_params, "auth_token": "valid-token"},
        id="test-123",
        a2a_context=a2a_context,
    )

    with (
        patch(
            "src.agentcore.reasoning.services.reasoning_jsonrpc.security_service"
        ) as mock_security,
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient"),
        patch(
            "src.agentcore.reasoning.services.reasoning_jsonrpc.BoundedContextEngine"
        ) as mock_engine_class,
        patch("src.agentcore.reasoning.services.reasoning_jsonrpc.logger") as mock_logger,
    ):
        mock_security.validate_token.return_value = valid_token_payload
        mock_engine = MagicMock()
        mock_engine.reason = AsyncMock(return_value=mock_reasoning_result)
        mock_engine_class.return_value = mock_engine

        # Execute
        result = await handle_bounded_reasoning(request)

        assert result["success"] is True

        # Verify A2A context was included in auth logs
        mock_logger.info.assert_any_call(
            "authentication_success",
            subject=valid_token_payload.sub,
            role=valid_token_payload.role.value,
            method="reasoning.bounded_context",
            trace_id="trace-abc",
        )
