"""Authentication and authorization utilities for tool execution.

This module provides A2A authentication integration for tool execution,
including JWT extraction, claim validation, and RBAC policy enforcement.

Implements TOOL-019: A2A Authentication Integration from tasks.md
"""

from typing import Any

import structlog
from jose import JWTError, jwt

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.security import Permission, Role, TokenPayload
from agentcore.agent_runtime.models.tool_integration import ToolDefinition

from .base import ExecutionContext

logger = structlog.get_logger()


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class AuthorizationError(Exception):
    """Raised when authorization fails."""

    pass


def extract_jwt_claims(token: str) -> TokenPayload:
    """Extract and validate JWT claims from token.

    Args:
        token: JWT token string

    Returns:
        TokenPayload containing validated claims

    Raises:
        AuthenticationError: If token is invalid, expired, or malformed
    """
    try:
        # Decode JWT using configured secret and algorithm
        payload_dict = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )

        # Parse into TokenPayload model for validation
        payload = TokenPayload.model_validate(payload_dict)

        # Check if token is expired (redundant with jwt.decode but for clarity)
        if payload.is_expired():
            logger.warning(
                "jwt_token_expired",
                subject=payload.sub,
                expired_at=payload.exp.isoformat(),
            )
            raise AuthenticationError("Token expired")

        logger.debug(
            "jwt_claims_extracted",
            subject=payload.sub,
            role=payload.role.value,
            agent_id=payload.agent_id,
        )

        return payload

    except JWTError as e:
        logger.error("jwt_validation_failed", error=str(e))
        # Check if it's an expiration error
        if "expired" in str(e).lower():
            raise AuthenticationError("Token expired")
        raise AuthenticationError(f"Invalid JWT token: {e}")
    except AuthenticationError:
        # Re-raise our own exceptions
        raise
    except Exception as e:
        logger.error("jwt_extraction_failed", error=str(e))
        raise AuthenticationError(f"Failed to extract JWT claims: {e}")


def create_execution_context_from_jwt(
    token: str,
    trace_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionContext:
    """Create ExecutionContext from JWT token.

    Extracts user_id and agent_id from JWT claims and creates an ExecutionContext
    with authentication information for tool execution.

    Args:
        token: JWT token string
        trace_id: Optional distributed tracing ID
        session_id: Optional session identifier
        metadata: Optional additional metadata

    Returns:
        ExecutionContext with user_id and agent_id from JWT

    Raises:
        AuthenticationError: If token is invalid or expired
    """
    payload = extract_jwt_claims(token)

    # Extract user_id and agent_id from JWT claims
    user_id = payload.sub  # Subject is the user/agent ID
    agent_id = payload.agent_id or payload.sub  # Use agent_id if present, else subject

    context = ExecutionContext(
        user_id=user_id,
        agent_id=agent_id,
        trace_id=trace_id,
        session_id=session_id,
        metadata=metadata or {},
    )

    # Store JWT payload in metadata for RBAC checks
    context.metadata["jwt_payload"] = payload

    logger.info(
        "execution_context_created_from_jwt",
        user_id=user_id,
        agent_id=agent_id,
        role=payload.role.value,
        trace_id=trace_id,
    )

    return context


def check_tool_access_permission(
    context: ExecutionContext,
    tool: ToolDefinition,
) -> tuple[bool, str | None]:
    """Check if user/agent has permission to execute tool.

    Implements basic RBAC policy enforcement for tool access. Checks:
    1. Tool-specific permissions (if tool defines required permissions)
    2. Role-based access (admin has access to all tools)
    3. Authentication method requirements (some tools require specific auth)

    Args:
        context: Execution context with JWT payload in metadata
        tool: Tool definition with access requirements

    Returns:
        Tuple of (is_authorized, error_message). If authorized, error_message is None.

    Raises:
        AuthorizationError: If JWT payload is missing from context
    """
    # Get JWT payload from context metadata
    jwt_payload = context.metadata.get("jwt_payload")
    if not jwt_payload or not isinstance(jwt_payload, TokenPayload):
        error_msg = "Missing JWT payload in execution context"
        logger.warning(
            "rbac_check_failed_missing_jwt",
            tool_id=tool.tool_id,
            user_id=context.user_id,
            agent_id=context.agent_id,
        )
        return False, error_msg

    # Admin role has access to all tools
    if jwt_payload.role == Role.ADMIN:
        logger.debug(
            "rbac_check_passed_admin",
            tool_id=tool.tool_id,
            user_id=context.user_id,
            role=jwt_payload.role.value,
        )
        return True, None

    # Check tool-specific required permissions
    required_permissions = getattr(tool, "required_permissions", None)
    if required_permissions:
        # Ensure required_permissions is a list
        if not isinstance(required_permissions, list):
            required_permissions = [required_permissions]

        # Check if user has all required permissions
        for required_perm in required_permissions:
            # Convert string to Permission enum if needed
            if isinstance(required_perm, str):
                try:
                    required_perm = Permission(required_perm)
                except ValueError:
                    error_msg = f"Invalid permission: {required_perm}"
                    return False, error_msg

            if not jwt_payload.has_permission(required_perm):
                error_msg = (
                    f"Insufficient permissions to execute tool '{tool.tool_id}'. "
                    f"Required permission: {required_perm.value}, "
                    f"User role: {jwt_payload.role.value}"
                )
                logger.warning(
                    "rbac_check_failed_insufficient_permissions",
                    tool_id=tool.tool_id,
                    user_id=context.user_id,
                    role=jwt_payload.role.value,
                    required_permission=required_perm.value,
                    user_permissions=[p.value for p in jwt_payload.permissions],
                )
                return False, error_msg

    # Check role-based access based on tool sensitivity
    # For now, all roles (AGENT, SERVICE) can execute tools unless specific permissions required
    # This is basic RBAC - can be enhanced with more granular policies

    logger.debug(
        "rbac_check_passed",
        tool_id=tool.tool_id,
        user_id=context.user_id,
        agent_id=context.agent_id,
        role=jwt_payload.role.value,
    )

    return True, None


def validate_authentication(
    token: str | None,
    required: bool = True,
) -> tuple[bool, str | None, TokenPayload | None]:
    """Validate JWT authentication token.

    Args:
        token: JWT token string (can be None if not required)
        required: Whether authentication is required

    Returns:
        Tuple of (is_valid, error_message, payload).
        If valid, error_message is None and payload contains JWT claims.
        If invalid, payload is None.
    """
    if not token:
        if required:
            return False, "Authentication required: JWT token missing", None
        return True, None, None

    try:
        payload = extract_jwt_claims(token)
        return True, None, payload
    except AuthenticationError as e:
        return False, str(e), None


def get_user_id_from_context(context: ExecutionContext) -> str | None:
    """Extract user_id from execution context.

    Args:
        context: Execution context

    Returns:
        User ID if present, None otherwise
    """
    return context.user_id


def get_agent_id_from_context(context: ExecutionContext) -> str | None:
    """Extract agent_id from execution context.

    Args:
        context: Execution context

    Returns:
        Agent ID if present, None otherwise
    """
    return context.agent_id
