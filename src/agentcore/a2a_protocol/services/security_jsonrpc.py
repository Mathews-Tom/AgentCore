"""
Security JSON-RPC Methods

JSON-RPC 2.0 methods for authentication, token management, RSA key management,
and rate limiting.
"""

from datetime import UTC, datetime
from typing import Any

import structlog

from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.models.security import (
    AuthenticationRequest,
    Permission,
    Role,
    SignedRequest,
    TokenType,
)
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.a2a_protocol.services.security_service import security_service

logger = structlog.get_logger()


@register_jsonrpc_method("auth.authenticate")
async def handle_authenticate(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Authenticate agent and issue JWT tokens.

    Method: auth.authenticate
    Params:
        - agent_id: string
        - credentials: object (e.g., {"api_key": "..."})
        - requested_permissions: array of Permission enums (optional)

    Returns:
        Authentication response with tokens
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: agent_id, credentials")

    try:
        agent_id = request.params.get("agent_id")
        credentials = request.params.get("credentials")

        if not agent_id or not credentials:
            raise ValueError("Missing required parameters: agent_id and/or credentials")

        requested_permissions = request.params.get("requested_permissions")
        if requested_permissions:
            requested_permissions = [Permission(p) for p in requested_permissions]

        # Create authentication request
        auth_request = AuthenticationRequest(
            agent_id=agent_id,
            credentials=credentials,
            requested_permissions=requested_permissions,
        )

        # Authenticate
        response = security_service.authenticate_agent(auth_request)

        logger.info(
            "Authentication attempt",
            agent_id=agent_id,
            success=response.success,
            method="auth.authenticate",
        )

        return response.model_dump()

    except Exception as e:
        logger.error("Authentication failed", error=str(e))
        raise


@register_jsonrpc_method("auth.validate_token")
async def handle_validate_token(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Validate JWT token.

    Method: auth.validate_token
    Params:
        - token: string

    Returns:
        Token payload if valid, error otherwise
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: token")

    token = request.params.get("token")
    if not token:
        raise ValueError("Missing required parameter: token")

    payload = security_service.validate_token(token)

    if not payload:
        raise ValueError("Invalid or expired token")

    logger.debug(
        "Token validated via JSON-RPC",
        subject=payload.sub,
        method="auth.validate_token",
    )

    return {"valid": True, "payload": payload.model_dump(mode="json")}


@register_jsonrpc_method("auth.check_permission")
async def handle_check_permission(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Check if token has required permission.

    Method: auth.check_permission
    Params:
        - token: string
        - permission: string (Permission enum)

    Returns:
        Permission check result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: token, permission")

    token = request.params.get("token")
    permission_str = request.params.get("permission")

    if not token or not permission_str:
        raise ValueError("Missing required parameters: token and/or permission")

    try:
        permission = Permission(permission_str)
        has_permission = security_service.check_permission(token, permission)

        logger.debug(
            "Permission checked via JSON-RPC",
            permission=permission_str,
            has_permission=has_permission,
            method="auth.check_permission",
        )

        return {"has_permission": has_permission, "permission": permission_str}

    except Exception as e:
        logger.error("Permission check failed", error=str(e))
        raise


@register_jsonrpc_method("security.generate_keypair")
async def handle_generate_keypair(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Generate RSA key pair for agent.

    Method: security.generate_keypair
    Params:
        - agent_id: string

    Returns:
        Public and private keys in PEM format
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    try:
        keys = security_service.generate_rsa_keypair(agent_id)

        logger.info(
            "RSA keypair generated via JSON-RPC",
            agent_id=agent_id,
            method="security.generate_keypair",
        )

        return {"success": True, "agent_id": agent_id, **keys}

    except Exception as e:
        logger.error("Keypair generation failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("security.register_public_key")
async def handle_register_public_key(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Register agent's public key.

    Method: security.register_public_key
    Params:
        - agent_id: string
        - public_key: string (PEM format)

    Returns:
        Registration confirmation
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameters required: agent_id, public_key")

    agent_id = request.params.get("agent_id")
    public_key = request.params.get("public_key")

    if not agent_id or not public_key:
        raise ValueError("Missing required parameters: agent_id and/or public_key")

    try:
        success = security_service.register_public_key(agent_id, public_key)

        if not success:
            raise ValueError("Failed to register public key")

        logger.info(
            "Public key registered via JSON-RPC",
            agent_id=agent_id,
            method="security.register_public_key",
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "message": "Public key registered successfully",
        }

    except Exception as e:
        logger.error("Public key registration failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("security.verify_signature")
async def handle_verify_signature(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Verify signed request.

    Method: security.verify_signature
    Params:
        - signed_request: SignedRequest object

    Returns:
        Verification result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: signed_request")

    try:
        signed_request_data = request.params.get("signed_request")
        if not signed_request_data:
            raise ValueError("Missing required parameter: signed_request")

        # Parse signed request
        signed_request = SignedRequest.model_validate(signed_request_data)

        # Verify signature
        is_valid = security_service.verify_signature(signed_request)

        logger.info(
            "Signature verified via JSON-RPC",
            agent_id=signed_request.agent_id,
            is_valid=is_valid,
            method="security.verify_signature",
        )

        return {
            "valid": is_valid,
            "agent_id": signed_request.agent_id,
            "timestamp": signed_request.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error("Signature verification failed", error=str(e))
        raise


@register_jsonrpc_method("security.check_rate_limit")
async def handle_check_rate_limit(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Check agent rate limit.

    Method: security.check_rate_limit
    Params:
        - agent_id: string
        - max_requests: number (optional)
        - window_seconds: number (optional, default 60)

    Returns:
        Rate limit check result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    max_requests = request.params.get("max_requests")
    window_seconds = request.params.get("window_seconds", 60)

    try:
        within_limit = security_service.check_rate_limit(
            agent_id=agent_id, max_requests=max_requests, window_seconds=window_seconds
        )

        rate_limit_info = security_service.get_rate_limit_info(agent_id)

        logger.debug(
            "Rate limit checked via JSON-RPC",
            agent_id=agent_id,
            within_limit=within_limit,
            method="security.check_rate_limit",
        )

        return {
            "within_limit": within_limit,
            "agent_id": agent_id,
            "requests_count": rate_limit_info.requests_count if rate_limit_info else 0,
            "max_requests": rate_limit_info.max_requests
            if rate_limit_info
            else max_requests,
            "remaining": rate_limit_info.get_remaining_requests()
            if rate_limit_info
            else 0,
            "reset_time": rate_limit_info.get_reset_time().isoformat()
            if rate_limit_info
            else None,
        }

    except Exception as e:
        logger.error("Rate limit check failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("security.get_rate_limit_info")
async def handle_get_rate_limit_info(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get rate limit information for agent.

    Method: security.get_rate_limit_info
    Params:
        - agent_id: string

    Returns:
        Rate limit information
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    rate_limit_info = security_service.get_rate_limit_info(agent_id)

    if not rate_limit_info:
        return {"agent_id": agent_id, "has_limit": False}

    logger.debug(
        "Rate limit info retrieved via JSON-RPC",
        agent_id=agent_id,
        method="security.get_rate_limit_info",
    )

    return {
        "agent_id": agent_id,
        "has_limit": True,
        **rate_limit_info.model_dump(mode="json"),
    }


@register_jsonrpc_method("security.reset_rate_limit")
async def handle_reset_rate_limit(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Reset rate limit for agent.

    Method: security.reset_rate_limit
    Params:
        - agent_id: string

    Returns:
        Reset confirmation
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if not agent_id:
        raise ValueError("Missing required parameter: agent_id")

    try:
        security_service.reset_rate_limit(agent_id)

        logger.info(
            "Rate limit reset via JSON-RPC",
            agent_id=agent_id,
            method="security.reset_rate_limit",
        )

        return {
            "success": True,
            "agent_id": agent_id,
            "message": "Rate limit reset successfully",
        }

    except Exception as e:
        logger.error("Rate limit reset failed", error=str(e), agent_id=agent_id)
        raise


@register_jsonrpc_method("security.get_stats")
async def handle_get_security_stats(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get security statistics.

    Method: security.get_stats
    Params: none

    Returns:
        Security statistics
    """
    stats = security_service.get_security_stats()

    logger.debug("Security stats retrieved via JSON-RPC", method="security.get_stats")

    return {"success": True, "stats": stats, "timestamp": datetime.now(UTC).isoformat()}


@register_jsonrpc_method("security.validate_agent_id")
async def handle_validate_agent_id(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Validate agent ID format.

    Method: security.validate_agent_id
    Params:
        - agent_id: string

    Returns:
        Validation result
    """
    if not request.params or not isinstance(request.params, dict):
        raise ValueError("Parameter required: agent_id")

    agent_id = request.params.get("agent_id")
    if agent_id is None:
        raise ValueError("Missing required parameter: agent_id")

    is_valid = security_service.validate_agent_id(agent_id)

    logger.debug(
        "Agent ID validated via JSON-RPC",
        agent_id=agent_id,
        is_valid=is_valid,
        method="security.validate_agent_id",
    )

    return {"valid": is_valid, "agent_id": agent_id}


# Log registration on import
logger.info(
    "Security JSON-RPC methods registered",
    methods=[
        "auth.authenticate",
        "auth.validate_token",
        "auth.check_permission",
        "security.generate_keypair",
        "security.register_public_key",
        "security.verify_signature",
        "security.check_rate_limit",
        "security.get_rate_limit_info",
        "security.reset_rate_limit",
        "security.get_stats",
        "security.validate_agent_id",
    ],
)
