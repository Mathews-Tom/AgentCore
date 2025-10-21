"""
FastAPI Authentication Dependencies

Dependency functions for protecting FastAPI routes with JWT authentication.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from gateway.auth.jwt import jwt_manager
from gateway.auth.models import TokenPayload, User, UserRole
from gateway.auth.session import session_manager

logger = structlog.get_logger()

# HTTP Bearer token scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_token_payload(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> TokenPayload:
    """
    Extract and validate JWT token from Authorization header.

    Args:
        credentials: HTTP Bearer credentials from request header

    Returns:
        Validated token payload

    Raises:
        HTTPException: If token is missing, invalid, or expired
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Validate access token
        payload = jwt_manager.validate_access_token(credentials.credentials)
        return payload

    except JWTError as e:
        logger.warning("Token validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    payload: Annotated[TokenPayload, Depends(get_token_payload)],
) -> User:
    """
    Get current authenticated user from token payload.

    Args:
        payload: Validated token payload

    Returns:
        User model with information from token

    Raises:
        HTTPException: If session is invalid
    """
    # Validate session
    session_valid = await session_manager.validate_session(payload.session_id)

    if not session_valid:
        logger.warning(
            "Invalid session",
            session_id=payload.session_id,
            user_id=payload.sub,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update session activity
    await session_manager.update_session_activity(payload.session_id)

    # Construct user from token payload
    user = User(
        id=payload.sub,
        username=payload.username,
        roles=payload.roles,
        is_active=True,
    )

    return user


async def require_role(required_role: UserRole, user: User) -> User:
    """
    Check if user has required role.

    Args:
        required_role: Required role for access
        user: Current user

    Returns:
        User if authorized

    Raises:
        HTTPException: If user doesn't have required role
    """
    if required_role not in user.roles:
        logger.warning(
            "Insufficient permissions",
            user_id=str(user.id),
            username=user.username,
            required_role=required_role,
            user_roles=user.roles,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Required role: {required_role}",
        )

    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    """
    Require admin role for access.

    Args:
        user: Current user

    Returns:
        User if authorized

    Raises:
        HTTPException: If user is not admin
    """
    if UserRole.ADMIN not in user.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user


def require_auth(user: Annotated[User, Depends(get_current_user)]) -> User:
    """
    Require any authenticated user.

    Args:
        user: Current user

    Returns:
        Authenticated user
    """
    return user
