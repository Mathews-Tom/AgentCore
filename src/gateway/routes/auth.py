"""
Authentication Routes

FastAPI endpoints for JWT authentication, token generation, and refresh.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from jose import JWTError

from gateway.auth.dependencies import get_current_user, require_auth
from gateway.auth.jwt import jwt_manager
from gateway.auth.models import AuthError, TokenRequest, TokenResponse, User, UserRole
from gateway.auth.session import session_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/auth", tags=["authentication"])


# Temporary user store for development (replace with real user service in production)
# This is a placeholder for GATE-002 implementation
DEMO_USERS = {
    "admin": {
        "id": "00000000-0000-0000-0000-000000000001",
        "username": "admin",
        "password": "admin123",  # In production, use proper password hashing
        "email": "admin@agentcore.ai",
        "roles": [UserRole.ADMIN, UserRole.USER],
    },
    "user": {
        "id": "00000000-0000-0000-0000-000000000002",
        "username": "user",
        "password": "user123",
        "email": "user@agentcore.ai",
        "roles": [UserRole.USER],
    },
    "service": {
        "id": "00000000-0000-0000-0000-000000000003",
        "username": "service",
        "password": "service123",
        "email": "service@agentcore.ai",
        "roles": [UserRole.SERVICE],
    },
}


async def authenticate_user(username: str, password: str) -> User | None:
    """
    Authenticate user credentials.

    This is a temporary implementation for GATE-002.
    In production, this should call a proper user service with password hashing.

    Args:
        username: Username
        password: Password

    Returns:
        User if authenticated, None otherwise
    """
    user_data = DEMO_USERS.get(username)

    if not user_data or user_data["password"] != password:
        return None

    return User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        roles=user_data["roles"],
        is_active=True,
    )


@router.post("/token", response_model=TokenResponse)
async def create_token(
    request: Request,
    token_request: TokenRequest,
) -> TokenResponse:
    """
    Generate JWT access and refresh tokens.

    Supports multiple grant types:
    - password: Username/password authentication
    - client_credentials: Service account authentication
    - refresh_token: Token refresh flow

    Args:
        request: FastAPI request object
        token_request: Token request parameters

    Returns:
        Token response with access and refresh tokens

    Raises:
        HTTPException: If authentication fails
    """
    # Get client information
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    try:
        if token_request.grant_type == "password":
            # Password grant - authenticate user
            if not token_request.username or not token_request.password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username and password required for password grant",
                )

            user = await authenticate_user(
                token_request.username,
                token_request.password,
            )

            if not user:
                logger.warning(
                    "Authentication failed",
                    username=token_request.username,
                    ip_address=client_ip,
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        elif token_request.grant_type == "client_credentials":
            # Client credentials grant - service account authentication
            if not token_request.client_id or not token_request.client_secret:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Client ID and secret required for client credentials grant",
                )

            # Authenticate service account (placeholder)
            user = await authenticate_user(
                token_request.client_id,
                token_request.client_secret,
            )

            if not user or UserRole.SERVICE not in user.roles:
                logger.warning(
                    "Service authentication failed",
                    client_id=token_request.client_id,
                    ip_address=client_ip,
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid client credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        elif token_request.grant_type == "refresh_token":
            # Refresh token grant - handled by separate endpoint
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Use /auth/refresh endpoint for token refresh",
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported grant type: {token_request.grant_type}",
            )

        # Create session
        session = await session_manager.create_session(
            user=user,
            ip_address=client_ip,
            user_agent=user_agent,
            metadata={"grant_type": token_request.grant_type},
        )

        # Generate tokens
        access_token = jwt_manager.create_access_token(
            user=user,
            session_id=session.session_id,
            scope=token_request.scope,
        )

        refresh_token = jwt_manager.create_refresh_token(
            user_id=str(user.id),
            session_id=session.session_id,
        )

        logger.info(
            "Token created",
            user_id=str(user.id),
            username=user.username,
            session_id=session.session_id,
            grant_type=token_request.grant_type,
        )

        # Calculate expiration times
        from datetime import timedelta

        from gateway.config import settings

        access_expire_seconds = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=access_expire_seconds,
            refresh_token=refresh_token,
            scope=token_request.scope,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token creation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token creation failed",
        ) from e


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: Request,
    refresh_token: str,
) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Args:
        request: FastAPI request object
        refresh_token: Valid refresh token

    Returns:
        New token response with fresh access token

    Raises:
        HTTPException: If refresh token is invalid or expired
    """
    try:
        # Validate refresh token
        payload = jwt_manager.validate_refresh_token(refresh_token)

        # Validate session
        session = await session_manager.get_session(payload.session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Reconstruct user from session
        user = User(
            id=session.user_id,
            username=session.username,
            roles=session.roles,
            is_active=True,
        )

        # Generate new access token
        access_token = jwt_manager.create_access_token(
            user=user,
            session_id=session.session_id,
        )

        # Update session activity
        await session_manager.update_session_activity(session.session_id)

        logger.info(
            "Token refreshed",
            user_id=session.user_id,
            username=session.username,
            session_id=session.session_id,
        )

        from datetime import timedelta

        from gateway.config import settings

        access_expire_seconds = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60

        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=access_expire_seconds,
            refresh_token=refresh_token,  # Return same refresh token
        )

    except JWTError as e:
        logger.warning("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired refresh token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        ) from e


@router.post("/logout")
async def logout(
    user: Annotated[User, Depends(require_auth)],
    request: Request,
) -> dict[str, str]:
    """
    Logout current user and invalidate session.

    Args:
        user: Current authenticated user
        request: FastAPI request object

    Returns:
        Success message
    """
    # Extract session ID from token
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        try:
            payload = jwt_manager.validate_access_token(token)
            await session_manager.delete_session(payload.session_id)

            logger.info(
                "User logged out",
                user_id=str(user.id),
                username=user.username,
                session_id=payload.session_id,
            )
        except JWTError:
            pass

    return {"message": "Successfully logged out"}


@router.get("/me", response_model=User)
async def get_current_user_info(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    Get current authenticated user information.

    Args:
        user: Current authenticated user

    Returns:
        User information
    """
    return user


@router.get("/sessions")
async def get_user_sessions(
    user: Annotated[User, Depends(require_auth)],
) -> dict[str, list[dict]]:
    """
    Get all active sessions for current user.

    Args:
        user: Current authenticated user

    Returns:
        List of active sessions
    """
    sessions = await session_manager.get_user_sessions(str(user.id))

    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat(),
                "expires_at": s.expires_at.isoformat(),
                "last_activity": s.last_activity.isoformat(),
                "ip_address": s.ip_address,
                "user_agent": s.user_agent,
            }
            for s in sessions
        ]
    }


@router.delete("/sessions/{session_id}")
async def delete_user_session(
    session_id: str,
    user: Annotated[User, Depends(require_auth)],
) -> dict[str, str]:
    """
    Delete specific user session.

    Args:
        session_id: Session identifier to delete
        user: Current authenticated user

    Returns:
        Success message

    Raises:
        HTTPException: If session not found or doesn't belong to user
    """
    # Verify session belongs to user
    session = await session_manager.get_session(session_id)

    if not session or session.user_id != str(user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    await session_manager.delete_session(session_id)

    logger.info(
        "Session deleted by user",
        user_id=str(user.id),
        session_id=session_id,
    )

    return {"message": "Session deleted successfully"}
