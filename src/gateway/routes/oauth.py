"""
OAuth Routes

FastAPI endpoints for OAuth 2.0/3.0 authentication flows including
authorization, callback, token exchange, and provider management.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse

from gateway.auth.jwt import jwt_manager
from gateway.auth.models import TokenResponse, User, UserRole
from gateway.auth.oauth.models import (
    OAuthCallbackRequest,
    OAuthError,
    OAuthProvider,
    PKCEChallengeMethod,
)
from gateway.auth.oauth.pkce import PKCEGenerator
from gateway.auth.oauth.registry import oauth_registry
from gateway.auth.oauth.scopes import ScopeManager
from gateway.auth.oauth.state import oauth_state_manager
from gateway.auth.session import session_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/oauth", tags=["oauth"])


@router.get("/authorize/{provider}")
async def oauth_authorize(
    provider: OAuthProvider,
    request: Request,
    scope: str | None = Query(None, description="Requested scopes (space-separated)"),
    redirect_after_login: str | None = Query(None, description="URL to redirect after successful login"),
) -> RedirectResponse:
    """
    Initiate OAuth authorization flow.

    Redirects user to OAuth provider's authorization endpoint with PKCE support.

    Args:
        provider: OAuth provider (google, github, etc.)
        request: FastAPI request object
        scope: Requested OAuth scopes
        redirect_after_login: Optional URL to redirect after login

    Returns:
        Redirect response to OAuth provider

    Raises:
        HTTPException: If provider not found or disabled
    """
    # Get provider instance
    provider_instance = oauth_registry.get_provider(provider)

    if not provider_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OAuth provider '{provider.value}' not found or disabled",
        )

    # Generate CSRF state
    state = oauth_state_manager.generate_state()

    # Generate PKCE challenge if supported
    code_challenge = None
    code_challenge_method = None
    code_verifier = None

    if provider_instance.supports_pkce():
        pkce_pair = PKCEGenerator.generate_pkce_pair(
            method=PKCEChallengeMethod.S256,
        )
        code_challenge = pkce_pair.code_challenge
        code_challenge_method = pkce_pair.code_challenge_method.value
        code_verifier = pkce_pair.code_verifier

    # Build redirect URI
    redirect_uri = f"{request.base_url}oauth/callback/{provider.value}"

    # Save OAuth state
    await oauth_state_manager.save_state(
        state=state,
        provider=provider,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        requested_scopes=scope,
        metadata={"redirect_after_login": redirect_after_login},
    )

    # Build authorization URL
    auth_url = provider_instance.build_authorization_url(
        state=state,
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )

    logger.info(
        "OAuth authorization initiated",
        provider=provider.value,
        state=state,
        has_pkce=code_challenge is not None,
    )

    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: OAuthProvider,
    request: Request,
    code: str | None = Query(None, description="Authorization code"),
    state: str | None = Query(None, description="State parameter"),
    error: str | None = Query(None, description="Error code"),
    error_description: str | None = Query(None, description="Error description"),
) -> RedirectResponse:
    """
    OAuth callback endpoint.

    Handles OAuth provider callback, exchanges code for tokens, and creates user session.

    Args:
        provider: OAuth provider
        request: FastAPI request object
        code: Authorization code from provider
        state: State parameter for CSRF validation
        error: Error code if authorization failed
        error_description: Human-readable error description

    Returns:
        Redirect response to application

    Raises:
        HTTPException: If callback processing fails
    """
    # Check for OAuth errors
    if error:
        logger.error(
            "OAuth authorization failed",
            provider=provider.value,
            error=error,
            error_description=error_description,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OAuth authorization failed: {error} - {error_description}",
        )

    # Validate required parameters
    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required parameters: code and state",
        )

    # Validate and consume state
    oauth_state = await oauth_state_manager.validate_and_consume_state(
        state=state,
        expected_provider=provider,
    )

    if not oauth_state:
        logger.error(
            "Invalid or expired OAuth state",
            provider=provider.value,
            state=state,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state. Please try again.",
        )

    # Get provider instance
    provider_instance = oauth_registry.get_provider(provider)

    if not provider_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OAuth provider '{provider.value}' not found or disabled",
        )

    try:
        # Exchange authorization code for tokens
        token_response = await provider_instance.exchange_code_for_token(
            code=code,
            redirect_uri=oauth_state.redirect_uri,
            code_verifier=oauth_state.code_verifier,
        )

        # Get user info from provider
        user_info = await provider_instance.get_user_info(token_response.access_token)

        # Create or update user
        # In production, this should check if user exists and create/update accordingly
        user = User(
            username=user_info.email or user_info.provider_user_id,
            email=user_info.email,
            roles=[UserRole.USER],
            is_active=True,
            metadata={
                "oauth_provider": provider.value,
                "oauth_user_id": user_info.provider_user_id,
                "oauth_email_verified": user_info.email_verified,
            },
        )

        # Create session
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        session = await session_manager.create_session(
            user=user,
            ip_address=client_ip,
            user_agent=user_agent,
            metadata={
                "oauth_provider": provider.value,
                "oauth_scopes": token_response.scope,
            },
        )

        # Create JWT access token
        access_token = jwt_manager.create_access_token(
            user=user,
            session_id=session.session_id,
            scope=token_response.scope,
        )

        logger.info(
            "OAuth login successful",
            provider=provider.value,
            user_id=str(user.id),
            username=user.username,
            session_id=session.session_id,
        )

        # Redirect to application
        redirect_url = oauth_state.metadata.get("redirect_after_login", "/")

        # In a real application, you'd typically set a secure cookie here
        # For now, we'll redirect with token as query parameter (not recommended for production)
        redirect_url_with_token = f"{redirect_url}?access_token={access_token}"

        return RedirectResponse(
            url=redirect_url_with_token,
            status_code=status.HTTP_302_FOUND,
        )

    except Exception as e:
        logger.error(
            "OAuth callback processing failed",
            provider=provider.value,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OAuth callback processing failed: {str(e)}",
        )


@router.post("/token/client_credentials")
async def oauth_client_credentials(
    provider: OAuthProvider,
    scope: str | None = None,
) -> TokenResponse:
    """
    OAuth client credentials flow.

    Get access token for machine-to-machine authentication.

    Args:
        provider: OAuth provider
        scope: Requested scopes (space-separated)

    Returns:
        Token response with access token

    Raises:
        HTTPException: If token request fails
    """
    # Get provider instance
    provider_instance = oauth_registry.get_provider(provider)

    if not provider_instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"OAuth provider '{provider.value}' not found or disabled",
        )

    try:
        # Get client credentials token
        token_response = await provider_instance.get_client_credentials_token(scope=scope)

        logger.info(
            "Client credentials token issued",
            provider=provider.value,
            scope=scope,
        )

        return TokenResponse(
            access_token=token_response.access_token,
            token_type=token_response.token_type,
            expires_in=token_response.expires_in,
            scope=token_response.scope,
        )

    except Exception as e:
        logger.error(
            "Client credentials token request failed",
            provider=provider.value,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token request failed: {str(e)}",
        )


@router.get("/providers")
async def list_oauth_providers() -> dict[str, list[dict[str, str]]]:
    """
    List available OAuth providers.

    Returns:
        List of enabled OAuth providers
    """
    enabled_providers = oauth_registry.list_providers(enabled_only=True)

    providers_info = [
        {
            "provider": provider.value,
            "name": provider.value.capitalize(),
            "authorize_url": f"/oauth/authorize/{provider.value}",
        }
        for provider in enabled_providers
    ]

    return {"providers": providers_info}


@router.get("/scopes")
async def list_oauth_scopes(
    resource: str | None = Query(None, description="Filter by resource type"),
) -> dict[str, list[dict]]:
    """
    List available OAuth scopes.

    Args:
        resource: Optional resource type filter (agent, task, user, admin)

    Returns:
        List of available scopes with descriptions
    """
    if resource:
        scopes = ScopeManager.get_scopes_for_resource(resource)
        scope_info = [ScopeManager.get_scope_info(scope) for scope in scopes]
    else:
        scope_info = ScopeManager.get_all_scopes()

    return {"scopes": scope_info}
