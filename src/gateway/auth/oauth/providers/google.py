"""
Google OAuth Provider

Google OAuth 2.0 implementation with OpenID Connect support.
"""

from __future__ import annotations

import httpx
import structlog

from gateway.auth.oauth.models import OAuthProvider, OAuthUserInfo
from gateway.auth.oauth.provider import OAuthProviderBase

logger = structlog.get_logger()


class GoogleOAuthProvider(OAuthProviderBase):
    """
    Google OAuth 2.0 provider.

    Implements Google-specific OAuth flows and user info retrieval.
    Supports OpenID Connect for identity verification.
    """

    # Google OAuth endpoints
    AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
    USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v3/userinfo"
    REVOCATION_ENDPOINT = "https://oauth2.googleapis.com/revoke"

    # Default scopes
    DEFAULT_SCOPES = [
        "openid",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
    ]

    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from Google.

        Args:
            access_token: Google access token

        Returns:
            OAuth user information

        Raises:
            httpx.HTTPError: If user info request fails
        """
        if not self.userinfo_endpoint:
            raise ValueError("User info endpoint not configured")

        logger.info("Fetching Google user info")

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()

            user_data = response.json()

        # Parse Google user info response
        return OAuthUserInfo(
            provider=OAuthProvider.GOOGLE,
            provider_user_id=user_data["sub"],  # Google user ID
            email=user_data.get("email"),
            email_verified=user_data.get("email_verified", False),
            name=user_data.get("name"),
            given_name=user_data.get("given_name"),
            family_name=user_data.get("family_name"),
            picture=user_data.get("picture"),
            locale=user_data.get("locale"),
            raw_data=user_data,
        )

    def build_authorization_url(
        self,
        state: str,
        redirect_uri: str | None = None,
        scope: str | None = None,
        code_challenge: str | None = None,
        code_challenge_method: str | None = None,
        additional_params: dict[str, str] | None = None,
    ) -> str:
        """
        Build Google OAuth authorization URL.

        Google-specific parameters:
        - access_type: "offline" for refresh token
        - prompt: "consent" to force consent screen
        - include_granted_scopes: "true" for incremental authorization

        Args:
            state: CSRF state parameter
            redirect_uri: Callback redirect URI
            scope: Requested scopes (space-separated)
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE challenge method
            additional_params: Additional parameters

        Returns:
            Authorization URL
        """
        # Google-specific defaults
        google_params = {
            "access_type": "offline",  # Request refresh token
            "include_granted_scopes": "true",  # Incremental authorization
        }

        if additional_params:
            google_params.update(additional_params)

        return super().build_authorization_url(
            state=state,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            additional_params=google_params,
        )
