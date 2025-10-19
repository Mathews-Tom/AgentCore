"""
GitHub OAuth Provider

GitHub OAuth 2.0 implementation for authentication and authorization.
"""

from __future__ import annotations

import httpx
import structlog

from gateway.auth.oauth.models import OAuthProvider, OAuthUserInfo
from gateway.auth.oauth.provider import OAuthProviderBase

logger = structlog.get_logger()


class GitHubOAuthProvider(OAuthProviderBase):
    """
    GitHub OAuth 2.0 provider.

    Implements GitHub-specific OAuth flows and user info retrieval.
    """

    # GitHub OAuth endpoints
    AUTHORIZATION_ENDPOINT = "https://github.com/login/oauth/authorize"
    TOKEN_ENDPOINT = "https://github.com/login/oauth/access_token"
    USERINFO_ENDPOINT = "https://api.github.com/user"
    EMAIL_ENDPOINT = "https://api.github.com/user/emails"

    # Default scopes
    DEFAULT_SCOPES = [
        "read:user",
        "user:email",
    ]

    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from GitHub.

        GitHub requires separate endpoint for email addresses.

        Args:
            access_token: GitHub access token

        Returns:
            OAuth user information

        Raises:
            httpx.HTTPError: If user info request fails
        """
        if not self.userinfo_endpoint:
            raise ValueError("User info endpoint not configured")

        logger.info("Fetching GitHub user info")

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient() as client:
            # Get user profile
            profile_response = await client.get(
                self.userinfo_endpoint,
                headers=headers,
            )
            profile_response.raise_for_status()
            user_data = profile_response.json()

            # Get user emails (GitHub stores emails separately)
            email_response = await client.get(
                self.EMAIL_ENDPOINT,
                headers=headers,
            )
            email_response.raise_for_status()
            emails = email_response.json()

        # Find primary verified email
        primary_email = None
        email_verified = False

        for email_info in emails:
            if email_info.get("primary") and email_info.get("verified"):
                primary_email = email_info["email"]
                email_verified = True
                break

        # Fallback to any verified email
        if not primary_email:
            for email_info in emails:
                if email_info.get("verified"):
                    primary_email = email_info["email"]
                    email_verified = True
                    break

        # Parse GitHub user info response
        return OAuthUserInfo(
            provider=OAuthProvider.GITHUB,
            provider_user_id=str(user_data["id"]),  # GitHub user ID
            email=primary_email or user_data.get("email"),
            email_verified=email_verified,
            name=user_data.get("name"),
            given_name=None,  # GitHub doesn't split names
            family_name=None,
            picture=user_data.get("avatar_url"),
            locale=None,  # GitHub doesn't provide locale
            raw_data={
                "profile": user_data,
                "emails": emails,
            },
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
        Build GitHub OAuth authorization URL.

        GitHub-specific parameters:
        - allow_signup: "true" to allow signup during OAuth flow

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
        # GitHub-specific defaults
        github_params = {
            "allow_signup": "true",  # Allow user signup
        }

        if additional_params:
            github_params.update(additional_params)

        return super().build_authorization_url(
            state=state,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            additional_params=github_params,
        )
