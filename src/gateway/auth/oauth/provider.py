"""
OAuth Provider Base Class

Abstract base class for OAuth providers with common functionality for
authorization code flow, token exchange, and user info retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlencode

import httpx
import structlog

from gateway.auth.oauth.models import (
    OAuthGrantType,
    OAuthProviderConfig,
    OAuthTokenResponse,
    OAuthUserInfo,
)
from gateway.auth.oauth.pkce import PKCEGenerator

logger = structlog.get_logger()


class OAuthProviderBase(ABC):
    """
    Abstract base class for OAuth providers.

    Implements common OAuth 2.0 flows with provider-specific customization.
    """

    def __init__(self, config: OAuthProviderConfig) -> None:
        """
        Initialize OAuth provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.client_id = config.client_id
        self.client_secret = config.client_secret
        self.authorization_endpoint = str(config.authorization_endpoint)
        self.token_endpoint = str(config.token_endpoint)
        self.userinfo_endpoint = str(config.userinfo_endpoint) if config.userinfo_endpoint else None
        self.revocation_endpoint = str(config.revocation_endpoint) if config.revocation_endpoint else None

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
        Build OAuth authorization URL.

        Args:
            state: CSRF state parameter
            redirect_uri: Callback redirect URI
            scope: Requested scopes (space-separated)
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE challenge method (S256 or plain)
            additional_params: Additional provider-specific parameters

        Returns:
            Authorization URL
        """
        params: dict[str, str] = {
            "client_id": self.client_id,
            "response_type": "code",
            "state": state,
            "redirect_uri": redirect_uri or self.config.redirect_uri,
        }

        # Add scopes
        if scope:
            params["scope"] = scope
        elif self.config.scopes:
            params["scope"] = " ".join(self.config.scopes)

        # Add PKCE parameters if provided
        if self.config.use_pkce and code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = code_challenge_method or "S256"

        # Add provider-specific parameters
        if additional_params:
            params.update(additional_params)

        # Add default additional params from config
        if self.config.additional_params:
            params.update(self.config.additional_params)

        # Build URL
        query_string = urlencode(params)
        return f"{self.authorization_endpoint}?{query_string}"

    async def exchange_code_for_token(
        self,
        code: str,
        redirect_uri: str | None = None,
        code_verifier: str | None = None,
    ) -> OAuthTokenResponse:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from provider
            redirect_uri: Redirect URI used in authorization request
            code_verifier: PKCE code verifier

        Returns:
            OAuth token response

        Raises:
            httpx.HTTPError: If token exchange fails
        """
        data = {
            "grant_type": OAuthGrantType.AUTHORIZATION_CODE.value,
            "code": code,
            "redirect_uri": redirect_uri or self.config.redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        # Add PKCE verifier if provided
        if code_verifier:
            data["code_verifier"] = code_verifier

        logger.info(
            "Exchanging authorization code for token",
            provider=self.config.provider.value,
            redirect_uri=data["redirect_uri"],
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_endpoint,
                data=data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

            token_data = response.json()

        return self._parse_token_response(token_data)

    async def refresh_access_token(self, refresh_token: str) -> OAuthTokenResponse:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New OAuth token response

        Raises:
            httpx.HTTPError: If token refresh fails
        """
        data = {
            "grant_type": OAuthGrantType.REFRESH_TOKEN.value,
            "refresh_token": refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        logger.info(
            "Refreshing access token",
            provider=self.config.provider.value,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_endpoint,
                data=data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

            token_data = response.json()

        return self._parse_token_response(token_data)

    async def get_client_credentials_token(
        self,
        scope: str | None = None,
    ) -> OAuthTokenResponse:
        """
        Get access token using client credentials flow.

        Args:
            scope: Requested scopes (space-separated)

        Returns:
            OAuth token response

        Raises:
            httpx.HTTPError: If token request fails
        """
        data = {
            "grant_type": OAuthGrantType.CLIENT_CREDENTIALS.value,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        if scope:
            data["scope"] = scope

        logger.info(
            "Requesting client credentials token",
            provider=self.config.provider.value,
            scope=scope,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_endpoint,
                data=data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()

            token_data = response.json()

        return self._parse_token_response(token_data)

    async def revoke_token(
        self,
        token: str,
        token_type_hint: str | None = None,
    ) -> bool:
        """
        Revoke access or refresh token.

        Args:
            token: Token to revoke
            token_type_hint: Type of token ("access_token" or "refresh_token")

        Returns:
            True if revocation successful, False otherwise
        """
        if not self.revocation_endpoint:
            logger.warning(
                "Token revocation not supported",
                provider=self.config.provider.value,
            )
            return False

        data = {
            "token": token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        if token_type_hint:
            data["token_type_hint"] = token_type_hint

        logger.info(
            "Revoking token",
            provider=self.config.provider.value,
            token_type=token_type_hint,
        )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.revocation_endpoint,
                    data=data,
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
                return True
        except httpx.HTTPError as e:
            logger.error(
                "Token revocation failed",
                provider=self.config.provider.value,
                error=str(e),
            )
            return False

    @abstractmethod
    async def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from provider.

        This method must be implemented by each provider as the
        user info endpoint structure varies between providers.

        Args:
            access_token: Access token for authentication

        Returns:
            OAuth user information

        Raises:
            httpx.HTTPError: If user info request fails
        """
        pass

    def _parse_token_response(self, token_data: dict[str, Any]) -> OAuthTokenResponse:
        """
        Parse token response from provider.

        Args:
            token_data: Raw token response from provider

        Returns:
            Parsed OAuth token response
        """
        return OAuthTokenResponse(
            access_token=token_data["access_token"],
            token_type=token_data.get("token_type", "Bearer"),
            expires_in=token_data["expires_in"],
            refresh_token=token_data.get("refresh_token"),
            scope=token_data.get("scope"),
            id_token=token_data.get("id_token"),
        )

    def supports_pkce(self) -> bool:
        """
        Check if provider supports PKCE.

        Returns:
            True if PKCE is enabled in configuration
        """
        return self.config.use_pkce

    def get_provider_name(self) -> str:
        """
        Get provider name.

        Returns:
            Provider name
        """
        return self.config.provider.value
