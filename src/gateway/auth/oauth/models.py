"""
OAuth Models

Pydantic models for OAuth 2.0/3.0 authentication flows, provider configuration,
and PKCE challenge/response.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class OAuthProvider(str, Enum):
    """Supported OAuth providers."""

    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    CUSTOM = "custom"


class OAuthGrantType(str, Enum):
    """OAuth grant types."""

    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    DEVICE_CODE = "device_code"


class PKCEChallengeMethod(str, Enum):
    """PKCE code challenge methods."""

    S256 = "S256"  # SHA-256 hash (recommended)
    PLAIN = "plain"  # Plain text (not recommended)


class OAuthProviderConfig(BaseModel):
    """OAuth provider configuration."""

    provider: OAuthProvider = Field(..., description="Provider identifier")
    client_id: str = Field(..., description="OAuth client ID")
    client_secret: str = Field(..., description="OAuth client secret")
    authorization_endpoint: HttpUrl = Field(..., description="Authorization endpoint URL")
    token_endpoint: HttpUrl = Field(..., description="Token endpoint URL")
    userinfo_endpoint: HttpUrl | None = Field(None, description="User info endpoint URL")
    revocation_endpoint: HttpUrl | None = Field(None, description="Token revocation endpoint URL")
    scopes: list[str] = Field(default_factory=list, description="Default scopes to request")
    redirect_uri: str = Field(..., description="Redirect URI for OAuth callback")
    use_pkce: bool = Field(default=True, description="Enable PKCE for authorization code flow")
    pkce_challenge_method: PKCEChallengeMethod = Field(
        default=PKCEChallengeMethod.S256,
        description="PKCE challenge method"
    )
    additional_params: dict[str, str] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )
    enabled: bool = Field(default=True, description="Provider enabled status")

    model_config = {"from_attributes": True}


class PKCEChallenge(BaseModel):
    """PKCE challenge and verifier pair."""

    code_verifier: str = Field(..., description="PKCE code verifier (random string)")
    code_challenge: str = Field(..., description="PKCE code challenge (hashed verifier)")
    code_challenge_method: PKCEChallengeMethod = Field(
        default=PKCEChallengeMethod.S256,
        description="Challenge method used"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Challenge creation time")
    expires_at: datetime = Field(..., description="Challenge expiration time")


class OAuthAuthorizationRequest(BaseModel):
    """OAuth authorization request parameters."""

    provider: OAuthProvider = Field(..., description="OAuth provider to use")
    scope: str | None = Field(None, description="Requested scopes (space-separated)")
    state: str = Field(..., description="CSRF protection state parameter")
    redirect_uri: str | None = Field(None, description="Custom redirect URI")
    code_challenge: str | None = Field(None, description="PKCE code challenge")
    code_challenge_method: PKCEChallengeMethod | None = Field(
        None,
        description="PKCE challenge method"
    )
    additional_params: dict[str, str] = Field(
        default_factory=dict,
        description="Additional provider-specific parameters"
    )


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request parameters."""

    code: str = Field(..., description="Authorization code from provider")
    state: str = Field(..., description="State parameter for CSRF validation")
    code_verifier: str | None = Field(None, description="PKCE code verifier")
    error: str | None = Field(None, description="Error code if authorization failed")
    error_description: str | None = Field(None, description="Error description")


class OAuthTokenExchangeRequest(BaseModel):
    """OAuth token exchange request."""

    grant_type: OAuthGrantType = Field(..., description="OAuth grant type")
    code: str | None = Field(None, description="Authorization code (for authorization_code grant)")
    redirect_uri: str | None = Field(None, description="Redirect URI (for authorization_code grant)")
    code_verifier: str | None = Field(None, description="PKCE code verifier")
    refresh_token: str | None = Field(None, description="Refresh token (for refresh_token grant)")
    scope: str | None = Field(None, description="Requested scope")
    client_id: str | None = Field(None, description="Client ID (for client_credentials grant)")
    client_secret: str | None = Field(None, description="Client secret (for client_credentials grant)")


class OAuthTokenResponse(BaseModel):
    """OAuth token response."""

    access_token: str = Field(..., description="OAuth access token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    refresh_token: str | None = Field(None, description="Refresh token")
    scope: str | None = Field(None, description="Granted scopes")
    id_token: str | None = Field(None, description="OpenID Connect ID token")
    issued_at: datetime = Field(default_factory=datetime.utcnow, description="Token issue time")


class OAuthUserInfo(BaseModel):
    """OAuth user information from provider."""

    provider: OAuthProvider = Field(..., description="OAuth provider")
    provider_user_id: str = Field(..., description="User ID from OAuth provider")
    email: str | None = Field(None, description="User email address")
    email_verified: bool = Field(default=False, description="Email verification status")
    name: str | None = Field(None, description="User full name")
    given_name: str | None = Field(None, description="User first name")
    family_name: str | None = Field(None, description="User last name")
    picture: str | None = Field(None, description="User profile picture URL")
    locale: str | None = Field(None, description="User locale")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Raw provider response")


class OAuthState(BaseModel):
    """OAuth state information stored in Redis."""

    state: str = Field(..., description="CSRF state parameter")
    provider: OAuthProvider = Field(..., description="OAuth provider")
    redirect_uri: str = Field(..., description="Callback redirect URI")
    code_verifier: str | None = Field(None, description="PKCE code verifier")
    requested_scopes: str | None = Field(None, description="Requested scopes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="State creation time")
    expires_at: datetime = Field(..., description="State expiration time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class OAuthError(BaseModel):
    """OAuth error response."""

    error: str = Field(..., description="Error code")
    error_description: str | None = Field(None, description="Human-readable error description")
    error_uri: str | None = Field(None, description="URI for error details")
    state: str | None = Field(None, description="State parameter if provided")
