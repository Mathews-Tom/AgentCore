"""
OAuth 3.0 Integration

OAuth 2.0/3.0 authentication with PKCE support, multiple providers,
and enterprise SSO integration.
"""

from gateway.auth.oauth.models import (
    OAuthAuthorizationRequest,
    OAuthCallbackRequest,
    OAuthProvider,
    OAuthProviderConfig,
    OAuthTokenExchangeRequest,
    OAuthTokenResponse,
    PKCEChallenge,
)
from gateway.auth.oauth.pkce import PKCEGenerator
from gateway.auth.oauth.provider import OAuthProviderBase
from gateway.auth.oauth.scopes import ScopeManager, ScopePermission

__all__ = [
    "OAuthProviderBase",
    "OAuthProvider",
    "OAuthProviderConfig",
    "OAuthAuthorizationRequest",
    "OAuthCallbackRequest",
    "OAuthTokenExchangeRequest",
    "OAuthTokenResponse",
    "PKCEChallenge",
    "PKCEGenerator",
    "ScopeManager",
    "ScopePermission",
]
