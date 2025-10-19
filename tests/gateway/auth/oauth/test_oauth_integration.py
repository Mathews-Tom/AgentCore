"""
Integration tests for OAuth flows.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from pydantic import HttpUrl

from gateway.auth.oauth.models import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthUserInfo,
    PKCEChallengeMethod,
)
from gateway.auth.oauth.pkce import PKCEGenerator
from gateway.auth.oauth.provider import OAuthProviderBase
from gateway.auth.oauth.providers.github import GitHubOAuthProvider
from gateway.auth.oauth.providers.google import GoogleOAuthProvider
from gateway.auth.oauth.registry import OAuthProviderRegistry
from gateway.auth.oauth.state import OAuthStateManager


# Skip TestOAuthProviderBase as it's an abstract class
# Testing concrete implementations instead (Google, GitHub)


class TestGoogleOAuthProvider:
    """Test Google OAuth provider."""

    @pytest.fixture
    def google_config(self) -> OAuthProviderConfig:
        """Create Google OAuth config."""
        return OAuthProviderConfig(
            provider=OAuthProvider.GOOGLE,
            client_id="google-client-id",
            client_secret="google-client-secret",
            authorization_endpoint=HttpUrl(GoogleOAuthProvider.AUTHORIZATION_ENDPOINT),
            token_endpoint=HttpUrl(GoogleOAuthProvider.TOKEN_ENDPOINT),
            userinfo_endpoint=HttpUrl(GoogleOAuthProvider.USERINFO_ENDPOINT),
            redirect_uri="http://localhost:8080/oauth/callback/google",
            scopes=GoogleOAuthProvider.DEFAULT_SCOPES,
            use_pkce=True,
        )

    @pytest.fixture
    def google_provider(self, google_config: OAuthProviderConfig) -> GoogleOAuthProvider:
        """Create Google OAuth provider."""
        return GoogleOAuthProvider(google_config)

    def test_google_provider_initialization(
        self,
        google_provider: GoogleOAuthProvider,
    ) -> None:
        """Test Google provider initialization."""
        assert google_provider.get_provider_name() == "google"

    def test_google_build_authorization_url(
        self,
        google_provider: GoogleOAuthProvider,
    ) -> None:
        """Test Google-specific authorization URL."""
        state = "test-state"
        auth_url = google_provider.build_authorization_url(state=state)

        assert "accounts.google.com" in auth_url
        assert "access_type=offline" in auth_url
        assert "include_granted_scopes=true" in auth_url


class TestGitHubOAuthProvider:
    """Test GitHub OAuth provider."""

    @pytest.fixture
    def github_config(self) -> OAuthProviderConfig:
        """Create GitHub OAuth config."""
        return OAuthProviderConfig(
            provider=OAuthProvider.GITHUB,
            client_id="github-client-id",
            client_secret="github-client-secret",
            authorization_endpoint=HttpUrl(GitHubOAuthProvider.AUTHORIZATION_ENDPOINT),
            token_endpoint=HttpUrl(GitHubOAuthProvider.TOKEN_ENDPOINT),
            userinfo_endpoint=HttpUrl(GitHubOAuthProvider.USERINFO_ENDPOINT),
            redirect_uri="http://localhost:8080/oauth/callback/github",
            scopes=GitHubOAuthProvider.DEFAULT_SCOPES,
            use_pkce=True,
        )

    @pytest.fixture
    def github_provider(self, github_config: OAuthProviderConfig) -> GitHubOAuthProvider:
        """Create GitHub OAuth provider."""
        return GitHubOAuthProvider(github_config)

    def test_github_provider_initialization(
        self,
        github_provider: GitHubOAuthProvider,
    ) -> None:
        """Test GitHub provider initialization."""
        assert github_provider.get_provider_name() == "github"

    def test_github_build_authorization_url(
        self,
        github_provider: GitHubOAuthProvider,
    ) -> None:
        """Test GitHub-specific authorization URL."""
        state = "test-state"
        auth_url = github_provider.build_authorization_url(state=state)

        assert "github.com" in auth_url
        assert "allow_signup=true" in auth_url


class TestOAuthProviderRegistry:
    """Test OAuth provider registry."""

    @pytest.fixture
    def registry(self) -> OAuthProviderRegistry:
        """Create fresh provider registry."""
        return OAuthProviderRegistry()

    @pytest.fixture
    def google_config(self) -> OAuthProviderConfig:
        """Create Google OAuth config."""
        return OAuthProviderConfig(
            provider=OAuthProvider.GOOGLE,
            client_id="test-client-id",
            client_secret="test-client-secret",
            authorization_endpoint=HttpUrl(GoogleOAuthProvider.AUTHORIZATION_ENDPOINT),
            token_endpoint=HttpUrl(GoogleOAuthProvider.TOKEN_ENDPOINT),
            userinfo_endpoint=HttpUrl(GoogleOAuthProvider.USERINFO_ENDPOINT),
            redirect_uri="http://localhost:8080/oauth/callback/google",
            scopes=GoogleOAuthProvider.DEFAULT_SCOPES,
        )

    def test_register_provider(
        self,
        registry: OAuthProviderRegistry,
        google_config: OAuthProviderConfig,
    ) -> None:
        """Test registering OAuth provider."""
        registry.register_provider(OAuthProvider.GOOGLE, google_config)

        assert registry.is_provider_enabled(OAuthProvider.GOOGLE)
        assert OAuthProvider.GOOGLE in registry.list_providers()

    def test_get_provider(
        self,
        registry: OAuthProviderRegistry,
        google_config: OAuthProviderConfig,
    ) -> None:
        """Test getting registered provider."""
        registry.register_provider(OAuthProvider.GOOGLE, google_config)

        provider = registry.get_provider(OAuthProvider.GOOGLE)

        assert provider is not None
        assert isinstance(provider, GoogleOAuthProvider)

    def test_get_provider_not_registered(self, registry: OAuthProviderRegistry) -> None:
        """Test getting non-registered provider."""
        provider = registry.get_provider(OAuthProvider.GOOGLE)

        assert provider is None

    def test_get_config(
        self,
        registry: OAuthProviderRegistry,
        google_config: OAuthProviderConfig,
    ) -> None:
        """Test getting provider config."""
        registry.register_provider(OAuthProvider.GOOGLE, google_config)

        config = registry.get_config(OAuthProvider.GOOGLE)

        assert config is not None
        assert config.client_id == google_config.client_id

    def test_disable_provider(
        self,
        registry: OAuthProviderRegistry,
        google_config: OAuthProviderConfig,
    ) -> None:
        """Test disabling provider."""
        registry.register_provider(OAuthProvider.GOOGLE, google_config)
        registry.disable_provider(OAuthProvider.GOOGLE)

        assert not registry.is_provider_enabled(OAuthProvider.GOOGLE)
        assert registry.get_provider(OAuthProvider.GOOGLE) is None

    def test_enable_provider(
        self,
        registry: OAuthProviderRegistry,
        google_config: OAuthProviderConfig,
    ) -> None:
        """Test enabling provider."""
        registry.register_provider(OAuthProvider.GOOGLE, google_config)
        registry.disable_provider(OAuthProvider.GOOGLE)
        registry.enable_provider(OAuthProvider.GOOGLE)

        assert registry.is_provider_enabled(OAuthProvider.GOOGLE)

    def test_unregister_provider(
        self,
        registry: OAuthProviderRegistry,
        google_config: OAuthProviderConfig,
    ) -> None:
        """Test unregistering provider."""
        registry.register_provider(OAuthProvider.GOOGLE, google_config)
        result = registry.unregister_provider(OAuthProvider.GOOGLE)

        assert result is True
        assert OAuthProvider.GOOGLE not in registry.list_providers(enabled_only=False)

    def test_list_providers_enabled_only(
        self,
        registry: OAuthProviderRegistry,
        google_config: OAuthProviderConfig,
    ) -> None:
        """Test listing only enabled providers."""
        registry.register_provider(OAuthProvider.GOOGLE, google_config)

        # Register GitHub but disable it
        github_config = OAuthProviderConfig(
            provider=OAuthProvider.GITHUB,
            client_id="test-id",
            client_secret="test-secret",
            authorization_endpoint=HttpUrl(GitHubOAuthProvider.AUTHORIZATION_ENDPOINT),
            token_endpoint=HttpUrl(GitHubOAuthProvider.TOKEN_ENDPOINT),
            redirect_uri="http://localhost:8080/callback",
            enabled=False,
        )
        registry.register_provider(OAuthProvider.GITHUB, github_config)

        enabled = registry.list_providers(enabled_only=True)

        assert OAuthProvider.GOOGLE in enabled
        assert OAuthProvider.GITHUB not in enabled


# Skip OAuth state manager tests that require Redis connection
# These would be tested in integration tests with Redis container
