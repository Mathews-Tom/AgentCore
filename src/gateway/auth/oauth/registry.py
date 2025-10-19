"""
OAuth Provider Registry

Central registry for OAuth providers with configuration management and
provider lifecycle.
"""

from __future__ import annotations

import structlog
from pydantic import HttpUrl

from gateway.auth.oauth.models import OAuthProvider, OAuthProviderConfig
from gateway.auth.oauth.provider import OAuthProviderBase
from gateway.auth.oauth.providers.github import GitHubOAuthProvider
from gateway.auth.oauth.providers.google import GoogleOAuthProvider
from gateway.config import settings

logger = structlog.get_logger()


class OAuthProviderRegistry:
    """
    OAuth provider registry.

    Manages OAuth provider instances with configuration-driven setup.
    """

    def __init__(self) -> None:
        """Initialize provider registry."""
        self._providers: dict[OAuthProvider, OAuthProviderBase] = {}
        self._configs: dict[OAuthProvider, OAuthProviderConfig] = {}

    def register_provider(
        self,
        provider: OAuthProvider,
        config: OAuthProviderConfig,
    ) -> None:
        """
        Register OAuth provider with configuration.

        Args:
            provider: Provider identifier
            config: Provider configuration
        """
        # Store configuration
        self._configs[provider] = config

        # Create provider instance
        if provider == OAuthProvider.GOOGLE:
            provider_instance = GoogleOAuthProvider(config)
        elif provider == OAuthProvider.GITHUB:
            provider_instance = GitHubOAuthProvider(config)
        else:
            # For custom providers, use base implementation
            provider_instance = OAuthProviderBase(config)

        self._providers[provider] = provider_instance

        logger.info(
            "OAuth provider registered",
            provider=provider.value,
            enabled=config.enabled,
        )

    def get_provider(self, provider: OAuthProvider) -> OAuthProviderBase | None:
        """
        Get OAuth provider instance.

        Args:
            provider: Provider identifier

        Returns:
            Provider instance if registered and enabled, None otherwise
        """
        if provider not in self._providers:
            logger.warning("OAuth provider not registered", provider=provider.value)
            return None

        config = self._configs[provider]
        if not config.enabled:
            logger.warning("OAuth provider disabled", provider=provider.value)
            return None

        return self._providers[provider]

    def get_config(self, provider: OAuthProvider) -> OAuthProviderConfig | None:
        """
        Get provider configuration.

        Args:
            provider: Provider identifier

        Returns:
            Provider configuration if registered, None otherwise
        """
        return self._configs.get(provider)

    def list_providers(self, enabled_only: bool = True) -> list[OAuthProvider]:
        """
        List registered providers.

        Args:
            enabled_only: Only return enabled providers

        Returns:
            List of provider identifiers
        """
        if enabled_only:
            return [
                provider
                for provider, config in self._configs.items()
                if config.enabled
            ]
        return list(self._providers.keys())

    def is_provider_enabled(self, provider: OAuthProvider) -> bool:
        """
        Check if provider is enabled.

        Args:
            provider: Provider identifier

        Returns:
            True if provider is registered and enabled
        """
        config = self._configs.get(provider)
        return config is not None and config.enabled

    def enable_provider(self, provider: OAuthProvider) -> bool:
        """
        Enable OAuth provider.

        Args:
            provider: Provider identifier

        Returns:
            True if provider was enabled, False if not found
        """
        config = self._configs.get(provider)
        if not config:
            return False

        config.enabled = True
        logger.info("OAuth provider enabled", provider=provider.value)
        return True

    def disable_provider(self, provider: OAuthProvider) -> bool:
        """
        Disable OAuth provider.

        Args:
            provider: Provider identifier

        Returns:
            True if provider was disabled, False if not found
        """
        config = self._configs.get(provider)
        if not config:
            return False

        config.enabled = False
        logger.info("OAuth provider disabled", provider=provider.value)
        return True

    def unregister_provider(self, provider: OAuthProvider) -> bool:
        """
        Unregister OAuth provider.

        Args:
            provider: Provider identifier

        Returns:
            True if provider was unregistered, False if not found
        """
        if provider not in self._providers:
            return False

        del self._providers[provider]
        del self._configs[provider]

        logger.info("OAuth provider unregistered", provider=provider.value)
        return True


# Global provider registry
oauth_registry = OAuthProviderRegistry()


def initialize_oauth_providers() -> None:
    """
    Initialize OAuth providers from configuration.

    Loads provider configurations from environment variables and registers them.
    """
    logger.info("Initializing OAuth providers")

    # Example: Register Google OAuth provider
    # In production, load these from configuration/database
    try:
        # Check if OAuth is enabled
        if not settings.OAUTH_ENABLED:
            logger.info("OAuth providers disabled in configuration")
            return

        # Google OAuth provider (if configured)
        google_client_id = getattr(settings, "OAUTH_GOOGLE_CLIENT_ID", None)
        google_client_secret = getattr(settings, "OAUTH_GOOGLE_CLIENT_SECRET", None)

        if google_client_id and google_client_secret:
            google_config = OAuthProviderConfig(
                provider=OAuthProvider.GOOGLE,
                client_id=google_client_id,
                client_secret=google_client_secret,
                authorization_endpoint=HttpUrl(GoogleOAuthProvider.AUTHORIZATION_ENDPOINT),
                token_endpoint=HttpUrl(GoogleOAuthProvider.TOKEN_ENDPOINT),
                userinfo_endpoint=HttpUrl(GoogleOAuthProvider.USERINFO_ENDPOINT),
                revocation_endpoint=HttpUrl(GoogleOAuthProvider.REVOCATION_ENDPOINT),
                scopes=GoogleOAuthProvider.DEFAULT_SCOPES,
                redirect_uri=getattr(
                    settings,
                    "OAUTH_GOOGLE_REDIRECT_URI",
                    f"http://{settings.HOST}:{settings.PORT}/oauth/callback/google"
                ),
                use_pkce=True,
            )
            oauth_registry.register_provider(OAuthProvider.GOOGLE, google_config)
            logger.info("Google OAuth provider registered")

        # GitHub OAuth provider (if configured)
        github_client_id = getattr(settings, "OAUTH_GITHUB_CLIENT_ID", None)
        github_client_secret = getattr(settings, "OAUTH_GITHUB_CLIENT_SECRET", None)

        if github_client_id and github_client_secret:
            github_config = OAuthProviderConfig(
                provider=OAuthProvider.GITHUB,
                client_id=github_client_id,
                client_secret=github_client_secret,
                authorization_endpoint=HttpUrl(GitHubOAuthProvider.AUTHORIZATION_ENDPOINT),
                token_endpoint=HttpUrl(GitHubOAuthProvider.TOKEN_ENDPOINT),
                userinfo_endpoint=HttpUrl(GitHubOAuthProvider.USERINFO_ENDPOINT),
                scopes=GitHubOAuthProvider.DEFAULT_SCOPES,
                redirect_uri=getattr(
                    settings,
                    "OAUTH_GITHUB_REDIRECT_URI",
                    f"http://{settings.HOST}:{settings.PORT}/oauth/callback/github"
                ),
                use_pkce=True,
            )
            oauth_registry.register_provider(OAuthProvider.GITHUB, github_config)
            logger.info("GitHub OAuth provider registered")

        enabled_providers = oauth_registry.list_providers(enabled_only=True)
        logger.info(
            "OAuth providers initialized",
            count=len(enabled_providers),
            providers=[p.value for p in enabled_providers],
        )

    except Exception as e:
        logger.error("Failed to initialize OAuth providers", error=str(e))
        raise
