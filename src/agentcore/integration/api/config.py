"""API configuration utilities.

Configuration loading and validation for API integrations.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import SecretStr

from agentcore.integration.api.exceptions import APIConfigurationError
from agentcore.integration.api.models import (
    APIConfig,
    AuthConfig,
    AuthScheme,
    RateLimitConfig,
    RetryConfig,
)


def load_api_config(
    name: str,
    base_url: str | None = None,
    auth_scheme: str | None = None,
    **kwargs: Any,
) -> APIConfig:
    """Load API configuration from environment and parameters.

    Args:
        name: API connector name
        base_url: Optional base URL (can be loaded from env)
        auth_scheme: Optional auth scheme (can be loaded from env)
        **kwargs: Additional configuration parameters

    Returns:
        APIConfig instance

    Raises:
        APIConfigurationError: If required configuration is missing
    """
    # Load from environment if not provided
    env_prefix = f"API_{name.upper().replace('-', '_')}_"

    if not base_url:
        base_url = os.getenv(f"{env_prefix}BASE_URL")

    if not base_url:
        raise APIConfigurationError(f"base_url is required for API connector '{name}'")

    # Load auth configuration
    if not auth_scheme:
        auth_scheme = os.getenv(f"{env_prefix}AUTH_SCHEME", "none")

    auth_config = _load_auth_config(name, auth_scheme, env_prefix)

    # Load rate limit config
    rate_limit_config = _load_rate_limit_config(env_prefix, kwargs)

    # Load retry config
    retry_config = _load_retry_config(env_prefix, kwargs)

    # Build default headers
    default_headers = kwargs.pop("default_headers", {})
    if env_user_agent := os.getenv(f"{env_prefix}USER_AGENT"):
        default_headers.setdefault("User-Agent", env_user_agent)

    # Create API config
    return APIConfig(
        name=name,
        base_url=base_url,
        timeout_seconds=int(os.getenv(f"{env_prefix}TIMEOUT", kwargs.pop("timeout_seconds", "30"))),
        auth=auth_config,
        rate_limit=rate_limit_config,
        retry=retry_config,
        default_headers=default_headers,
        verify_ssl=_get_bool_env(f"{env_prefix}VERIFY_SSL", kwargs.pop("verify_ssl", True)),
        follow_redirects=_get_bool_env(f"{env_prefix}FOLLOW_REDIRECTS", kwargs.pop("follow_redirects", True)),
        max_redirects=int(os.getenv(f"{env_prefix}MAX_REDIRECTS", kwargs.pop("max_redirects", "10"))),
        pool_connections=int(os.getenv(f"{env_prefix}POOL_CONNECTIONS", kwargs.pop("pool_connections", "10"))),
        pool_maxsize=int(os.getenv(f"{env_prefix}POOL_MAXSIZE", kwargs.pop("pool_maxsize", "10"))),
        pool_keepalive=int(os.getenv(f"{env_prefix}POOL_KEEPALIVE", kwargs.pop("pool_keepalive", "60"))),
        description=kwargs.pop("description", None),
        tags=kwargs.pop("tags", []),
        metadata=kwargs.pop("metadata", {}),
    )


def _load_auth_config(name: str, auth_scheme: str, env_prefix: str) -> AuthConfig:
    """Load authentication configuration.

    Args:
        name: API connector name
        auth_scheme: Authentication scheme
        env_prefix: Environment variable prefix

    Returns:
        AuthConfig instance

    Raises:
        APIConfigurationError: If auth configuration is invalid
    """
    try:
        scheme = AuthScheme(auth_scheme.lower())
    except ValueError:
        raise APIConfigurationError(
            f"Invalid auth scheme '{auth_scheme}' for API connector '{name}'"
        )

    if scheme == AuthScheme.NONE:
        return AuthConfig(scheme=AuthScheme.NONE)

    if scheme == AuthScheme.BEARER:
        token = os.getenv(f"{env_prefix}TOKEN")
        if not token:
            raise APIConfigurationError(
                f"Bearer token not found for API connector '{name}' (expected {env_prefix}TOKEN)"
            )
        return AuthConfig(scheme=AuthScheme.BEARER, token=SecretStr(token))

    if scheme == AuthScheme.BASIC:
        username = os.getenv(f"{env_prefix}USERNAME")
        password = os.getenv(f"{env_prefix}PASSWORD")
        if not username or not password:
            raise APIConfigurationError(
                f"Basic auth credentials not found for API connector '{name}'"
            )
        return AuthConfig(
            scheme=AuthScheme.BASIC,
            username=username,
            password=SecretStr(password),
        )

    if scheme == AuthScheme.API_KEY:
        api_key = os.getenv(f"{env_prefix}API_KEY")
        if not api_key:
            raise APIConfigurationError(
                f"API key not found for API connector '{name}' (expected {env_prefix}API_KEY)"
            )
        api_key_header = os.getenv(f"{env_prefix}API_KEY_HEADER", "X-API-Key")
        api_key_query_param = os.getenv(f"{env_prefix}API_KEY_QUERY_PARAM")
        return AuthConfig(
            scheme=AuthScheme.API_KEY,
            api_key=SecretStr(api_key),
            api_key_header=api_key_header,
            api_key_query_param=api_key_query_param,
        )

    if scheme == AuthScheme.OAUTH2:
        oauth2_token_url = os.getenv(f"{env_prefix}OAUTH2_TOKEN_URL")
        oauth2_client_id = os.getenv(f"{env_prefix}OAUTH2_CLIENT_ID")
        oauth2_client_secret = os.getenv(f"{env_prefix}OAUTH2_CLIENT_SECRET")

        if not all([oauth2_token_url, oauth2_client_id, oauth2_client_secret]):
            raise APIConfigurationError(
                f"OAuth2 configuration incomplete for API connector '{name}'"
            )

        return AuthConfig(
            scheme=AuthScheme.OAUTH2,
            oauth2_token_url=oauth2_token_url,
            oauth2_client_id=oauth2_client_id,
            oauth2_client_secret=SecretStr(oauth2_client_secret),
            oauth2_scope=os.getenv(f"{env_prefix}OAUTH2_SCOPE"),
        )

    raise APIConfigurationError(f"Unsupported auth scheme: {scheme}")


def _load_rate_limit_config(env_prefix: str, kwargs: dict[str, Any]) -> RateLimitConfig:
    """Load rate limit configuration.

    Args:
        env_prefix: Environment variable prefix
        kwargs: Additional configuration parameters

    Returns:
        RateLimitConfig instance
    """
    return RateLimitConfig(
        enabled=_get_bool_env(
            f"{env_prefix}RATE_LIMIT_ENABLED",
            kwargs.pop("rate_limit_enabled", True),
        ),
        requests_per_window=int(
            os.getenv(
                f"{env_prefix}RATE_LIMIT_REQUESTS",
                kwargs.pop("rate_limit_requests", "100"),
            )
        ),
        window_seconds=int(
            os.getenv(
                f"{env_prefix}RATE_LIMIT_WINDOW",
                kwargs.pop("rate_limit_window", "60"),
            )
        ),
        burst_size=_get_optional_int_env(
            f"{env_prefix}RATE_LIMIT_BURST",
            kwargs.pop("rate_limit_burst", None),
        ),
        per_endpoint=_get_bool_env(
            f"{env_prefix}RATE_LIMIT_PER_ENDPOINT",
            kwargs.pop("rate_limit_per_endpoint", False),
        ),
        per_tenant=_get_bool_env(
            f"{env_prefix}RATE_LIMIT_PER_TENANT",
            kwargs.pop("rate_limit_per_tenant", False),
        ),
    )


def _load_retry_config(env_prefix: str, kwargs: dict[str, Any]) -> RetryConfig:
    """Load retry configuration.

    Args:
        env_prefix: Environment variable prefix
        kwargs: Additional configuration parameters

    Returns:
        RetryConfig instance
    """
    return RetryConfig(
        enabled=_get_bool_env(
            f"{env_prefix}RETRY_ENABLED",
            kwargs.pop("retry_enabled", True),
        ),
        max_attempts=int(
            os.getenv(
                f"{env_prefix}RETRY_MAX_ATTEMPTS",
                kwargs.pop("retry_max_attempts", "3"),
            )
        ),
        initial_backoff_ms=int(
            os.getenv(
                f"{env_prefix}RETRY_INITIAL_BACKOFF",
                kwargs.pop("retry_initial_backoff_ms", "1000"),
            )
        ),
        max_backoff_ms=int(
            os.getenv(
                f"{env_prefix}RETRY_MAX_BACKOFF",
                kwargs.pop("retry_max_backoff_ms", "60000"),
            )
        ),
        backoff_multiplier=float(
            os.getenv(
                f"{env_prefix}RETRY_BACKOFF_MULTIPLIER",
                kwargs.pop("retry_backoff_multiplier", "2.0"),
            )
        ),
        jitter=_get_bool_env(
            f"{env_prefix}RETRY_JITTER",
            kwargs.pop("retry_jitter", True),
        ),
    )


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean value from environment or default.

    Args:
        key: Environment variable key
        default: Default value

    Returns:
        Boolean value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _get_optional_int_env(key: str, default: int | None) -> int | None:
    """Get optional integer value from environment.

    Args:
        key: Environment variable key
        default: Default value

    Returns:
        Integer value or None
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default
