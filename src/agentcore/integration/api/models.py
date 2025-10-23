"""API integration models and configuration.

Pydantic models for API client configuration, requests, and responses.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr


class AuthScheme(str, Enum):
    """Supported authentication schemes."""

    NONE = "none"
    BEARER = "bearer"
    BASIC = "basic"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"


class HTTPMethod(str, Enum):
    """Supported HTTP methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class AuthConfig(BaseModel):
    """Authentication configuration for API requests.

    Supports multiple authentication schemes:
    - Bearer token
    - Basic auth (username/password)
    - API key (header or query parameter)
    - OAuth2 (with token refresh)
    """

    scheme: AuthScheme = Field(default=AuthScheme.NONE, description="Authentication scheme")

    # Bearer token
    token: SecretStr | None = Field(default=None, description="Bearer token or API token")

    # Basic auth
    username: str | None = Field(default=None, description="Basic auth username")
    password: SecretStr | None = Field(default=None, description="Basic auth password")

    # API key
    api_key: SecretStr | None = Field(default=None, description="API key value")
    api_key_header: str | None = Field(default="X-API-Key", description="API key header name")
    api_key_query_param: str | None = Field(default=None, description="API key query parameter name")

    # OAuth2
    oauth2_token_url: str | None = Field(default=None, description="OAuth2 token endpoint URL")
    oauth2_client_id: str | None = Field(default=None, description="OAuth2 client ID")
    oauth2_client_secret: SecretStr | None = Field(default=None, description="OAuth2 client secret")
    oauth2_scope: str | None = Field(default=None, description="OAuth2 scope")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration.

    Uses token bucket algorithm for rate limiting:
    - Tokens are added at a fixed rate (rate per window)
    - Each request consumes one token
    - Requests are blocked when bucket is empty
    """

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_window: int = Field(default=100, description="Maximum requests per window")
    window_seconds: int = Field(default=60, description="Time window in seconds")
    burst_size: int | None = Field(default=None, description="Maximum burst size (defaults to requests_per_window)")
    per_endpoint: bool = Field(default=False, description="Rate limit per endpoint")
    per_tenant: bool = Field(default=False, description="Rate limit per tenant")


class RetryConfig(BaseModel):
    """Retry policy configuration.

    Implements exponential backoff with jitter for failed requests.
    """

    enabled: bool = Field(default=True, description="Enable retry logic")
    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    initial_backoff_ms: int = Field(default=1000, description="Initial backoff in milliseconds")
    max_backoff_ms: int = Field(default=60000, description="Maximum backoff in milliseconds")
    backoff_multiplier: float = Field(default=2.0, description="Backoff multiplier for exponential backoff")
    jitter: bool = Field(default=True, description="Add random jitter to backoff")
    retry_on_status_codes: list[int] = Field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504],
        description="HTTP status codes to retry on",
    )
    retry_on_connection_errors: bool = Field(default=True, description="Retry on connection errors")


class APIConfig(BaseModel):
    """Base configuration for API connectors.

    Contains all configuration needed to connect to and interact with an API.
    """

    name: str = Field(description="API connector name")
    base_url: str = Field(description="Base URL for API endpoint")
    timeout_seconds: int = Field(default=30, description="Request timeout in seconds")

    # Authentication
    auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication configuration")

    # Rate limiting
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig, description="Rate limiting configuration")

    # Retry policy
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry policy configuration")

    # Request settings
    default_headers: dict[str, str] = Field(default_factory=dict, description="Default request headers")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    max_redirects: int = Field(default=10, description="Maximum number of redirects")

    # Connection pooling
    pool_connections: int = Field(default=10, description="Number of connection pool connections")
    pool_maxsize: int = Field(default=10, description="Maximum size of connection pool")
    pool_keepalive: int = Field(default=60, description="Keep-alive timeout in seconds")

    # Metadata
    description: str | None = Field(default=None, description="API connector description")
    tags: list[str] = Field(default_factory=list, description="Connector tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class APIRequest(BaseModel):
    """API request model."""

    method: HTTPMethod = Field(description="HTTP method")
    url: str = Field(description="Request URL (can be relative to base_url)")
    headers: dict[str, str] = Field(default_factory=dict, description="Request headers")
    params: dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    body: dict[str, Any] | str | bytes | None = Field(default=None, description="Request body")
    timeout: int | None = Field(default=None, description="Override timeout for this request")
    retry: bool = Field(default=True, description="Enable retry for this request")

    # Metadata
    request_id: str | None = Field(default=None, description="Request tracking ID")
    tenant_id: str | None = Field(default=None, description="Tenant ID for rate limiting")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional request metadata")


class APIResponse(BaseModel):
    """API response model."""

    status_code: int = Field(description="HTTP status code")
    headers: dict[str, str] = Field(description="Response headers")
    body: Any = Field(description="Response body (parsed)")
    raw_body: str = Field(description="Raw response body")

    # Timing
    request_time: datetime = Field(description="Request timestamp")
    response_time: datetime = Field(description="Response timestamp")
    duration_ms: float = Field(description="Request duration in milliseconds")

    # Retry information
    attempt_number: int = Field(default=1, description="Attempt number (1 for first attempt)")
    total_attempts: int = Field(default=1, description="Total attempts made")

    # Request metadata
    request: APIRequest = Field(description="Original request")

    # Response metadata
    cached: bool = Field(default=False, description="Response served from cache")
    rate_limited: bool = Field(default=False, description="Request was rate limited")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")


class TransformationRule(BaseModel):
    """Data transformation rule for response processing."""

    rule_type: Literal["extract", "map", "filter", "validate"] = Field(
        description="Type of transformation rule"
    )
    source_path: str = Field(description="JSONPath or XPath to source data")
    target_path: str | None = Field(default=None, description="Target path in transformed output")
    transformation: str | None = Field(default=None, description="Transformation expression")
    validation_schema: dict[str, Any] | None = Field(default=None, description="JSON schema for validation")


class ResponseTransformation(BaseModel):
    """Response transformation configuration."""

    rules: list[TransformationRule] = Field(default_factory=list, description="Transformation rules")
    output_format: Literal["json", "dict", "list", "custom"] = Field(
        default="dict",
        description="Output format after transformation",
    )
    error_handling: Literal["raise", "skip", "default"] = Field(
        default="raise",
        description="How to handle transformation errors",
    )
    default_value: Any = Field(default=None, description="Default value if transformation fails")
