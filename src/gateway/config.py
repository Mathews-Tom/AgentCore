"""
Gateway Layer Configuration

Environment-based configuration management for the API gateway.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class GatewaySettings(BaseSettings):
    """Gateway configuration loaded from environment variables."""

    # Application
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8080, description="Server port")

    # CORS Configuration
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://localhost:8001"],
        description="Allowed CORS origins"
    )

    # Gateway Configuration
    GATEWAY_NAME: str = Field(default="AgentCore Gateway", description="Gateway service name")
    GATEWAY_VERSION: str = Field(default="0.1.0", description="Gateway version")
    MAX_REQUEST_SIZE: int = Field(default=10_485_760, description="Max request size in bytes (10MB)")
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout in seconds")

    # Backend Service URLs (placeholders for future routing)
    A2A_PROTOCOL_URL: str = Field(
        default="http://localhost:8001",
        description="A2A Protocol service URL"
    )
    AGENT_RUNTIME_URL: str = Field(
        default="http://localhost:8002",
        description="Agent Runtime service URL"
    )

    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REDIS_URL: str = Field(
        default="redis://localhost:6379/2",
        description="Redis URL for rate limiting"
    )
    RATE_LIMIT_ALGORITHM: str = Field(
        default="sliding_window",
        description="Rate limit algorithm (sliding_window, fixed_window, token_bucket, leaky_bucket)"
    )

    # Default Rate Limit Policies
    RATE_LIMIT_CLIENT_IP_LIMIT: int = Field(default=1000, description="Requests per minute per client IP")
    RATE_LIMIT_CLIENT_IP_WINDOW: int = Field(default=60, description="Client IP rate limit window in seconds")
    RATE_LIMIT_ENDPOINT_LIMIT: int = Field(default=100, description="Requests per minute per endpoint")
    RATE_LIMIT_ENDPOINT_WINDOW: int = Field(default=60, description="Endpoint rate limit window in seconds")
    RATE_LIMIT_USER_LIMIT: int = Field(default=5000, description="Requests per minute per authenticated user")
    RATE_LIMIT_USER_WINDOW: int = Field(default=60, description="User rate limit window in seconds")

    # DDoS Protection
    DDOS_PROTECTION_ENABLED: bool = Field(default=True, description="Enable DDoS protection")
    DDOS_GLOBAL_REQUESTS_PER_SECOND: int = Field(default=10000, description="Global requests per second")
    DDOS_GLOBAL_REQUESTS_PER_MINUTE: int = Field(default=500000, description="Global requests per minute")
    DDOS_IP_REQUESTS_PER_SECOND: int = Field(default=100, description="Per-IP requests per second")
    DDOS_IP_REQUESTS_PER_MINUTE: int = Field(default=1000, description="Per-IP requests per minute")
    DDOS_BURST_THRESHOLD_MULTIPLIER: float = Field(default=5.0, description="Burst detection threshold multiplier")
    DDOS_BURST_WINDOW_SECONDS: int = Field(default=10, description="Burst detection window in seconds")
    DDOS_AUTO_BLOCKING_ENABLED: bool = Field(default=True, description="Enable automatic IP blocking")
    DDOS_AUTO_BLOCK_DURATION_SECONDS: int = Field(default=3600, description="Auto-block duration in seconds")
    DDOS_AUTO_BLOCK_THRESHOLD: int = Field(default=10, description="Violations before auto-block")

    # Rate Limiting Exempt Paths
    RATE_LIMIT_EXEMPT_PATHS: list[str] = Field(
        default=["/health", "/metrics", "/.well-known/"],
        description="Paths exempt from rate limiting"
    )

    # JWT Authentication
    JWT_ALGORITHM: str = Field(default="RS256", description="JWT signing algorithm (RS256 for RSA)")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, description="Access token expiration in minutes")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration in days")
    JWT_ISSUER: str = Field(default="agentcore-gateway", description="JWT token issuer")
    JWT_AUDIENCE: str = Field(default="agentcore-api", description="JWT token audience")

    # RSA Key Management
    RSA_PRIVATE_KEY_PATH: str = Field(
        default="/tmp/agentcore_rsa_private.pem",
        description="Path to RSA private key file"
    )
    RSA_PUBLIC_KEY_PATH: str = Field(
        default="/tmp/agentcore_rsa_public.pem",
        description="Path to RSA public key file"
    )
    RSA_KEY_SIZE: int = Field(default=2048, description="RSA key size in bits")
    RSA_KEY_ROTATION_DAYS: int = Field(default=90, description="Days before RSA key rotation")

    # Session Management
    SESSION_REDIS_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Redis URL for session storage"
    )
    SESSION_MAX_AGE_HOURS: int = Field(default=24, description="Maximum session age in hours")
    SESSION_CLEANUP_INTERVAL_MINUTES: int = Field(
        default=60,
        description="Session cleanup interval in minutes"
    )

    # OAuth 2.0/3.0 Configuration
    OAUTH_ENABLED: bool = Field(default=False, description="Enable OAuth providers")
    OAUTH_PROVIDERS: list[str] = Field(default=[], description="Enabled OAuth providers (google, github, microsoft)")

    # Google OAuth
    OAUTH_GOOGLE_CLIENT_ID: str | None = Field(None, description="Google OAuth client ID")
    OAUTH_GOOGLE_CLIENT_SECRET: str | None = Field(None, description="Google OAuth client secret")
    OAUTH_GOOGLE_REDIRECT_URI: str | None = Field(None, description="Google OAuth redirect URI")

    # GitHub OAuth
    OAUTH_GITHUB_CLIENT_ID: str | None = Field(None, description="GitHub OAuth client ID")
    OAUTH_GITHUB_CLIENT_SECRET: str | None = Field(None, description="GitHub OAuth client secret")
    OAUTH_GITHUB_REDIRECT_URI: str | None = Field(None, description="GitHub OAuth redirect URI")

    # Microsoft OAuth
    OAUTH_MICROSOFT_CLIENT_ID: str | None = Field(None, description="Microsoft OAuth client ID")
    OAUTH_MICROSOFT_CLIENT_SECRET: str | None = Field(None, description="Microsoft OAuth client secret")
    OAUTH_MICROSOFT_REDIRECT_URI: str | None = Field(None, description="Microsoft OAuth redirect URI")
    OAUTH_MICROSOFT_TENANT_ID: str | None = Field(None, description="Microsoft OAuth tenant ID")

    # OAuth State Management
    OAUTH_STATE_TTL_MINUTES: int = Field(default=10, description="OAuth state TTL in minutes")

    # Enterprise SSO Configuration
    SSO_ENABLED: bool = Field(default=False, description="Enable enterprise SSO")
    SSO_LDAP_ENABLED: bool = Field(default=False, description="Enable LDAP authentication")
    SSO_SAML_ENABLED: bool = Field(default=False, description="Enable SAML authentication")

    # LDAP Configuration
    LDAP_SERVER_URI: str | None = Field(None, description="LDAP server URI")
    LDAP_BIND_DN: str | None = Field(None, description="LDAP bind DN")
    LDAP_BIND_PASSWORD: str | None = Field(None, description="LDAP bind password")
    LDAP_BASE_DN: str | None = Field(None, description="LDAP base DN")
    LDAP_USE_TLS: bool = Field(default=True, description="Use TLS for LDAP connections")

    # SAML Configuration
    SAML_ENTITY_ID: str | None = Field(None, description="SAML entity ID")
    SAML_SSO_URL: str | None = Field(None, description="SAML SSO URL")
    SAML_X509_CERT: str | None = Field(None, description="SAML X.509 certificate")

    # Real-time Communication Configuration
    REALTIME_ENABLED: bool = Field(default=True, description="Enable WebSocket and SSE support")
    REALTIME_MAX_CONNECTIONS: int = Field(default=10000, description="Maximum concurrent connections")
    REALTIME_HEARTBEAT_INTERVAL: int = Field(default=30, description="Heartbeat interval in seconds")
    REALTIME_CONNECTION_TIMEOUT: int = Field(default=300, description="Connection timeout in seconds")
    REALTIME_KEEPALIVE_INTERVAL: int = Field(default=30, description="SSE keepalive interval in seconds")
    REALTIME_EVENT_QUEUE_SIZE: int = Field(default=10000, description="Event queue size")

    model_config = {
        "env_file": ".env",
        "env_prefix": "GATEWAY_",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Global settings instance
settings = GatewaySettings()
