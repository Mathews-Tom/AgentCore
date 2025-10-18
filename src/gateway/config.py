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

    # Rate Limiting (placeholder for future implementation)
    RATE_LIMIT_ENABLED: bool = Field(default=False, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Requests per minute")

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

    model_config = {
        "env_file": ".env",
        "env_prefix": "GATEWAY_",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Global settings instance
settings = GatewaySettings()
