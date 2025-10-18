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

    # OAuth 3.0 (prepare for GATE-003)
    OAUTH_ENABLED: bool = Field(default=False, description="Enable OAuth 3.0 providers")
    OAUTH_PROVIDERS: list[str] = Field(default=[], description="Enabled OAuth providers")

    model_config = {
        "env_file": ".env",
        "env_prefix": "GATEWAY_",
        "case_sensitive": True,
        "extra": "ignore",
    }


# Global settings instance
settings = GatewaySettings()
