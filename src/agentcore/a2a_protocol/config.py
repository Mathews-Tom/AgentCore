"""
A2A Protocol Layer Configuration

Environment-based configuration management for the A2A protocol layer.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8001, description="Server port")

    # CORS
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins",
    )

    # A2A Protocol
    A2A_PROTOCOL_VERSION: str = Field(default="0.2", description="A2A protocol version")
    MAX_CONCURRENT_CONNECTIONS: int = Field(
        default=1000, description="Max WebSocket connections"
    )
    MESSAGE_TIMEOUT_SECONDS: int = Field(
        default=30, description="Message timeout in seconds"
    )
    AGENT_DISCOVERY_TTL: int = Field(
        default=300, description="Agent discovery TTL in seconds"
    )
    TASK_EXECUTION_TIMEOUT: int = Field(
        default=3600, description="Task execution timeout in seconds"
    )

    # Security
    JWT_SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production", description="JWT secret key"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_EXPIRATION_HOURS: int = Field(
        default=24, description="JWT token expiration in hours"
    )

    # Database
    DATABASE_URL: str = Field(
        default="",
        description="PostgreSQL connection URL (overrides individual settings)",
    )
    POSTGRES_USER: str = Field(default="agentcore", description="PostgreSQL username")
    POSTGRES_PASSWORD: str = Field(
        default="password", description="PostgreSQL password"
    )
    POSTGRES_HOST: str = Field(default="localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(default=5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field(
        default="agentcore", description="PostgreSQL database name"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=10, description="Database connection pool size"
    )
    DATABASE_MAX_OVERFLOW: int = Field(
        default=20, description="Database connection pool max overflow"
    )
    DATABASE_POOL_TIMEOUT: int = Field(
        default=30, description="Database connection pool timeout in seconds"
    )
    DATABASE_POOL_RECYCLE: int = Field(
        default=3600, description="Database connection recycle time in seconds"
    )

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    REDIS_CLUSTER_URLS: list[str] = Field(
        default=[], description="Redis cluster URLs (overrides REDIS_URL if provided)"
    )

    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")

    # Bounded Context Reasoning
    REASONING_MAX_ITERATIONS: int = Field(
        default=5, ge=1, le=50, description="Maximum reasoning iterations (1-50)"
    )
    REASONING_CHUNK_SIZE: int = Field(
        default=8192,
        ge=1024,
        le=32768,
        description="Reasoning context chunk size in tokens (1024-32768)",
    )
    REASONING_CARRYOVER_SIZE: int = Field(
        default=4096,
        ge=512,
        le=16384,
        description="Carryover context size in tokens (512-16384)",
    )
    REASONING_DEFAULT_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Default LLM temperature for reasoning (0.0-2.0)",
    )
    REASONING_ENABLE_METRICS: bool = Field(
        default=True, description="Enable reasoning metrics collection"
    )
    REASONING_ENABLE_TRACING: bool = Field(
        default=True, description="Enable reasoning trace_id propagation"
    )
    REASONING_INPUT_SANITIZATION: bool = Field(
        default=True, description="Enable input sanitization for reasoning queries"
    )

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",  # Ignore extra environment variables
    }


# Global settings instance
settings = Settings()
