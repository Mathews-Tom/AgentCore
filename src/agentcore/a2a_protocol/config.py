"""
A2A Protocol Layer Configuration

Environment-based configuration management for the A2A protocol layer.
"""

from typing import Any

from pydantic import Field, field_validator
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
        default=10, description="Database connection pool size (minimum connections)"
    )
    DATABASE_MAX_OVERFLOW: int = Field(
        default=40, description="Database connection pool max overflow (allows up to 50 total)"
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

    # Qdrant Vector Database (Memory System)
    QDRANT_URL: str = Field(
        default="http://localhost:6333", description="Qdrant HTTP API connection URL"
    )
    QDRANT_GRPC_URL: str | None = Field(
        default=None, description="Qdrant gRPC API connection URL (optional, for better performance)"
    )
    QDRANT_API_KEY: str | None = Field(
        default=None, description="Qdrant API key (required for cloud deployment)"
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="agentcore_memories", description="Qdrant collection name for memory storage"
    )
    QDRANT_VECTOR_SIZE: int = Field(
        default=1536, ge=1, description="Vector dimension size for embeddings (1536 for OpenAI text-embedding-3-small)"
    )
    QDRANT_DISTANCE: str = Field(
        default="Cosine", description="Distance metric for vector similarity (Cosine, Euclid, or Dot)"
    )
    QDRANT_TIMEOUT: int = Field(
        default=30, gt=0, description="Qdrant request timeout in seconds"
    )
    QDRANT_MAX_RETRIES: int = Field(
        default=3, ge=0, description="Maximum retry attempts for Qdrant requests"
    )

    # Neo4j Graph Database (Memory System - Knowledge Graphs)
    NEO4J_URI: str = Field(
        default="bolt://localhost:7687", description="Neo4j Bolt protocol URI"
    )
    NEO4J_USER: str = Field(
        default="neo4j", description="Neo4j username"
    )
    NEO4J_PASSWORD: str = Field(
        default="password", description="Neo4j password"
    )
    NEO4J_DATABASE: str = Field(
        default="agentcore", description="Neo4j database name"
    )
    NEO4J_MAX_CONNECTION_LIFETIME: int = Field(
        default=3600, gt=0, description="Neo4j connection lifetime in seconds"
    )
    NEO4J_MAX_CONNECTION_POOL_SIZE: int = Field(
        default=50, ge=1, description="Maximum Neo4j connection pool size"
    )
    NEO4J_CONNECTION_ACQUISITION_TIMEOUT: int = Field(
        default=60, gt=0, description="Neo4j connection acquisition timeout in seconds"
    )
    NEO4J_ENCRYPTED: bool = Field(
        default=False, description="Enable TLS encryption for Neo4j connections"
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

    # LLM Service Configuration
    ALLOWED_MODELS: list[str] = Field(
        default=[
            # OpenAI models
            "gpt-5-mini",
            "gpt-5",
            "gpt-5-pro",
            # Anthropic models
            "claude-haiku-4-5-20251001",
            "claude-sonnet-4-5-20250929",
            "claude-opus-4-1-20250805",
            # Gemini models
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        description="List of allowed LLM models for the service",
    )
    LLM_DEFAULT_MODEL: str = Field(
        default="gpt-5-mini", description="Default LLM model to use"
    )

    OPENAI_API_KEY: str | None = Field(
        default=None, description="OpenAI API key for GPT models"
    )
    ANTHROPIC_API_KEY: str | None = Field(
        default=None, description="Anthropic API key for Claude models"
    )
    GEMINI_API_KEY: str | None = Field(
        default=None, description="Google Gemini API key"
    )

    LLM_REQUEST_TIMEOUT: float = Field(
        default=60.0, gt=0, description="LLM request timeout in seconds (must be >0)"
    )
    LLM_MAX_RETRIES: int = Field(
        default=3, ge=0, description="Maximum retry attempts for LLM requests (>=0)"
    )
    LLM_RETRY_EXPONENTIAL_BASE: float = Field(
        default=2.0,
        gt=1,
        description="Exponential backoff base for retry delays (must be >1)",
    )
    LLM_MAX_RETRY_DELAY: float = Field(
        default=32.0,
        gt=0,
        description="Maximum retry delay in seconds for rate limit backoff (must be >0)",
    )
    LLM_RATE_LIMIT_QUEUE_SIZE: int = Field(
        default=100, ge=0, description="Maximum queue size for rate-limited requests (>=0)"
    )

    # Coordination Service (Ripple Effect Protocol)
    COORDINATION_ENABLE_REP: bool = Field(
        default=True, description="Enable Ripple Effect Protocol coordination"
    )
    COORDINATION_SIGNAL_TTL: int = Field(
        default=60, gt=0, description="Default signal time-to-live in seconds (must be >0)"
    )
    COORDINATION_MAX_HISTORY_SIZE: int = Field(
        default=100, ge=1, description="Maximum signal history size per agent (>=1)"
    )
    COORDINATION_CLEANUP_INTERVAL: int = Field(
        default=300, gt=0, description="Signal cleanup interval in seconds (must be >0)"
    )

    # Routing Optimization Weights (must sum to 1.0)
    ROUTING_WEIGHT_LOAD: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Weight for load score (0.0-1.0)"
    )
    ROUTING_WEIGHT_CAPACITY: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Weight for capacity score (0.0-1.0)"
    )
    ROUTING_WEIGHT_QUALITY: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Weight for quality score (0.0-1.0)"
    )
    ROUTING_WEIGHT_COST: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight for cost score (0.0-1.0)"
    )
    ROUTING_WEIGHT_AVAILABILITY: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight for availability score (0.0-1.0)"
    )

    @field_validator("ROUTING_WEIGHT_AVAILABILITY")
    @classmethod
    def validate_weights_sum_to_one(cls, v: float, info: Any) -> float:
        """Ensure routing weights sum to 1.0 (±0.01 tolerance)."""
        # Get all weight values from the validation context
        weights = [
            info.data.get("ROUTING_WEIGHT_LOAD", 0.25),
            info.data.get("ROUTING_WEIGHT_CAPACITY", 0.25),
            info.data.get("ROUTING_WEIGHT_QUALITY", 0.20),
            info.data.get("ROUTING_WEIGHT_COST", 0.15),
            v,  # ROUTING_WEIGHT_AVAILABILITY (current field)
        ]
        total = sum(weights)

        # Allow small tolerance for floating point precision
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Routing weights must sum to 1.0 (got {total:.3f}). "
                f"Weights: load={weights[0]}, capacity={weights[1]}, "
                f"quality={weights[2]}, cost={weights[3]}, availability={weights[4]}"
            )
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",  # Ignore extra environment variables
    }


# Global settings instance
settings = Settings()
