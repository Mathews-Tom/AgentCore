"""
Reasoning Strategy Framework Configuration.

Environment-based configuration management for the reasoning framework,
supporting multi-level strategy configuration and strategy-specific parameters.
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings


class BoundedContextStrategyConfig(BaseSettings):
    """Configuration for bounded context reasoning strategy."""

    model_config = ConfigDict(env_prefix="REASONING_BOUNDED_CONTEXT_", extra="ignore")

    default_chunk_size: int = Field(
        default=8192,
        ge=1024,
        le=32768,
        description="Default maximum tokens per iteration",
    )
    default_carryover_size: int = Field(
        default=4096,
        ge=512,
        le=16384,
        description="Default tokens to carry forward between iterations",
    )
    default_max_iterations: int = Field(
        default=5, ge=1, le=50, description="Default maximum reasoning iterations"
    )
    max_allowed_iterations: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum iterations allowed (hard limit)",
    )

    @field_validator("default_carryover_size")
    @classmethod
    def validate_carryover_less_than_chunk(cls, v: int, info) -> int:
        """Ensure default_carryover_size < default_chunk_size."""
        if "default_chunk_size" in info.data and v >= info.data["default_chunk_size"]:
            raise ValueError(
                f"default_carryover_size ({v}) must be less than "
                f"default_chunk_size ({info.data['default_chunk_size']})"
            )
        return v


class ChainOfThoughtStrategyConfig(BaseSettings):
    """Configuration for chain of thought reasoning strategy."""

    model_config = ConfigDict(env_prefix="REASONING_CHAIN_OF_THOUGHT_", extra="ignore")

    default_max_tokens: int = Field(
        default=32768,
        ge=1024,
        le=128000,
        description="Default maximum tokens for reasoning",
    )


class ReActStrategyConfig(BaseSettings):
    """Configuration for ReAct (Reasoning + Acting) strategy."""

    model_config = ConfigDict(env_prefix="REASONING_REACT_", extra="ignore")

    default_max_tool_calls: int = Field(
        default=10, ge=1, le=50, description="Default maximum tool calls allowed"
    )
    default_max_tokens: int = Field(
        default=16384,
        ge=1024,
        le=128000,
        description="Default maximum tokens for reasoning",
    )


class ReasoningConfig(BaseSettings):
    """
    Main configuration for the reasoning strategy framework.

    Supports:
    - System-level default strategy selection
    - Enabled strategies list
    - Strategy-specific configuration
    - Performance and monitoring settings
    """

    # Default strategy (null/empty = no default, must be specified per request)
    default_strategy: str | None = Field(
        default=None,
        description="System-level default reasoning strategy (e.g., 'chain_of_thought', 'bounded_context')",
    )

    # Enabled strategies (empty list = no strategies enabled)
    enabled_strategies: list[str] = Field(
        default_factory=lambda: ["chain_of_thought", "bounded_context", "react"],
        description="List of enabled reasoning strategies",
    )

    # Strategy-specific configurations
    bounded_context: BoundedContextStrategyConfig = Field(
        default_factory=BoundedContextStrategyConfig,
        description="Bounded context strategy configuration",
    )
    chain_of_thought: ChainOfThoughtStrategyConfig = Field(
        default_factory=ChainOfThoughtStrategyConfig,
        description="Chain of thought strategy configuration",
    )
    react: ReActStrategyConfig = Field(
        default_factory=ReActStrategyConfig,
        description="ReAct strategy configuration",
    )

    # Performance settings
    max_concurrent_requests: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum concurrent reasoning requests",
    )
    strategy_selection_timeout_ms: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Timeout for strategy selection in milliseconds",
    )
    enable_metrics: bool = Field(
        default=True, description="Enable Prometheus metrics collection"
    )
    enable_trace_logging: bool = Field(
        default=False,
        description="Enable detailed trace logging for debugging (may contain PII)",
    )

    # LLM Provider settings (shared across strategies)
    llm_provider: str = Field(
        default="openai", description="LLM provider (openai, anthropic, etc.)"
    )
    llm_model: str = Field(
        default="gpt-4.1", description="Default LLM model for reasoning"
    )
    llm_api_key_env: str = Field(
        default="OPENAI_API_KEY", description="Environment variable for LLM API key"
    )
    llm_timeout_seconds: int = Field(
        default=60, ge=1, le=300, description="Timeout for LLM API calls in seconds"
    )
    llm_max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum retries for LLM API calls"
    )

    @field_validator("default_strategy")
    @classmethod
    def validate_default_strategy_enabled(cls, v: str | None, info) -> str | None:
        """Ensure default strategy is in enabled_strategies list if specified."""
        if v is not None:
            enabled = info.data.get("enabled_strategies", [])
            if v not in enabled:
                raise ValueError(
                    f"default_strategy '{v}' must be in enabled_strategies list: {enabled}"
                )
        return v

    model_config = ConfigDict(
        env_prefix="REASONING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        """
        Get configuration for a specific strategy.

        Args:
            strategy_name: Name of the strategy (e.g., "bounded_context")

        Returns:
            dict: Strategy configuration as dictionary

        Raises:
            ValueError: If strategy_name is not recognized

        Example:
            >>> config = ReasoningConfig()
            >>> bc_config = config.get_strategy_config("bounded_context")
            >>> print(bc_config["default_chunk_size"])
            8192
        """
        if strategy_name == "bounded_context":
            return self.bounded_context.model_dump()
        elif strategy_name == "chain_of_thought":
            return self.chain_of_thought.model_dump()
        elif strategy_name == "react":
            return self.react.model_dump()
        else:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Supported: bounded_context, chain_of_thought, react"
            )

    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """
        Check if a strategy is enabled.

        Args:
            strategy_name: Name of the strategy

        Returns:
            bool: True if enabled, False otherwise

        Example:
            >>> config = ReasoningConfig()
            >>> if config.is_strategy_enabled("bounded_context"):
            ...     # Use bounded context
            ...     pass
        """
        return strategy_name in self.enabled_strategies


# Global configuration instance
reasoning_config = ReasoningConfig()
