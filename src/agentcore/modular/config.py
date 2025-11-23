"""
Module-Specific Model Configuration

Provides per-module LLM model selection and cost tracking for the modular agent core.
Enables optimization of cost vs. performance by using different model sizes for each module.

Configuration hierarchy:
1. Environment variables (highest priority)
2. Programmatic configuration
3. Default values (lowest priority)

Environment Variables:
- MODULAR_PLANNER_MODEL: Model for Planner module (default: gpt-5)
- MODULAR_EXECUTOR_MODEL: Model for Executor module (default: gpt-4.1)
- MODULAR_VERIFIER_MODEL: Model for Verifier module (default: gpt-4.1-mini)
- MODULAR_GENERATOR_MODEL: Model for Generator module (default: gpt-4.1)

Cost Tracking:
Each module tracks token usage for cost attribution:
- Input tokens
- Output tokens
- Total tokens
- Estimated cost (based on model pricing)

Example:
    >>> config = ModularConfig()
    >>> config.get_model_for_module("planner")
    'gpt-5'
    >>> config.track_token_usage("planner", input_tokens=100, output_tokens=50)
    >>> config.get_module_costs()
    {'planner': {'tokens': 150, 'estimated_cost_usd': 0.0015}}
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class ModuleName(str, Enum):
    """Valid module names in the modular agent core."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    VERIFIER = "verifier"
    GENERATOR = "generator"


class ModelTier(str, Enum):
    """Model tier classification for cost optimization."""

    LARGE = "large"  # gpt-5, gpt-5-pro - complex reasoning
    MEDIUM = "medium"  # gpt-4.1 - balanced performance
    SMALL = "small"  # gpt-4.1-mini, gpt-5-mini - simple tasks


# Model pricing (cost per 1M tokens in USD)
# Source: OpenAI pricing as of 2025-01
MODEL_PRICING: dict[str, dict[str, float]] = {
    # GPT-5 Series
    "gpt-5": {"input": 5.0, "output": 15.0},
    "gpt-5-mini": {"input": 1.0, "output": 3.0},
    "gpt-5-pro": {"input": 10.0, "output": 30.0},
    # GPT-4.1 Series
    "gpt-4.1": {"input": 2.5, "output": 7.5},
    "gpt-4.1-mini": {"input": 0.5, "output": 1.5},
}

# Allowed models (from CLAUDE.md requirements)
ALLOWED_MODELS: frozenset[str] = frozenset({
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5",
    "gpt-5-mini",
})


class TokenUsage(BaseModel):
    """Token usage tracking for a single module."""

    input_tokens: int = Field(default=0, ge=0, description="Total input tokens consumed")
    output_tokens: int = Field(default=0, ge=0, description="Total output tokens consumed")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens (input + output)")
    estimated_cost_usd: float = Field(
        default=0.0, ge=0.0, description="Estimated cost in USD based on model pricing"
    )
    invocation_count: int = Field(
        default=0, ge=0, description="Number of LLM invocations"
    )

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> None:
        """
        Add token usage and update cost estimate.

        Args:
            input_tokens: Input tokens for this invocation
            output_tokens: Output tokens for this invocation
            model: Model used for this invocation

        Raises:
            ValueError: If token counts are negative
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative")

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens = self.input_tokens + self.output_tokens
        self.invocation_count += 1

        # Update cost estimate
        if model in MODEL_PRICING:
            pricing = MODEL_PRICING[model]
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            self.estimated_cost_usd += input_cost + output_cost


class ModuleModelConfig(BaseModel):
    """Model configuration for a single module."""

    module_name: ModuleName = Field(..., description="Module name")
    model: str = Field(..., description="LLM model to use")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature (0.0-2.0)"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens for response (optional)"
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Token usage tracking"
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate that model is in allowed list."""
        if v not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{v}' is not allowed. Allowed models: {sorted(ALLOWED_MODELS)}"
            )
        return v


class ModularConfigSettings(BaseSettings):
    """Settings for modular agent configuration loaded from environment."""

    # Per-module model configuration
    MODULAR_PLANNER_MODEL: str = Field(
        default="gpt-5",
        description="Model for Planner module (large model for complex reasoning)",
    )
    MODULAR_EXECUTOR_MODEL: str = Field(
        default="gpt-4.1",
        description="Model for Executor module (medium model for tool invocation)",
    )
    MODULAR_VERIFIER_MODEL: str = Field(
        default="gpt-4.1-mini",
        description="Model for Verifier module (small model for validation)",
    )
    MODULAR_GENERATOR_MODEL: str = Field(
        default="gpt-4.1",
        description="Model for Generator module (medium model for response synthesis)",
    )

    # Per-module temperature settings (optional overrides)
    MODULAR_PLANNER_TEMPERATURE: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for Planner"
    )
    MODULAR_EXECUTOR_TEMPERATURE: float = Field(
        default=0.5, ge=0.0, le=2.0, description="Temperature for Executor"
    )
    MODULAR_VERIFIER_TEMPERATURE: float = Field(
        default=0.3, ge=0.0, le=2.0, description="Temperature for Verifier"
    )
    MODULAR_GENERATOR_TEMPERATURE: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Temperature for Generator"
    )

    # Per-module max tokens (optional)
    MODULAR_PLANNER_MAX_TOKENS: int | None = Field(
        default=4096, ge=1, description="Max tokens for Planner"
    )
    MODULAR_EXECUTOR_MAX_TOKENS: int | None = Field(
        default=2048, ge=1, description="Max tokens for Executor"
    )
    MODULAR_VERIFIER_MAX_TOKENS: int | None = Field(
        default=1024, ge=1, description="Max tokens for Verifier"
    )
    MODULAR_GENERATOR_MAX_TOKENS: int | None = Field(
        default=2048, ge=1, description="Max tokens for Generator"
    )

    @field_validator(
        "MODULAR_PLANNER_MODEL",
        "MODULAR_EXECUTOR_MODEL",
        "MODULAR_VERIFIER_MODEL",
        "MODULAR_GENERATOR_MODEL",
    )
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate that model is in allowed list."""
        if v not in ALLOWED_MODELS:
            raise ValueError(
                f"Model '{v}' is not allowed. Allowed models: {sorted(ALLOWED_MODELS)}"
            )
        return v

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore",
    }


class ModularConfig:
    """
    Configuration manager for modular agent core.

    Manages per-module LLM model selection and cost tracking.
    Configuration is loaded from environment variables with sensible defaults.

    Default Model Assignments (optimized for cost vs. performance):
    - Planner: gpt-5 (large model for complex reasoning and planning)
    - Executor: gpt-4.1 (medium model for tool invocation)
    - Verifier: gpt-4.1-mini (small model for validation tasks)
    - Generator: gpt-4.1 (medium model for response synthesis)

    Cost Optimization Strategy:
    - Use larger models (gpt-5) for complex reasoning tasks (Planner)
    - Use medium models (gpt-4.1) for balanced tasks (Executor, Generator)
    - Use smaller models (gpt-4.1-mini) for simple validation (Verifier)
    - Target: 30-40% cost reduction vs. using gpt-5 for all modules

    Example:
        >>> config = ModularConfig()
        >>> planner_model = config.get_model_for_module("planner")
        >>> print(planner_model)
        'gpt-5'
        >>> config.track_token_usage("planner", input_tokens=100, output_tokens=50)
        >>> costs = config.get_module_costs()
        >>> print(costs["planner"]["estimated_cost_usd"])
        0.0015
    """

    def __init__(
        self,
        settings: ModularConfigSettings | None = None,
    ) -> None:
        """
        Initialize modular configuration.

        Args:
            settings: Optional settings override (defaults to loading from environment)
        """
        self.settings = settings or ModularConfigSettings()
        self._module_configs: dict[ModuleName, ModuleModelConfig] = {}

        # Initialize module configurations
        self._initialize_module_configs()

    def _initialize_module_configs(self) -> None:
        """Initialize per-module configurations from settings."""
        # Planner: Large model for complex reasoning
        self._module_configs[ModuleName.PLANNER] = ModuleModelConfig(
            module_name=ModuleName.PLANNER,
            model=self.settings.MODULAR_PLANNER_MODEL,
            temperature=self.settings.MODULAR_PLANNER_TEMPERATURE,
            max_tokens=self.settings.MODULAR_PLANNER_MAX_TOKENS,
        )

        # Executor: Medium model for tool invocation
        self._module_configs[ModuleName.EXECUTOR] = ModuleModelConfig(
            module_name=ModuleName.EXECUTOR,
            model=self.settings.MODULAR_EXECUTOR_MODEL,
            temperature=self.settings.MODULAR_EXECUTOR_TEMPERATURE,
            max_tokens=self.settings.MODULAR_EXECUTOR_MAX_TOKENS,
        )

        # Verifier: Small model for validation
        self._module_configs[ModuleName.VERIFIER] = ModuleModelConfig(
            module_name=ModuleName.VERIFIER,
            model=self.settings.MODULAR_VERIFIER_MODEL,
            temperature=self.settings.MODULAR_VERIFIER_TEMPERATURE,
            max_tokens=self.settings.MODULAR_VERIFIER_MAX_TOKENS,
        )

        # Generator: Medium model for response synthesis
        self._module_configs[ModuleName.GENERATOR] = ModuleModelConfig(
            module_name=ModuleName.GENERATOR,
            model=self.settings.MODULAR_GENERATOR_MODEL,
            temperature=self.settings.MODULAR_GENERATOR_TEMPERATURE,
            max_tokens=self.settings.MODULAR_GENERATOR_MAX_TOKENS,
        )

    def get_model_for_module(self, module: str | ModuleName) -> str:
        """
        Get LLM model for a specific module.

        Args:
            module: Module name (string or ModuleName enum)

        Returns:
            Model identifier (e.g., "gpt-5", "gpt-4.1-mini")

        Raises:
            ValueError: If module name is invalid

        Example:
            >>> config = ModularConfig()
            >>> config.get_model_for_module("planner")
            'gpt-5'
            >>> config.get_model_for_module(ModuleName.EXECUTOR)
            'gpt-4.1'
        """
        module_name = self._normalize_module_name(module)
        return self._module_configs[module_name].model

    def get_temperature_for_module(self, module: str | ModuleName) -> float:
        """
        Get LLM temperature for a specific module.

        Args:
            module: Module name (string or ModuleName enum)

        Returns:
            Temperature value (0.0-2.0)

        Raises:
            ValueError: If module name is invalid
        """
        module_name = self._normalize_module_name(module)
        return self._module_configs[module_name].temperature

    def get_max_tokens_for_module(self, module: str | ModuleName) -> int | None:
        """
        Get max tokens for a specific module.

        Args:
            module: Module name (string or ModuleName enum)

        Returns:
            Max tokens or None if not set

        Raises:
            ValueError: If module name is invalid
        """
        module_name = self._normalize_module_name(module)
        return self._module_configs[module_name].max_tokens

    def get_module_config(self, module: str | ModuleName) -> ModuleModelConfig:
        """
        Get complete configuration for a module.

        Args:
            module: Module name (string or ModuleName enum)

        Returns:
            Module configuration with model, temperature, and usage

        Raises:
            ValueError: If module name is invalid
        """
        module_name = self._normalize_module_name(module)
        return self._module_configs[module_name]

    def track_token_usage(
        self,
        module: str | ModuleName,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """
        Track token usage for a module.

        Automatically updates cost estimates based on model pricing.

        Args:
            module: Module name (string or ModuleName enum)
            input_tokens: Input tokens consumed
            output_tokens: Output tokens consumed

        Raises:
            ValueError: If module name is invalid or token counts are negative

        Example:
            >>> config = ModularConfig()
            >>> config.track_token_usage("planner", input_tokens=100, output_tokens=50)
            >>> usage = config.get_token_usage("planner")
            >>> print(usage.total_tokens)
            150
        """
        module_name = self._normalize_module_name(module)
        config = self._module_configs[module_name]
        config.token_usage.add_usage(input_tokens, output_tokens, config.model)

    def get_token_usage(self, module: str | ModuleName) -> TokenUsage:
        """
        Get token usage for a module.

        Args:
            module: Module name (string or ModuleName enum)

        Returns:
            Token usage with cost estimate

        Raises:
            ValueError: If module name is invalid
        """
        module_name = self._normalize_module_name(module)
        return self._module_configs[module_name].token_usage

    def get_module_costs(self) -> dict[str, dict[str, Any]]:
        """
        Get cost summary for all modules.

        Returns:
            Dict mapping module names to cost information

        Example:
            >>> config = ModularConfig()
            >>> config.track_token_usage("planner", 100, 50)
            >>> costs = config.get_module_costs()
            >>> print(costs)
            {
                'planner': {
                    'tokens': 150,
                    'invocations': 1,
                    'estimated_cost_usd': 0.0015
                },
                ...
            }
        """
        costs = {}
        for module_name, config in self._module_configs.items():
            usage = config.token_usage
            costs[module_name.value] = {
                "tokens": usage.total_tokens,
                "invocations": usage.invocation_count,
                "estimated_cost_usd": usage.estimated_cost_usd,
            }
        return costs

    def get_total_cost(self) -> float:
        """
        Get total estimated cost across all modules.

        Returns:
            Total cost in USD

        Example:
            >>> config = ModularConfig()
            >>> config.track_token_usage("planner", 100, 50)
            >>> config.track_token_usage("executor", 200, 100)
            >>> total = config.get_total_cost()
            >>> print(f"${total:.4f}")
            $0.0045
        """
        return sum(
            config.token_usage.estimated_cost_usd
            for config in self._module_configs.values()
        )

    def reset_usage_stats(self, module: str | ModuleName | None = None) -> None:
        """
        Reset token usage statistics.

        Args:
            module: Module name to reset (or None to reset all modules)

        Raises:
            ValueError: If module name is invalid
        """
        if module is None:
            # Reset all modules
            for config in self._module_configs.values():
                config.token_usage = TokenUsage()
        else:
            # Reset specific module
            module_name = self._normalize_module_name(module)
            self._module_configs[module_name].token_usage = TokenUsage()

    def _normalize_module_name(self, module: str | ModuleName) -> ModuleName:
        """
        Normalize module name to ModuleName enum.

        Args:
            module: Module name as string or enum

        Returns:
            ModuleName enum value

        Raises:
            ValueError: If module name is invalid
        """
        if isinstance(module, ModuleName):
            return module

        # Convert string to lowercase for case-insensitive lookup
        try:
            return ModuleName(module.lower())
        except ValueError:
            valid_names = [m.value for m in ModuleName]
            raise ValueError(
                f"Invalid module name '{module}'. Valid names: {valid_names}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Configuration dict with all module settings

        Example:
            >>> config = ModularConfig()
            >>> config_dict = config.to_dict()
            >>> print(config_dict["planner"]["model"])
            'gpt-5'
        """
        return {
            module_name.value: {
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "token_usage": config.token_usage.model_dump(),
            }
            for module_name, config in self._module_configs.items()
        }


# Global configuration instance
_global_config: ModularConfig | None = None


def get_modular_config() -> ModularConfig:
    """
    Get global modular configuration instance.

    Lazy initialization - creates config on first access.

    Returns:
        Global ModularConfig instance

    Example:
        >>> config = get_modular_config()
        >>> model = config.get_model_for_module("planner")
    """
    global _global_config
    if _global_config is None:
        _global_config = ModularConfig()
    return _global_config


def reset_modular_config() -> None:
    """
    Reset global modular configuration instance.

    Useful for testing or when configuration needs to be reloaded.
    """
    global _global_config
    _global_config = None
