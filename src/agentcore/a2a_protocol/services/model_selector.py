"""Runtime model selector for intelligent tier-based selection.

This module implements the ModelSelector class that provides intelligent model
selection based on task complexity and configured tiers. It enables cost optimization
by mapping task requirements to appropriate model tiers (FAST, BALANCED, PREMIUM).

The selector implements the Strategy pattern for flexible model selection with:
- Tier-to-model mapping (FAST → gpt-5-mini, BALANCED → gpt-5, PREMIUM → gpt-5)
- Complexity-to-tier mapping (low → FAST, medium → BALANCED, high → PREMIUM)
- Provider preference support for multi-provider failover
- Fallback model selection if preferred model unavailable
- Selection rationale logging for observability
- Configuration validation to ensure all tiers have models

This is a P1 (nice-to-have) cost optimization feature that enables intelligent
model selection based on task requirements rather than hardcoded model choices.

Example:
    ```python
    from agentcore.a2a_protocol.services.model_selector import ModelSelector
    from agentcore.a2a_protocol.models.llm import ModelTier

    # Create selector with provider preference
    selector = ModelSelector(provider_preference=["openai", "anthropic", "gemini"])

    # Select by tier
    model = selector.select_model(ModelTier.FAST)  # "gpt-5-mini"

    # Select by complexity
    model = selector.select_model_by_complexity("low")  # "gpt-5-mini"
    model = selector.select_model_by_complexity("medium")  # "gpt-5"
    model = selector.select_model_by_complexity("high")  # "gpt-5"

    # Validate configuration
    is_valid = selector.validate_configuration()  # True if all tiers mapped
    ```

Error Handling:
    - ValueError: Raised when no models available for tier or invalid complexity
    - RuntimeError: Raised when configuration validation fails
"""

from __future__ import annotations

import logging

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.llm import ModelTier
from agentcore.a2a_protocol.services.llm_service import MODEL_PROVIDER_MAP

# Tier-to-model mapping configuration
# Each tier maps to a list of candidate models in priority order
# Models must be in ALLOWED_MODELS configuration to be selected
TIER_MODEL_MAP: dict[ModelTier, list[str]] = {
    ModelTier.FAST: [
        "gpt-5-mini",
        "claude-haiku-4-5-20251001",
        "gemini-2.5-flash-lite",
    ],
    ModelTier.BALANCED: [
        "gpt-5",
        "claude-sonnet-4-5-20250929",
        "gemini-2.5-flash",
    ],
    ModelTier.PREMIUM: [
        "gpt-5-pro",  # Most capable OpenAI reasoning model
        "claude-opus-4-1-20250805",
        "gemini-2.5-pro",
    ],
}

# Complexity-to-tier mapping configuration
# Maps task complexity strings to ModelTier enums
COMPLEXITY_TIER_MAP: dict[str, ModelTier] = {
    "low": ModelTier.FAST,
    "medium": ModelTier.BALANCED,
    "high": ModelTier.PREMIUM,
}


class ModelSelector:
    """Intelligent model selector for tier-based cost optimization.

    This class implements the Strategy pattern for flexible model selection based on
    task complexity and configured tiers. It enables cost optimization by selecting
    appropriate models from tier mappings while respecting provider preferences and
    ALLOWED_MODELS governance.

    The selector provides two selection strategies:
    1. Direct tier selection: select_model(tier) for explicit tier choice
    2. Complexity-based selection: select_model_by_complexity(complexity) for simplified API

    Features:
    - Tier-to-model mapping with multiple candidates per tier
    - Complexity-to-tier mapping for simplified selection
    - Provider preference ordering (e.g., prefer OpenAI over Anthropic)
    - Fallback selection if preferred model not in ALLOWED_MODELS
    - Selection rationale logging for observability
    - Configuration validation to ensure all tiers have models

    Attributes:
        provider_preference: List of provider names in preference order (e.g., ["openai", "anthropic"])
        logger: Structured logger for selection rationale and warnings
    """

    def __init__(self, provider_preference: list[str] | None = None) -> None:
        """Initialize model selector with optional provider preference.

        Args:
            provider_preference: List of provider names in preference order
                (e.g., ["openai", "anthropic", "gemini"]). If None, uses
                natural order from TIER_MODEL_MAP. Provider names must match
                Provider enum values: "openai", "anthropic", "gemini".
        """
        self.provider_preference = provider_preference
        self.logger = logging.getLogger(__name__)

    def select_model(self, tier: ModelTier) -> str:
        """Select model for the specified tier with provider preference and fallback.

        This method implements the core model selection logic:
        1. Get candidate models from TIER_MODEL_MAP[tier]
        2. Filter by ALLOWED_MODELS configuration for governance
        3. Apply provider preference if specified
        4. Return first match (highest priority)
        5. Log selection rationale for observability
        6. Raise error if no model available

        Args:
            tier: Model tier to select from (FAST, BALANCED, or PREMIUM)

        Returns:
            Model identifier string (e.g., "gpt-5-mini", "claude-sonnet-4-5-20250929")

        Raises:
            ValueError: When no models available for tier after filtering by ALLOWED_MODELS

        Example:
            >>> selector = ModelSelector(provider_preference=["openai"])
            >>> model = selector.select_model(ModelTier.FAST)
            >>> print(model)
            "gpt-5-mini"
        """
        # Get candidate models from tier mapping
        candidate_models = TIER_MODEL_MAP[tier]

        # Filter by ALLOWED_MODELS for governance enforcement
        allowed_models = [
            model for model in candidate_models if model in settings.ALLOWED_MODELS
        ]

        if not allowed_models:
            self.logger.error(
                "No allowed models for tier",
                extra={
                    "tier": tier.value,
                    "candidate_models": candidate_models,
                    "allowed_models": settings.ALLOWED_MODELS,
                },
            )
            raise ValueError(
                f"No allowed models for tier '{tier.value}'. "
                f"Candidates: {candidate_models}, Allowed: {settings.ALLOWED_MODELS}"
            )

        # Apply provider preference if specified
        if self.provider_preference:
            preferred_models = self._filter_by_preference(allowed_models)
            if preferred_models:
                selected_model = preferred_models[0]
            else:
                # Fallback to first allowed model if no preferred provider available
                selected_model = allowed_models[0]
                self.logger.warning(
                    "Provider preference not satisfied, using fallback",
                    extra={
                        "tier": tier.value,
                        "provider_preference": self.provider_preference,
                        "fallback_model": selected_model,
                    },
                )
        else:
            # No preference, use first allowed model
            selected_model = allowed_models[0]

        # Log selection rationale for observability
        provider = MODEL_PROVIDER_MAP.get(selected_model, "unknown")
        self.logger.info(
            "Model selected",
            extra={
                "tier": tier.value,
                "selected_model": selected_model,
                "provider": provider.value if hasattr(provider, "value") else str(provider),
                "candidate_count": len(candidate_models),
                "allowed_count": len(allowed_models),
            },
        )

        return selected_model

    def select_model_by_complexity(self, complexity: str) -> str:
        """Select model by task complexity level with automatic tier mapping.

        This method provides a simplified selection API that maps complexity strings
        to model tiers and delegates to select_model(). It's designed for callers
        who don't want to deal with tier enums directly.

        Complexity mapping:
        - "low" → ModelTier.FAST (e.g., gpt-5-mini)
        - "medium" → ModelTier.BALANCED (e.g., gpt-5)
        - "high" → ModelTier.PREMIUM (e.g., gpt-5-pro)

        Args:
            complexity: Task complexity level ("low", "medium", or "high")

        Returns:
            Model identifier string selected for the mapped tier

        Raises:
            ValueError: When complexity is not recognized (not in ["low", "medium", "high"])

        Example:
            >>> selector = ModelSelector()
            >>> model = selector.select_model_by_complexity("low")
            >>> print(model)
            "gpt-5-mini"
        """
        # Map complexity to tier
        tier = COMPLEXITY_TIER_MAP.get(complexity.lower())

        if tier is None:
            valid_complexities = list(COMPLEXITY_TIER_MAP.keys())
            self.logger.error(
                "Invalid complexity level",
                extra={
                    "complexity": complexity,
                    "valid_complexities": valid_complexities,
                },
            )
            raise ValueError(
                f"Invalid complexity level '{complexity}'. "
                f"Valid levels: {valid_complexities}"
            )

        # Delegate to tier-based selection
        self.logger.debug(
            "Complexity mapped to tier",
            extra={"complexity": complexity, "tier": tier.value},
        )

        return self.select_model(tier)

    def validate_configuration(self) -> bool:
        """Validate configuration ensures all tiers have at least one allowed model.

        This method checks that every tier in TIER_MODEL_MAP has at least one model
        that's in ALLOWED_MODELS configuration. It logs warnings for tiers with no
        allowed models, which would cause select_model() to fail at runtime.

        This should be called during service initialization to detect configuration
        issues early before production use.

        Returns:
            True if all tiers have at least one allowed model, False otherwise

        Example:
            >>> selector = ModelSelector()
            >>> is_valid = selector.validate_configuration()
            >>> if not is_valid:
            ...     print("Configuration has missing tier mappings!")
        """
        all_valid = True

        for tier, candidate_models in TIER_MODEL_MAP.items():
            allowed_models = [
                model for model in candidate_models if model in settings.ALLOWED_MODELS
            ]

            if not allowed_models:
                all_valid = False
                self.logger.warning(
                    "Tier has no allowed models",
                    extra={
                        "tier": tier.value,
                        "candidate_models": candidate_models,
                        "allowed_models": settings.ALLOWED_MODELS,
                    },
                )

        if all_valid:
            self.logger.info(
                "Configuration validation passed",
                extra={
                    "tiers": [tier.value for tier in TIER_MODEL_MAP.keys()],
                    "allowed_models": settings.ALLOWED_MODELS,
                },
            )
        else:
            self.logger.error(
                "Configuration validation failed - some tiers have no allowed models",
                extra={"allowed_models": settings.ALLOWED_MODELS},
            )

        return all_valid

    def _filter_by_preference(self, models: list[str]) -> list[str]:
        """Filter and sort models by provider preference.

        This private method implements provider preference logic by:
        1. Determining provider for each model using MODEL_PROVIDER_MAP
        2. Sorting models by provider preference order
        3. Returning sorted list with preferred providers first

        Args:
            models: List of model identifiers to filter and sort

        Returns:
            List of models sorted by provider preference (preferred providers first)

        Example:
            >>> selector = ModelSelector(provider_preference=["openai", "anthropic"])
            >>> models = ["claude-haiku-4-5-20251001", "gpt-5-mini", "gemini-2.5-flash-lite"]
            >>> sorted_models = selector._filter_by_preference(models)
            >>> print(sorted_models)
            ["gpt-5-mini", "claude-haiku-4-5-20251001", "gemini-2.5-flash-lite"]
        """
        if not self.provider_preference:
            return models

        # Create preference index map (lower index = higher priority)
        preference_index = {
            provider: i for i, provider in enumerate(self.provider_preference)
        }

        # Sort models by provider preference
        def get_preference_order(model: str) -> int:
            """Get preference order for a model (lower = higher priority)."""
            provider = MODEL_PROVIDER_MAP.get(model)
            if provider is None or self.provider_preference is None:
                # Unknown provider or no preference, put at end
                return len(self.provider_preference) if self.provider_preference else 999

            provider_name = provider.value
            # Return preference index if in preference list, otherwise put at end
            return preference_index.get(provider_name, len(self.provider_preference))

        sorted_models = sorted(models, key=get_preference_order)

        self.logger.debug(
            "Models filtered by provider preference",
            extra={
                "original_order": models,
                "sorted_order": sorted_models,
                "provider_preference": self.provider_preference,
            },
        )

        return sorted_models
