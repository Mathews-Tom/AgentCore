"""LLM Service provider registry for multi-provider LLM operations.

This module implements the ProviderRegistry that manages model-to-provider mapping
and provider instance lifecycle. It unifies OpenAI, Anthropic, and Gemini clients
under a single selection interface.

The registry implements three key patterns:
- Registry Pattern: Central mapping of models to providers
- Singleton Pattern: Single instance per provider type
- Lazy Initialization: Providers created only when first requested

Features:
- Model-to-provider mapping for all supported models
- Provider instance management (singleton per provider)
- Lazy initialization (providers created on first request)
- Configuration-driven API key loading
- Model validation against ALLOWED_MODELS
- Missing API key detection and error handling

Example:
    ```python
    from agentcore.a2a_protocol.services.llm_service import ProviderRegistry
    from agentcore.a2a_protocol.models.llm import LLMRequest

    registry = ProviderRegistry(timeout=60.0, max_retries=3)

    # Get provider for a specific model
    client = registry.get_provider_for_model("gpt-4.1-mini")
    request = LLMRequest(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    response = await client.complete(request)

    # List all available models
    models = registry.list_available_models()
    print(models)  # ["gpt-4.1-mini", "claude-3-5-haiku-20241022", ...]
    ```

Error Handling:
    - ValueError: Raised when model is unknown or not in ALLOWED_MODELS
    - RuntimeError: Raised when API key is not configured for a provider
"""

from __future__ import annotations

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.models.llm import ModelNotAllowedError, Provider
from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
from agentcore.a2a_protocol.services.llm_client_base import LLMClient
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI

# Model-to-provider mapping
# This mapping defines which provider handles each model
MODEL_PROVIDER_MAP: dict[str, Provider] = {
    # OpenAI models
    "gpt-4.1": Provider.OPENAI,
    "gpt-4.1-mini": Provider.OPENAI,
    "gpt-5": Provider.OPENAI,
    "gpt-5-mini": Provider.OPENAI,
    # Anthropic models
    "claude-3-5-sonnet": Provider.ANTHROPIC,
    "claude-3-5-haiku-20241022": Provider.ANTHROPIC,
    "claude-3-opus": Provider.ANTHROPIC,
    # Gemini models
    "gemini-2.0-flash-exp": Provider.GEMINI,
    "gemini-1.5-pro": Provider.GEMINI,
    "gemini-1.5-flash": Provider.GEMINI,
}


class ProviderRegistry:
    """Provider registry managing model-to-provider mapping and provider instances.

    This class implements the central registry for all LLM providers. It manages
    the lifecycle of provider instances using singleton pattern and lazy initialization.

    The registry ensures that:
    - Each provider is instantiated at most once (singleton)
    - Providers are created only when needed (lazy initialization)
    - Models are validated against ALLOWED_MODELS configuration
    - API keys are validated before provider creation
    - Provider selection is deterministic and configuration-driven

    Attributes:
        _instances: Singleton cache of provider instances (class variable)
        timeout: Request timeout in seconds (default 60.0)
        max_retries: Maximum retry attempts on transient errors (default 3)
    """

    # Class variable for singleton instances
    _instances: dict[Provider, LLMClient] = {}

    def __init__(self, timeout: float = 60.0, max_retries: int = 3) -> None:
        """Initialize provider registry with timeout and retry configuration.

        Args:
            timeout: Request timeout in seconds (default 60.0)
            max_retries: Maximum number of retry attempts (default 3)
        """
        self.timeout = timeout
        self.max_retries = max_retries

    def get_provider_for_model(self, model: str) -> LLMClient:
        """Get provider client for the specified model.

        This method implements the core provider selection logic. It:
        1. Validates model is in ALLOWED_MODELS configuration
        2. Looks up provider in MODEL_PROVIDER_MAP
        3. Creates provider instance if not already cached (lazy initialization)
        4. Returns cached instance if already created (singleton)

        Args:
            model: Model identifier (e.g., "gpt-4.1-mini", "claude-3-5-haiku-20241022")

        Returns:
            LLMClient instance for the provider that handles this model

        Raises:
            ModelNotAllowedError: When model is not in ALLOWED_MODELS configuration
            ValueError: When model is unknown (not in MODEL_PROVIDER_MAP)
            RuntimeError: When API key is not configured for the provider
        """
        # Validate model is in ALLOWED_MODELS
        if model not in settings.ALLOWED_MODELS:
            raise ModelNotAllowedError(model, settings.ALLOWED_MODELS)

        # Look up provider in mapping
        provider = MODEL_PROVIDER_MAP.get(model)
        if provider is None:
            raise ValueError(
                f"Unknown model: {model}. Available models: {list(MODEL_PROVIDER_MAP.keys())}"
            )

        # Create provider instance if not already cached (lazy initialization)
        if provider not in self._instances:
            self._instances[provider] = self._create_provider(provider)

        return self._instances[provider]

    def list_available_models(self) -> list[str]:
        """List all available models based on ALLOWED_MODELS configuration.

        Returns intersection of MODEL_PROVIDER_MAP keys and ALLOWED_MODELS.
        This ensures only models that are both supported and allowed are listed.

        Returns:
            List of available model identifiers sorted alphabetically
        """
        # Get intersection of mapped models and allowed models
        available = set(MODEL_PROVIDER_MAP.keys()) & set(settings.ALLOWED_MODELS)
        return sorted(available)

    def _create_provider(self, provider: Provider) -> LLMClient:
        """Create provider client instance with appropriate API key.

        This private method handles provider instantiation with validation:
        1. Retrieves API key from settings based on provider type
        2. Validates API key is configured (not None)
        3. Creates provider instance with timeout and retry settings

        Args:
            provider: Provider enum value (OPENAI, ANTHROPIC, or GEMINI)

        Returns:
            LLMClient instance for the specified provider

        Raises:
            RuntimeError: When API key is not configured for the provider
        """
        # Map provider to API key and client class
        if provider == Provider.OPENAI:
            api_key = settings.OPENAI_API_KEY
            if api_key is None:
                raise RuntimeError(
                    "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
                )
            return LLMClientOpenAI(
                api_key=api_key, timeout=self.timeout, max_retries=self.max_retries
            )

        if provider == Provider.ANTHROPIC:
            api_key = settings.ANTHROPIC_API_KEY
            if api_key is None:
                raise RuntimeError(
                    "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
                )
            return LLMClientAnthropic(
                api_key=api_key, timeout=self.timeout, max_retries=self.max_retries
            )

        if provider == Provider.GEMINI:
            api_key = settings.GOOGLE_API_KEY
            if api_key is None:
                raise RuntimeError(
                    "Google API key not configured. Set GOOGLE_API_KEY environment variable."
                )
            return LLMClientGemini(
                api_key=api_key, timeout=self.timeout, max_retries=self.max_retries
            )

        # This should never happen if Provider enum is complete
        raise ValueError(f"Unknown provider: {provider}")
