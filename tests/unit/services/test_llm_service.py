"""Unit tests for LLM Service provider registry.

This module tests the ProviderRegistry class that manages model-to-provider mapping
and provider instance lifecycle. Tests cover:
- Provider selection for each model
- Lazy initialization pattern
- Singleton pattern (instance reuse)
- Model validation against ALLOWED_MODELS
- API key validation
- Error handling for unknown models and missing API keys

Target: 100% code coverage
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentcore.a2a_protocol.models.llm import ModelNotAllowedError, Provider
from agentcore.a2a_protocol.services.llm_client_anthropic import LLMClientAnthropic
from agentcore.a2a_protocol.services.llm_client_gemini import LLMClientGemini
from agentcore.a2a_protocol.services.llm_client_openai import LLMClientOpenAI
from agentcore.a2a_protocol.services.llm_service import (
    MODEL_PROVIDER_MAP,
    ProviderRegistry,
)


class TestProviderRegistry:
    """Test suite for ProviderRegistry class."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    def test_initialization(self) -> None:
        """Test registry initialization with custom timeout and retries."""
        registry = ProviderRegistry(timeout=30.0, max_retries=5)
        assert registry.timeout == 30.0
        assert registry.max_retries == 5

    def test_initialization_defaults(self) -> None:
        """Test registry initialization with default values."""
        registry = ProviderRegistry()
        assert registry.timeout == 60.0
        assert registry.max_retries == 3

    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientOpenAI")
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_get_provider_for_openai_model(
        self, mock_settings: MagicMock, mock_openai_class: MagicMock
    ) -> None:
        """Test provider selection for OpenAI models."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_openai_instance = MagicMock(spec=LLMClientOpenAI)
        mock_openai_class.return_value = mock_openai_instance

        registry = ProviderRegistry(timeout=30.0, max_retries=5)

        # Execute
        client = registry.get_provider_for_model("gpt-4.1-mini")

        # Verify
        assert client is mock_openai_instance
        mock_openai_class.assert_called_once_with(
            api_key="sk-test-key", timeout=30.0, max_retries=5
        )

    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientAnthropic")
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_get_provider_for_anthropic_model(
        self, mock_settings: MagicMock, mock_anthropic_class: MagicMock
    ) -> None:
        """Test provider selection for Anthropic models."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["claude-3-5-haiku-20241022"]
        mock_settings.ANTHROPIC_API_KEY = "sk-ant-test-key"
        mock_anthropic_instance = MagicMock(spec=LLMClientAnthropic)
        mock_anthropic_class.return_value = mock_anthropic_instance

        registry = ProviderRegistry(timeout=45.0, max_retries=2)

        # Execute
        client = registry.get_provider_for_model("claude-3-5-haiku-20241022")

        # Verify
        assert client is mock_anthropic_instance
        mock_anthropic_class.assert_called_once_with(
            api_key="sk-ant-test-key", timeout=45.0, max_retries=2
        )

    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientGemini")
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_get_provider_for_gemini_model(
        self, mock_settings: MagicMock, mock_gemini_class: MagicMock
    ) -> None:
        """Test provider selection for Gemini models."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["gemini-1.5-flash"]
        mock_settings.GEMINI_API_KEY = "google-test-key"
        mock_gemini_instance = MagicMock(spec=LLMClientGemini)
        mock_gemini_class.return_value = mock_gemini_instance

        registry = ProviderRegistry(timeout=90.0, max_retries=1)

        # Execute
        client = registry.get_provider_for_model("gemini-1.5-flash")

        # Verify
        assert client is mock_gemini_instance
        mock_gemini_class.assert_called_once_with(
            api_key="google-test-key", timeout=90.0, max_retries=1
        )

    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientOpenAI")
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_lazy_initialization(
        self, mock_settings: MagicMock, mock_openai_class: MagicMock
    ) -> None:
        """Test provider is created only on first request (lazy initialization)."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_openai_instance = MagicMock(spec=LLMClientOpenAI)
        mock_openai_class.return_value = mock_openai_instance

        registry = ProviderRegistry()

        # Verify provider not created yet
        assert Provider.OPENAI not in ProviderRegistry._instances

        # Execute - first call should create provider
        client = registry.get_provider_for_model("gpt-4.1-mini")

        # Verify provider created
        assert Provider.OPENAI in ProviderRegistry._instances
        assert client is mock_openai_instance
        mock_openai_class.assert_called_once()

    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientOpenAI")
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_singleton_pattern(
        self, mock_settings: MagicMock, mock_openai_class: MagicMock
    ) -> None:
        """Test same provider instance is returned on subsequent requests (singleton)."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini", "gpt-5-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_openai_instance = MagicMock(spec=LLMClientOpenAI)
        mock_openai_class.return_value = mock_openai_instance

        registry = ProviderRegistry()

        # Execute - multiple requests for different models from same provider
        client1 = registry.get_provider_for_model("gpt-4.1-mini")
        client2 = registry.get_provider_for_model("gpt-5-mini")

        # Verify same instance returned
        assert client1 is client2
        assert client1 is mock_openai_instance
        # Provider created only once
        mock_openai_class.assert_called_once()

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_model_not_in_allowed_models(self, mock_settings: MagicMock) -> None:
        """Test error when model is not in ALLOWED_MODELS configuration."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]

        registry = ProviderRegistry()

        # Execute & Verify
        with pytest.raises(ModelNotAllowedError) as exc_info:
            registry.get_provider_for_model("gpt-5")

        assert exc_info.value.model == "gpt-5"
        assert exc_info.value.allowed == ["gpt-4.1-mini"]
        assert "gpt-5" in str(exc_info.value)

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_unknown_model(self, mock_settings: MagicMock) -> None:
        """Test error when model is not in MODEL_PROVIDER_MAP."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["unknown-model-xyz"]

        registry = ProviderRegistry()

        # Execute & Verify
        with pytest.raises(ValueError) as exc_info:
            registry.get_provider_for_model("unknown-model-xyz")

        assert "Unknown model: unknown-model-xyz" in str(exc_info.value)
        assert "Available models:" in str(exc_info.value)

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_missing_openai_api_key(self, mock_settings: MagicMock) -> None:
        """Test error when OpenAI API key is not configured."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = None

        registry = ProviderRegistry()

        # Execute & Verify
        with pytest.raises(RuntimeError) as exc_info:
            registry.get_provider_for_model("gpt-4.1-mini")

        assert "OpenAI API key not configured" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_missing_anthropic_api_key(self, mock_settings: MagicMock) -> None:
        """Test error when Anthropic API key is not configured."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["claude-3-5-haiku-20241022"]
        mock_settings.ANTHROPIC_API_KEY = None

        registry = ProviderRegistry()

        # Execute & Verify
        with pytest.raises(RuntimeError) as exc_info:
            registry.get_provider_for_model("claude-3-5-haiku-20241022")

        assert "Anthropic API key not configured" in str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_missing_google_api_key(self, mock_settings: MagicMock) -> None:
        """Test error when Google API key is not configured."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["gemini-1.5-flash"]
        mock_settings.GEMINI_API_KEY = None

        registry = ProviderRegistry()

        # Execute & Verify
        with pytest.raises(RuntimeError) as exc_info:
            registry.get_provider_for_model("gemini-1.5-flash")

        assert "Gemini API key not configured" in str(exc_info.value)
        assert "GEMINI_API_KEY" in str(exc_info.value)

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_list_available_models_all_allowed(self, mock_settings: MagicMock) -> None:
        """Test list_available_models when all models are allowed."""
        # Setup
        mock_settings.ALLOWED_MODELS = list(MODEL_PROVIDER_MAP.keys())

        registry = ProviderRegistry()

        # Execute
        models = registry.list_available_models()

        # Verify - all models should be available and sorted
        expected = sorted(MODEL_PROVIDER_MAP.keys())
        assert models == expected

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_list_available_models_subset_allowed(
        self, mock_settings: MagicMock
    ) -> None:
        """Test list_available_models when only subset of models are allowed."""
        # Setup
        mock_settings.ALLOWED_MODELS = [
            "gpt-4.1-mini",
            "claude-3-5-haiku-20241022",
            "gemini-1.5-flash",
        ]

        registry = ProviderRegistry()

        # Execute
        models = registry.list_available_models()

        # Verify - only allowed models that are in the map
        expected = sorted(
            [
                "gpt-4.1-mini",
                "claude-3-5-haiku-20241022",
                "gemini-1.5-flash",
            ]
        )
        assert models == expected

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_list_available_models_no_overlap(self, mock_settings: MagicMock) -> None:
        """Test list_available_models when ALLOWED_MODELS has no overlap with map."""
        # Setup
        mock_settings.ALLOWED_MODELS = ["unknown-model-1", "unknown-model-2"]

        registry = ProviderRegistry()

        # Execute
        models = registry.list_available_models()

        # Verify - empty list since no models are both mapped and allowed
        assert models == []

    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientOpenAI")
    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientAnthropic")
    @patch("agentcore.a2a_protocol.services.llm_service.LLMClientGemini")
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_all_providers_created(
        self,
        mock_settings: MagicMock,
        mock_gemini_class: MagicMock,
        mock_anthropic_class: MagicMock,
        mock_openai_class: MagicMock,
    ) -> None:
        """Test all three providers can be created successfully."""
        # Setup
        mock_settings.ALLOWED_MODELS = [
            "gpt-4.1-mini",
            "claude-3-5-haiku-20241022",
            "gemini-1.5-flash",
        ]
        mock_settings.OPENAI_API_KEY = "sk-test-openai"
        mock_settings.ANTHROPIC_API_KEY = "sk-ant-test-anthropic"
        mock_settings.GEMINI_API_KEY = "google-test-key"

        mock_openai_instance = MagicMock(spec=LLMClientOpenAI)
        mock_anthropic_instance = MagicMock(spec=LLMClientAnthropic)
        mock_gemini_instance = MagicMock(spec=LLMClientGemini)

        mock_openai_class.return_value = mock_openai_instance
        mock_anthropic_class.return_value = mock_anthropic_instance
        mock_gemini_class.return_value = mock_gemini_instance

        registry = ProviderRegistry()

        # Execute - request model from each provider
        openai_client = registry.get_provider_for_model("gpt-4.1-mini")
        anthropic_client = registry.get_provider_for_model("claude-3-5-haiku-20241022")
        gemini_client = registry.get_provider_for_model("gemini-1.5-flash")

        # Verify all providers created
        assert openai_client is mock_openai_instance
        assert anthropic_client is mock_anthropic_instance
        assert gemini_client is mock_gemini_instance

        # Verify all provider classes called
        mock_openai_class.assert_called_once()
        mock_anthropic_class.assert_called_once()
        mock_gemini_class.assert_called_once()

        # Verify all providers in instance cache
        assert len(ProviderRegistry._instances) == 3
        assert Provider.OPENAI in ProviderRegistry._instances
        assert Provider.ANTHROPIC in ProviderRegistry._instances
        assert Provider.GEMINI in ProviderRegistry._instances


class TestModelProviderMap:
    """Test suite for MODEL_PROVIDER_MAP configuration."""

    def test_all_openai_models_mapped(self) -> None:
        """Test all OpenAI models are mapped to OPENAI provider."""
        openai_models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini"]
        for model in openai_models:
            assert MODEL_PROVIDER_MAP[model] == Provider.OPENAI

    def test_all_anthropic_models_mapped(self) -> None:
        """Test all Anthropic models are mapped to ANTHROPIC provider."""
        anthropic_models = [
            "claude-3-5-sonnet",
            "claude-3-5-haiku-20241022",
            "claude-3-opus",
        ]
        for model in anthropic_models:
            assert MODEL_PROVIDER_MAP[model] == Provider.ANTHROPIC

    def test_all_gemini_models_mapped(self) -> None:
        """Test all Gemini models are mapped to GEMINI provider."""
        gemini_models = [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
        for model in gemini_models:
            assert MODEL_PROVIDER_MAP[model] == Provider.GEMINI

    def test_total_model_count(self) -> None:
        """Test total number of models in mapping."""
        # 4 OpenAI + 3 Anthropic + 3 Gemini = 10 total
        assert len(MODEL_PROVIDER_MAP) == 10


class TestLLMService:
    """Test suite for LLMService facade class."""

    def setup_method(self) -> None:
        """Clear singleton instances before each test."""
        ProviderRegistry._instances = {}

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_initialization_defaults(self, mock_settings: MagicMock) -> None:
        """Test LLMService initialization with default values."""
        from agentcore.a2a_protocol.services.llm_service import LLMService

        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        service = LLMService()

        assert service.timeout == 60.0
        assert service.max_retries == 3
        assert service.registry is not None

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_initialization_custom_values(self, mock_settings: MagicMock) -> None:
        """Test LLMService initialization with custom timeout and retries."""
        from agentcore.a2a_protocol.services.llm_service import LLMService

        service = LLMService(timeout=30.0, max_retries=5)

        assert service.timeout == 30.0
        assert service.max_retries == 5

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    async def test_complete_model_governance_rejection(
        self, mock_settings: MagicMock
    ) -> None:
        """Test that complete() rejects models not in ALLOWED_MODELS."""
        from agentcore.a2a_protocol.models.llm import LLMRequest
        from agentcore.a2a_protocol.services.llm_service import LLMService

        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        service = LLMService()
        request = LLMRequest(
            model="gpt-5",  # Not in ALLOWED_MODELS
            messages=[{"role": "user", "content": "test"}],
            trace_id="trace-123",
        )

        with pytest.raises(ModelNotAllowedError) as exc_info:
            await service.complete(request)

        assert exc_info.value.model == "gpt-5"
        assert exc_info.value.allowed == ["gpt-4.1-mini"]

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    async def test_complete_success(self, mock_settings: MagicMock) -> None:
        """Test successful completion with provider."""
        from agentcore.a2a_protocol.models.llm import (
            LLMRequest,
            LLMResponse,
            LLMUsage,
        )
        from agentcore.a2a_protocol.services.llm_service import LLMService

        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        mock_response = LLMResponse(
            content="Test response",
            usage=LLMUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            latency_ms=100,
            provider="openai",
            model="gpt-4.1-mini",
            trace_id="trace-123",
        )

        service = LLMService()

        # Mock the provider's complete method
        with patch.object(
            service.registry.get_provider_for_model("gpt-4.1-mini"),
            "complete",
            return_value=mock_response,
        ) as mock_complete:
            request = LLMRequest(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "Hello"}],
                trace_id="trace-123",
                source_agent="agent-001",
            )

            response = await service.complete(request)

            # Verify
            assert response == mock_response
            mock_complete.assert_called_once_with(request)

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    async def test_complete_provider_error(self, mock_settings: MagicMock) -> None:
        """Test error handling when provider raises ProviderError."""
        from agentcore.a2a_protocol.models.llm import LLMRequest, ProviderError
        from agentcore.a2a_protocol.services.llm_service import LLMService

        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        service = LLMService()
        provider = service.registry.get_provider_for_model("gpt-4.1-mini")

        # Mock provider to raise error
        original_error = Exception("API Error")
        with patch.object(
            provider, "complete", side_effect=ProviderError("openai", original_error)
        ):
            request = LLMRequest(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "test"}],
            )

            with pytest.raises(ProviderError) as exc_info:
                await service.complete(request)

            assert exc_info.value.provider == "openai"
            assert exc_info.value.original_error == original_error

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    async def test_complete_provider_timeout(self, mock_settings: MagicMock) -> None:
        """Test error handling when provider raises ProviderTimeoutError."""
        from agentcore.a2a_protocol.models.llm import (
            LLMRequest,
            ProviderTimeoutError,
        )
        from agentcore.a2a_protocol.services.llm_service import LLMService

        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        service = LLMService()
        provider = service.registry.get_provider_for_model("gpt-4.1-mini")

        # Mock provider to raise timeout
        with patch.object(
            provider,
            "complete",
            side_effect=ProviderTimeoutError("openai", 60.0),
        ):
            request = LLMRequest(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "test"}],
            )

            with pytest.raises(ProviderTimeoutError) as exc_info:
                await service.complete(request)

            assert exc_info.value.provider == "openai"
            assert exc_info.value.timeout_seconds == 60.0

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    async def test_stream_model_governance_rejection(
        self, mock_settings: MagicMock
    ) -> None:
        """Test that stream() rejects models not in ALLOWED_MODELS."""
        from agentcore.a2a_protocol.models.llm import LLMRequest
        from agentcore.a2a_protocol.services.llm_service import LLMService

        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        service = LLMService()
        request = LLMRequest(
            model="claude-3-opus",  # Not in ALLOWED_MODELS
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )

        with pytest.raises(ModelNotAllowedError) as exc_info:
            async for _ in service.stream(request):
                pass

        assert exc_info.value.model == "claude-3-opus"

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    async def test_stream_success(self, mock_settings: MagicMock) -> None:
        """Test successful streaming with provider."""
        from agentcore.a2a_protocol.models.llm import LLMRequest
        from agentcore.a2a_protocol.services.llm_service import LLMService

        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        service = LLMService()
        provider = service.registry.get_provider_for_model("gpt-4.1-mini")

        # Create async generator for mock
        async def mock_stream_generator(request: LLMRequest) -> object:
            for token in ["Hello", " ", "World"]:
                yield token

        # Mock the provider's stream method
        with patch.object(provider, "stream", side_effect=mock_stream_generator):
            request = LLMRequest(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
                trace_id="trace-xyz",
            )

            tokens = []
            async for token in service.stream(request):
                tokens.append(token)

            # Verify
            assert tokens == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    async def test_stream_provider_error(self, mock_settings: MagicMock) -> None:
        """Test error handling when provider stream raises ProviderError."""
        from agentcore.a2a_protocol.models.llm import LLMRequest, ProviderError
        from agentcore.a2a_protocol.services.llm_service import LLMService

        # Setup
        mock_settings.ALLOWED_MODELS = ["gpt-4.1-mini"]
        mock_settings.OPENAI_API_KEY = "sk-test-key"
        mock_settings.LLM_REQUEST_TIMEOUT = 60.0
        mock_settings.LLM_MAX_RETRIES = 3

        service = LLMService()
        provider = service.registry.get_provider_for_model("gpt-4.1-mini")

        # Create async generator that raises error
        async def mock_stream_with_error(request: LLMRequest) -> object:
            yield "Hello"
            raise ProviderError("openai", Exception("Stream error"))

        # Mock the provider's stream method
        with patch.object(provider, "stream", side_effect=mock_stream_with_error):
            request = LLMRequest(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
            )

            with pytest.raises(ProviderError) as exc_info:
                async for _ in service.stream(request):
                    pass

            assert exc_info.value.provider == "openai"

    @patch("agentcore.a2a_protocol.services.llm_service.settings")
    def test_global_singleton_instance(self, mock_settings: MagicMock) -> None:
        """Test that global llm_service singleton exists."""
        from agentcore.a2a_protocol.services.llm_service import (
            LLMService,
            llm_service,
        )

        # Verify global instance exists and is correct type
        assert llm_service is not None
        assert isinstance(llm_service, LLMService)
