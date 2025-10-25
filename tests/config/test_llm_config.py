"""Tests for LLM service configuration management."""

from __future__ import annotations

import pytest

from agentcore.a2a_protocol.config import Settings


class TestLLMConfig:
    """Test suite for LLM service configuration settings."""

    def test_default_llm_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default LLM configuration values are set correctly."""
        # Clear API keys from environment to test defaults
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Create settings without loading from .env file
        settings = Settings(_env_file=None)

        assert settings.ALLOWED_MODELS == [
            "gpt-4.1-mini",
            "gpt-5-mini",
            "claude-3-5-haiku-20241022",
            "gemini-1.5-flash",
        ]
        assert settings.LLM_DEFAULT_MODEL == "gpt-4.1-mini"
        assert settings.OPENAI_API_KEY is None
        assert settings.ANTHROPIC_API_KEY is None
        assert settings.GOOGLE_API_KEY is None
        assert settings.LLM_REQUEST_TIMEOUT == 60.0
        assert settings.LLM_MAX_RETRIES == 3
        assert settings.LLM_RETRY_EXPONENTIAL_BASE == 2.0

    def test_llm_env_var_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading LLM configuration from environment variables."""
        monkeypatch.setenv("LLM_DEFAULT_MODEL", "gpt-5-mini")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-456")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-key-789")
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "120.0")
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        monkeypatch.setenv("LLM_RETRY_EXPONENTIAL_BASE", "3.0")

        settings = Settings()

        assert settings.LLM_DEFAULT_MODEL == "gpt-5-mini"
        assert settings.OPENAI_API_KEY == "sk-test-key-123"
        assert settings.ANTHROPIC_API_KEY == "sk-ant-test-key-456"
        assert settings.GOOGLE_API_KEY == "AIza-test-key-789"
        assert settings.LLM_REQUEST_TIMEOUT == 120.0
        assert settings.LLM_MAX_RETRIES == 5
        assert settings.LLM_RETRY_EXPONENTIAL_BASE == 3.0

    def test_llm_request_timeout_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test LLM_REQUEST_TIMEOUT must be greater than 0."""
        # Test valid positive value
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "30.0")
        settings = Settings()
        assert settings.LLM_REQUEST_TIMEOUT == 30.0

        # Test zero value (should fail)
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "0.0")
        with pytest.raises(ValueError, match="greater than 0"):
            Settings()

        # Test negative value (should fail)
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "-10.0")
        with pytest.raises(ValueError, match="greater than 0"):
            Settings()

    def test_llm_max_retries_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test LLM_MAX_RETRIES must be greater than or equal to 0."""
        # Test zero value (should pass)
        monkeypatch.setenv("LLM_MAX_RETRIES", "0")
        settings = Settings()
        assert settings.LLM_MAX_RETRIES == 0

        # Test positive value
        monkeypatch.setenv("LLM_MAX_RETRIES", "10")
        settings = Settings()
        assert settings.LLM_MAX_RETRIES == 10

        # Test negative value (should fail)
        monkeypatch.setenv("LLM_MAX_RETRIES", "-1")
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            Settings()

    def test_llm_retry_exponential_base_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test LLM_RETRY_EXPONENTIAL_BASE must be greater than 1."""
        # Test valid value (>1)
        monkeypatch.setenv("LLM_RETRY_EXPONENTIAL_BASE", "2.5")
        settings = Settings()
        assert settings.LLM_RETRY_EXPONENTIAL_BASE == 2.5

        # Test value equal to 1 (should fail)
        monkeypatch.setenv("LLM_RETRY_EXPONENTIAL_BASE", "1.0")
        with pytest.raises(ValueError, match="greater than 1"):
            Settings()

        # Test value less than 1 (should fail)
        monkeypatch.setenv("LLM_RETRY_EXPONENTIAL_BASE", "0.5")
        with pytest.raises(ValueError, match="greater than 1"):
            Settings()

    def test_api_keys_optional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that API keys are optional and can be None."""
        # Clear any existing API keys from environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        # Create settings without loading from .env file
        settings = Settings(_env_file=None)

        assert settings.OPENAI_API_KEY is None
        assert settings.ANTHROPIC_API_KEY is None
        assert settings.GOOGLE_API_KEY is None

    def test_allowed_models_contains_valid_models(self) -> None:
        """Test ALLOWED_MODELS contains expected model identifiers."""
        settings = Settings()

        # Verify all expected models are present
        expected_models = [
            "gpt-4.1-mini",
            "gpt-5-mini",
            "claude-3-5-haiku-20241022",
            "gemini-1.5-flash",
        ]

        for model in expected_models:
            assert (
                model in settings.ALLOWED_MODELS
            ), f"Expected model {model} not in ALLOWED_MODELS"

    def test_llm_config_independence(self) -> None:
        """Test that LLM config is independent of other settings."""
        settings = Settings()

        # Verify LLM settings exist alongside other settings
        assert hasattr(settings, "DEBUG")
        assert hasattr(settings, "DATABASE_URL")
        assert hasattr(settings, "LLM_DEFAULT_MODEL")
        assert hasattr(settings, "ALLOWED_MODELS")

        # Verify other settings don't interfere
        assert settings.DEBUG in (True, False)
        assert isinstance(settings.LLM_DEFAULT_MODEL, str)

    def test_llm_config_completeness(self) -> None:
        """Test all required LLM configuration fields are present."""
        settings = Settings()

        required_fields = [
            "ALLOWED_MODELS",
            "LLM_DEFAULT_MODEL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "LLM_REQUEST_TIMEOUT",
            "LLM_MAX_RETRIES",
            "LLM_RETRY_EXPONENTIAL_BASE",
        ]

        for field in required_fields:
            assert hasattr(
                settings, field
            ), f"Missing required LLM config field: {field}"

    def test_llm_default_model_in_allowed_models(self) -> None:
        """Test that LLM_DEFAULT_MODEL is in ALLOWED_MODELS list."""
        settings = Settings()

        assert (
            settings.LLM_DEFAULT_MODEL in settings.ALLOWED_MODELS
        ), f"Default model {settings.LLM_DEFAULT_MODEL} not in ALLOWED_MODELS"

    def test_allowed_models_list_type(self) -> None:
        """Test that ALLOWED_MODELS is a list of strings."""
        settings = Settings()

        assert isinstance(settings.ALLOWED_MODELS, list)
        assert all(isinstance(model, str) for model in settings.ALLOWED_MODELS)
        assert len(settings.ALLOWED_MODELS) > 0

    def test_llm_timeout_accepts_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that LLM_REQUEST_TIMEOUT accepts float values."""
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "45.5")
        settings = Settings()
        assert settings.LLM_REQUEST_TIMEOUT == 45.5
        assert isinstance(settings.LLM_REQUEST_TIMEOUT, float)

    def test_llm_exponential_base_accepts_float(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that LLM_RETRY_EXPONENTIAL_BASE accepts float values."""
        monkeypatch.setenv("LLM_RETRY_EXPONENTIAL_BASE", "1.5")
        settings = Settings()
        assert settings.LLM_RETRY_EXPONENTIAL_BASE == 1.5
        assert isinstance(settings.LLM_RETRY_EXPONENTIAL_BASE, float)
