"""Tests for Portkey configuration management."""

from __future__ import annotations

import os

import pytest

from agentcore.integration.portkey.config import PortkeyConfig
from agentcore.integration.portkey.exceptions import PortkeyConfigurationError


class TestPortkeyConfig:
    """Test suite for PortkeyConfig."""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that default values are set correctly."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")

        config = PortkeyConfig()

        assert config.api_key == "test-api-key"
        assert config.base_url == "https://api.portkey.ai/v1"
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.enable_logging is True
        assert config.enable_tracing is True

    def test_env_var_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("PORTKEY_API_KEY", "custom-api-key")
        monkeypatch.setenv("PORTKEY_BASE_URL", "https://custom.portkey.ai")
        monkeypatch.setenv("PORTKEY_TIMEOUT", "120.0")
        monkeypatch.setenv("PORTKEY_MAX_RETRIES", "5")
        monkeypatch.setenv("PORTKEY_DEFAULT_PROVIDER", "openai")

        config = PortkeyConfig()

        assert config.api_key == "custom-api-key"
        assert config.base_url == "https://custom.portkey.ai"
        assert config.timeout == 120.0
        assert config.max_retries == 5
        assert config.default_provider == "openai"

    def test_validate_required_fields_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test validation passes when required fields are present."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")

        config = PortkeyConfig()
        config.validate_required_fields()  # Should not raise

    def test_validate_required_fields_missing_api_key(self) -> None:
        """Test validation fails when API key is missing."""
        config = PortkeyConfig(api_key=None)

        with pytest.raises(
            PortkeyConfigurationError,
            match="PORTKEY_API_KEY environment variable is required",
        ):
            config.validate_required_fields()

    def test_get_provider_config_with_default_virtual_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting provider config with default virtual key."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")
        monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "default-virtual-key")

        config = PortkeyConfig()
        provider_config = config.get_provider_config("openai")

        assert provider_config["provider"] == "openai"
        assert provider_config["virtual_key"] == "default-virtual-key"

    def test_get_provider_config_with_provider_specific_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting provider config with provider-specific virtual key."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")
        monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "default-virtual-key")
        monkeypatch.setenv("PORTKEY_OPENAI_VIRTUAL_KEY", "openai-virtual-key")

        config = PortkeyConfig()
        provider_config = config.get_provider_config("openai")

        assert provider_config["provider"] == "openai"
        assert provider_config["virtual_key"] == "openai-virtual-key"

    def test_merge_with_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test merging override parameters with defaults."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")

        config = PortkeyConfig(
            custom_headers={"X-Custom": "header1"},
            metadata={"key1": "value1"},
        )

        overrides = {
            "timeout": 90.0,
            "custom_headers": {"X-Override": "header2"},
            "metadata": {"key2": "value2"},
        }

        merged = config.merge_with_defaults(overrides)

        assert merged["api_key"] == "test-api-key"
        assert merged["timeout"] == 90.0
        assert merged["custom_headers"] == {
            "X-Custom": "header1",
            "X-Override": "header2",
        }
        assert merged["metadata"] == {"key1": "value1", "key2": "value2"}

    def test_from_env_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating config from environment variables."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")

        config = PortkeyConfig.from_env()

        assert config.api_key == "test-api-key"

    def test_from_env_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test from_env fails when API key is missing."""
        # Clear any existing PORTKEY_API_KEY
        monkeypatch.delenv("PORTKEY_API_KEY", raising=False)

        with pytest.raises(
            PortkeyConfigurationError,
            match="PORTKEY_API_KEY environment variable is required",
        ):
            PortkeyConfig.from_env()

    def test_custom_headers_and_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test setting custom headers and metadata."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")

        config = PortkeyConfig(
            custom_headers={"X-Custom-Header": "value"},
            metadata={"environment": "test"},
        )

        assert config.custom_headers == {"X-Custom-Header": "value"}
        assert config.metadata == {"environment": "test"}

    def test_cost_and_caching_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test cost optimization and caching settings."""
        monkeypatch.setenv("PORTKEY_API_KEY", "test-api-key")
        monkeypatch.setenv("PORTKEY_ENABLE_CACHING", "false")
        monkeypatch.setenv("PORTKEY_CACHE_TTL", "7200")
        monkeypatch.setenv("PORTKEY_MAX_COST_PER_REQUEST", "0.5")

        config = PortkeyConfig()

        assert config.enable_caching is False
        assert config.cache_ttl == 7200
        assert config.max_cost_per_request == 0.5
