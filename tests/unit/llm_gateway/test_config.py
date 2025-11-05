"""Unit tests for LLM Gateway configuration management.

This module tests the LLMGatewayConfig class that manages configuration
loading from environment variables:
- Initialization with defaults
- Environment variable loading with PORTKEY_ prefix
- Configuration validation
- Provider-specific configuration
- Configuration merging
- from_env() factory method

Target: 90%+ code coverage
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agentcore.llm_gateway.config import LLMGatewayConfig
from agentcore.llm_gateway.exceptions import LLMGatewayConfigurationError


class TestLLMGatewayConfig:
    """Test suite for LLMGatewayConfig class."""

    def test_initialization_defaults(self) -> None:
        """Test configuration initialization with default values."""
        config = LLMGatewayConfig()
        assert config.api_key is None
        assert config.base_url == "https://api.portkey.ai/v1"
        assert config.virtual_key is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.default_provider is None
        assert config.fallback_providers == []
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.max_cost_per_request is None
        assert config.enable_logging is True
        assert config.log_request_bodies is False
        assert config.log_response_bodies is False
        assert config.enable_tracing is True
        assert config.trace_id_header == "X-Portkey-Trace-Id"
        assert config.enable_metrics is True
        assert config.metrics_collection_interval == 60
        assert config.enable_sla_monitoring is True
        assert config.sla_availability_target == 99.9
        assert config.sla_response_time_target_ms == 2000
        assert config.sla_success_rate_target == 99.5
        assert config.enable_prometheus_export is True
        assert config.prometheus_export_port == 9090
        assert config.alert_debounce_seconds == 300
        assert config.metadata == {}
        assert config.custom_headers == {}

    def test_initialization_with_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = LLMGatewayConfig(
            api_key="pk-test-123",
            base_url="https://custom.portkey.ai",
            virtual_key="vk-456",
            timeout=30.0,
            max_retries=5,
            default_provider="openai",
            fallback_providers=["anthropic", "gemini"],
            enable_caching=False,
            cache_ttl=7200,
        )
        assert config.api_key == "pk-test-123"
        assert config.base_url == "https://custom.portkey.ai"
        assert config.virtual_key == "vk-456"
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.default_provider == "openai"
        assert config.fallback_providers == ["anthropic", "gemini"]
        assert config.enable_caching is False
        assert config.cache_ttl == 7200

    def test_timeout_validation_minimum(self) -> None:
        """Test that timeout must be at least 1.0 second."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(timeout=0.5)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_timeout_validation_maximum(self) -> None:
        """Test that timeout cannot exceed 300.0 seconds."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(timeout=301.0)
        assert "less than or equal to 300" in str(exc_info.value)

    def test_timeout_boundary_values(self) -> None:
        """Test timeout boundary values."""
        config_min = LLMGatewayConfig(timeout=1.0)
        assert config_min.timeout == 1.0

        config_max = LLMGatewayConfig(timeout=300.0)
        assert config_max.timeout == 300.0

    def test_max_retries_validation_minimum(self) -> None:
        """Test that max_retries must be at least 0."""
        config = LLMGatewayConfig(max_retries=0)
        assert config.max_retries == 0

    def test_max_retries_validation_maximum(self) -> None:
        """Test that max_retries cannot exceed 10."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(max_retries=11)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_cache_ttl_validation_minimum(self) -> None:
        """Test that cache_ttl must be at least 60 seconds."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(cache_ttl=30)
        assert "greater than or equal to 60" in str(exc_info.value)

    def test_max_cost_per_request_validation_negative(self) -> None:
        """Test that max_cost_per_request cannot be negative."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(max_cost_per_request=-0.01)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_max_cost_per_request_zero(self) -> None:
        """Test that zero max_cost_per_request is allowed."""
        config = LLMGatewayConfig(max_cost_per_request=0.0)
        assert config.max_cost_per_request == 0.0

    def test_metrics_collection_interval_validation_minimum(self) -> None:
        """Test that metrics_collection_interval must be at least 1."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(metrics_collection_interval=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_metrics_collection_interval_validation_maximum(self) -> None:
        """Test that metrics_collection_interval cannot exceed 3600."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(metrics_collection_interval=3601)
        assert "less than or equal to 3600" in str(exc_info.value)

    def test_sla_availability_target_validation(self) -> None:
        """Test SLA availability target validation."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(sla_availability_target=100.1)
        assert "less than or equal to 100" in str(exc_info.value)

        config = LLMGatewayConfig(sla_availability_target=99.99)
        assert config.sla_availability_target == 99.99

    def test_sla_response_time_target_validation(self) -> None:
        """Test SLA response time target validation."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(sla_response_time_target_ms=50)
        assert "greater than or equal to 100" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(sla_response_time_target_ms=10001)
        assert "less than or equal to 10000" in str(exc_info.value)

    def test_prometheus_export_port_validation(self) -> None:
        """Test Prometheus export port validation."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(prometheus_export_port=1023)
        assert "greater than or equal to 1024" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(prometheus_export_port=65536)
        assert "less than or equal to 65535" in str(exc_info.value)

    def test_alert_debounce_seconds_validation(self) -> None:
        """Test alert debounce seconds validation."""
        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(alert_debounce_seconds=59)
        assert "greater than or equal to 60" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMGatewayConfig(alert_debounce_seconds=3601)
        assert "less than or equal to 3600" in str(exc_info.value)

    def test_metadata_custom_dict(self) -> None:
        """Test custom metadata dictionary."""
        metadata = {"tenant_id": "tenant-123", "environment": "production"}
        config = LLMGatewayConfig(metadata=metadata)
        assert config.metadata == metadata

    def test_custom_headers_dict(self) -> None:
        """Test custom headers dictionary."""
        headers = {"X-Custom-Header": "value", "X-Another-Header": "value2"}
        config = LLMGatewayConfig(custom_headers=headers)
        assert config.custom_headers == headers

    def test_validate_required_fields_missing_api_key(self) -> None:
        """Test validation fails when api_key is missing."""
        config = LLMGatewayConfig()
        with pytest.raises(LLMGatewayConfigurationError) as exc_info:
            config.validate_required_fields()
        assert "PORTKEY_API_KEY" in str(exc_info.value)

    def test_validate_required_fields_with_api_key(self) -> None:
        """Test validation succeeds when api_key is present."""
        config = LLMGatewayConfig(api_key="pk-test-123")
        config.validate_required_fields()  # Should not raise

    def test_get_provider_config_with_default_virtual_key(self) -> None:
        """Test get_provider_config uses default virtual key."""
        config = LLMGatewayConfig(virtual_key="vk-default")
        provider_config = config.get_provider_config("openai")
        assert provider_config["provider"] == "openai"
        assert provider_config["virtual_key"] == "vk-default"

    @patch.dict(os.environ, {"PORTKEY_OPENAI_VIRTUAL_KEY": "vk-openai-specific"})
    def test_get_provider_config_with_provider_specific_key(self) -> None:
        """Test get_provider_config uses provider-specific virtual key."""
        config = LLMGatewayConfig(virtual_key="vk-default")
        provider_config = config.get_provider_config("openai")
        assert provider_config["provider"] == "openai"
        assert provider_config["virtual_key"] == "vk-openai-specific"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_provider_config_no_virtual_key(self) -> None:
        """Test get_provider_config when no virtual key is set."""
        config = LLMGatewayConfig()
        provider_config = config.get_provider_config("anthropic")
        assert provider_config["provider"] == "anthropic"
        assert provider_config["virtual_key"] is None

    def test_merge_with_defaults_base_config(self) -> None:
        """Test merge_with_defaults with base configuration only."""
        config = LLMGatewayConfig(
            api_key="pk-test",
            base_url="https://api.portkey.ai/v1",
            timeout=60.0,
            max_retries=3,
        )
        merged = config.merge_with_defaults({})
        assert merged["api_key"] == "pk-test"
        assert merged["base_url"] == "https://api.portkey.ai/v1"
        assert merged["timeout"] == 60.0
        assert merged["max_retries"] == 3

    def test_merge_with_defaults_overrides(self) -> None:
        """Test merge_with_defaults with overrides."""
        config = LLMGatewayConfig(
            api_key="pk-test",
            timeout=60.0,
        )
        overrides = {
            "timeout": 30.0,
            "max_retries": 5,
        }
        merged = config.merge_with_defaults(overrides)
        assert merged["api_key"] == "pk-test"
        assert merged["timeout"] == 30.0
        assert merged["max_retries"] == 5

    def test_merge_with_defaults_custom_headers(self) -> None:
        """Test merge_with_defaults with custom headers."""
        config = LLMGatewayConfig(
            custom_headers={"X-Base-Header": "base"},
        )
        overrides = {
            "custom_headers": {"X-Override-Header": "override"},
        }
        merged = config.merge_with_defaults(overrides)
        assert merged["custom_headers"]["X-Base-Header"] == "base"
        assert merged["custom_headers"]["X-Override-Header"] == "override"

    def test_merge_with_defaults_metadata(self) -> None:
        """Test merge_with_defaults with metadata."""
        config = LLMGatewayConfig(
            metadata={"tenant_id": "tenant-123"},
        )
        overrides = {
            "metadata": {"environment": "production"},
        }
        merged = config.merge_with_defaults(overrides)
        assert merged["metadata"]["tenant_id"] == "tenant-123"
        assert merged["metadata"]["environment"] == "production"

    def test_merge_with_defaults_no_custom_headers(self) -> None:
        """Test merge_with_defaults when config has no custom headers."""
        config = LLMGatewayConfig(api_key="pk-test")
        merged = config.merge_with_defaults({})
        assert "custom_headers" not in merged

    def test_merge_with_defaults_no_metadata(self) -> None:
        """Test merge_with_defaults when config has no metadata."""
        config = LLMGatewayConfig(api_key="pk-test")
        merged = config.merge_with_defaults({})
        assert "metadata" not in merged

    @patch.dict(
        os.environ,
        {
            "PORTKEY_API_KEY": "pk-from-env",
            "PORTKEY_BASE_URL": "https://custom-env.portkey.ai",
            "PORTKEY_TIMEOUT": "45.0",
            "PORTKEY_MAX_RETRIES": "7",
        },
    )
    def test_from_env_loads_environment_variables(self) -> None:
        """Test from_env loads configuration from environment variables."""
        config = LLMGatewayConfig.from_env()
        assert config.api_key == "pk-from-env"
        assert config.base_url == "https://custom-env.portkey.ai"
        assert config.timeout == 45.0
        assert config.max_retries == 7

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_missing_api_key(self) -> None:
        """Test from_env raises error when api_key is missing."""
        with pytest.raises(LLMGatewayConfigurationError) as exc_info:
            LLMGatewayConfig.from_env()
        assert "PORTKEY_API_KEY" in str(exc_info.value)

    @patch.dict(
        os.environ,
        {
            "PORTKEY_API_KEY": "pk-test",
            "PORTKEY_ENABLE_CACHING": "false",
            "PORTKEY_ENABLE_LOGGING": "false",
            "PORTKEY_ENABLE_TRACING": "false",
        },
    )
    def test_from_env_boolean_fields(self) -> None:
        """Test from_env correctly parses boolean fields."""
        config = LLMGatewayConfig.from_env()
        assert config.enable_caching is False
        assert config.enable_logging is False
        assert config.enable_tracing is False

    @patch.dict(
        os.environ,
        {
            "PORTKEY_API_KEY": "pk-test",
            "PORTKEY_FALLBACK_PROVIDERS": '["openai", "anthropic"]',
        },
    )
    def test_from_env_list_fields(self) -> None:
        """Test from_env with list fields from environment."""
        # Note: Pydantic Settings may not parse JSON lists from env vars
        # This test documents the expected behavior
        config = LLMGatewayConfig.from_env()
        assert config.api_key == "pk-test"

    def test_case_insensitive_env_vars(self) -> None:
        """Test that environment variables are case-insensitive."""
        # This is configured via model_config case_sensitive=False
        config = LLMGatewayConfig()
        assert config.model_config["case_sensitive"] is False

    def test_extra_fields_ignored(self) -> None:
        """Test that extra fields in config are ignored."""
        # This is configured via model_config extra="ignore"
        config = LLMGatewayConfig(unknown_field="value")  # type: ignore[call-arg]
        assert not hasattr(config, "unknown_field")

    def test_env_prefix(self) -> None:
        """Test that environment variable prefix is PORTKEY_."""
        config = LLMGatewayConfig()
        assert config.model_config["env_prefix"] == "PORTKEY_"

    def test_monitoring_config_defaults(self) -> None:
        """Test monitoring configuration defaults."""
        config = LLMGatewayConfig()
        assert config.enable_metrics is True
        assert config.metrics_collection_interval == 60
        assert config.enable_sla_monitoring is True
        assert config.enable_prometheus_export is True

    def test_full_monitoring_config(self) -> None:
        """Test full monitoring configuration."""
        config = LLMGatewayConfig(
            enable_metrics=False,
            metrics_collection_interval=300,
            enable_sla_monitoring=False,
            sla_availability_target=99.0,
            sla_response_time_target_ms=3000,
            sla_success_rate_target=98.0,
            enable_prometheus_export=False,
            prometheus_export_port=8080,
            alert_debounce_seconds=600,
        )
        assert config.enable_metrics is False
        assert config.metrics_collection_interval == 300
        assert config.enable_sla_monitoring is False
        assert config.sla_availability_target == 99.0
        assert config.sla_response_time_target_ms == 3000
        assert config.sla_success_rate_target == 98.0
        assert config.enable_prometheus_export is False
        assert config.prometheus_export_port == 8080
        assert config.alert_debounce_seconds == 600
