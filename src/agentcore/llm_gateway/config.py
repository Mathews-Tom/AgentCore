"""Portkey integration configuration management.

Handles loading and validation of Portkey configuration from environment
variables and configuration files.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentcore.integration.portkey.exceptions import PortkeyConfigurationError


class PortkeyConfig(BaseSettings):
    """Configuration for Portkey AI Gateway integration.

    Loads configuration from environment variables with fallback to
    sensible defaults. All sensitive values should be loaded from
    environment variables or secure secret stores.
    """

    model_config = SettingsConfigDict(
        env_prefix="PORTKEY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Authentication
    api_key: str | None = Field(
        default=None,
        description="Portkey API key for authentication",
    )

    base_url: str = Field(
        default="https://api.portkey.ai/v1",
        description="Portkey API base URL",
    )

    virtual_key: str | None = Field(
        default=None,
        description="Default virtual key for provider authentication",
    )

    # Request Configuration
    timeout: float = Field(
        default=60.0,
        description="Default request timeout in seconds",
        ge=1.0,
        le=300.0,
    )

    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=0,
        le=10,
    )

    # Routing and Load Balancing
    default_provider: str | None = Field(
        default=None,
        description="Default LLM provider (e.g., 'openai', 'anthropic')",
    )

    fallback_providers: list[str] = Field(
        default_factory=list,
        description="Fallback providers in order of preference",
    )

    # Cost Optimization
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching for cost optimization",
    )

    cache_ttl: int = Field(
        default=3600,
        description="Cache time-to-live in seconds",
        ge=60,
    )

    max_cost_per_request: float | None = Field(
        default=None,
        description="Maximum cost per request in USD (for budget control)",
        ge=0.0,
    )

    # Monitoring and Observability
    enable_logging: bool = Field(
        default=True,
        description="Enable request/response logging",
    )

    log_request_bodies: bool = Field(
        default=False,
        description="Log full request bodies (may contain sensitive data)",
    )

    log_response_bodies: bool = Field(
        default=False,
        description="Log full response bodies",
    )

    enable_tracing: bool = Field(
        default=True,
        description="Enable distributed tracing for requests",
    )

    trace_id_header: str = Field(
        default="X-Portkey-Trace-Id",
        description="Header name for trace ID propagation",
    )

    # Performance Monitoring (INT-005)
    enable_metrics: bool = Field(
        default=True,
        description="Enable performance metrics collection",
    )

    metrics_collection_interval: int = Field(
        default=60,
        description="Metrics aggregation interval in seconds",
        ge=1,
        le=3600,
    )

    enable_sla_monitoring: bool = Field(
        default=True,
        description="Enable SLA monitoring and alerts",
    )

    sla_availability_target: float = Field(
        default=99.9,
        description="Target availability percentage for SLA",
        ge=0.0,
        le=100.0,
    )

    sla_response_time_target_ms: int = Field(
        default=2000,
        description="Target response time in milliseconds for SLA",
        ge=100,
        le=10000,
    )

    sla_success_rate_target: float = Field(
        default=99.5,
        description="Target success rate percentage for SLA",
        ge=0.0,
        le=100.0,
    )

    enable_prometheus_export: bool = Field(
        default=True,
        description="Enable Prometheus metrics export",
    )

    prometheus_export_port: int = Field(
        default=9090,
        description="Port for Prometheus metrics export",
        ge=1024,
        le=65535,
    )

    alert_debounce_seconds: int = Field(
        default=300,
        description="Seconds between duplicate performance alerts",
        ge=60,
        le=3600,
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to include with requests",
    )

    custom_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Custom HTTP headers to include with requests",
    )

    def validate_required_fields(self) -> None:
        """Validate that required configuration is present.

        Raises:
            PortkeyConfigurationError: If required configuration is missing
        """
        if not self.api_key:
            raise PortkeyConfigurationError(
                "PORTKEY_API_KEY environment variable is required"
            )

    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')

        Returns:
            Provider-specific configuration dictionary
        """
        # Look for provider-specific virtual key
        virtual_key_env = f"PORTKEY_{provider.upper()}_VIRTUAL_KEY"
        virtual_key = os.getenv(virtual_key_env) or self.virtual_key

        return {
            "provider": provider,
            "virtual_key": virtual_key,
        }

    def merge_with_defaults(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge override parameters with default configuration.

        Args:
            overrides: Dictionary of configuration overrides

        Returns:
            Merged configuration dictionary
        """
        base_config = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        # Merge custom headers
        if self.custom_headers:
            base_config["custom_headers"] = {**self.custom_headers}
            if "custom_headers" in overrides:
                base_config["custom_headers"].update(overrides["custom_headers"])

        # Merge metadata
        if self.metadata:
            base_config["metadata"] = {**self.metadata}
            if "metadata" in overrides:
                base_config["metadata"].update(overrides["metadata"])

        # Apply other overrides
        for key, value in overrides.items():
            if key not in ("custom_headers", "metadata"):
                base_config[key] = value

        return base_config

    @classmethod
    def from_env(cls) -> PortkeyConfig:
        """Create configuration from environment variables.

        Returns:
            Configured PortkeyConfig instance

        Raises:
            PortkeyConfigurationError: If required configuration is missing
        """
        config = cls()
        config.validate_required_fields()
        return config
