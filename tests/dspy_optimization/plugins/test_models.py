"""
Tests for plugin models
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentcore.dspy_optimization.plugins.models import (
    PluginCapability,
    PluginConfig,
    PluginMetadata,
    PluginRegistration,
    PluginStatus,
    PluginValidationResult,
)


class TestPluginMetadata:
    """Tests for PluginMetadata model"""

    def test_valid_metadata(self) -> None:
        """Test creating valid plugin metadata"""
        metadata = PluginMetadata(
            name="test_optimizer",
            version="1.0.0",
            author="Test Author",
            description="Test optimizer plugin",
            capabilities=[PluginCapability.MULTI_OBJECTIVE],
            tags=["test", "optimization"],
        )

        assert metadata.name == "test_optimizer"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.description == "Test optimizer plugin"
        assert PluginCapability.MULTI_OBJECTIVE in metadata.capabilities
        assert "test" in metadata.tags

    def test_metadata_with_dependencies(self) -> None:
        """Test metadata with dependencies"""
        metadata = PluginMetadata(
            name="advanced_optimizer",
            version="2.0.0",
            author="Test Author",
            description="Advanced optimizer",
            dependencies=["numpy>=1.20.0", "scipy>=1.7.0"],
        )

        assert len(metadata.dependencies) == 2
        assert "numpy>=1.20.0" in metadata.dependencies

    def test_metadata_defaults(self) -> None:
        """Test default values"""
        metadata = PluginMetadata(
            name="minimal_optimizer",
            version="1.0.0",
            author="Test Author",
            description="Minimal optimizer",
        )

        assert metadata.capabilities == []
        assert metadata.dependencies == []
        assert metadata.tags == []
        assert metadata.documentation_url is None
        assert metadata.license == "MIT"
        assert metadata.requires_python == ">=3.12"


class TestPluginConfig:
    """Tests for PluginConfig model"""

    def test_valid_config(self) -> None:
        """Test creating valid plugin config"""
        config = PluginConfig(
            plugin_name="test_optimizer",
            enabled=True,
            priority=200,
            timeout_seconds=3600,
            max_memory_mb=2048,
            parameters={"learning_rate": 0.01},
        )

        assert config.plugin_name == "test_optimizer"
        assert config.enabled is True
        assert config.priority == 200
        assert config.timeout_seconds == 3600
        assert config.max_memory_mb == 2048
        assert config.parameters["learning_rate"] == 0.01

    def test_config_defaults(self) -> None:
        """Test default config values"""
        config = PluginConfig(plugin_name="test_optimizer")

        assert config.enabled is True
        assert config.priority == 100
        assert config.timeout_seconds == 7200
        assert config.max_memory_mb == 4096
        assert config.parameters == {}

    def test_priority_bounds(self) -> None:
        """Test priority must be within bounds"""
        with pytest.raises(ValidationError):
            PluginConfig(plugin_name="test", priority=-1)

        with pytest.raises(ValidationError):
            PluginConfig(plugin_name="test", priority=1001)

    def test_timeout_bounds(self) -> None:
        """Test timeout must be at least 60 seconds"""
        with pytest.raises(ValidationError):
            PluginConfig(plugin_name="test", timeout_seconds=30)

    def test_memory_bounds(self) -> None:
        """Test max memory must be at least 128 MB"""
        with pytest.raises(ValidationError):
            PluginConfig(plugin_name="test", max_memory_mb=64)


class TestPluginValidationResult:
    """Tests for PluginValidationResult model"""

    def test_valid_result(self) -> None:
        """Test creating valid validation result"""
        result = PluginValidationResult(
            plugin_name="test_optimizer",
            is_valid=True,
            errors=[],
            warnings=["Missing documentation"],
            checks_passed=4,
            checks_total=5,
        )

        assert result.plugin_name == "test_optimizer"
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.checks_passed == 4
        assert result.checks_total == 5

    def test_invalid_result(self) -> None:
        """Test validation result with errors"""
        result = PluginValidationResult(
            plugin_name="bad_optimizer",
            is_valid=False,
            errors=["Missing required method", "Invalid version"],
            warnings=[],
            checks_passed=2,
            checks_total=5,
        )

        assert result.is_valid is False
        assert len(result.errors) == 2
        assert "Missing required method" in result.errors


class TestPluginRegistration:
    """Tests for PluginRegistration model"""

    def test_valid_registration(self) -> None:
        """Test creating valid plugin registration"""
        metadata = PluginMetadata(
            name="test_optimizer",
            version="1.0.0",
            author="Test Author",
            description="Test optimizer",
        )
        config = PluginConfig(plugin_name="test_optimizer")

        registration = PluginRegistration(
            metadata=metadata,
            config=config,
            status=PluginStatus.ACTIVE,
        )

        assert registration.metadata.name == "test_optimizer"
        assert registration.config.plugin_name == "test_optimizer"
        assert registration.status == PluginStatus.ACTIVE
        assert registration.usage_count == 0
        assert registration.last_used is None

    def test_registration_defaults(self) -> None:
        """Test default registration values"""
        metadata = PluginMetadata(
            name="test_optimizer",
            version="1.0.0",
            author="Test Author",
            description="Test optimizer",
        )
        config = PluginConfig(plugin_name="test_optimizer")

        registration = PluginRegistration(metadata=metadata, config=config)

        assert registration.status == PluginStatus.REGISTERED
        assert registration.registered_at is not None
        assert registration.last_used is None
        assert registration.usage_count == 0
        assert registration.error_message is None
