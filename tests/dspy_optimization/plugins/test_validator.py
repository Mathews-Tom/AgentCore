"""
Tests for plugin validator
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.plugins.interface import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginCapability,
    PluginConfig,
    PluginMetadata,
    PluginValidationResult,
)
from agentcore.dspy_optimization.plugins.validator import PluginValidator


class WellDocumentedPlugin(OptimizerPlugin):
    """Well-documented plugin with proper docstrings"""

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name="well_documented",
            version="1.0.0",
            author="Test Author",
            description="Well-documented plugin",
            capabilities=[PluginCapability.GRADIENT_FREE],
            documentation_url="https://example.com/docs",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        """Create optimizer instance"""
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        """Validate plugin"""
        validator = PluginValidator()
        return validator.validate(self)


class PoorlyDocumentedPlugin(OptimizerPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="poorly_documented",
            version="1.0.0",
            author="Test",
            description="Poorly documented",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        validator = PluginValidator()
        return validator.validate(self)


class InvalidNamePlugin(OptimizerPlugin):
    """Plugin with invalid name"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="invalid name!@#",
            version="1.0.0",
            author="Test",
            description="Invalid name",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        validator = PluginValidator()
        return validator.validate(self)


class InvalidVersionPlugin(OptimizerPlugin):
    """Plugin with invalid version"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="invalid_version",
            version="v1.0",
            author="Test",
            description="Invalid version",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        validator = PluginValidator()
        return validator.validate(self)


class MissingMethodPlugin(OptimizerPlugin):
    """Plugin missing required method"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="missing_method",
            version="1.0.0",
            author="Test",
            description="Missing method",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    # Missing validate method


class BrokenOptimizerPlugin(OptimizerPlugin):
    """Plugin that creates invalid optimizer"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="broken_optimizer",
            version="1.0.0",
            author="Test",
            description="Broken optimizer",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        # Returns wrong type
        return "not an optimizer"  # type: ignore

    def validate(self) -> PluginValidationResult:
        validator = PluginValidator()
        return validator.validate(self)


@pytest.fixture
def validator() -> PluginValidator:
    """Create plugin validator"""
    return PluginValidator()


class TestPluginValidator:
    """Tests for PluginValidator"""

    def test_validate_well_documented_plugin(
        self, validator: PluginValidator
    ) -> None:
        """Test validating well-documented plugin"""
        plugin = WellDocumentedPlugin()
        result = validator.validate(plugin)

        assert result.plugin_name == "well_documented"
        assert result.is_valid is True
        assert result.checks_passed == result.checks_total
        assert len(result.errors) == 0

    def test_validate_poorly_documented_plugin(
        self, validator: PluginValidator
    ) -> None:
        """Test validating poorly documented plugin"""
        plugin = PoorlyDocumentedPlugin()
        result = validator.validate(plugin)

        assert result.plugin_name == "poorly_documented"
        assert result.is_valid is True  # Documentation warnings don't fail validation
        assert len(result.warnings) > 0
        assert any("docstring" in w.lower() for w in result.warnings)

    def test_validate_invalid_name(self, validator: PluginValidator) -> None:
        """Test validating plugin with invalid name"""
        plugin = InvalidNamePlugin()
        result = validator.validate(plugin)

        assert result.is_valid is False
        assert any("name" in e.lower() for e in result.errors)

    def test_validate_invalid_version(self, validator: PluginValidator) -> None:
        """Test validating plugin with invalid version"""
        plugin = InvalidVersionPlugin()
        result = validator.validate(plugin)

        assert result.is_valid is False
        assert any("version" in e.lower() for e in result.errors)

    def test_validate_broken_optimizer(self, validator: PluginValidator) -> None:
        """Test validating plugin that creates invalid optimizer"""
        plugin = BrokenOptimizerPlugin()
        result = validator.validate(plugin)

        assert result.is_valid is False
        assert any("BaseOptimizer" in e for e in result.errors)

    def test_check_metadata(self, validator: PluginValidator) -> None:
        """Test metadata check"""
        plugin = WellDocumentedPlugin()
        check_result = validator._check_metadata(plugin)

        assert check_result["passed"] is True
        assert len(check_result["errors"]) == 0

    def test_check_interface(self, validator: PluginValidator) -> None:
        """Test interface check"""
        plugin = WellDocumentedPlugin()
        check_result = validator._check_interface(plugin)

        assert check_result["passed"] is True
        assert len(check_result["errors"]) == 0

    def test_check_optimizer_creation(self, validator: PluginValidator) -> None:
        """Test optimizer creation check"""
        plugin = WellDocumentedPlugin()
        check_result = validator._check_optimizer_creation(plugin)

        assert check_result["passed"] is True
        assert len(check_result["errors"]) == 0

    def test_check_config(self, validator: PluginValidator) -> None:
        """Test config check"""
        plugin = WellDocumentedPlugin()
        check_result = validator._check_config(plugin)

        assert check_result["passed"] is True

    def test_check_documentation(self, validator: PluginValidator) -> None:
        """Test documentation check"""
        plugin = WellDocumentedPlugin()
        check_result = validator._check_documentation(plugin)

        assert check_result["passed"] is True

    def test_is_valid_semver(self) -> None:
        """Test semantic version validation"""
        assert PluginValidator._is_valid_semver("1.0.0") is True
        assert PluginValidator._is_valid_semver("1.2.3") is True
        assert PluginValidator._is_valid_semver("0.0.1") is True
        assert PluginValidator._is_valid_semver("v1.0.0") is False
        assert PluginValidator._is_valid_semver("1.0") is False
        assert PluginValidator._is_valid_semver("1") is False
        assert PluginValidator._is_valid_semver("1.0.0.0") is False

    def test_validation_checks_count(self, validator: PluginValidator) -> None:
        """Test that validator has correct number of checks"""
        assert len(validator.checks) == 5
        check_names = [name for name, _ in validator.checks]
        assert "metadata" in check_names
        assert "interface" in check_names
        assert "optimizer_creation" in check_names
        assert "config" in check_names
        assert "documentation" in check_names
