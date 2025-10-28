"""
Plugin validation framework
"""

from __future__ import annotations

import inspect
from typing import Any

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.plugins.interface import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginValidationResult,
)


class PluginValidator:
    """
    Validator for optimizer plugins

    Performs comprehensive validation checks on plugin implementation,
    metadata, and compatibility.
    """

    def __init__(self) -> None:
        """Initialize plugin validator"""
        self.checks: list[tuple[str, callable]] = [
            ("metadata", self._check_metadata),
            ("interface", self._check_interface),
            ("optimizer_creation", self._check_optimizer_creation),
            ("config", self._check_config),
            ("documentation", self._check_documentation),
        ]

    def validate(self, plugin: OptimizerPlugin) -> PluginValidationResult:
        """
        Validate plugin implementation

        Args:
            plugin: Plugin to validate

        Returns:
            PluginValidationResult with validation details
        """
        metadata = plugin.get_metadata()
        errors: list[str] = []
        warnings: list[str] = []
        checks_passed = 0

        for check_name, check_func in self.checks:
            try:
                check_result = check_func(plugin)
                if check_result["passed"]:
                    checks_passed += 1
                errors.extend(check_result.get("errors", []))
                warnings.extend(check_result.get("warnings", []))
            except Exception as e:
                errors.append(f"{check_name} check failed: {str(e)}")

        is_valid = len(errors) == 0
        checks_total = len(self.checks)

        return PluginValidationResult(
            plugin_name=metadata.name,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            checks_passed=checks_passed,
            checks_total=checks_total,
        )

    def _check_metadata(
        self, plugin: OptimizerPlugin
    ) -> dict[str, Any]:
        """Check plugin metadata"""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            metadata = plugin.get_metadata()

            # Required fields
            if not metadata.name:
                errors.append("Plugin name is required")
            elif not metadata.name.replace("_", "").replace("-", "").isalnum():
                errors.append("Plugin name must be alphanumeric (underscores/hyphens allowed)")

            if not metadata.version:
                errors.append("Plugin version is required")
            elif not self._is_valid_semver(metadata.version):
                errors.append("Plugin version must follow semantic versioning (e.g., 1.0.0)")

            if not metadata.author:
                warnings.append("Plugin author is recommended")

            if not metadata.description:
                warnings.append("Plugin description is recommended")

            if not metadata.capabilities:
                warnings.append("Plugin capabilities list is empty")

            if not metadata.documentation_url:
                warnings.append("Plugin documentation URL is recommended")

        except Exception as e:
            errors.append(f"Failed to get metadata: {str(e)}")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _check_interface(
        self, plugin: OptimizerPlugin
    ) -> dict[str, Any]:
        """Check plugin interface implementation"""
        errors: list[str] = []
        warnings: list[str] = []

        # Check required methods
        required_methods = ["get_metadata", "create_optimizer", "validate"]
        for method_name in required_methods:
            if not hasattr(plugin, method_name):
                errors.append(f"Missing required method: {method_name}")
            elif not callable(getattr(plugin, method_name)):
                errors.append(f"Method {method_name} is not callable")

        # Check optional hooks
        optional_methods = ["on_load", "on_unload", "get_default_config"]
        for method_name in optional_methods:
            if hasattr(plugin, method_name):
                if not callable(getattr(plugin, method_name)):
                    warnings.append(f"Method {method_name} exists but is not callable")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _check_optimizer_creation(
        self, plugin: OptimizerPlugin
    ) -> dict[str, Any]:
        """Check optimizer creation capability"""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            config = plugin.get_default_config()
            optimizer = plugin.create_optimizer(config)

            # Check if optimizer is BaseOptimizer instance
            if not isinstance(optimizer, BaseOptimizer):
                errors.append(
                    f"Created optimizer must be BaseOptimizer instance, got {type(optimizer)}"
                )

            # Check if optimizer has required methods
            required_methods = ["optimize", "get_algorithm_name"]
            for method_name in required_methods:
                if not hasattr(optimizer, method_name):
                    errors.append(f"Optimizer missing required method: {method_name}")

        except Exception as e:
            errors.append(f"Failed to create optimizer: {str(e)}")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _check_config(
        self, plugin: OptimizerPlugin
    ) -> dict[str, Any]:
        """Check plugin configuration"""
        errors: list[str] = []
        warnings: list[str] = []

        try:
            config = plugin.get_default_config()

            if config.priority < 0 or config.priority > 1000:
                warnings.append("Plugin priority should be between 0 and 1000")

            if config.timeout_seconds < 60:
                warnings.append("Plugin timeout should be at least 60 seconds")

            if config.max_memory_mb < 128:
                warnings.append("Plugin max memory should be at least 128 MB")

        except Exception as e:
            errors.append(f"Failed to get default config: {str(e)}")

        return {
            "passed": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _check_documentation(
        self, plugin: OptimizerPlugin
    ) -> dict[str, Any]:
        """Check plugin documentation"""
        errors: list[str] = []
        warnings: list[str] = []

        # Check class docstring
        if not plugin.__class__.__doc__:
            warnings.append("Plugin class should have docstring")

        # Check method docstrings
        methods_to_check = ["create_optimizer", "validate"]
        for method_name in methods_to_check:
            method = getattr(plugin, method_name, None)
            if method and not method.__doc__:
                warnings.append(f"Method {method_name} should have docstring")

        return {
            "passed": True,  # Documentation warnings don't fail validation
            "errors": errors,
            "warnings": warnings,
        }

    @staticmethod
    def _is_valid_semver(version: str) -> bool:
        """Check if version follows semantic versioning"""
        parts = version.split(".")
        if len(parts) != 3:
            return False
        return all(part.isdigit() for part in parts)
