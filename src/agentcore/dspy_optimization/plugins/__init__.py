"""
Plugin system for custom optimizer algorithms

Provides extensible architecture for registering and using custom
optimization algorithms in the DSPy optimization engine.
"""

from __future__ import annotations

from agentcore.dspy_optimization.plugins.comparison import (
    AlgorithmComparison,
    MetricComparison,
    PerformanceComparator,
)
from agentcore.dspy_optimization.plugins.interface import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginCapability,
    PluginConfig,
    PluginMetadata,
    PluginRegistration,
    PluginStatus,
    PluginValidationResult,
)
from agentcore.dspy_optimization.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
    PluginRegistryError,
    PluginValidationError,
    get_plugin_registry,
)
from agentcore.dspy_optimization.plugins.validator import PluginValidator

__all__ = [
    # Interface
    "OptimizerPlugin",
    # Models
    "PluginCapability",
    "PluginConfig",
    "PluginMetadata",
    "PluginRegistration",
    "PluginStatus",
    "PluginValidationResult",
    # Registry
    "PluginRegistry",
    "PluginRegistryError",
    "PluginNotFoundError",
    "PluginAlreadyRegisteredError",
    "PluginValidationError",
    "get_plugin_registry",
    # Validator
    "PluginValidator",
    # Comparison
    "PerformanceComparator",
    "AlgorithmComparison",
    "MetricComparison",
]
