"""
Plugin interface for custom optimizer algorithms
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.plugins.models import (
    PluginConfig,
    PluginMetadata,
    PluginValidationResult,
)


class OptimizerPlugin(ABC):
    """
    Base interface for optimizer plugins

    All custom optimizer plugins must implement this interface to be
    registered and used in the optimization pipeline.
    """

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """
        Get plugin metadata

        Returns:
            PluginMetadata with name, version, capabilities, etc.
        """
        pass

    @abstractmethod
    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        """
        Create optimizer instance

        Args:
            config: Plugin configuration
            **kwargs: Additional optimizer arguments (e.g., llm)

        Returns:
            Configured BaseOptimizer instance
        """
        pass

    @abstractmethod
    def validate(self) -> PluginValidationResult:
        """
        Validate plugin implementation

        Returns:
            PluginValidationResult with validation status and details
        """
        pass

    def on_load(self) -> None:
        """
        Hook called when plugin is loaded

        Override to perform initialization tasks like loading models,
        connecting to services, etc.
        """
        pass

    def on_unload(self) -> None:
        """
        Hook called when plugin is unloaded

        Override to perform cleanup tasks like closing connections,
        releasing resources, etc.
        """
        pass

    def get_default_config(self) -> PluginConfig:
        """
        Get default plugin configuration

        Returns:
            Default PluginConfig
        """
        metadata = self.get_metadata()
        return PluginConfig(
            plugin_name=metadata.name,
            enabled=True,
            priority=100,
        )
