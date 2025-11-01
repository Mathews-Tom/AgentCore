"""
Plugin registry for custom optimizer algorithms
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.plugins.interface import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginConfig,
    PluginMetadata,
    PluginRegistration,
    PluginStatus,
)


class PluginRegistryError(Exception):
    """Base exception for plugin registry errors"""

    pass


class PluginNotFoundError(PluginRegistryError):
    """Raised when plugin is not found"""

    pass


class PluginAlreadyRegisteredError(PluginRegistryError):
    """Raised when attempting to register duplicate plugin"""

    pass


class PluginValidationError(PluginRegistryError):
    """Raised when plugin validation fails"""

    pass


class PluginRegistry:
    """
    Registry for managing optimizer plugins

    Handles plugin registration, discovery, lifecycle management,
    and optimizer instance creation.
    """

    def __init__(self) -> None:
        """Initialize plugin registry"""
        self._plugins: dict[str, PluginRegistration] = {}
        self._plugin_instances: dict[str, OptimizerPlugin] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        plugin: OptimizerPlugin,
        config: PluginConfig | None = None,
        validate: bool = True,
    ) -> PluginRegistration:
        """
        Register a new optimizer plugin

        Args:
            plugin: OptimizerPlugin instance
            config: Optional plugin configuration
            validate: Whether to validate plugin before registration

        Returns:
            PluginRegistration record

        Raises:
            PluginAlreadyRegisteredError: If plugin name already registered
            PluginValidationError: If validation fails
        """
        async with self._lock:
            metadata = plugin.get_metadata()

            # Check if already registered
            if metadata.name in self._plugins:
                raise PluginAlreadyRegisteredError(
                    f"Plugin '{metadata.name}' is already registered"
                )

            # Validate plugin if requested
            if validate:
                validation_result = plugin.validate()
                if not validation_result.is_valid:
                    raise PluginValidationError(
                        f"Plugin '{metadata.name}' validation failed: "
                        f"{', '.join(validation_result.errors)}"
                    )

            # Use provided config or default
            plugin_config = config or plugin.get_default_config()

            # Create registration record
            registration = PluginRegistration(
                metadata=metadata,
                config=plugin_config,
                status=PluginStatus.REGISTERED,
            )

            # Store plugin and registration
            self._plugin_instances[metadata.name] = plugin
            self._plugins[metadata.name] = registration

            # Call on_load hook
            try:
                plugin.on_load()
                registration.status = PluginStatus.ACTIVE
            except Exception as e:
                registration.status = PluginStatus.ERROR
                registration.error_message = str(e)
                raise PluginRegistryError(
                    f"Failed to load plugin '{metadata.name}': {e}"
                ) from e

            return registration

    async def unregister(self, plugin_name: str) -> None:
        """
        Unregister an optimizer plugin

        Args:
            plugin_name: Name of plugin to unregister

        Raises:
            PluginNotFoundError: If plugin not found
        """
        async with self._lock:
            if plugin_name not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")

            # Call on_unload hook
            plugin = self._plugin_instances[plugin_name]
            try:
                plugin.on_unload()
            except Exception as e:
                # Log but don't fail unregistration
                print(f"Warning: Error during plugin unload: {e}")

            # Remove from registry
            del self._plugins[plugin_name]
            del self._plugin_instances[plugin_name]

    async def get_optimizer(self, plugin_name: str, **kwargs: Any) -> BaseOptimizer:
        """
        Create optimizer instance from plugin

        Args:
            plugin_name: Name of plugin
            **kwargs: Additional arguments for optimizer creation

        Returns:
            BaseOptimizer instance

        Raises:
            PluginNotFoundError: If plugin not found
        """
        async with self._lock:
            if plugin_name not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")

            registration = self._plugins[plugin_name]

            # Check if plugin is active
            if registration.status != PluginStatus.ACTIVE:
                raise PluginRegistryError(
                    f"Plugin '{plugin_name}' is not active (status: {registration.status})"
                )

            # Update usage stats
            registration.last_used = datetime.now(UTC)
            registration.usage_count += 1

            # Create optimizer
            plugin = self._plugin_instances[plugin_name]
            try:
                optimizer = plugin.create_optimizer(registration.config, **kwargs)
                return optimizer
            except Exception as e:
                registration.status = PluginStatus.ERROR
                registration.error_message = str(e)
                raise PluginRegistryError(
                    f"Failed to create optimizer from plugin '{plugin_name}': {e}"
                ) from e

    def list_plugins(
        self, status: PluginStatus | None = None
    ) -> list[PluginRegistration]:
        """
        List registered plugins

        Args:
            status: Optional status filter

        Returns:
            List of plugin registrations
        """
        plugins = list(self._plugins.values())
        if status:
            plugins = [p for p in plugins if p.status == status]
        return plugins

    def get_plugin_info(self, plugin_name: str) -> PluginRegistration:
        """
        Get plugin registration info

        Args:
            plugin_name: Name of plugin

        Returns:
            PluginRegistration record

        Raises:
            PluginNotFoundError: If plugin not found
        """
        if plugin_name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")
        return self._plugins[plugin_name]

    async def discover_plugins(self, plugin_dir: Path) -> list[str]:
        """
        Discover and auto-register plugins from directory

        Args:
            plugin_dir: Directory containing plugin modules

        Returns:
            List of registered plugin names
        """
        if not plugin_dir.exists() or not plugin_dir.is_dir():
            raise ValueError(f"Invalid plugin directory: {plugin_dir}")

        registered = []

        # Find all Python files in directory
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue

            try:
                # Import module
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find OptimizerPlugin subclasses
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, OptimizerPlugin)
                            and obj is not OptimizerPlugin
                        ):
                            # Instantiate and register
                            plugin = obj()
                            registration = await self.register(plugin)
                            registered.append(registration.metadata.name)

            except Exception as e:
                print(f"Warning: Failed to load plugin from {plugin_file}: {e}")

        return registered

    async def update_config(self, plugin_name: str, config: PluginConfig) -> None:
        """
        Update plugin configuration

        Args:
            plugin_name: Name of plugin
            config: New configuration

        Raises:
            PluginNotFoundError: If plugin not found
        """
        async with self._lock:
            if plugin_name not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{plugin_name}' not found")

            registration = self._plugins[plugin_name]
            registration.config = config


# Global plugin registry instance
_global_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """
    Get global plugin registry instance

    Returns:
        Global PluginRegistry singleton
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry
