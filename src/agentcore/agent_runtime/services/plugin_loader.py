"""Plugin loader service for dynamic plugin management."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from ..models.plugin import (
    PluginConfig,
    PluginLoadError,
    PluginMetadata,
    PluginState,
    PluginStatus,
    PluginType,
    PluginValidationError,
)
from .plugin_validator import PluginValidator

logger = structlog.get_logger()


class PluginLoader:
    """Service for loading and managing plugins dynamically."""

    def __init__(
        self,
        plugin_directory: Path,
        validator: PluginValidator,
        enable_auto_load: bool = True,
    ) -> None:
        """
        Initialize plugin loader.

        Args:
            plugin_directory: Directory containing plugin packages
            validator: Plugin security validator
            enable_auto_load: Enable auto-loading of plugins
        """
        self._plugin_directory = plugin_directory
        self._validator = validator
        self._enable_auto_load = enable_auto_load
        self._plugins: dict[str, PluginState] = {}
        self._loaded_modules: dict[str, Any] = {}

        # Ensure plugin directory exists
        self._plugin_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            "plugin_loader_initialized",
            plugin_directory=str(plugin_directory),
            auto_load=enable_auto_load,
        )

    async def discover_plugins(self) -> list[PluginMetadata]:
        """
        Discover all available plugins in plugin directory.

        Returns:
            List of discovered plugin metadata

        Raises:
            OSError: If plugin directory cannot be read
        """
        discovered: list[PluginMetadata] = []

        logger.info("discovering_plugins", directory=str(self._plugin_directory))

        for plugin_path in self._plugin_directory.iterdir():
            if not plugin_path.is_dir():
                continue

            # Look for plugin manifest
            manifest_path = plugin_path / "plugin.json"
            if not manifest_path.exists():
                logger.debug(
                    "plugin_manifest_not_found",
                    plugin_path=str(plugin_path),
                )
                continue

            try:
                # Load and parse manifest
                with manifest_path.open("r") as f:
                    manifest_data = json.load(f)

                metadata = PluginMetadata(**manifest_data)
                discovered.append(metadata)

                logger.info(
                    "plugin_discovered",
                    plugin_id=metadata.plugin_id,
                    version=metadata.version,
                    type=metadata.plugin_type,
                )

            except Exception as e:
                logger.error(
                    "plugin_manifest_parse_error",
                    plugin_path=str(plugin_path),
                    error=str(e),
                )
                continue

        logger.info("plugin_discovery_complete", count=len(discovered))

        return discovered

    async def load_plugin(
        self,
        plugin_id: str,
        config: PluginConfig | None = None,
        validate: bool = True,
    ) -> PluginState:
        """
        Load and initialize a plugin.

        Args:
            plugin_id: Plugin identifier
            config: Plugin configuration (uses defaults if None)
            validate: Perform security validation before loading

        Returns:
            Plugin state after loading

        Raises:
            PluginLoadError: If plugin loading fails
            PluginValidationError: If plugin validation fails
        """
        logger.info("loading_plugin", plugin_id=plugin_id, validate=validate)

        # Check if already loaded
        if plugin_id in self._plugins:
            existing_state = self._plugins[plugin_id]
            if existing_state.status in (
                PluginStatus.LOADED,
                PluginStatus.ACTIVE,
            ):
                logger.warning("plugin_already_loaded", plugin_id=plugin_id)
                return existing_state

        try:
            # Find plugin directory
            plugin_path = self._find_plugin_path(plugin_id)
            if plugin_path is None:
                raise PluginLoadError(
                    plugin_id=plugin_id,
                    message="Plugin directory not found",
                )

            # Load metadata
            metadata = await self._load_metadata(plugin_path)

            # Create default config if not provided
            if config is None:
                config = PluginConfig(plugin_id=plugin_id)

            # Initialize plugin state
            state = PluginState(
                plugin_id=plugin_id,
                status=PluginStatus.LOADING,
                metadata=metadata,
                config=config,
            )
            self._plugins[plugin_id] = state

            # Validate plugin if requested
            if validate:
                validation_result = await self._validator.validate_plugin(
                    plugin_path=plugin_path,
                    metadata=metadata,
                )

                if not validation_result.valid:
                    state.status = PluginStatus.FAILED
                    state.error_message = "; ".join(validation_result.errors)
                    raise PluginValidationError(
                        plugin_id=plugin_id,
                        validation_result=validation_result,
                    )

                logger.info(
                    "plugin_validation_passed",
                    plugin_id=plugin_id,
                    security_score=validation_result.security_score,
                    risk_level=validation_result.risk_level,
                )

            # Load plugin module
            module = await self._load_module(
                plugin_id=plugin_id,
                entry_point=metadata.entry_point,
                plugin_path=plugin_path,
            )

            # Initialize plugin instance
            instance = await self._initialize_plugin(
                module=module,
                metadata=metadata,
                config=config,
            )

            # Update state
            state.instance = instance
            state.status = PluginStatus.LOADED
            state.load_time = datetime.now(UTC)

            logger.info(
                "plugin_loaded_successfully",
                plugin_id=plugin_id,
                version=metadata.version,
                type=metadata.plugin_type,
            )

            return state

        except Exception as e:
            # Update state on failure
            if plugin_id in self._plugins:
                state = self._plugins[plugin_id]
                state.status = PluginStatus.FAILED
                state.error_message = str(e)

            logger.error(
                "plugin_load_failed",
                plugin_id=plugin_id,
                error=str(e),
                error_type=type(e).__name__,
            )

            if isinstance(e, (PluginLoadError, PluginValidationError)):
                raise
            raise PluginLoadError(
                plugin_id=plugin_id,
                message=str(e),
                original_error=e,
            ) from e

    async def unload_plugin(self, plugin_id: str) -> None:
        """
        Unload a plugin and cleanup resources.

        Args:
            plugin_id: Plugin identifier

        Raises:
            KeyError: If plugin not found
        """
        if plugin_id not in self._plugins:
            raise KeyError(f"Plugin {plugin_id} not loaded")

        logger.info("unloading_plugin", plugin_id=plugin_id)

        state = self._plugins[plugin_id]
        state.status = PluginStatus.UNLOADING

        try:
            # Call plugin cleanup if available
            if state.instance and hasattr(state.instance, "cleanup"):
                await state.instance.cleanup()

            # Remove from loaded modules
            if plugin_id in self._loaded_modules:
                del self._loaded_modules[plugin_id]

            # Update state
            state.instance = None
            state.status = PluginStatus.UNLOADED

            logger.info("plugin_unloaded_successfully", plugin_id=plugin_id)

        except Exception as e:
            state.status = PluginStatus.FAILED
            state.error_message = f"Cleanup failed: {e}"

            logger.error(
                "plugin_unload_failed",
                plugin_id=plugin_id,
                error=str(e),
            )
            raise

    async def reload_plugin(
        self,
        plugin_id: str,
        validate: bool = True,
    ) -> PluginState:
        """
        Reload a plugin (unload then load).

        Args:
            plugin_id: Plugin identifier
            validate: Perform security validation

        Returns:
            Plugin state after reloading
        """
        logger.info("reloading_plugin", plugin_id=plugin_id)

        # Get current config before unloading
        config = None
        if plugin_id in self._plugins:
            config = self._plugins[plugin_id].config
            await self.unload_plugin(plugin_id)

        # Reload with same config
        return await self.load_plugin(
            plugin_id=plugin_id,
            config=config,
            validate=validate,
        )

    async def activate_plugin(self, plugin_id: str) -> None:
        """
        Activate a loaded plugin.

        Args:
            plugin_id: Plugin identifier

        Raises:
            KeyError: If plugin not found
            ValueError: If plugin not in LOADED status
        """
        if plugin_id not in self._plugins:
            raise KeyError(f"Plugin {plugin_id} not loaded")

        state = self._plugins[plugin_id]

        if state.status != PluginStatus.LOADED:
            raise ValueError(
                f"Plugin {plugin_id} must be LOADED to activate (current: {state.status})"
            )

        # Call activate hook if available
        if state.instance and hasattr(state.instance, "activate"):
            await state.instance.activate()

        state.status = PluginStatus.ACTIVE

        logger.info("plugin_activated", plugin_id=plugin_id)

    async def deactivate_plugin(self, plugin_id: str) -> None:
        """
        Deactivate an active plugin.

        Args:
            plugin_id: Plugin identifier

        Raises:
            KeyError: If plugin not found
        """
        if plugin_id not in self._plugins:
            raise KeyError(f"Plugin {plugin_id} not loaded")

        state = self._plugins[plugin_id]

        # Call deactivate hook if available
        if state.instance and hasattr(state.instance, "deactivate"):
            await state.instance.deactivate()

        state.status = PluginStatus.INACTIVE

        logger.info("plugin_deactivated", plugin_id=plugin_id)

    def get_plugin_state(self, plugin_id: str) -> PluginState:
        """
        Get current state of a plugin.

        Args:
            plugin_id: Plugin identifier

        Returns:
            Plugin state

        Raises:
            KeyError: If plugin not found
        """
        if plugin_id not in self._plugins:
            raise KeyError(f"Plugin {plugin_id} not loaded")

        return self._plugins[plugin_id]

    def list_plugins(
        self,
        status_filter: PluginStatus | None = None,
        type_filter: PluginType | None = None,
    ) -> list[PluginState]:
        """
        List all loaded plugins with optional filtering.

        Args:
            status_filter: Filter by status
            type_filter: Filter by type

        Returns:
            List of plugin states
        """
        plugins = list(self._plugins.values())

        if status_filter:
            plugins = [p for p in plugins if p.status == status_filter]

        if type_filter:
            plugins = [p for p in plugins if p.metadata.plugin_type == type_filter]

        return plugins

    async def auto_load_plugins(self) -> list[PluginState]:
        """
        Auto-load all plugins configured for auto-loading.

        Returns:
            List of loaded plugin states
        """
        if not self._enable_auto_load:
            logger.info("auto_load_disabled")
            return []

        logger.info("auto_loading_plugins")

        loaded_plugins: list[PluginState] = []
        discovered = await self.discover_plugins()

        for metadata in discovered:
            try:
                # Check for plugin config file
                config_path = (
                    self._plugin_directory / metadata.plugin_id / "config.json"
                )
                config = None

                if config_path.exists():
                    with config_path.open("r") as f:
                        config_data = json.load(f)
                        config = PluginConfig(**config_data)

                # Only auto-load if config enables it
                if config and config.auto_load:
                    state = await self.load_plugin(
                        plugin_id=metadata.plugin_id,
                        config=config,
                    )
                    loaded_plugins.append(state)

            except Exception as e:
                logger.error(
                    "auto_load_plugin_failed",
                    plugin_id=metadata.plugin_id,
                    error=str(e),
                )
                continue

        logger.info(
            "auto_load_complete",
            loaded_count=len(loaded_plugins),
            total_discovered=len(discovered),
        )

        return loaded_plugins

    def _find_plugin_path(self, plugin_id: str) -> Path | None:
        """Find plugin directory path by ID."""
        # Try direct lookup first
        plugin_path = self._plugin_directory / plugin_id
        if plugin_path.exists():
            return plugin_path

        # Search all directories for matching plugin.json
        for path in self._plugin_directory.iterdir():
            if not path.is_dir():
                continue

            manifest_path = path / "plugin.json"
            if manifest_path.exists():
                try:
                    with manifest_path.open("r") as f:
                        manifest = json.load(f)
                        if manifest.get("plugin_id") == plugin_id:
                            return path
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to decode manifest for plugin at {path}: {e}"
                    )
                    continue
                except OSError as e:
                    logger.warning(f"Failed to read manifest for plugin at {path}: {e}")
                    continue

        return None

    async def _load_metadata(self, plugin_path: Path) -> PluginMetadata:
        """Load plugin metadata from manifest file."""
        manifest_path = plugin_path / "plugin.json"

        if not manifest_path.exists():
            raise PluginLoadError(
                plugin_id=str(plugin_path.name),
                message="plugin.json not found",
            )

        with manifest_path.open("r") as f:
            manifest_data = json.load(f)

        return PluginMetadata(**manifest_data)

    async def _load_module(
        self,
        plugin_id: str,
        entry_point: str,
        plugin_path: Path,
    ) -> Any:
        """Load plugin module from entry point."""
        # Add plugin path to sys.path temporarily
        plugin_path_str = str(plugin_path)
        if plugin_path_str not in sys.path:
            sys.path.insert(0, plugin_path_str)

        try:
            # Import the entry point module
            module = importlib.import_module(entry_point)

            # Store loaded module
            self._loaded_modules[plugin_id] = module

            return module

        except Exception as e:
            raise PluginLoadError(
                plugin_id=plugin_id,
                message=f"Failed to import entry point '{entry_point}': {e}",
                original_error=e,
            ) from e

        finally:
            # Remove from sys.path
            if plugin_path_str in sys.path:
                sys.path.remove(plugin_path_str)

    async def _initialize_plugin(
        self,
        module: Any,
        metadata: PluginMetadata,
        config: PluginConfig,
    ) -> Any:
        """Initialize plugin instance from loaded module."""
        # Look for Plugin class or create_plugin function
        if hasattr(module, "Plugin"):
            plugin_class = module.Plugin
            instance = plugin_class(
                metadata=metadata,
                config=config.config,
            )
        elif hasattr(module, "create_plugin"):
            create_fn = module.create_plugin
            instance = await create_fn(
                metadata=metadata,
                config=config.config,
            )
        else:
            raise PluginLoadError(
                plugin_id=metadata.plugin_id,
                message="Plugin must define 'Plugin' class or 'create_plugin' function",
            )

        # Call initialize hook if available
        if hasattr(instance, "initialize"):
            await instance.initialize()

        return instance
