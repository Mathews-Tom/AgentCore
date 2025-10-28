"""
Tests for plugin registry
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.plugins.interface import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginCapability,
    PluginConfig,
    PluginMetadata,
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


class ValidPlugin(OptimizerPlugin):
    """Valid test plugin"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="valid_plugin",
            version="1.0.0",
            author="Test",
            description="Valid plugin",
            capabilities=[PluginCapability.GRADIENT_FREE],
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        return PluginValidationResult(
            plugin_name="valid_plugin",
            is_valid=True,
            checks_passed=5,
            checks_total=5,
        )


class InvalidPlugin(OptimizerPlugin):
    """Invalid test plugin that fails validation"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="invalid_plugin",
            version="1.0.0",
            author="Test",
            description="Invalid plugin",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        return PluginValidationResult(
            plugin_name="invalid_plugin",
            is_valid=False,
            errors=["Validation failed"],
            checks_passed=0,
            checks_total=5,
        )


class FailingLoadPlugin(OptimizerPlugin):
    """Plugin that fails during on_load"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="failing_load",
            version="1.0.0",
            author="Test",
            description="Failing plugin",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        return PluginValidationResult(
            plugin_name="failing_load",
            is_valid=True,
            checks_passed=5,
            checks_total=5,
        )

    def on_load(self) -> None:
        raise RuntimeError("Load failed")


@pytest.fixture
def registry() -> PluginRegistry:
    """Create a fresh plugin registry for each test"""
    return PluginRegistry()


class TestPluginRegistry:
    """Tests for PluginRegistry"""

    @pytest.mark.asyncio
    async def test_register_valid_plugin(self, registry: PluginRegistry) -> None:
        """Test registering a valid plugin"""
        plugin = ValidPlugin()
        registration = await registry.register(plugin)

        assert registration.metadata.name == "valid_plugin"
        assert registration.status == PluginStatus.ACTIVE
        assert registration.usage_count == 0

    @pytest.mark.asyncio
    async def test_register_with_custom_config(
        self, registry: PluginRegistry
    ) -> None:
        """Test registering plugin with custom config"""
        plugin = ValidPlugin()
        config = PluginConfig(
            plugin_name="valid_plugin",
            priority=200,
            timeout_seconds=3600,
        )

        registration = await registry.register(plugin, config=config)

        assert registration.config.priority == 200
        assert registration.config.timeout_seconds == 3600

    @pytest.mark.asyncio
    async def test_register_duplicate_plugin(
        self, registry: PluginRegistry
    ) -> None:
        """Test registering duplicate plugin raises error"""
        plugin1 = ValidPlugin()
        plugin2 = ValidPlugin()

        await registry.register(plugin1)

        with pytest.raises(PluginAlreadyRegisteredError):
            await registry.register(plugin2)

    @pytest.mark.asyncio
    async def test_register_invalid_plugin(
        self, registry: PluginRegistry
    ) -> None:
        """Test registering invalid plugin raises error"""
        plugin = InvalidPlugin()

        with pytest.raises(PluginValidationError):
            await registry.register(plugin)

    @pytest.mark.asyncio
    async def test_register_skip_validation(
        self, registry: PluginRegistry
    ) -> None:
        """Test registering invalid plugin with validation skipped"""
        plugin = InvalidPlugin()
        registration = await registry.register(plugin, validate=False)

        assert registration.metadata.name == "invalid_plugin"
        assert registration.status == PluginStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_register_failing_load(self, registry: PluginRegistry) -> None:
        """Test plugin that fails during on_load"""
        plugin = FailingLoadPlugin()

        with pytest.raises(PluginRegistryError):
            await registry.register(plugin)

    @pytest.mark.asyncio
    async def test_unregister_plugin(self, registry: PluginRegistry) -> None:
        """Test unregistering a plugin"""
        plugin = ValidPlugin()
        await registry.register(plugin)

        await registry.unregister("valid_plugin")

        with pytest.raises(PluginNotFoundError):
            registry.get_plugin_info("valid_plugin")

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, registry: PluginRegistry) -> None:
        """Test unregistering nonexistent plugin raises error"""
        with pytest.raises(PluginNotFoundError):
            await registry.unregister("nonexistent")

    @pytest.mark.asyncio
    async def test_get_optimizer(self, registry: PluginRegistry) -> None:
        """Test getting optimizer from plugin"""
        plugin = ValidPlugin()
        await registry.register(plugin)

        optimizer = await registry.get_optimizer("valid_plugin")

        assert isinstance(optimizer, BaseOptimizer)

    @pytest.mark.asyncio
    async def test_get_optimizer_updates_stats(
        self, registry: PluginRegistry
    ) -> None:
        """Test getting optimizer updates usage stats"""
        plugin = ValidPlugin()
        await registry.register(plugin)

        await registry.get_optimizer("valid_plugin")
        await registry.get_optimizer("valid_plugin")

        info = registry.get_plugin_info("valid_plugin")
        assert info.usage_count == 2
        assert info.last_used is not None

    @pytest.mark.asyncio
    async def test_get_optimizer_nonexistent(
        self, registry: PluginRegistry
    ) -> None:
        """Test getting optimizer for nonexistent plugin"""
        with pytest.raises(PluginNotFoundError):
            await registry.get_optimizer("nonexistent")

    def test_list_plugins(self, registry: PluginRegistry) -> None:
        """Test listing all plugins"""
        assert registry.list_plugins() == []

    @pytest.mark.asyncio
    async def test_list_plugins_with_status_filter(
        self, registry: PluginRegistry
    ) -> None:
        """Test listing plugins with status filter"""
        plugin = ValidPlugin()
        await registry.register(plugin)

        active_plugins = registry.list_plugins(status=PluginStatus.ACTIVE)
        assert len(active_plugins) == 1
        assert active_plugins[0].metadata.name == "valid_plugin"

        inactive_plugins = registry.list_plugins(status=PluginStatus.INACTIVE)
        assert len(inactive_plugins) == 0

    @pytest.mark.asyncio
    async def test_get_plugin_info(self, registry: PluginRegistry) -> None:
        """Test getting plugin info"""
        plugin = ValidPlugin()
        await registry.register(plugin)

        info = registry.get_plugin_info("valid_plugin")

        assert info.metadata.name == "valid_plugin"
        assert info.status == PluginStatus.ACTIVE

    def test_get_plugin_info_nonexistent(self, registry: PluginRegistry) -> None:
        """Test getting info for nonexistent plugin"""
        with pytest.raises(PluginNotFoundError):
            registry.get_plugin_info("nonexistent")

    @pytest.mark.asyncio
    async def test_update_config(self, registry: PluginRegistry) -> None:
        """Test updating plugin configuration"""
        plugin = ValidPlugin()
        await registry.register(plugin)

        new_config = PluginConfig(
            plugin_name="valid_plugin",
            priority=500,
            timeout_seconds=1800,
        )

        await registry.update_config("valid_plugin", new_config)

        info = registry.get_plugin_info("valid_plugin")
        assert info.config.priority == 500
        assert info.config.timeout_seconds == 1800

    @pytest.mark.asyncio
    async def test_update_config_nonexistent(
        self, registry: PluginRegistry
    ) -> None:
        """Test updating config for nonexistent plugin"""
        config = PluginConfig(plugin_name="nonexistent")

        with pytest.raises(PluginNotFoundError):
            await registry.update_config("nonexistent", config)


class TestGlobalRegistry:
    """Tests for global registry singleton"""

    def test_get_plugin_registry(self) -> None:
        """Test getting global registry singleton"""
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()

        assert registry1 is registry2
