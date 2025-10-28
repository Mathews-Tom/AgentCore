"""
Tests for plugin discovery
"""

from __future__ import annotations

import tempfile
from pathlib import Path
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
from agentcore.dspy_optimization.plugins.registry import PluginRegistry


class TestDiscoveryPlugin(OptimizerPlugin):
    """Plugin for discovery testing"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="discovery_plugin",
            version="1.0.0",
            author="Test",
            description="Discovery test plugin",
            capabilities=[PluginCapability.GRADIENT_FREE],
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

        return MIPROv2Optimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        return PluginValidationResult(
            plugin_name="discovery_plugin",
            is_valid=True,
            checks_passed=5,
            checks_total=5,
        )


@pytest.fixture
def registry() -> PluginRegistry:
    """Create plugin registry"""
    return PluginRegistry()


@pytest.fixture
def plugin_dir() -> Path:
    """Create temporary plugin directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPluginDiscovery:
    """Tests for plugin discovery"""

    @pytest.mark.asyncio
    async def test_discover_plugins_empty_directory(
        self, registry: PluginRegistry, plugin_dir: Path
    ) -> None:
        """Test discovering plugins in empty directory"""
        registered = await registry.discover_plugins(plugin_dir)
        assert registered == []

    @pytest.mark.asyncio
    async def test_discover_plugins_invalid_directory(
        self, registry: PluginRegistry
    ) -> None:
        """Test discovering plugins in invalid directory"""
        with pytest.raises(ValueError):
            await registry.discover_plugins(Path("/nonexistent/directory"))

    @pytest.mark.asyncio
    async def test_discover_plugins_ignores_private_files(
        self, registry: PluginRegistry, plugin_dir: Path
    ) -> None:
        """Test that private files are ignored"""
        # Create a private file
        private_file = plugin_dir / "_private.py"
        private_file.write_text(
            """
from agentcore.dspy_optimization.plugins import OptimizerPlugin

class PrivatePlugin(OptimizerPlugin):
    pass
"""
        )

        registered = await registry.discover_plugins(plugin_dir)
        assert registered == []

    @pytest.mark.asyncio
    async def test_discover_multiple_plugins(
        self, registry: PluginRegistry, plugin_dir: Path
    ) -> None:
        """Test discovering multiple plugins"""
        # Create two plugin files
        plugin1_file = plugin_dir / "plugin1.py"
        plugin1_file.write_text(
            """
from typing import Any
from agentcore.dspy_optimization.plugins import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginMetadata, PluginConfig, PluginValidationResult, PluginCapability
)
from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.algorithms.miprov2 import MIPROv2Optimizer

class DiscoveredPlugin1(OptimizerPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="discovered_plugin1",
            version="1.0.0",
            author="Test",
            description="Discovered plugin 1",
            capabilities=[PluginCapability.GRADIENT_FREE],
        )

    def create_optimizer(self, config, **kwargs):
        return MIPROv2Optimizer(**kwargs)

    def validate(self):
        return PluginValidationResult(
            plugin_name="discovered_plugin1",
            is_valid=True,
            checks_passed=5,
            checks_total=5,
        )
"""
        )

        plugin2_file = plugin_dir / "plugin2.py"
        plugin2_file.write_text(
            """
from typing import Any
from agentcore.dspy_optimization.plugins import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginMetadata, PluginConfig, PluginValidationResult, PluginCapability
)
from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.algorithms.gepa import GEPAOptimizer

class DiscoveredPlugin2(OptimizerPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="discovered_plugin2",
            version="1.0.0",
            author="Test",
            description="Discovered plugin 2",
            capabilities=[PluginCapability.EVOLUTIONARY],
        )

    def create_optimizer(self, config, **kwargs):
        return GEPAOptimizer(**kwargs)

    def validate(self):
        return PluginValidationResult(
            plugin_name="discovered_plugin2",
            is_valid=True,
            checks_passed=5,
            checks_total=5,
        )
"""
        )

        registered = await registry.discover_plugins(plugin_dir)

        assert len(registered) == 2
        assert "discovered_plugin1" in registered
        assert "discovered_plugin2" in registered

        # Verify plugins are actually registered
        info1 = registry.get_plugin_info("discovered_plugin1")
        assert info1.metadata.name == "discovered_plugin1"

        info2 = registry.get_plugin_info("discovered_plugin2")
        assert info2.metadata.name == "discovered_plugin2"
