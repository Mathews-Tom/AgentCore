"""Tests for plugin system architecture."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentcore.agent_runtime.models.plugin import (
    PluginCapability,
    PluginConfig,
    PluginDependency,
    PluginLoadError,
    PluginMarketplaceInfo,
    PluginMetadata,
    PluginPermissions,
    PluginState,
    PluginStatus,
    PluginType,
    PluginValidationError,
    PluginValidationResult,
    PluginVersionConflictError,
)
from agentcore.agent_runtime.services.audit_logger import AuditLogger
from agentcore.agent_runtime.services.plugin_loader import PluginLoader
from agentcore.agent_runtime.services.plugin_registry import PluginRegistry
from agentcore.agent_runtime.services.plugin_validator import PluginValidator
from agentcore.agent_runtime.services.plugin_version_manager import PluginVersionManager


class TestPluginModels:
    """Test plugin data models."""

    def test_plugin_metadata_creation(self):
        """Test creating plugin metadata."""
        metadata = PluginMetadata(
            plugin_id="com.example.test",
            name="Test Plugin",
            version="1.0.0",
            description="A test plugin",
            author="Test Author",
            plugin_type=PluginType.TOOL,
            entry_point="test_plugin.main",
        )

        assert metadata.plugin_id == "com.example.test"
        assert metadata.name == "Test Plugin"
        assert metadata.version == "1.0.0"
        assert metadata.plugin_type == PluginType.TOOL

    def test_plugin_metadata_validation(self):
        """Test plugin metadata validation."""
        # Invalid plugin ID (no dots)
        with pytest.raises(ValueError, match="reverse DNS pattern"):
            PluginMetadata(
                plugin_id="invalid",
                name="Test",
                version="1.0.0",
                description="Test",
                author="Test",
                plugin_type=PluginType.TOOL,
                entry_point="test",
            )

    def test_plugin_config_creation(self):
        """Test creating plugin configuration."""
        config = PluginConfig(
            plugin_id="com.example.test",
            enabled=True,
            auto_load=True,
            priority=100,
            config={"key": "value"},
        )

        assert config.plugin_id == "com.example.test"
        assert config.enabled is True
        assert config.auto_load is True
        assert config.config["key"] == "value"

    def test_plugin_state_lifecycle(self):
        """Test plugin state transitions."""
        metadata = PluginMetadata(
            plugin_id="com.example.test",
            name="Test",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.TOOL,
            entry_point="test",
        )
        config = PluginConfig(plugin_id="com.example.test")

        state = PluginState(
            plugin_id="com.example.test",
            metadata=metadata,
            config=config,
        )

        assert state.status == PluginStatus.UNLOADED
        assert state.usage_count == 0

        # Simulate loading
        state.status = PluginStatus.LOADED
        state.load_time = datetime.now(UTC)

        assert state.status == PluginStatus.LOADED
        assert state.load_time is not None


class TestPluginVersionManager:
    """Test version management and compatibility checking."""

    def test_version_comparison(self):
        """Test semantic version comparison."""
        manager = PluginVersionManager()

        assert manager.compare_versions("1.0.0", "1.0.0") == 0
        assert manager.compare_versions("1.0.0", "1.0.1") == -1
        assert manager.compare_versions("1.0.1", "1.0.0") == 1
        assert manager.compare_versions("2.0.0", "1.9.9") == 1

    def test_version_parsing(self):
        """Test version string parsing."""
        manager = PluginVersionManager()

        assert manager._parse_version("1.2.3") == (1, 2, 3)
        assert manager._parse_version("v1.2.3") == (1, 2, 3)
        assert manager._parse_version("1.2") == (1, 2, 0)

        with pytest.raises(ValueError):
            manager._parse_version("invalid")

    def test_constraint_satisfaction(self):
        """Test version constraint checking."""
        manager = PluginVersionManager()

        # Wildcard
        assert manager._satisfies_constraint("1.0.0", "*") is True

        # Greater than or equal
        assert manager._satisfies_constraint("1.5.0", ">=1.0.0") is True
        assert manager._satisfies_constraint("0.9.0", ">=1.0.0") is False

        # Less than or equal
        assert manager._satisfies_constraint("1.0.0", "<=1.5.0") is True
        assert manager._satisfies_constraint("2.0.0", "<=1.5.0") is False

        # Exact match
        assert manager._satisfies_constraint("1.0.0", "==1.0.0") is True
        assert manager._satisfies_constraint("1.0.1", "==1.0.0") is False

        # Caret (backward compatible)
        assert manager._satisfies_constraint("1.2.0", "^1.0.0") is True
        assert manager._satisfies_constraint("2.0.0", "^1.0.0") is False

    def test_runtime_compatibility(self):
        """Test runtime version compatibility checking."""
        manager = PluginVersionManager(runtime_version="1.5.0")

        # Compatible plugin
        metadata = PluginMetadata(
            plugin_id="com.example.test",
            name="Test",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.TOOL,
            entry_point="test",
            min_runtime_version="1.0.0",
            max_runtime_version="2.0.0",
        )

        is_compatible, reason = manager.check_runtime_compatibility(metadata)
        assert is_compatible is True

        # Incompatible (runtime too old)
        metadata.min_runtime_version = "2.0.0"
        is_compatible, reason = manager.check_runtime_compatibility(metadata)
        assert is_compatible is False
        assert "below minimum" in reason

    def test_dependency_resolution(self):
        """Test plugin dependency resolution."""
        manager = PluginVersionManager()

        metadata = PluginMetadata(
            plugin_id="com.example.main",
            name="Main Plugin",
            version="1.0.0",
            description="Main",
            author="Test",
            plugin_type=PluginType.TOOL,
            entry_point="main",
            dependencies=[
                PluginDependency(
                    plugin_id="com.example.dep1",
                    version_constraint=">=1.0.0",
                    optional=False,
                ),
                PluginDependency(
                    plugin_id="com.example.dep2",
                    version_constraint="^1.0.0",
                    optional=True,
                ),
            ],
        )

        # All dependencies satisfied
        available = {
            "com.example.dep1": "1.2.0",
            "com.example.dep2": "1.1.0",
        }

        all_satisfied, missing = manager.resolve_dependencies(metadata, available)
        assert all_satisfied is True
        assert len(missing) == 0

        # Required dependency missing
        available = {
            "com.example.dep2": "1.1.0",
        }

        all_satisfied, missing = manager.resolve_dependencies(metadata, available)
        assert all_satisfied is False
        assert len(missing) == 1
        assert "com.example.dep1" in missing[0]

    def test_latest_version_selection(self):
        """Test selecting latest version from list."""
        manager = PluginVersionManager()

        versions = ["1.0.0", "1.2.0", "1.1.0", "2.0.0", "1.2.1"]
        latest = manager.get_latest_version(versions)

        assert latest == "2.0.0"

    def test_backward_compatibility(self):
        """Test backward compatibility checking."""
        manager = PluginVersionManager()

        # Same major version, increased minor/patch
        assert manager.is_backward_compatible("1.0.0", "1.1.0") is True
        assert manager.is_backward_compatible("1.0.0", "1.0.1") is True

        # Different major version
        assert manager.is_backward_compatible("1.0.0", "2.0.0") is False

        # Downgrade
        assert manager.is_backward_compatible("1.1.0", "1.0.0") is False


class TestPluginValidator:
    """Test plugin security validation."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary plugin directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def validator(self):
        """Create plugin validator."""
        return PluginValidator(
            max_file_size_mb=10,
            enable_code_scanning=True,
            enable_checksum_validation=False,
        )

    def test_structure_validation(self, temp_plugin_dir, validator):
        """Test plugin structure validation."""
        # Missing plugin.json
        errors = validator._validate_structure(temp_plugin_dir)
        assert any("plugin.json" in e for e in errors)

        # Create valid structure
        (temp_plugin_dir / "plugin.json").write_text("{}")
        errors = validator._validate_structure(temp_plugin_dir)
        assert len(errors) == 0

    def test_metadata_validation(self, validator):
        """Test metadata validation."""
        # Valid metadata
        metadata = PluginMetadata(
            plugin_id="com.example.test",
            name="Test",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.TOOL,
            entry_point="test.main",
        )

        errors = validator._validate_metadata(metadata)
        assert len(errors) == 0

        # Invalid entry point
        metadata.entry_point = "invalid entry"
        errors = validator._validate_metadata(metadata)
        assert any("Invalid entry point" in e for e in errors)

    @pytest.mark.asyncio
    async def test_code_scanning(self, temp_plugin_dir, validator):
        """Test static code analysis."""
        metadata = PluginMetadata(
            plugin_id="com.example.test",
            name="Test",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.TOOL,
            entry_point="test",
        )

        # Create safe Python file
        safe_code = """
def hello():
    return "Hello, World!"
"""
        (temp_plugin_dir / "safe.py").write_text(safe_code)

        errors, warnings = await validator._scan_code(temp_plugin_dir, metadata)
        assert len(errors) == 0

        # Create dangerous Python file
        dangerous_code = """
import os
import subprocess

def dangerous():
    os.system("rm -rf /")
    subprocess.call(["ls"])
"""
        (temp_plugin_dir / "dangerous.py").write_text(dangerous_code)

        errors, warnings = await validator._scan_code(temp_plugin_dir, metadata)
        assert len(errors) > 0
        assert any("Dangerous import" in e for e in errors)

    @pytest.mark.asyncio
    async def test_permission_validation(self, validator):
        """Test permission validation."""
        # Broad filesystem permissions
        metadata = PluginMetadata(
            plugin_id="com.example.test",
            name="Test",
            version="1.0.0",
            description="Test",
            author="Test",
            plugin_type=PluginType.TOOL,
            entry_point="test",
            permissions=PluginPermissions(
                filesystem_write=["/*"],
                network_hosts=["*"],
            ),
        )

        warnings = await validator._validate_permissions(metadata)
        assert len(warnings) > 0
        assert any("root filesystem" in w for w in warnings)
        assert any("all network hosts" in w for w in warnings)


@pytest.mark.asyncio
class TestPluginLoader:
    """Test plugin loading and management."""

    @pytest.fixture
    def temp_plugin_dir(self):
        """Create temporary plugin directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def validator(self):
        """Create validator."""
        return PluginValidator(enable_code_scanning=False)

    @pytest.fixture
    def loader(self, temp_plugin_dir, validator):
        """Create plugin loader."""
        return PluginLoader(
            plugin_directory=temp_plugin_dir,
            validator=validator,
            enable_auto_load=False,
        )

    def _create_test_plugin(
        self,
        plugin_dir: Path,
        plugin_id: str = "com.example.test",
    ):
        """Create a minimal test plugin."""
        plugin_path = plugin_dir / plugin_id
        plugin_path.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest = {
            "plugin_id": plugin_id,
            "name": "Test Plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "author": "Test",
            "plugin_type": "tool",
            "entry_point": "plugin",
            "capabilities": [],
            "dependencies": [],
            "permissions": {},
        }

        (plugin_path / "plugin.json").write_text(json.dumps(manifest))

        # Create simple Python module
        code = """
class Plugin:
    def __init__(self, metadata, config):
        self.metadata = metadata
        self.config = config

    async def initialize(self):
        pass

    async def cleanup(self):
        pass
"""
        (plugin_path / "plugin.py").write_text(code)

        return plugin_path

    async def test_discover_plugins(self, temp_plugin_dir, loader):
        """Test plugin discovery."""
        # Create test plugins
        self._create_test_plugin(temp_plugin_dir, "com.example.plugin1")
        self._create_test_plugin(temp_plugin_dir, "com.example.plugin2")

        discovered = await loader.discover_plugins()

        assert len(discovered) == 2
        plugin_ids = {p.plugin_id for p in discovered}
        assert "com.example.plugin1" in plugin_ids
        assert "com.example.plugin2" in plugin_ids

    async def test_load_plugin(self, temp_plugin_dir, loader):
        """Test loading a plugin."""
        plugin_id = "com.example.test"
        self._create_test_plugin(temp_plugin_dir, plugin_id)

        state = await loader.load_plugin(
            plugin_id=plugin_id,
            validate=False,
        )

        assert state.plugin_id == plugin_id
        assert state.status == PluginStatus.LOADED
        assert state.instance is not None

    async def test_load_nonexistent_plugin(self, loader):
        """Test loading non-existent plugin."""
        with pytest.raises(PluginLoadError, match="not found"):
            await loader.load_plugin(
                plugin_id="com.example.nonexistent",
                validate=False,
            )

    async def test_unload_plugin(self, temp_plugin_dir, loader):
        """Test unloading a plugin."""
        plugin_id = "com.example.test"
        self._create_test_plugin(temp_plugin_dir, plugin_id)

        # Load plugin
        await loader.load_plugin(plugin_id, validate=False)

        # Unload plugin
        await loader.unload_plugin(plugin_id)

        state = loader.get_plugin_state(plugin_id)
        assert state.status == PluginStatus.UNLOADED
        assert state.instance is None

    async def test_reload_plugin(self, temp_plugin_dir, loader):
        """Test reloading a plugin."""
        plugin_id = "com.example.test"
        self._create_test_plugin(temp_plugin_dir, plugin_id)

        # Load and reload
        await loader.load_plugin(plugin_id, validate=False)
        state = await loader.reload_plugin(plugin_id, validate=False)

        assert state.status == PluginStatus.LOADED

    async def test_activate_deactivate_plugin(self, temp_plugin_dir, loader):
        """Test plugin activation/deactivation."""
        plugin_id = "com.example.test"
        self._create_test_plugin(temp_plugin_dir, plugin_id)

        await loader.load_plugin(plugin_id, validate=False)

        # Activate
        await loader.activate_plugin(plugin_id)
        state = loader.get_plugin_state(plugin_id)
        assert state.status == PluginStatus.ACTIVE

        # Deactivate
        await loader.deactivate_plugin(plugin_id)
        state = loader.get_plugin_state(plugin_id)
        assert state.status == PluginStatus.INACTIVE

    async def test_list_plugins(self, temp_plugin_dir, loader):
        """Test listing loaded plugins."""
        self._create_test_plugin(temp_plugin_dir, "com.example.plugin1")
        self._create_test_plugin(temp_plugin_dir, "com.example.plugin2")

        await loader.load_plugin("com.example.plugin1", validate=False)
        await loader.load_plugin("com.example.plugin2", validate=False)

        plugins = loader.list_plugins()
        assert len(plugins) == 2

        # Filter by status
        await loader.activate_plugin("com.example.plugin1")
        active_plugins = loader.list_plugins(status_filter=PluginStatus.ACTIVE)
        assert len(active_plugins) == 1


def test_plugin_models_serialization():
    """Test plugin model JSON serialization."""
    metadata = PluginMetadata(
        plugin_id="com.example.test",
        name="Test",
        version="1.0.0",
        description="Test",
        author="Test",
        plugin_type=PluginType.TOOL,
        entry_point="test",
    )

    # Should serialize without errors
    json_str = metadata.model_dump_json()
    assert json_str is not None

    # Should deserialize
    loaded = PluginMetadata.model_validate_json(json_str)
    assert loaded.plugin_id == metadata.plugin_id
