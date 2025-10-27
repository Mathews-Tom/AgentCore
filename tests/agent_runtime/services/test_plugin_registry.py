"""
Unit tests for Plugin Registry service.

Tests for marketplace integration, plugin download/install, and registry management.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from zipfile import ZipFile

import httpx
import pytest

from agentcore.agent_runtime.models.plugin import (
    PluginLoadError,
    PluginMarketplaceInfo,
    PluginMetadata,
    PluginType,
    PluginValidationResult)
from agentcore.agent_runtime.services.plugin_registry import PluginRegistry
from agentcore.agent_runtime.services.plugin_validator import PluginValidator


@pytest.fixture
def temp_plugin_dir(tmp_path):
    """Create temporary plugin directory."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    return plugin_dir


@pytest.fixture
def mock_validator():
    """Create mock plugin validator."""
    validator = Mock(spec=PluginValidator)
    validator.validate_plugin = AsyncMock(
        return_value=PluginValidationResult(
            valid=True,
            security_score=95.0,
            risk_level="low")
    )
    return validator


@pytest.fixture
async def plugin_registry(temp_plugin_dir, mock_validator):
    """Create plugin registry instance."""
    registry = PluginRegistry(
        plugin_directory=temp_plugin_dir,
        validator=mock_validator,
        marketplace_url="https://test.marketplace.io",
        enable_marketplace=True)
    yield registry
    await registry.close()


@pytest.fixture
def sample_plugin_metadata():
    """Sample plugin metadata for testing."""
    return {
        "plugin_id": "com.example.test",
        "name": "Test Plugin",
        "version": "1.0.0",
        "description": "Test plugin for unit tests",
        "author": "Test Author",
        "license": "MIT",
        "plugin_type": "tool",
        "entry_point": "test_plugin.main",
    }


@pytest.fixture
def sample_marketplace_info():
    """Sample marketplace info for testing."""
    return {
        "plugin_id": "com.example.test",
        "marketplace_url": "https://test.marketplace.io/plugins/com.example.test",
        "download_url": "https://test.marketplace.io/download/com.example.test.zip",
        "version": "1.0.0",
        "checksum": "abc123def456",
        "downloads_count": 100,
        "rating": 4.5,
        "verified": True,
        "last_updated": datetime.now(UTC).isoformat(),
    }


class TestPluginRegistryInitialization:
    """Test PluginRegistry initialization."""

    async def test_init_creates_directory(self, tmp_path, mock_validator):
        """Test that initialization creates plugin directory."""
        plugin_dir = tmp_path / "new_plugins"
        assert not plugin_dir.exists()

        registry = PluginRegistry(
            plugin_directory=plugin_dir,
            validator=mock_validator)

        assert plugin_dir.exists()
        assert plugin_dir.is_dir()
        await registry.close()

    async def test_init_with_custom_marketplace(self, tmp_path, mock_validator):
        """Test initialization with custom marketplace URL."""
        registry = PluginRegistry(
            plugin_directory=tmp_path / "plugins",
            validator=mock_validator,
            marketplace_url="https://custom.marketplace.io",
            enable_marketplace=False)

        assert registry._marketplace_url == "https://custom.marketplace.io"
        assert registry._enable_marketplace is False
        await registry.close()


class TestMarketplaceSearch:
    """Test marketplace search functionality."""

    @pytest.mark.asyncio
    async def test_search_marketplace_basic(
        self, plugin_registry, sample_marketplace_info
    ):
        """Test basic marketplace search."""
        with patch.object(
            plugin_registry._http_client,
            "get",
            new_callable=AsyncMock) as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "plugins": [sample_marketplace_info],
            }
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            results = await plugin_registry.search_marketplace(query="test")

            assert len(results) == 1
            assert results[0].plugin_id == "com.example.test"
            assert results[0].version == "1.0.0"
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_marketplace_with_filters(
        self, plugin_registry, sample_marketplace_info
    ):
        """Test marketplace search with type and tags filters."""
        with patch.object(
            plugin_registry._http_client,
            "get",
            new_callable=AsyncMock) as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"plugins": [sample_marketplace_info]}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            await plugin_registry.search_marketplace(
                query="analytics",
                plugin_type=PluginType.TOOL,
                tags=["data", "visualization"],
                limit=10)

            call_args = mock_get.call_args
            params = call_args.kwargs["params"]
            assert params["q"] == "analytics"
            assert params["type"] == "tool"
            assert params["tags"] == "data,visualization"
            assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_search_marketplace_disabled(self, temp_plugin_dir, mock_validator):
        """Test that search fails when marketplace is disabled."""
        registry = PluginRegistry(
            plugin_directory=temp_plugin_dir,
            validator=mock_validator,
            enable_marketplace=False)

        with pytest.raises(RuntimeError, match="Marketplace integration is disabled"):
            await registry.search_marketplace(query="test")

        await registry.close()

    @pytest.mark.asyncio
    async def test_search_marketplace_http_error(self, plugin_registry):
        """Test handling of HTTP errors during search."""
        with patch.object(
            plugin_registry._http_client,
            "get",
            new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError(
                message="Not found",
                request=Mock(),
                response=Mock(status_code=404))

            with pytest.raises(httpx.HTTPStatusError):
                await plugin_registry.search_marketplace(query="nonexistent")

    @pytest.mark.asyncio
    async def test_search_marketplace_empty_results(self, plugin_registry):
        """Test search with no results."""
        with patch.object(
            plugin_registry._http_client,
            "get",
            new_callable=AsyncMock) as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"plugins": []}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            results = await plugin_registry.search_marketplace(query="nonexistent")

            assert len(results) == 0


class TestPluginDownload:
    """Test plugin download functionality."""

    @pytest.mark.asyncio
    async def test_download_plugin_success(
        self,
        plugin_registry,
        sample_plugin_metadata,
        sample_marketplace_info,
        tmp_path):
        """Test successful plugin download."""
        # Create mock plugin package
        temp_zip = tmp_path / "test_plugin.zip"
        plugin_dir = tmp_path / "plugin_content"
        plugin_dir.mkdir()

        # Create plugin.json
        manifest_path = plugin_dir / "plugin.json"
        manifest_path.write_text(json.dumps(sample_plugin_metadata))

        # Create test file
        (plugin_dir / "main.py").write_text("# Test plugin")

        # Create zip file
        with ZipFile(temp_zip, "w") as zipf:
            zipf.write(manifest_path, "plugin.json")
            zipf.write(plugin_dir / "main.py", "main.py")

        # Mock marketplace info
        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.return_value = PluginMarketplaceInfo(
                **sample_marketplace_info
            )

            # Mock download
            with patch.object(
                plugin_registry,
                "_download_package",
                new_callable=AsyncMock) as mock_download:

                async def mock_download_impl(download_url, dest_dir):
                    # Extract zip to dest_dir
                    with ZipFile(temp_zip, "r") as zipf:
                        zipf.extractall(dest_dir)

                mock_download.side_effect = mock_download_impl

                # Mock checksum calculation
                with patch.object(
                    plugin_registry,
                    "_calculate_directory_checksum",
                    new_callable=AsyncMock) as mock_checksum:
                    mock_checksum.return_value = "abc123def456"

                    result_path = await plugin_registry.download_plugin(
                        plugin_id="com.example.test"
                    )

                    assert result_path.exists()
                    assert result_path.name == "com.example.test"
                    assert (result_path / "plugin.json").exists()

    @pytest.mark.asyncio
    async def test_download_plugin_marketplace_disabled(
        self, temp_plugin_dir, mock_validator
    ):
        """Test download fails when marketplace disabled."""
        registry = PluginRegistry(
            plugin_directory=temp_plugin_dir,
            validator=mock_validator,
            enable_marketplace=False)

        with pytest.raises(RuntimeError, match="Marketplace integration is disabled"):
            await registry.download_plugin(plugin_id="com.example.test")

        await registry.close()

    @pytest.mark.asyncio
    async def test_download_plugin_checksum_mismatch(
        self, plugin_registry, sample_marketplace_info
    ):
        """Test download fails on checksum mismatch."""
        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.return_value = PluginMarketplaceInfo(
                **sample_marketplace_info
            )

            with patch.object(
                plugin_registry,
                "_download_package",
                new_callable=AsyncMock):
                with patch.object(
                    plugin_registry,
                    "_calculate_directory_checksum",
                    new_callable=AsyncMock) as mock_checksum:
                    mock_checksum.return_value = "wrong_checksum"

                    with pytest.raises(PluginLoadError, match="Checksum mismatch"):
                        await plugin_registry.download_plugin(
                            plugin_id="com.example.test"
                        )

    @pytest.mark.asyncio
    async def test_download_plugin_missing_manifest(
        self, plugin_registry, sample_marketplace_info, tmp_path
    ):
        """Test download fails when plugin.json is missing."""
        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.return_value = PluginMarketplaceInfo(
                **sample_marketplace_info
            )

            with patch.object(
                plugin_registry,
                "_download_package",
                new_callable=AsyncMock):
                with patch.object(
                    plugin_registry,
                    "_calculate_directory_checksum",
                    new_callable=AsyncMock) as mock_checksum:
                    mock_checksum.return_value = "abc123def456"

                    with pytest.raises(PluginLoadError, match="missing plugin.json"):
                        await plugin_registry.download_plugin(
                            plugin_id="com.example.test"
                        )

    @pytest.mark.asyncio
    async def test_download_plugin_validation_failure(
        self, plugin_registry, sample_plugin_metadata, sample_marketplace_info, tmp_path
    ):
        """Test download fails on validation failure."""
        # Create mock plugin package with manifest
        temp_zip = tmp_path / "test_plugin.zip"
        plugin_dir = tmp_path / "plugin_content"
        plugin_dir.mkdir()

        manifest_path = plugin_dir / "plugin.json"
        manifest_path.write_text(json.dumps(sample_plugin_metadata))

        with ZipFile(temp_zip, "w") as zipf:
            zipf.write(manifest_path, "plugin.json")

        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.return_value = PluginMarketplaceInfo(
                **sample_marketplace_info
            )

            with patch.object(
                plugin_registry,
                "_download_package",
                new_callable=AsyncMock) as mock_download:

                async def mock_download_impl(download_url, dest_dir):
                    with ZipFile(temp_zip, "r") as zipf:
                        zipf.extractall(dest_dir)

                mock_download.side_effect = mock_download_impl

                with patch.object(
                    plugin_registry,
                    "_calculate_directory_checksum",
                    new_callable=AsyncMock) as mock_checksum:
                    mock_checksum.return_value = "abc123def456"

                    # Make validator return invalid result
                    plugin_registry._validator.validate_plugin.return_value = (
                        PluginValidationResult(
                            valid=False,
                            errors=["Security vulnerability detected"],
                            security_score=20.0,
                            risk_level="high")
                    )

                    with pytest.raises(
                        PluginLoadError, match="Plugin validation failed"
                    ):
                        await plugin_registry.download_plugin(
                            plugin_id="com.example.test"
                        )

    @pytest.mark.asyncio
    async def test_download_plugin_skip_validation(
        self, plugin_registry, sample_plugin_metadata, sample_marketplace_info, tmp_path
    ):
        """Test downloading plugin without validation."""
        # Create mock plugin package
        temp_zip = tmp_path / "test_plugin.zip"
        plugin_dir = tmp_path / "plugin_content"
        plugin_dir.mkdir()

        manifest_path = plugin_dir / "plugin.json"
        manifest_path.write_text(json.dumps(sample_plugin_metadata))

        with ZipFile(temp_zip, "w") as zipf:
            zipf.write(manifest_path, "plugin.json")

        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.return_value = PluginMarketplaceInfo(
                **sample_marketplace_info
            )

            with patch.object(
                plugin_registry,
                "_download_package",
                new_callable=AsyncMock) as mock_download:

                async def mock_download_impl(download_url, dest_dir):
                    with ZipFile(temp_zip, "r") as zipf:
                        zipf.extractall(dest_dir)

                mock_download.side_effect = mock_download_impl

                with patch.object(
                    plugin_registry,
                    "_calculate_directory_checksum",
                    new_callable=AsyncMock) as mock_checksum:
                    mock_checksum.return_value = "abc123def456"

                    result_path = await plugin_registry.download_plugin(
                        plugin_id="com.example.test",
                        validate=False)

                    assert result_path.exists()
                    # Validator should not have been called
                    plugin_registry._validator.validate_plugin.assert_not_called()


class TestPluginInstallUninstall:
    """Test plugin install and uninstall functionality."""

    @pytest.mark.asyncio
    async def test_install_plugin(self, plugin_registry):
        """Test plugin installation (download + validate)."""
        with patch.object(
            plugin_registry,
            "download_plugin",
            new_callable=AsyncMock) as mock_download:
            mock_download.return_value = Path("/fake/path/com.example.test")

            result = await plugin_registry.install_plugin(
                plugin_id="com.example.test",
                version="1.0.0")

            assert result == Path("/fake/path/com.example.test")
            mock_download.assert_called_once_with(
                plugin_id="com.example.test",
                version="1.0.0",
                validate=True)

    @pytest.mark.asyncio
    async def test_uninstall_plugin_success(
        self, plugin_registry, sample_plugin_metadata
    ):
        """Test successful plugin uninstallation."""
        # Create fake installed plugin
        plugin_dir = plugin_registry._plugin_directory / "com.example.test"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(json.dumps(sample_plugin_metadata))

        assert plugin_dir.exists()

        await plugin_registry.uninstall_plugin(plugin_id="com.example.test")

        assert not plugin_dir.exists()

    @pytest.mark.asyncio
    async def test_uninstall_plugin_not_found(self, plugin_registry):
        """Test uninstall fails when plugin not installed."""
        with pytest.raises(FileNotFoundError, match="not installed"):
            await plugin_registry.uninstall_plugin(plugin_id="nonexistent.plugin")


class TestListInstalledPlugins:
    """Test listing installed plugins."""

    @pytest.mark.asyncio
    async def test_list_installed_plugins_empty(self, plugin_registry):
        """Test listing when no plugins installed."""
        installed = await plugin_registry.list_installed_plugins()

        assert len(installed) == 0

    @pytest.mark.asyncio
    async def test_list_installed_plugins(
        self, plugin_registry, sample_plugin_metadata
    ):
        """Test listing installed plugins."""
        # Create multiple plugins
        for i in range(3):
            plugin_data = sample_plugin_metadata.copy()
            plugin_data["plugin_id"] = f"com.example.plugin{i}"
            plugin_data["version"] = f"1.{i}.0"

            plugin_dir = plugin_registry._plugin_directory / f"com.example.plugin{i}"
            plugin_dir.mkdir()
            (plugin_dir / "plugin.json").write_text(json.dumps(plugin_data))

        installed = await plugin_registry.list_installed_plugins()

        assert len(installed) == 3
        assert all(isinstance(p, PluginMetadata) for p in installed)
        plugin_ids = [p.plugin_id for p in installed]
        assert "com.example.plugin0" in plugin_ids
        assert "com.example.plugin1" in plugin_ids
        assert "com.example.plugin2" in plugin_ids

    @pytest.mark.asyncio
    async def test_list_installed_plugins_ignores_invalid(
        self, plugin_registry, sample_plugin_metadata
    ):
        """Test that invalid plugins are skipped with warning."""
        # Create valid plugin
        plugin_dir1 = plugin_registry._plugin_directory / "com.example.valid"
        plugin_dir1.mkdir()
        (plugin_dir1 / "plugin.json").write_text(json.dumps(sample_plugin_metadata))

        # Create invalid plugin (bad JSON)
        plugin_dir2 = plugin_registry._plugin_directory / "com.example.invalid"
        plugin_dir2.mkdir()
        (plugin_dir2 / "plugin.json").write_text("invalid json{")

        # Create directory without manifest
        plugin_dir3 = plugin_registry._plugin_directory / "com.example.no_manifest"
        plugin_dir3.mkdir()

        # Create non-directory file
        (plugin_registry._plugin_directory / "random_file.txt").write_text("test")

        installed = await plugin_registry.list_installed_plugins()

        # Only the valid plugin should be returned
        assert len(installed) == 1
        assert installed[0].plugin_id == "com.example.test"


class TestCheckUpdates:
    """Test checking for plugin updates."""

    @pytest.mark.asyncio
    async def test_check_updates_available(
        self, plugin_registry, sample_plugin_metadata, sample_marketplace_info
    ):
        """Test checking updates when updates are available."""
        # Install plugin with older version
        plugin_data = sample_plugin_metadata.copy()
        plugin_data["version"] = "0.9.0"
        plugin_dir = plugin_registry._plugin_directory / "com.example.test"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(json.dumps(plugin_data))

        # Mock marketplace response with newer version
        marketplace_info = sample_marketplace_info.copy()
        marketplace_info["version"] = "1.0.0"

        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.return_value = PluginMarketplaceInfo(**marketplace_info)

            updates = await plugin_registry.check_updates()

            assert len(updates) == 1
            assert "com.example.test" in updates
            assert updates["com.example.test"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_check_updates_none_available(
        self, plugin_registry, sample_plugin_metadata, sample_marketplace_info
    ):
        """Test checking updates when no updates available."""
        # Install plugin with current version
        plugin_dir = plugin_registry._plugin_directory / "com.example.test"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(json.dumps(sample_plugin_metadata))

        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.return_value = PluginMarketplaceInfo(
                **sample_marketplace_info
            )

            updates = await plugin_registry.check_updates()

            assert len(updates) == 0

    @pytest.mark.asyncio
    async def test_check_updates_marketplace_disabled(
        self, temp_plugin_dir, mock_validator
    ):
        """Test check updates fails when marketplace disabled."""
        registry = PluginRegistry(
            plugin_directory=temp_plugin_dir,
            validator=mock_validator,
            enable_marketplace=False)

        with pytest.raises(RuntimeError, match="Marketplace integration is disabled"):
            await registry.check_updates()

        await registry.close()

    @pytest.mark.asyncio
    async def test_check_updates_handles_errors(
        self, plugin_registry, sample_plugin_metadata
    ):
        """Test that individual plugin check errors are handled gracefully."""
        # Install two plugins
        for i in range(2):
            plugin_data = sample_plugin_metadata.copy()
            plugin_data["plugin_id"] = f"com.example.plugin{i}"
            plugin_dir = plugin_registry._plugin_directory / f"com.example.plugin{i}"
            plugin_dir.mkdir()
            (plugin_dir / "plugin.json").write_text(json.dumps(plugin_data))

        # Mock marketplace to fail for first plugin, succeed for second
        call_count = 0

        async def mock_get_info_side_effect(plugin_id, version=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.HTTPError("Network error")
            return PluginMarketplaceInfo(**sample_marketplace_info)

        with patch.object(
            plugin_registry,
            "_get_marketplace_info",
            new_callable=AsyncMock) as mock_get_info:
            mock_get_info.side_effect = mock_get_info_side_effect

            updates = await plugin_registry.check_updates()

            # Should continue checking despite first failure
            assert isinstance(updates, dict)


class TestHelperMethods:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_get_marketplace_info_success(
        self, plugin_registry, sample_marketplace_info
    ):
        """Test getting marketplace info."""
        with patch.object(
            plugin_registry._http_client,
            "get",
            new_callable=AsyncMock) as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_marketplace_info
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            info = await plugin_registry._get_marketplace_info(
                plugin_id="com.example.test"
            )

            assert info.plugin_id == "com.example.test"
            assert info.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_marketplace_info_with_version(
        self, plugin_registry, sample_marketplace_info
    ):
        """Test getting specific version from marketplace."""
        with patch.object(
            plugin_registry._http_client,
            "get",
            new_callable=AsyncMock) as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_marketplace_info
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            await plugin_registry._get_marketplace_info(
                plugin_id="com.example.test",
                version="1.0.0")

            call_args = mock_get.call_args
            url = call_args[0][0]
            assert "/versions/1.0.0" in url

    @pytest.mark.asyncio
    async def test_calculate_directory_checksum(self, plugin_registry, tmp_path):
        """Test directory checksum calculation."""
        test_dir = tmp_path / "test_checksum"
        test_dir.mkdir()

        # Create test files
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file3.txt").write_text("content3")

        checksum = await plugin_registry._calculate_directory_checksum(test_dir)

        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest length

        # Same directory should produce same checksum
        checksum2 = await plugin_registry._calculate_directory_checksum(test_dir)
        assert checksum == checksum2

    @pytest.mark.asyncio
    async def test_close_http_client(self, plugin_registry):
        """Test closing HTTP client."""
        with patch.object(
            plugin_registry._http_client,
            "aclose",
            new_callable=AsyncMock) as mock_close:
            await plugin_registry.close()

            mock_close.assert_called_once()


class TestDownloadPackage:
    """Test package download and extraction."""

    @pytest.mark.asyncio
    async def test_download_package_success(self, plugin_registry, tmp_path):
        """Test successful package download and extraction."""
        # Create a test zip file
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        (source_dir / "test.txt").write_text("test content")

        zip_path = tmp_path / "test.zip"
        with ZipFile(zip_path, "w") as zipf:
            zipf.write(source_dir / "test.txt", "test.txt")

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        # Mock streaming response
        class MockResponse:
            def __init__(self, zip_content):
                self.zip_content = zip_content
                self.status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_bytes(self, chunk_size):
                yield self.zip_content

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        with open(zip_path, "rb") as f:
            zip_content = f.read()

        with patch.object(
            plugin_registry._http_client,
            "stream",
            return_value=MockResponse(zip_content)):
            await plugin_registry._download_package(
                download_url="https://test.com/plugin.zip",
                dest_dir=dest_dir)

            # Verify extraction
            assert (dest_dir / "test.txt").exists()
            assert (dest_dir / "test.txt").read_text() == "test content"
            assert not (dest_dir / "package.zip").exists()  # Archive should be removed
