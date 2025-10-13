"""Plugin registry and marketplace integration service."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import structlog

from ..models.plugin import (
    PluginLoadError,
    PluginMarketplaceInfo,
    PluginMetadata,
    PluginType,
    PluginValidationResult,
)
from .plugin_validator import PluginValidator

logger = structlog.get_logger()


class PluginRegistry:
    """Service for managing plugin registry and marketplace integration."""

    def __init__(
        self,
        plugin_directory: Path,
        validator: PluginValidator,
        marketplace_url: str = "https://plugins.agentcore.io",
        enable_marketplace: bool = True,
    ) -> None:
        """
        Initialize plugin registry.

        Args:
            plugin_directory: Directory for storing plugins
            validator: Plugin security validator
            marketplace_url: Marketplace API base URL
            enable_marketplace: Enable marketplace integration
        """
        self._plugin_directory = plugin_directory
        self._validator = validator
        self._marketplace_url = marketplace_url
        self._enable_marketplace = enable_marketplace
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Ensure plugin directory exists
        self._plugin_directory.mkdir(parents=True, exist_ok=True)

        logger.info(
            "plugin_registry_initialized",
            plugin_directory=str(plugin_directory),
            marketplace_url=marketplace_url,
            marketplace_enabled=enable_marketplace,
        )

    async def search_marketplace(
        self,
        query: str = "",
        plugin_type: PluginType | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[PluginMarketplaceInfo]:
        """
        Search marketplace for plugins.

        Args:
            query: Search query (plugin name, description)
            plugin_type: Filter by plugin type
            tags: Filter by tags
            limit: Maximum results

        Returns:
            List of marketplace plugin info

        Raises:
            RuntimeError: If marketplace is disabled
            httpx.HTTPError: If marketplace request fails
        """
        if not self._enable_marketplace:
            raise RuntimeError("Marketplace integration is disabled")

        logger.info(
            "searching_marketplace",
            query=query,
            plugin_type=plugin_type,
            tags=tags,
            limit=limit,
        )

        # Build search parameters
        params: dict[str, Any] = {
            "limit": limit,
        }

        if query:
            params["q"] = query

        if plugin_type:
            params["type"] = plugin_type.value

        if tags:
            params["tags"] = ",".join(tags)

        try:
            # Make marketplace API request
            response = await self._http_client.get(
                urljoin(self._marketplace_url, "/api/v1/plugins/search"),
                params=params,
            )
            response.raise_for_status()

            # Parse results
            data = response.json()
            results = [
                PluginMarketplaceInfo(**item) for item in data.get("plugins", [])
            ]

            logger.info(
                "marketplace_search_complete",
                query=query,
                results_count=len(results),
            )

            return results

        except httpx.HTTPError as e:
            logger.error(
                "marketplace_search_failed",
                error=str(e),
                status_code=getattr(e.response, "status_code", None),
            )
            raise

    async def download_plugin(
        self,
        plugin_id: str,
        version: str | None = None,
        validate: bool = True,
    ) -> Path:
        """
        Download plugin from marketplace.

        Args:
            plugin_id: Plugin identifier
            version: Specific version (uses latest if None)
            validate: Validate plugin after download

        Returns:
            Path to downloaded plugin directory

        Raises:
            RuntimeError: If marketplace is disabled
            PluginLoadError: If download or validation fails
        """
        if not self._enable_marketplace:
            raise RuntimeError("Marketplace integration is disabled")

        logger.info(
            "downloading_plugin",
            plugin_id=plugin_id,
            version=version,
        )

        try:
            # Get plugin info from marketplace
            info = await self._get_marketplace_info(plugin_id, version)

            # Download plugin package
            temp_dir = Path(tempfile.mkdtemp(prefix="plugin_download_"))

            try:
                await self._download_package(
                    download_url=info.download_url,
                    dest_dir=temp_dir,
                )

                # Verify checksum
                actual_checksum = await self._calculate_directory_checksum(
                    temp_dir
                )

                if actual_checksum != info.checksum:
                    raise PluginLoadError(
                        plugin_id=plugin_id,
                        message=f"Checksum mismatch: expected {info.checksum}, got {actual_checksum}",
                    )

                # Load and validate metadata
                manifest_path = temp_dir / "plugin.json"
                if not manifest_path.exists():
                    raise PluginLoadError(
                        plugin_id=plugin_id,
                        message="Downloaded plugin missing plugin.json",
                    )

                with manifest_path.open("r") as f:
                    metadata = PluginMetadata(**json.load(f))

                # Validate plugin if requested
                if validate:
                    validation_result = await self._validator.validate_plugin(
                        plugin_path=temp_dir,
                        metadata=metadata,
                        expected_checksum=info.checksum,
                    )

                    if not validation_result.valid:
                        raise PluginLoadError(
                            plugin_id=plugin_id,
                            message=f"Plugin validation failed: {'; '.join(validation_result.errors)}",
                        )

                # Move to plugin directory
                final_path = self._plugin_directory / plugin_id
                if final_path.exists():
                    shutil.rmtree(final_path)

                shutil.move(str(temp_dir), str(final_path))

                logger.info(
                    "plugin_downloaded_successfully",
                    plugin_id=plugin_id,
                    version=metadata.version,
                    path=str(final_path),
                )

                return final_path

            finally:
                # Cleanup temp directory if it still exists
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            logger.error(
                "plugin_download_failed",
                plugin_id=plugin_id,
                error=str(e),
            )

            if isinstance(e, PluginLoadError):
                raise
            raise PluginLoadError(
                plugin_id=plugin_id,
                message=f"Download failed: {e}",
                original_error=e,
            ) from e

    async def install_plugin(
        self,
        plugin_id: str,
        version: str | None = None,
    ) -> Path:
        """
        Install plugin from marketplace (download + verify).

        Args:
            plugin_id: Plugin identifier
            version: Specific version (uses latest if None)

        Returns:
            Path to installed plugin directory
        """
        logger.info("installing_plugin", plugin_id=plugin_id, version=version)

        # Download and validate
        plugin_path = await self.download_plugin(
            plugin_id=plugin_id,
            version=version,
            validate=True,
        )

        logger.info(
            "plugin_installed_successfully",
            plugin_id=plugin_id,
            path=str(plugin_path),
        )

        return plugin_path

    async def uninstall_plugin(self, plugin_id: str) -> None:
        """
        Uninstall plugin (remove from directory).

        Args:
            plugin_id: Plugin identifier

        Raises:
            FileNotFoundError: If plugin not found
        """
        logger.info("uninstalling_plugin", plugin_id=plugin_id)

        plugin_path = self._plugin_directory / plugin_id

        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin {plugin_id} not installed")

        shutil.rmtree(plugin_path)

        logger.info("plugin_uninstalled_successfully", plugin_id=plugin_id)

    async def list_installed_plugins(self) -> list[PluginMetadata]:
        """
        List all installed plugins.

        Returns:
            List of installed plugin metadata
        """
        installed: list[PluginMetadata] = []

        for plugin_dir in self._plugin_directory.iterdir():
            if not plugin_dir.is_dir():
                continue

            manifest_path = plugin_dir / "plugin.json"
            if not manifest_path.exists():
                continue

            try:
                with manifest_path.open("r") as f:
                    metadata = PluginMetadata(**json.load(f))
                    installed.append(metadata)
            except Exception as e:
                logger.warning(
                    "failed_to_load_plugin_metadata",
                    plugin_dir=str(plugin_dir),
                    error=str(e),
                )

        return installed

    async def check_updates(self) -> dict[str, str]:
        """
        Check for available updates for installed plugins.

        Returns:
            Dict of plugin_id -> latest_version for plugins with updates

        Raises:
            RuntimeError: If marketplace is disabled
        """
        if not self._enable_marketplace:
            raise RuntimeError("Marketplace integration is disabled")

        logger.info("checking_plugin_updates")

        updates: dict[str, str] = {}
        installed = await self.list_installed_plugins()

        for metadata in installed:
            try:
                # Get latest version from marketplace
                info = await self._get_marketplace_info(metadata.plugin_id)

                # Compare versions (simple string comparison for now)
                if info.version != metadata.version:
                    updates[metadata.plugin_id] = info.version

            except Exception as e:
                logger.warning(
                    "failed_to_check_update",
                    plugin_id=metadata.plugin_id,
                    error=str(e),
                )

        logger.info("update_check_complete", updates_count=len(updates))

        return updates

    async def _get_marketplace_info(
        self,
        plugin_id: str,
        version: str | None = None,
    ) -> PluginMarketplaceInfo:
        """Get plugin info from marketplace API."""
        endpoint = f"/api/v1/plugins/{plugin_id}"

        if version:
            endpoint += f"/versions/{version}"

        response = await self._http_client.get(
            urljoin(self._marketplace_url, endpoint)
        )
        response.raise_for_status()

        data = response.json()
        return PluginMarketplaceInfo(**data)

    async def _download_package(
        self,
        download_url: str,
        dest_dir: Path,
    ) -> None:
        """Download and extract plugin package."""
        # For simplicity, assume package is a zip or tar.gz
        # In production, this would handle different archive formats

        temp_file = dest_dir / "package.zip"

        # Download file
        async with self._http_client.stream("GET", download_url) as response:
            response.raise_for_status()

            with temp_file.open("wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)

        # Extract archive
        import zipfile

        if zipfile.is_zipfile(temp_file):
            with zipfile.ZipFile(temp_file, "r") as zip_ref:
                zip_ref.extractall(dest_dir)

        # Remove archive
        temp_file.unlink()

    async def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate SHA-256 checksum of all files in directory."""
        sha256 = hashlib.sha256()

        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.name != "package.zip":
                with file_path.open("rb") as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)

        return sha256.hexdigest()

    async def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        await self._http_client.aclose()
