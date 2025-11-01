"""Azure Blob Storage adapter implementation.

Provides Azure Blob-specific implementation with azure-storage-blob
for file upload/download, metadata management, and signed URL generation.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator

import structlog

from agentcore.integration.storage.base import (
    AccessControl,
    AccessLevel,
    DownloadResult,
    StorageAdapter,
    StorageConfig,
    StorageMetadata,
    StorageObject,
    UploadResult,
)

logger = structlog.get_logger(__name__)


class AzureBlobAdapter(StorageAdapter):
    """Azure Blob Storage adapter with azure-storage-blob.

    High-performance async Azure Blob adapter using azure-storage-blob (async)
    for async operations with streaming support and SAS token generation.
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize Azure Blob adapter.

        Args:
            config: Storage configuration
        """
        super().__init__(config)
        self._client: Any = None
        self._container_client: Any = None

        logger.info(
            "azure_blob_adapter_initialized",
            container=config.bucket_name,
        )

    async def connect(self) -> None:
        """Establish connection to Azure Blob Storage."""
        if self._connected:
            return

        try:
            # Import azure-storage-blob for async operations
            try:
                from azure.storage.blob.aio import BlobServiceClient
            except ImportError as e:
                raise ImportError(
                    "azure-storage-blob is required for Azure Blob adapter. "
                    "Install it with: pip install azure-storage-blob"
                ) from e

            # Create service client
            if self.config.connection_string:
                connection_string = self.config.connection_string.get_secret_value()
                self._client = BlobServiceClient.from_connection_string(
                    connection_string
                )
            else:
                # Use account key authentication
                account_url = f"https://{self.config.access_key}.blob.core.windows.net"
                from azure.storage.blob import BlobServiceClient as SyncClient

                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=self.config.secret_key.get_secret_value()
                    if self.config.secret_key
                    else None,
                )

            # Get container client
            self._container_client = self._client.get_container_client(
                self.config.bucket_name
            )

            # Test connection by checking container exists
            exists = await self._container_client.exists()
            if not exists:
                # Create container if it doesn't exist
                await self._container_client.create_container()

            self._connected = True

            logger.info(
                "azure_blob_connected",
                container=self.config.bucket_name,
            )

        except Exception as e:
            logger.error(
                "azure_blob_connection_failed",
                error=str(e),
                container=self.config.bucket_name,
            )
            raise

    async def disconnect(self) -> None:
        """Close Azure Blob connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

        self._container_client = None
        self._connected = False

        logger.info(
            "azure_blob_disconnected",
            container=self.config.bucket_name,
        )

    async def upload_file(
        self,
        key: str,
        content: bytes,
        metadata: StorageMetadata | None = None,
        access_control: AccessControl | None = None,
    ) -> UploadResult:
        """Upload file to Azure Blob Storage.

        Args:
            key: Blob name
            content: File content as bytes
            metadata: Optional metadata
            access_control: Optional access control

        Returns:
            Upload result with ETag and version
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        start_time = time.time()

        try:
            blob_client = self._container_client.get_blob_client(key)

            # Prepare upload parameters
            upload_kwargs: dict[str, Any] = {}

            if metadata:
                upload_kwargs["content_settings"] = {
                    "content_type": metadata.content_type,
                }
                if metadata.content_encoding:
                    upload_kwargs["content_settings"]["content_encoding"] = (
                        metadata.content_encoding
                    )
                if metadata.content_language:
                    upload_kwargs["content_settings"]["content_language"] = (
                        metadata.content_language
                    )
                if metadata.cache_control:
                    upload_kwargs["content_settings"]["cache_control"] = (
                        metadata.cache_control
                    )
                if metadata.content_disposition:
                    upload_kwargs["content_settings"]["content_disposition"] = (
                        metadata.content_disposition
                    )
                if metadata.custom_metadata:
                    upload_kwargs["metadata"] = metadata.custom_metadata

            # Azure Blob doesn't have direct ACL like S3
            # Public access is controlled at container level
            # We'll set public access for individual blobs if needed
            if access_control and access_control.access_level != AccessLevel.PRIVATE:
                # Note: Azure requires container-level public access settings
                logger.warning(
                    "azure_blob_public_access_requires_container_config",
                    key=key,
                )

            # Upload blob
            response = await blob_client.upload_blob(
                content,
                overwrite=True,
                **upload_kwargs,
            )

            upload_time_ms = int((time.time() - start_time) * 1000)

            # Get blob properties for version info
            properties = await blob_client.get_blob_properties()

            # Generate URL if needed
            url = None
            if access_control and access_control.access_level != AccessLevel.PRIVATE:
                url = await self.generate_signed_url(
                    key, expiry_seconds=access_control.signed_url_expiry
                )

            result = UploadResult(
                key=key,
                etag=response["etag"].strip('"'),
                version_id=properties.get("version_id"),
                size=len(content),
                upload_time_ms=upload_time_ms,
                url=url,
            )

            logger.info(
                "azure_blob_file_uploaded",
                key=key,
                size=len(content),
                upload_time_ms=upload_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "azure_blob_upload_failed",
                error=str(e),
                key=key,
                size=len(content),
            )
            raise

    async def upload_file_streaming(
        self,
        key: str,
        content: AsyncIterator[bytes],
        size: int,
        metadata: StorageMetadata | None = None,
        access_control: AccessControl | None = None,
    ) -> UploadResult:
        """Upload large file using streaming.

        Args:
            key: Blob name
            content: Async iterator of file chunks
            size: Total file size
            metadata: Optional metadata
            access_control: Optional access control

        Returns:
            Upload result with ETag and version
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        start_time = time.time()

        try:
            blob_client = self._container_client.get_blob_client(key)

            # Prepare upload parameters
            upload_kwargs: dict[str, Any] = {"length": size}

            if metadata:
                upload_kwargs["content_settings"] = {
                    "content_type": metadata.content_type,
                }
                if metadata.custom_metadata:
                    upload_kwargs["metadata"] = metadata.custom_metadata

            # Create async generator from iterator
            async def content_generator() -> AsyncIterator[bytes]:
                async for chunk in content:
                    yield chunk

            # Upload blob with streaming
            response = await blob_client.upload_blob(
                content_generator(),
                overwrite=True,
                **upload_kwargs,
            )

            upload_time_ms = int((time.time() - start_time) * 1000)

            # Get blob properties
            properties = await blob_client.get_blob_properties()

            result = UploadResult(
                key=key,
                etag=response["etag"].strip('"'),
                version_id=properties.get("version_id"),
                size=size,
                upload_time_ms=upload_time_ms,
                url=None,
            )

            logger.info(
                "azure_blob_file_uploaded_streaming",
                key=key,
                size=size,
                upload_time_ms=upload_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "azure_blob_streaming_upload_failed",
                error=str(e),
                key=key,
                size=size,
            )
            raise

    async def download_file(
        self,
        key: str,
        version_id: str | None = None,
        byte_range: tuple[int, int] | None = None,
    ) -> DownloadResult:
        """Download file from Azure Blob Storage.

        Args:
            key: Blob name
            version_id: Optional version to download
            byte_range: Optional byte range (start, end)

        Returns:
            Download result with content and metadata
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        start_time = time.time()

        try:
            blob_client = self._container_client.get_blob_client(key)

            # Prepare download parameters
            download_kwargs: dict[str, Any] = {}

            if version_id:
                download_kwargs["version_id"] = version_id

            if byte_range:
                download_kwargs["offset"] = byte_range[0]
                download_kwargs["length"] = byte_range[1] - byte_range[0] + 1

            # Download blob
            download_stream = await blob_client.download_blob(**download_kwargs)
            content = await download_stream.readall()
            properties = download_stream.properties

            download_time_ms = int((time.time() - start_time) * 1000)

            # Extract metadata
            content_settings = properties.content_settings or {}
            metadata = StorageMetadata(
                content_type=content_settings.get("content_type", "application/octet-stream"),
                content_encoding=content_settings.get("content_encoding"),
                content_language=content_settings.get("content_language"),
                cache_control=content_settings.get("cache_control"),
                content_disposition=content_settings.get("content_disposition"),
                custom_metadata=properties.metadata or {},
            )

            result = DownloadResult(
                key=key,
                content=content,
                metadata=metadata,
                version_id=properties.get("version_id"),
                etag=properties.etag.strip('"'),
                size=properties.size,
                download_time_ms=download_time_ms,
            )

            logger.info(
                "azure_blob_file_downloaded",
                key=key,
                size=len(content),
                download_time_ms=download_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "azure_blob_download_failed",
                error=str(e),
                key=key,
            )
            raise

    async def download_file_streaming(
        self,
        key: str,
        version_id: str | None = None,
        byte_range: tuple[int, int] | None = None,
    ) -> AsyncIterator[bytes]:
        """Download large file using streaming.

        Args:
            key: Blob name
            version_id: Optional version
            byte_range: Optional byte range

        Yields:
            File content chunks
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        try:
            blob_client = self._container_client.get_blob_client(key)

            download_kwargs: dict[str, Any] = {}

            if version_id:
                download_kwargs["version_id"] = version_id

            if byte_range:
                download_kwargs["offset"] = byte_range[0]
                download_kwargs["length"] = byte_range[1] - byte_range[0] + 1

            download_stream = await blob_client.download_blob(**download_kwargs)

            # Stream content in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            async for chunk in download_stream.chunks():
                yield chunk

            logger.info(
                "azure_blob_file_downloaded_streaming",
                key=key,
            )

        except Exception as e:
            logger.error(
                "azure_blob_streaming_download_failed",
                error=str(e),
                key=key,
            )
            raise

    async def delete_file(
        self,
        key: str,
        version_id: str | None = None,
    ) -> bool:
        """Delete file from Azure Blob Storage.

        Args:
            key: Blob name
            version_id: Optional version to delete

        Returns:
            True if deletion was successful
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        try:
            blob_client = self._container_client.get_blob_client(key)

            delete_kwargs: dict[str, Any] = {}
            if version_id:
                delete_kwargs["version_id"] = version_id

            await blob_client.delete_blob(**delete_kwargs)

            logger.info(
                "azure_blob_file_deleted",
                key=key,
                version_id=version_id,
            )

            return True

        except Exception as e:
            logger.error(
                "azure_blob_delete_failed",
                error=str(e),
                key=key,
            )
            raise

    async def list_files(
        self,
        prefix: str | None = None,
        max_results: int = 1000,
    ) -> list[StorageObject]:
        """List files in Azure Blob container.

        Args:
            prefix: Optional blob name prefix filter
            max_results: Maximum results

        Returns:
            List of storage objects
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        try:
            list_kwargs: dict[str, Any] = {}
            if prefix:
                list_kwargs["name_starts_with"] = prefix

            objects = []
            async for blob in self._container_client.list_blobs(**list_kwargs):
                if len(objects) >= max_results:
                    break

                # Extract metadata
                content_settings = blob.content_settings or {}
                metadata = StorageMetadata(
                    content_type=content_settings.get("content_type", "application/octet-stream"),
                    custom_metadata=blob.metadata or {},
                )

                storage_obj = StorageObject(
                    key=blob.name,
                    size=blob.size,
                    etag=blob.etag.strip('"'),
                    last_modified=blob.last_modified,
                    metadata=metadata,
                    version_id=blob.version_id,
                    storage_class=blob.blob_tier,
                )

                objects.append(storage_obj)

            logger.info(
                "azure_blob_files_listed",
                count=len(objects),
                prefix=prefix,
            )

            return objects

        except Exception as e:
            logger.error(
                "azure_blob_list_failed",
                error=str(e),
                prefix=prefix,
            )
            raise

    async def get_metadata(
        self,
        key: str,
        version_id: str | None = None,
    ) -> StorageObject:
        """Get blob metadata from Azure Blob Storage.

        Args:
            key: Blob name
            version_id: Optional version

        Returns:
            Storage object with metadata
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        try:
            blob_client = self._container_client.get_blob_client(key)

            kwargs: dict[str, Any] = {}
            if version_id:
                kwargs["version_id"] = version_id

            properties = await blob_client.get_blob_properties(**kwargs)

            # Extract metadata
            content_settings = properties.content_settings or {}
            metadata = StorageMetadata(
                content_type=content_settings.get("content_type", "application/octet-stream"),
                content_encoding=content_settings.get("content_encoding"),
                content_language=content_settings.get("content_language"),
                cache_control=content_settings.get("cache_control"),
                content_disposition=content_settings.get("content_disposition"),
                custom_metadata=properties.metadata or {},
            )

            storage_obj = StorageObject(
                key=key,
                size=properties.size,
                etag=properties.etag.strip('"'),
                last_modified=properties.last_modified,
                metadata=metadata,
                version_id=properties.get("version_id"),
                storage_class=properties.blob_tier,
            )

            logger.info(
                "azure_blob_metadata_retrieved",
                key=key,
            )

            return storage_obj

        except Exception as e:
            logger.error(
                "azure_blob_get_metadata_failed",
                error=str(e),
                key=key,
            )
            raise

    async def update_metadata(
        self,
        key: str,
        metadata: StorageMetadata,
        version_id: str | None = None,
    ) -> bool:
        """Update blob metadata in Azure Blob Storage.

        Args:
            key: Blob name
            metadata: New metadata
            version_id: Optional version

        Returns:
            True if update was successful
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        try:
            blob_client = self._container_client.get_blob_client(key)

            # Update content settings
            content_settings = {
                "content_type": metadata.content_type,
            }
            if metadata.content_encoding:
                content_settings["content_encoding"] = metadata.content_encoding
            if metadata.cache_control:
                content_settings["cache_control"] = metadata.cache_control

            await blob_client.set_http_headers(content_settings=content_settings)

            # Update custom metadata if provided
            if metadata.custom_metadata:
                await blob_client.set_blob_metadata(metadata=metadata.custom_metadata)

            logger.info(
                "azure_blob_metadata_updated",
                key=key,
            )

            return True

        except Exception as e:
            logger.error(
                "azure_blob_update_metadata_failed",
                error=str(e),
                key=key,
            )
            raise

    async def generate_signed_url(
        self,
        key: str,
        expiry_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate SAS token URL for temporary access.

        Args:
            key: Blob name
            expiry_seconds: URL expiration time
            method: HTTP method (GET, PUT, DELETE)

        Returns:
            SAS token URL
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        try:
            from azure.storage.blob import (
                BlobSasPermissions,
                generate_blob_sas,
            )

            blob_client = self._container_client.get_blob_client(key)

            # Map method to permissions
            permissions = BlobSasPermissions(read=False, write=False, delete=False)
            if method.upper() == "GET":
                permissions.read = True
            elif method.upper() == "PUT":
                permissions.write = True
            elif method.upper() == "DELETE":
                permissions.delete = True

            # Generate SAS token
            expiry = datetime.now(UTC) + timedelta(seconds=expiry_seconds)

            # Get account credentials
            account_name = self.config.access_key or ""
            account_key = (
                self.config.secret_key.get_secret_value()
                if self.config.secret_key
                else ""
            )

            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.config.bucket_name,
                blob_name=key,
                account_key=account_key,
                permission=permissions,
                expiry=expiry,
            )

            url = f"{blob_client.url}?{sas_token}"

            logger.info(
                "azure_blob_signed_url_generated",
                key=key,
                expiry_seconds=expiry_seconds,
                method=method,
            )

            return url

        except Exception as e:
            logger.error(
                "azure_blob_generate_signed_url_failed",
                error=str(e),
                key=key,
            )
            raise

    async def copy_file(
        self,
        source_key: str,
        destination_key: str,
        source_version_id: str | None = None,
    ) -> UploadResult:
        """Copy blob within Azure Blob Storage.

        Args:
            source_key: Source blob name
            destination_key: Destination blob name
            source_version_id: Optional source version

        Returns:
            Upload result for copied blob
        """
        if not self._connected or self._container_client is None:
            raise RuntimeError("Azure Blob not connected. Call connect() first.")

        start_time = time.time()

        try:
            source_blob = self._container_client.get_blob_client(source_key)
            dest_blob = self._container_client.get_blob_client(destination_key)

            # Get source URL
            source_url = source_blob.url
            if source_version_id:
                source_url = f"{source_url}?versionid={source_version_id}"

            # Start copy operation
            copy_result = await dest_blob.start_copy_from_url(source_url)

            # Wait for copy to complete
            properties = await dest_blob.get_blob_properties()
            while properties.copy.status == "pending":
                await asyncio.sleep(1)
                properties = await dest_blob.get_blob_properties()

            if properties.copy.status != "success":
                raise RuntimeError(
                    f"Copy failed with status: {properties.copy.status}"
                )

            upload_time_ms = int((time.time() - start_time) * 1000)

            result = UploadResult(
                key=destination_key,
                etag=properties.etag.strip('"'),
                version_id=properties.get("version_id"),
                size=properties.size,
                upload_time_ms=upload_time_ms,
                url=None,
            )

            logger.info(
                "azure_blob_file_copied",
                source_key=source_key,
                destination_key=destination_key,
                upload_time_ms=upload_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "azure_blob_copy_failed",
                error=str(e),
                source_key=source_key,
                destination_key=destination_key,
            )
            raise

    async def health_check(self) -> bool:
        """Check Azure Blob connection health.

        Returns:
            True if connection is healthy
        """
        if not self._connected or self._container_client is None:
            return False

        try:
            exists = await self._container_client.exists()
            return exists

        except Exception as e:
            logger.warning(
                "azure_blob_health_check_failed",
                error=str(e),
            )
            return False
