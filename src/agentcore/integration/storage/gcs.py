"""Google Cloud Storage adapter implementation.

Provides GCS-specific implementation with google-cloud-storage
for file upload/download, metadata management, and signed URL generation.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
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


class GCSAdapter(StorageAdapter):
    """Google Cloud Storage adapter with google-cloud-storage.

    High-performance async GCS adapter using google-cloud-storage
    with streaming support and signed URL generation.
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize GCS adapter.

        Args:
            config: Storage configuration
        """
        super().__init__(config)
        self._client: Any = None
        self._bucket: Any = None

        logger.info(
            "gcs_adapter_initialized",
            bucket=config.bucket_name,
        )

    async def connect(self) -> None:
        """Establish connection to Google Cloud Storage."""
        if self._connected:
            return

        try:
            # Import google-cloud-storage
            try:
                from google.cloud import storage
            except ImportError as e:
                raise ImportError(
                    "google-cloud-storage is required for GCS adapter. "
                    "Install it with: pip install google-cloud-storage"
                ) from e

            # Create client
            if self.config.credentials_file:
                self._client = storage.Client.from_service_account_json(
                    self.config.credentials_file
                )
            elif self.config.access_key and self.config.secret_key:
                # Use HMAC keys
                from google.auth.credentials import Credentials

                credentials = storage.Client.create_anonymous_client()
                self._client = storage.Client(
                    project=self.config.access_key,
                    credentials=credentials,
                )
            else:
                # Use default credentials
                self._client = storage.Client()

            # Get bucket
            self._bucket = self._client.bucket(self.config.bucket_name)

            # Test connection
            _ = self._bucket.exists()

            self._connected = True

            logger.info(
                "gcs_connected",
                bucket=self.config.bucket_name,
            )

        except Exception as e:
            logger.error(
                "gcs_connection_failed",
                error=str(e),
                bucket=self.config.bucket_name,
            )
            raise

    async def disconnect(self) -> None:
        """Close GCS connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

        self._bucket = None
        self._connected = False

        logger.info(
            "gcs_disconnected",
            bucket=self.config.bucket_name,
        )

    async def upload_file(
        self,
        key: str,
        content: bytes,
        metadata: StorageMetadata | None = None,
        access_control: AccessControl | None = None,
    ) -> UploadResult:
        """Upload file to GCS.

        Args:
            key: Object name
            content: File content as bytes
            metadata: Optional metadata
            access_control: Optional access control

        Returns:
            Upload result with ETag and generation
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        start_time = time.time()

        try:
            blob = self._bucket.blob(key)

            # Set metadata
            if metadata:
                blob.content_type = metadata.content_type
                if metadata.content_encoding:
                    blob.content_encoding = metadata.content_encoding
                if metadata.content_language:
                    blob.content_language = metadata.content_language
                if metadata.cache_control:
                    blob.cache_control = metadata.cache_control
                if metadata.content_disposition:
                    blob.content_disposition = metadata.content_disposition
                if metadata.custom_metadata:
                    blob.metadata = metadata.custom_metadata

            # Set ACL
            if access_control:
                if access_control.access_level == AccessLevel.PUBLIC_READ:
                    blob.make_public()

            # Upload blob
            # Note: google-cloud-storage doesn't have native async support
            # We'll use sync methods but wrap in executor for async compatibility
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                executor,
                blob.upload_from_string,
                content,
            )

            upload_time_ms = int((time.time() - start_time) * 1000)

            # Reload to get updated properties
            await loop.run_in_executor(executor, blob.reload)

            # Generate URL if needed
            url = None
            if access_control and access_control.access_level != AccessLevel.PRIVATE:
                if access_control.access_level == AccessLevel.PUBLIC_READ:
                    url = blob.public_url
                else:
                    url = await self.generate_signed_url(
                        key, expiry_seconds=access_control.signed_url_expiry
                    )

            result = UploadResult(
                key=key,
                etag=blob.etag,
                version_id=str(blob.generation) if blob.generation else None,
                size=len(content),
                upload_time_ms=upload_time_ms,
                url=url,
            )

            logger.info(
                "gcs_file_uploaded",
                key=key,
                size=len(content),
                upload_time_ms=upload_time_ms,
                generation=blob.generation,
            )

            return result

        except Exception as e:
            logger.error(
                "gcs_upload_failed",
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
        """Upload large file using resumable upload.

        Args:
            key: Object name
            content: Async iterator of file chunks
            size: Total file size
            metadata: Optional metadata
            access_control: Optional access control

        Returns:
            Upload result with ETag and generation
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        start_time = time.time()

        try:
            blob = self._bucket.blob(key)

            # Set metadata
            if metadata:
                blob.content_type = metadata.content_type
                if metadata.custom_metadata:
                    blob.metadata = metadata.custom_metadata

            # Collect chunks (GCS doesn't have native async streaming)
            chunks = []
            async for chunk in content:
                chunks.append(chunk)

            full_content = b"".join(chunks)

            # Upload using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                executor,
                blob.upload_from_string,
                full_content,
            )

            upload_time_ms = int((time.time() - start_time) * 1000)

            # Reload properties
            await loop.run_in_executor(executor, blob.reload)

            result = UploadResult(
                key=key,
                etag=blob.etag,
                version_id=str(blob.generation) if blob.generation else None,
                size=size,
                upload_time_ms=upload_time_ms,
                url=None,
            )

            logger.info(
                "gcs_file_uploaded_streaming",
                key=key,
                size=size,
                upload_time_ms=upload_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "gcs_streaming_upload_failed",
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
        """Download file from GCS.

        Args:
            key: Object name
            version_id: Optional generation to download
            byte_range: Optional byte range (start, end)

        Returns:
            Download result with content and metadata
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        start_time = time.time()

        try:
            # Get blob with specific generation if provided
            if version_id:
                blob = self._bucket.blob(key, generation=int(version_id))
            else:
                blob = self._bucket.blob(key)

            # Download using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            # Download content
            if byte_range:
                content = await loop.run_in_executor(
                    executor,
                    blob.download_as_bytes,
                    None,  # client
                    byte_range[0],  # start
                    byte_range[1],  # end
                )
            else:
                content = await loop.run_in_executor(
                    executor,
                    blob.download_as_bytes,
                )

            download_time_ms = int((time.time() - start_time) * 1000)

            # Reload to get metadata
            await loop.run_in_executor(executor, blob.reload)

            # Extract metadata
            metadata = StorageMetadata(
                content_type=blob.content_type or "application/octet-stream",
                content_encoding=blob.content_encoding,
                content_language=blob.content_language,
                cache_control=blob.cache_control,
                content_disposition=blob.content_disposition,
                custom_metadata=blob.metadata or {},
            )

            result = DownloadResult(
                key=key,
                content=content,
                metadata=metadata,
                version_id=str(blob.generation) if blob.generation else None,
                etag=blob.etag,
                size=blob.size,
                download_time_ms=download_time_ms,
            )

            logger.info(
                "gcs_file_downloaded",
                key=key,
                size=len(content),
                download_time_ms=download_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "gcs_download_failed",
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
            key: Object name
            version_id: Optional generation
            byte_range: Optional byte range

        Yields:
            File content chunks
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        try:
            # Download entire file (GCS doesn't have native async streaming)
            result = await self.download_file(key, version_id, byte_range)

            # Yield in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            for i in range(0, len(result.content), chunk_size):
                yield result.content[i : i + chunk_size]

            logger.info(
                "gcs_file_downloaded_streaming",
                key=key,
            )

        except Exception as e:
            logger.error(
                "gcs_streaming_download_failed",
                error=str(e),
                key=key,
            )
            raise

    async def delete_file(
        self,
        key: str,
        version_id: str | None = None,
    ) -> bool:
        """Delete file from GCS.

        Args:
            key: Object name
            version_id: Optional generation to delete

        Returns:
            True if deletion was successful
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        try:
            if version_id:
                blob = self._bucket.blob(key, generation=int(version_id))
            else:
                blob = self._bucket.blob(key)

            # Delete using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(executor, blob.delete)

            logger.info(
                "gcs_file_deleted",
                key=key,
                version_id=version_id,
            )

            return True

        except Exception as e:
            logger.error(
                "gcs_delete_failed",
                error=str(e),
                key=key,
            )
            raise

    async def list_files(
        self,
        prefix: str | None = None,
        max_results: int = 1000,
    ) -> list[StorageObject]:
        """List files in GCS bucket.

        Args:
            prefix: Optional object name prefix filter
            max_results: Maximum results

        Returns:
            List of storage objects
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        try:
            # List blobs using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            blobs = await loop.run_in_executor(
                executor,
                lambda: list(self._bucket.list_blobs(prefix=prefix, max_results=max_results)),
            )

            objects = []
            for blob in blobs:
                metadata = StorageMetadata(
                    content_type=blob.content_type or "application/octet-stream",
                    custom_metadata=blob.metadata or {},
                )

                storage_obj = StorageObject(
                    key=blob.name,
                    size=blob.size,
                    etag=blob.etag,
                    last_modified=blob.updated,
                    metadata=metadata,
                    version_id=str(blob.generation) if blob.generation else None,
                    storage_class=blob.storage_class,
                )

                objects.append(storage_obj)

            logger.info(
                "gcs_files_listed",
                count=len(objects),
                prefix=prefix,
            )

            return objects

        except Exception as e:
            logger.error(
                "gcs_list_failed",
                error=str(e),
                prefix=prefix,
            )
            raise

    async def get_metadata(
        self,
        key: str,
        version_id: str | None = None,
    ) -> StorageObject:
        """Get object metadata from GCS.

        Args:
            key: Object name
            version_id: Optional generation

        Returns:
            Storage object with metadata
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        try:
            if version_id:
                blob = self._bucket.blob(key, generation=int(version_id))
            else:
                blob = self._bucket.blob(key)

            # Reload using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(executor, blob.reload)

            metadata = StorageMetadata(
                content_type=blob.content_type or "application/octet-stream",
                content_encoding=blob.content_encoding,
                content_language=blob.content_language,
                cache_control=blob.cache_control,
                content_disposition=blob.content_disposition,
                custom_metadata=blob.metadata or {},
            )

            storage_obj = StorageObject(
                key=key,
                size=blob.size,
                etag=blob.etag,
                last_modified=blob.updated,
                metadata=metadata,
                version_id=str(blob.generation) if blob.generation else None,
                storage_class=blob.storage_class,
            )

            logger.info(
                "gcs_metadata_retrieved",
                key=key,
            )

            return storage_obj

        except Exception as e:
            logger.error(
                "gcs_get_metadata_failed",
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
        """Update object metadata in GCS.

        Args:
            key: Object name
            metadata: New metadata
            version_id: Optional generation

        Returns:
            True if update was successful
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        try:
            if version_id:
                blob = self._bucket.blob(key, generation=int(version_id))
            else:
                blob = self._bucket.blob(key)

            # Update metadata
            blob.content_type = metadata.content_type
            if metadata.content_encoding:
                blob.content_encoding = metadata.content_encoding
            if metadata.cache_control:
                blob.cache_control = metadata.cache_control
            if metadata.custom_metadata:
                blob.metadata = metadata.custom_metadata

            # Patch using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(executor, blob.patch)

            logger.info(
                "gcs_metadata_updated",
                key=key,
            )

            return True

        except Exception as e:
            logger.error(
                "gcs_update_metadata_failed",
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
        """Generate signed URL for temporary access.

        Args:
            key: Object name
            expiry_seconds: URL expiration time
            method: HTTP method (GET, PUT, DELETE)

        Returns:
            Signed URL
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        try:
            blob = self._bucket.blob(key)

            # Generate signed URL using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            url = await loop.run_in_executor(
                executor,
                blob.generate_signed_url,
                timedelta(seconds=expiry_seconds),
                method,
            )

            logger.info(
                "gcs_signed_url_generated",
                key=key,
                expiry_seconds=expiry_seconds,
                method=method,
            )

            return url

        except Exception as e:
            logger.error(
                "gcs_generate_signed_url_failed",
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
        """Copy file within GCS.

        Args:
            source_key: Source object name
            destination_key: Destination object name
            source_version_id: Optional source generation

        Returns:
            Upload result for copied object
        """
        if not self._connected or self._bucket is None:
            raise RuntimeError("GCS not connected. Call connect() first.")

        start_time = time.time()

        try:
            if source_version_id:
                source_blob = self._bucket.blob(source_key, generation=int(source_version_id))
            else:
                source_blob = self._bucket.blob(source_key)

            dest_blob = self._bucket.blob(destination_key)

            # Copy using executor
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(
                executor,
                self._bucket.copy_blob,
                source_blob,
                self._bucket,
                destination_key,
            )

            upload_time_ms = int((time.time() - start_time) * 1000)

            # Reload destination blob
            await loop.run_in_executor(executor, dest_blob.reload)

            result = UploadResult(
                key=destination_key,
                etag=dest_blob.etag,
                version_id=str(dest_blob.generation) if dest_blob.generation else None,
                size=dest_blob.size,
                upload_time_ms=upload_time_ms,
                url=None,
            )

            logger.info(
                "gcs_file_copied",
                source_key=source_key,
                destination_key=destination_key,
                upload_time_ms=upload_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "gcs_copy_failed",
                error=str(e),
                source_key=source_key,
                destination_key=destination_key,
            )
            raise

    async def health_check(self) -> bool:
        """Check GCS connection health.

        Returns:
            True if connection is healthy
        """
        if not self._connected or self._bucket is None:
            return False

        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)
            loop = asyncio.get_event_loop()

            exists = await loop.run_in_executor(executor, self._bucket.exists)
            return exists

        except Exception as e:
            logger.warning(
                "gcs_health_check_failed",
                error=str(e),
            )
            return False
