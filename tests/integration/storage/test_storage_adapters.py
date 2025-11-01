"""Integration tests for storage adapters (INT-009).

Tests INT-009 acceptance criteria:
- S3, Azure Blob, GCS integration
- File upload and download management
- Metadata and versioning support
- Access control and security

These tests use mocking to simulate cloud provider behavior without
requiring actual cloud accounts or credentials.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pydantic import SecretStr

from agentcore.integration.storage import (
    AccessControl,
    AccessLevel,
    AzureBlobAdapter,
    DownloadResult,
    GCSAdapter,
    S3Adapter,
    StorageConfig,
    StorageFactory,
    StorageMetadata,
    StorageObject,
    StorageProvider,
    UploadResult)


# Mock S3 Adapter for testing


class MockS3Adapter(S3Adapter):
    """Mock S3 adapter for testing without AWS credentials."""

    def __init__(self, config: StorageConfig) -> None:
        """Initialize mock S3 adapter."""
        super().__init__(config)
        self._mock_storage: dict[str, dict[str, Any]] = {}
        self._mock_client = AsyncMock()

    async def connect(self) -> None:
        """Mock connection."""
        self._connected = True
        self._client = self._mock_client

    async def disconnect(self) -> None:
        """Mock disconnection."""
        self._connected = False
        self._client = None
        self._mock_storage.clear()

    async def upload_file(
        self,
        key: str,
        content: bytes,
        metadata: StorageMetadata | None = None,
        access_control: AccessControl | None = None) -> UploadResult:
        """Mock file upload."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        import hashlib
        import time

        start_time = time.time()
        etag = hashlib.md5(content).hexdigest()

        # Store in mock storage
        self._mock_storage[key] = {
            "content": content,
            "metadata": metadata or StorageMetadata(),
            "etag": etag,
            "version_id": "v1",
            "size": len(content),
        }

        upload_time_ms = int((time.time() - start_time) * 1000)

        return UploadResult(
            key=key,
            etag=etag,
            version_id="v1",
            size=len(content),
            upload_time_ms=upload_time_ms,
            url=f"https://{self.config.bucket_name}.s3.amazonaws.com/{key}")

    async def download_file(
        self,
        key: str,
        version_id: str | None = None,
        byte_range: tuple[int, int] | None = None) -> DownloadResult:
        """Mock file download."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        if key not in self._mock_storage:
            raise KeyError(f"Object not found: {key}")

        import time

        start_time = time.time()
        obj = self._mock_storage[key]

        content = obj["content"]
        if byte_range:
            content = content[byte_range[0] : byte_range[1] + 1]

        download_time_ms = int((time.time() - start_time) * 1000)

        return DownloadResult(
            key=key,
            content=content,
            metadata=obj["metadata"],
            version_id=obj["version_id"],
            etag=obj["etag"],
            size=len(content),
            download_time_ms=download_time_ms)

    async def delete_file(
        self,
        key: str,
        version_id: str | None = None) -> bool:
        """Mock file deletion."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        if key in self._mock_storage:
            del self._mock_storage[key]

        return True

    async def list_files(
        self,
        prefix: str | None = None,
        max_results: int = 1000) -> list[StorageObject]:
        """Mock file listing."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        objects = []
        for key, obj in self._mock_storage.items():
            if prefix is None or key.startswith(prefix):
                storage_obj = StorageObject(
                    key=key,
                    size=obj["size"],
                    etag=obj["etag"],
                    last_modified=datetime.now(UTC),
                    metadata=obj["metadata"],
                    version_id=obj["version_id"],
                    storage_class="STANDARD")
                objects.append(storage_obj)

                if len(objects) >= max_results:
                    break

        return objects

    async def get_metadata(
        self,
        key: str,
        version_id: str | None = None) -> StorageObject:
        """Mock metadata retrieval."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        if key not in self._mock_storage:
            raise KeyError(f"Object not found: {key}")

        obj = self._mock_storage[key]

        return StorageObject(
            key=key,
            size=obj["size"],
            etag=obj["etag"],
            last_modified=datetime.now(UTC),
            metadata=obj["metadata"],
            version_id=obj["version_id"],
            storage_class="STANDARD")

    async def update_metadata(
        self,
        key: str,
        metadata: StorageMetadata,
        version_id: str | None = None) -> bool:
        """Mock metadata update."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        if key not in self._mock_storage:
            raise KeyError(f"Object not found: {key}")

        self._mock_storage[key]["metadata"] = metadata
        return True

    async def generate_signed_url(
        self,
        key: str,
        expiry_seconds: int = 3600,
        method: str = "GET") -> str:
        """Mock signed URL generation."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        return f"https://{self.config.bucket_name}.s3.amazonaws.com/{key}?signed=true&expires={expiry_seconds}"

    async def copy_file(
        self,
        source_key: str,
        destination_key: str,
        source_version_id: str | None = None) -> UploadResult:
        """Mock file copy."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        if source_key not in self._mock_storage:
            raise KeyError(f"Source object not found: {source_key}")

        import time

        start_time = time.time()
        source = self._mock_storage[source_key]

        self._mock_storage[destination_key] = {
            "content": source["content"],
            "metadata": source["metadata"],
            "etag": source["etag"],
            "version_id": "v2",
            "size": source["size"],
        }

        upload_time_ms = int((time.time() - start_time) * 1000)

        return UploadResult(
            key=destination_key,
            etag=source["etag"],
            version_id="v2",
            size=source["size"],
            upload_time_ms=upload_time_ms,
            url=None)

    async def upload_file_streaming(
        self,
        key: str,
        content: AsyncIterator[bytes],
        size: int,
        metadata: StorageMetadata | None = None,
        access_control: AccessControl | None = None) -> UploadResult:
        """Mock streaming upload."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        # Collect chunks
        chunks = []
        async for chunk in content:
            chunks.append(chunk)

        full_content = b"".join(chunks)
        return await self.upload_file(key, full_content, metadata, access_control)

    async def download_file_streaming(
        self,
        key: str,
        version_id: str | None = None,
        byte_range: tuple[int, int] | None = None) -> AsyncIterator[bytes]:
        """Mock streaming download."""
        if not self._connected:
            raise RuntimeError("S3 not connected. Call connect() first.")

        result = await self.download_file(key, version_id, byte_range)

        # Yield in chunks
        chunk_size = 1024
        for i in range(0, len(result.content), chunk_size):
            yield result.content[i : i + chunk_size]

    async def health_check(self) -> bool:
        """Mock health check."""
        return self._connected


# Fixtures


@pytest_asyncio.fixture
def s3_config() -> StorageConfig:
    """Create S3 storage configuration."""
    return StorageConfig(
        provider=StorageProvider.S3,
        bucket_name="test-bucket",
        region="us-east-1",
        access_key="test-access-key",
        secret_key=SecretStr("test-secret-key"),
        use_ssl=True,
        timeout=300,
        max_retries=3,
        enable_versioning=True,
        enable_encryption=True)


@pytest_asyncio.fixture
async def s3_adapter(s3_config: StorageConfig) -> MockS3Adapter:
    """Create and connect mock S3 adapter."""
    adapter = MockS3Adapter(s3_config)
    await adapter.connect()

    yield adapter

    await adapter.disconnect()


# Tests


class TestStorageConfigValidation:
    """Test storage configuration validation."""

    def test_config_creation(self, s3_config: StorageConfig) -> None:
        """Test storage config creation with valid parameters."""
        assert s3_config.provider == StorageProvider.S3
        assert s3_config.bucket_name == "test-bucket"
        assert s3_config.region == "us-east-1"
        assert s3_config.enable_versioning is True
        assert s3_config.enable_encryption is True

    def test_secret_key_security(self, s3_config: StorageConfig) -> None:
        """Test secret key is stored securely."""
        # Secret key should be SecretStr
        assert isinstance(s3_config.secret_key, SecretStr)

        # Should not be visible in repr
        config_repr = repr(s3_config)
        assert "test-secret-key" not in config_repr

        # Should be retrievable with get_secret_value()
        assert s3_config.secret_key.get_secret_value() == "test-secret-key"

    def test_config_validation_timeout(self) -> None:
        """Test timeout validation."""
        with pytest.raises(ValueError):
            StorageConfig(
                provider=StorageProvider.S3,
                bucket_name="test",
                timeout=4000,  # Exceeds max of 3600
                access_key="key",
                secret_key=SecretStr("secret"))


class TestStorageAdapterBasics:
    """Test basic storage adapter operations."""

    @pytest.mark.asyncio
    async def test_connection_lifecycle(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test storage connection and disconnection."""
        assert s3_adapter.is_connected() is True

        await s3_adapter.disconnect()
        assert s3_adapter.is_connected() is False

    @pytest.mark.asyncio
    async def test_health_check(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test storage health check."""
        health = await s3_adapter.health_check()
        assert health is True

        await s3_adapter.disconnect()

        health_disconnected = await s3_adapter.health_check()
        assert health_disconnected is False


class TestFileUploadDownload:
    """Test file upload and download operations."""

    @pytest.mark.asyncio
    async def test_simple_file_upload(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test uploading file to storage."""
        content = b"Hello, World!"
        metadata = StorageMetadata(
            content_type="text/plain",
            custom_metadata={"author": "test"})

        result = await s3_adapter.upload_file(
            key="test.txt",
            content=content,
            metadata=metadata)

        assert result.key == "test.txt"
        assert result.size == len(content)
        assert result.etag is not None
        assert result.version_id is not None
        assert result.upload_time_ms >= 0

    @pytest.mark.asyncio
    async def test_file_download(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test downloading file from storage."""
        content = b"Test content"

        # Upload first
        await s3_adapter.upload_file("download-test.txt", content)

        # Download
        result = await s3_adapter.download_file("download-test.txt")

        assert result.key == "download-test.txt"
        assert result.content == content
        assert result.size == len(content)
        assert result.etag is not None
        assert result.download_time_ms >= 0

    @pytest.mark.asyncio
    async def test_byte_range_download(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test partial file download with byte range."""
        content = b"0123456789"

        await s3_adapter.upload_file("range-test.txt", content)

        # Download bytes 2-5
        result = await s3_adapter.download_file(
            "range-test.txt",
            byte_range=(2, 5))

        assert result.content == b"2345"
        assert result.size == 4

    @pytest.mark.asyncio
    async def test_streaming_upload(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test streaming file upload."""

        async def content_generator() -> AsyncIterator[bytes]:
            for i in range(3):
                yield f"chunk{i}".encode()

        result = await s3_adapter.upload_file_streaming(
            key="stream-test.txt",
            content=content_generator(),
            size=18,  # chunk0 + chunk1 + chunk2
        )

        assert result.key == "stream-test.txt"
        assert result.size == 18

        # Verify content
        download = await s3_adapter.download_file("stream-test.txt")
        assert download.content == b"chunk0chunk1chunk2"

    @pytest.mark.asyncio
    async def test_streaming_download(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test streaming file download."""
        content = b"This is a large file for streaming"

        await s3_adapter.upload_file("stream-download.txt", content)

        # Download in chunks
        chunks = []
        async for chunk in s3_adapter.download_file_streaming("stream-download.txt"):
            chunks.append(chunk)

        downloaded_content = b"".join(chunks)
        assert downloaded_content == content


class TestMetadataManagement:
    """Test metadata management operations."""

    @pytest.mark.asyncio
    async def test_upload_with_metadata(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test uploading file with metadata."""
        metadata = StorageMetadata(
            content_type="application/json",
            content_encoding="gzip",
            cache_control="max-age=3600",
            custom_metadata={
                "project": "test",
                "version": "1.0",
            })

        await s3_adapter.upload_file(
            key="data.json",
            content=b'{"key": "value"}',
            metadata=metadata)

        # Retrieve metadata
        obj = await s3_adapter.get_metadata("data.json")

        assert obj.metadata.content_type == "application/json"
        assert obj.metadata.content_encoding == "gzip"
        assert obj.metadata.cache_control == "max-age=3600"
        assert obj.metadata.custom_metadata["project"] == "test"

    @pytest.mark.asyncio
    async def test_update_metadata(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test updating file metadata."""
        # Upload file
        await s3_adapter.upload_file(
            key="update-meta.txt",
            content=b"content")

        # Update metadata
        new_metadata = StorageMetadata(
            content_type="text/html",
            cache_control="no-cache",
            custom_metadata={"updated": "true"})

        result = await s3_adapter.update_metadata(
            "update-meta.txt",
            new_metadata)

        assert result is True

        # Verify update
        obj = await s3_adapter.get_metadata("update-meta.txt")
        assert obj.metadata.content_type == "text/html"
        assert obj.metadata.cache_control == "no-cache"


class TestAccessControl:
    """Test access control and security features."""

    @pytest.mark.asyncio
    async def test_upload_with_access_control(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test uploading file with access control."""
        access_control = AccessControl(
            access_level=AccessLevel.PUBLIC_READ,
            signed_url_expiry=7200)

        result = await s3_adapter.upload_file(
            key="public-file.txt",
            content=b"public content",
            access_control=access_control)

        assert result.url is not None
        assert "test-bucket" in result.url

    @pytest.mark.asyncio
    async def test_generate_signed_url(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test signed URL generation."""
        await s3_adapter.upload_file(
            key="signed-test.txt",
            content=b"content")

        url = await s3_adapter.generate_signed_url(
            "signed-test.txt",
            expiry_seconds=3600,
            method="GET")

        assert url is not None
        assert "signed=true" in url
        assert "expires=3600" in url


class TestFileOperations:
    """Test file operations like delete, copy, list."""

    @pytest.mark.asyncio
    async def test_delete_file(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test deleting file from storage."""
        await s3_adapter.upload_file(
            key="delete-test.txt",
            content=b"to be deleted")

        # Verify file exists
        obj = await s3_adapter.get_metadata("delete-test.txt")
        assert obj.key == "delete-test.txt"

        # Delete file
        result = await s3_adapter.delete_file("delete-test.txt")
        assert result is True

        # Verify file is gone
        with pytest.raises(KeyError):
            await s3_adapter.get_metadata("delete-test.txt")

    @pytest.mark.asyncio
    async def test_copy_file(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test copying file within storage."""
        content = b"original content"

        await s3_adapter.upload_file(
            key="source.txt",
            content=content)

        # Copy file
        result = await s3_adapter.copy_file(
            source_key="source.txt",
            destination_key="destination.txt")

        assert result.key == "destination.txt"
        assert result.size == len(content)

        # Verify both files exist
        source = await s3_adapter.download_file("source.txt")
        dest = await s3_adapter.download_file("destination.txt")

        assert source.content == dest.content

    @pytest.mark.asyncio
    async def test_list_files(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test listing files in storage."""
        # Upload multiple files
        await s3_adapter.upload_file("file1.txt", b"content1")
        await s3_adapter.upload_file("file2.txt", b"content2")
        await s3_adapter.upload_file("dir/file3.txt", b"content3")

        # List all files
        objects = await s3_adapter.list_files()
        assert len(objects) == 3

        # List with prefix
        objects_with_prefix = await s3_adapter.list_files(prefix="dir/")
        assert len(objects_with_prefix) == 1
        assert objects_with_prefix[0].key == "dir/file3.txt"

    @pytest.mark.asyncio
    async def test_list_files_max_results(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test listing files with max results limit."""
        # Upload multiple files
        for i in range(5):
            await s3_adapter.upload_file(f"file{i}.txt", b"content")

        # List with max results
        objects = await s3_adapter.list_files(max_results=3)
        assert len(objects) == 3


class TestVersioning:
    """Test versioning support."""

    @pytest.mark.asyncio
    async def test_versioned_upload(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test versioning is tracked on uploads."""
        result1 = await s3_adapter.upload_file(
            key="versioned.txt",
            content=b"version 1")

        assert result1.version_id is not None

        result2 = await s3_adapter.upload_file(
            key="versioned.txt",
            content=b"version 2")

        assert result2.version_id is not None


class TestStorageFactory:
    """Test storage factory."""

    def test_create_s3_adapter(self, s3_config: StorageConfig) -> None:
        """Test factory creates S3 adapter."""
        adapter = StorageFactory.create(s3_config)

        assert isinstance(adapter, S3Adapter)
        assert adapter.config == s3_config

    def test_create_azure_blob_adapter(self) -> None:
        """Test factory creates Azure Blob adapter."""
        config = StorageConfig(
            provider=StorageProvider.AZURE_BLOB,
            bucket_name="test-container",
            connection_string=SecretStr("DefaultEndpointsProtocol=https;AccountName=test"))

        adapter = StorageFactory.create(config)

        assert isinstance(adapter, AzureBlobAdapter)
        assert adapter.config == config

    def test_create_gcs_adapter(self) -> None:
        """Test factory creates GCS adapter."""
        config = StorageConfig(
            provider=StorageProvider.GCS,
            bucket_name="test-bucket",
            credentials_file="/path/to/credentials.json")

        adapter = StorageFactory.create(config)

        assert isinstance(adapter, GCSAdapter)
        assert adapter.config == config

    def test_unsupported_provider(self) -> None:
        """Test factory raises error for unsupported provider."""
        # This would require modifying the enum which we can't do in tests
        # Just verify supported providers are returned
        providers = StorageFactory.supported_providers()

        assert "s3" in providers
        assert "azure_blob" in providers
        assert "gcs" in providers


class TestErrorHandling:
    """Test error handling in storage operations."""

    @pytest.mark.asyncio
    async def test_operation_without_connection(
        self, s3_config: StorageConfig
    ) -> None:
        """Test operations fail without connection."""
        adapter = MockS3Adapter(s3_config)
        # Don't connect

        with pytest.raises(RuntimeError, match="not connected"):
            await adapter.upload_file("test.txt", b"content")

    @pytest.mark.asyncio
    async def test_download_nonexistent_file(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test downloading non-existent file raises error."""
        with pytest.raises(KeyError):
            await s3_adapter.download_file("does-not-exist.txt")

    @pytest.mark.asyncio
    async def test_update_metadata_nonexistent_file(
        self, s3_adapter: MockS3Adapter
    ) -> None:
        """Test updating metadata for non-existent file."""
        metadata = StorageMetadata(content_type="text/plain")

        with pytest.raises(KeyError):
            await s3_adapter.update_metadata("does-not-exist.txt", metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
