"""Base storage adapter interface and models.

Defines the abstract interface for all cloud storage adapters with
file upload/download, metadata management, versioning, and access control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field, SecretStr


class StorageProvider(str, Enum):
    """Supported cloud storage providers."""

    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


class AccessLevel(str, Enum):
    """Access control levels for storage objects."""

    PRIVATE = "private"
    PUBLIC_READ = "public_read"
    PUBLIC_READ_WRITE = "public_read_write"


class StorageConfig(BaseModel):
    """Storage provider configuration.

    Encapsulates connection parameters with secure credential handling.
    """

    provider: StorageProvider = Field(
        description="Storage provider type (s3, azure_blob, gcs)",
    )
    bucket_name: str = Field(
        description="Bucket/container name",
    )
    region: str | None = Field(
        default=None,
        description="Storage region (required for S3, optional for others)",
    )
    endpoint_url: str | None = Field(
        default=None,
        description="Custom endpoint URL (for S3-compatible services)",
    )
    access_key: str | None = Field(
        default=None,
        description="Access key ID (S3, GCS with HMAC)",
    )
    secret_key: SecretStr | None = Field(
        default=None,
        description="Secret access key (encrypted in storage)",
    )
    connection_string: SecretStr | None = Field(
        default=None,
        description="Connection string (Azure Blob)",
    )
    credentials_file: str | None = Field(
        default=None,
        description="Path to credentials file (GCS service account)",
    )
    use_ssl: bool = Field(
        default=True,
        description="Enable SSL/TLS encryption",
    )
    timeout: int = Field(
        default=300,
        description="Request timeout in seconds",
        ge=1,
        le=3600,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed operations",
        ge=0,
        le=10,
    )
    enable_versioning: bool = Field(
        default=True,
        description="Enable object versioning if provider supports it",
    )
    enable_encryption: bool = Field(
        default=True,
        description="Enable server-side encryption",
    )


class StorageMetadata(BaseModel):
    """Storage object metadata.

    Encapsulates metadata, tags, and custom headers for storage objects.
    """

    content_type: str = Field(
        default="application/octet-stream",
        description="MIME type of the object",
    )
    content_encoding: str | None = Field(
        default=None,
        description="Content encoding (e.g., gzip)",
    )
    content_language: str | None = Field(
        default=None,
        description="Content language",
    )
    cache_control: str | None = Field(
        default=None,
        description="Cache-Control header",
    )
    content_disposition: str | None = Field(
        default=None,
        description="Content-Disposition header",
    )
    custom_metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Custom key-value metadata",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Object tags for organization and filtering",
    )


class AccessControl(BaseModel):
    """Access control configuration for storage objects."""

    access_level: AccessLevel = Field(
        default=AccessLevel.PRIVATE,
        description="Access level for the object",
    )
    signed_url_expiry: int = Field(
        default=3600,
        description="Signed URL expiration time in seconds",
        ge=60,
        le=604800,  # Max 7 days
    )


class StorageObject(BaseModel):
    """Storage object information.

    Represents a stored object with its metadata and properties.
    """

    key: str = Field(
        description="Object key (path) in storage",
    )
    size: int = Field(
        description="Object size in bytes",
        ge=0,
    )
    etag: str = Field(
        description="Entity tag (version identifier)",
    )
    last_modified: datetime = Field(
        description="Last modification timestamp",
    )
    metadata: StorageMetadata = Field(
        description="Object metadata",
    )
    version_id: str | None = Field(
        default=None,
        description="Version ID if versioning is enabled",
    )
    storage_class: str | None = Field(
        default=None,
        description="Storage class/tier",
    )


class UploadResult(BaseModel):
    """File upload operation result.

    Encapsulates upload results with metadata for verification and tracking.
    """

    key: str = Field(
        description="Object key in storage",
    )
    etag: str = Field(
        description="Entity tag of uploaded object",
    )
    version_id: str | None = Field(
        default=None,
        description="Version ID if versioning is enabled",
    )
    size: int = Field(
        description="Uploaded file size in bytes",
        ge=0,
    )
    upload_time_ms: int = Field(
        description="Upload time in milliseconds",
        ge=0,
    )
    url: str | None = Field(
        default=None,
        description="Public or signed URL to access the object",
    )


class DownloadResult(BaseModel):
    """File download operation result.

    Encapsulates download results with metadata and content.
    """

    key: str = Field(
        description="Object key in storage",
    )
    content: bytes = Field(
        description="Downloaded file content",
    )
    metadata: StorageMetadata = Field(
        description="Object metadata",
    )
    version_id: str | None = Field(
        default=None,
        description="Version ID if versioning is enabled",
    )
    etag: str = Field(
        description="Entity tag",
    )
    size: int = Field(
        description="Downloaded file size in bytes",
        ge=0,
    )
    download_time_ms: int = Field(
        description="Download time in milliseconds",
        ge=0,
    )


class StorageAdapter(ABC):
    """Abstract base class for cloud storage adapters.

    Defines the interface that all storage provider-specific adapters must implement.
    Provides file upload/download, metadata management, versioning, and access control.
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize storage adapter.

        Args:
            config: Storage provider configuration
        """
        self.config = config
        self._client: Any = None
        self._connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to storage provider.

        Must be called before performing storage operations.
        """
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to storage provider.

        Gracefully closes all connections and releases resources.
        """
        ...

    @abstractmethod
    async def upload_file(
        self,
        key: str,
        content: bytes,
        metadata: StorageMetadata | None = None,
        access_control: AccessControl | None = None,
    ) -> UploadResult:
        """Upload file to storage.

        Args:
            key: Object key (path) in storage
            content: File content as bytes
            metadata: Optional metadata for the object
            access_control: Optional access control settings

        Returns:
            Upload result with ETag, version, and URL

        Raises:
            StorageError: If upload fails
            TimeoutError: If upload exceeds timeout
        """
        ...

    @abstractmethod
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
            key: Object key (path) in storage
            content: Async iterator of file chunks
            size: Total file size in bytes
            metadata: Optional metadata for the object
            access_control: Optional access control settings

        Returns:
            Upload result with ETag, version, and URL

        Raises:
            StorageError: If upload fails
            TimeoutError: If upload exceeds timeout
        """
        ...

    @abstractmethod
    async def download_file(
        self,
        key: str,
        version_id: str | None = None,
        byte_range: tuple[int, int] | None = None,
    ) -> DownloadResult:
        """Download file from storage.

        Args:
            key: Object key (path) in storage
            version_id: Optional specific version to download
            byte_range: Optional byte range (start, end) for partial download

        Returns:
            Download result with content and metadata

        Raises:
            StorageError: If download fails or object not found
            TimeoutError: If download exceeds timeout
        """
        ...

    @abstractmethod
    async def download_file_streaming(
        self,
        key: str,
        version_id: str | None = None,
        byte_range: tuple[int, int] | None = None,
    ) -> AsyncIterator[bytes]:
        """Download large file using streaming.

        Args:
            key: Object key (path) in storage
            version_id: Optional specific version to download
            byte_range: Optional byte range for partial download

        Yields:
            File content chunks

        Raises:
            StorageError: If download fails or object not found
            TimeoutError: If download exceeds timeout
        """
        ...

    @abstractmethod
    async def delete_file(
        self,
        key: str,
        version_id: str | None = None,
    ) -> bool:
        """Delete file from storage.

        Args:
            key: Object key (path) in storage
            version_id: Optional specific version to delete

        Returns:
            True if deletion was successful

        Raises:
            StorageError: If deletion fails
        """
        ...

    @abstractmethod
    async def list_files(
        self,
        prefix: str | None = None,
        max_results: int = 1000,
    ) -> list[StorageObject]:
        """List files in storage.

        Args:
            prefix: Optional key prefix to filter results
            max_results: Maximum number of results to return

        Returns:
            List of storage objects

        Raises:
            StorageError: If listing fails
        """
        ...

    @abstractmethod
    async def get_metadata(
        self,
        key: str,
        version_id: str | None = None,
    ) -> StorageObject:
        """Get object metadata without downloading content.

        Args:
            key: Object key (path) in storage
            version_id: Optional specific version

        Returns:
            Storage object information with metadata

        Raises:
            StorageError: If object not found or metadata retrieval fails
        """
        ...

    @abstractmethod
    async def update_metadata(
        self,
        key: str,
        metadata: StorageMetadata,
        version_id: str | None = None,
    ) -> bool:
        """Update object metadata.

        Args:
            key: Object key (path) in storage
            metadata: New metadata to apply
            version_id: Optional specific version

        Returns:
            True if update was successful

        Raises:
            StorageError: If update fails
        """
        ...

    @abstractmethod
    async def generate_signed_url(
        self,
        key: str,
        expiry_seconds: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate signed URL for temporary access.

        Args:
            key: Object key (path) in storage
            expiry_seconds: URL expiration time in seconds
            method: HTTP method (GET, PUT, DELETE)

        Returns:
            Signed URL for temporary access

        Raises:
            StorageError: If URL generation fails
        """
        ...

    @abstractmethod
    async def copy_file(
        self,
        source_key: str,
        destination_key: str,
        source_version_id: str | None = None,
    ) -> UploadResult:
        """Copy file within storage.

        Args:
            source_key: Source object key
            destination_key: Destination object key
            source_version_id: Optional source version

        Returns:
            Upload result for the copied object

        Raises:
            StorageError: If copy fails
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if storage connection is healthy.

        Returns:
            True if connection is active and responsive
        """
        ...

    def is_connected(self) -> bool:
        """Check if adapter is connected.

        Returns:
            True if connection is active
        """
        return self._connected
