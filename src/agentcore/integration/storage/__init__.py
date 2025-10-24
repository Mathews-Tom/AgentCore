"""Storage adapter framework for cloud storage providers.

Provides unified interface for AWS S3, Azure Blob Storage, and Google Cloud Storage
with file upload/download, metadata management, versioning, and access control.
"""

from agentcore.integration.storage.base import (
    AccessControl,
    AccessLevel,
    DownloadResult,
    StorageAdapter,
    StorageConfig,
    StorageMetadata,
    StorageObject,
    StorageProvider,
    UploadResult,
)
from agentcore.integration.storage.azure_blob import AzureBlobAdapter
from agentcore.integration.storage.factory import StorageFactory
from agentcore.integration.storage.gcs import GCSAdapter
from agentcore.integration.storage.s3 import S3Adapter

__all__ = [
    "AccessControl",
    "AccessLevel",
    "AzureBlobAdapter",
    "DownloadResult",
    "GCSAdapter",
    "S3Adapter",
    "StorageAdapter",
    "StorageConfig",
    "StorageFactory",
    "StorageMetadata",
    "StorageObject",
    "StorageProvider",
    "UploadResult",
]
