"""Storage adapter framework for cloud storage providers.

Provides unified interface for AWS S3, Azure Blob Storage, and Google Cloud Storage
with file upload/download, metadata management, versioning, and access control.
"""

from agentcore.integration.storage.base import (
    StorageAdapter,
    StorageConfig,
    StorageObject,
    StorageMetadata,
    UploadResult,
    DownloadResult,
    AccessControl,
)
from agentcore.integration.storage.factory import StorageFactory
from agentcore.integration.storage.s3 import S3Adapter
from agentcore.integration.storage.azure_blob import AzureBlobAdapter
from agentcore.integration.storage.gcs import GCSAdapter

__all__ = [
    "StorageAdapter",
    "StorageConfig",
    "StorageObject",
    "StorageMetadata",
    "UploadResult",
    "DownloadResult",
    "AccessControl",
    "StorageFactory",
    "S3Adapter",
    "AzureBlobAdapter",
    "GCSAdapter",
]
