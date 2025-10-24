"""Storage adapter factory for creating provider-specific adapters.

Provides factory pattern for instantiating storage adapters based on provider type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agentcore.integration.storage.base import StorageAdapter, StorageConfig, StorageProvider

if TYPE_CHECKING:
    from agentcore.integration.storage.azure_blob import AzureBlobAdapter
    from agentcore.integration.storage.gcs import GCSAdapter
    from agentcore.integration.storage.s3 import S3Adapter

logger = structlog.get_logger(__name__)


class StorageFactory:
    """Factory for creating storage adapters.

    Instantiates the appropriate storage adapter based on provider configuration.
    """

    @staticmethod
    def create(config: StorageConfig) -> StorageAdapter:
        """Create storage adapter based on provider type.

        Args:
            config: Storage configuration with provider type

        Returns:
            Provider-specific storage adapter instance

        Raises:
            ValueError: If provider type is not supported
        """
        provider = config.provider

        if provider == StorageProvider.S3:
            from agentcore.integration.storage.s3 import S3Adapter

            logger.info(
                "storage_adapter_created",
                provider="s3",
                bucket=config.bucket_name,
            )
            return S3Adapter(config)

        elif provider == StorageProvider.AZURE_BLOB:
            from agentcore.integration.storage.azure_blob import AzureBlobAdapter

            logger.info(
                "storage_adapter_created",
                provider="azure_blob",
                container=config.bucket_name,
            )
            return AzureBlobAdapter(config)

        elif provider == StorageProvider.GCS:
            from agentcore.integration.storage.gcs import GCSAdapter

            logger.info(
                "storage_adapter_created",
                provider="gcs",
                bucket=config.bucket_name,
            )
            return GCSAdapter(config)

        else:
            raise ValueError(
                f"Unsupported storage provider: {provider}. "
                f"Supported providers: {', '.join(p.value for p in StorageProvider)}"
            )

    @staticmethod
    def supported_providers() -> list[str]:
        """Get list of supported storage provider types.

        Returns:
            List of provider type strings
        """
        return [provider.value for provider in StorageProvider]
