"""AWS S3 storage adapter implementation.

Provides S3-specific implementation with boto3 for file upload/download,
metadata management, versioning, and signed URL generation.
"""

from __future__ import annotations

import time
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


class S3Adapter(StorageAdapter):
    """AWS S3 storage adapter with boto3.

    High-performance async S3 adapter using aioboto3 for async operations
    with streaming support, versioning, and signed URL generation.
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize S3 adapter.

        Args:
            config: Storage configuration
        """
        super().__init__(config)
        self._client: Any = None
        self._resource: Any = None

        logger.info(
            "s3_adapter_initialized",
            bucket=config.bucket_name,
            region=config.region,
            endpoint=config.endpoint_url,
        )

    async def connect(self) -> None:
        """Establish connection to S3."""
        if self._connected:
            return

        try:
            # Import aioboto3 for async S3 operations
            try:
                import aioboto3
            except ImportError as e:
                raise ImportError(
                    "aioboto3 is required for S3 adapter. "
                    "Install it with: pip install aioboto3"
                ) from e

            # Create session
            session = aioboto3.Session(
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=(
                    self.config.secret_key.get_secret_value()
                    if self.config.secret_key
                    else None
                ),
                region_name=self.config.region,
            )

            # Create async client and resource
            client_config = {
                "region_name": self.config.region,
                "use_ssl": self.config.use_ssl,
            }

            if self.config.endpoint_url:
                client_config["endpoint_url"] = self.config.endpoint_url

            self._session = session
            self._client = await session.client("s3", **client_config).__aenter__()
            self._resource = await session.resource("s3", **client_config).__aenter__()

            # Test connection by listing bucket
            await self._client.head_bucket(Bucket=self.config.bucket_name)

            self._connected = True

            logger.info(
                "s3_connected",
                bucket=self.config.bucket_name,
                region=self.config.region,
            )

        except Exception as e:
            logger.error(
                "s3_connection_failed",
                error=str(e),
                bucket=self.config.bucket_name,
            )
            raise

    async def disconnect(self) -> None:
        """Close S3 connection."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

        if self._resource is not None:
            await self._resource.__aexit__(None, None, None)
            self._resource = None

        self._connected = False

        logger.info(
            "s3_disconnected",
            bucket=self.config.bucket_name,
        )

    async def upload_file(
        self,
        key: str,
        content: bytes,
        metadata: StorageMetadata | None = None,
        access_control: AccessControl | None = None,
    ) -> UploadResult:
        """Upload file to S3.

        Args:
            key: Object key in S3
            content: File content as bytes
            metadata: Optional metadata
            access_control: Optional access control

        Returns:
            Upload result with ETag and version
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        start_time = time.time()

        try:
            # Prepare upload parameters
            extra_args: dict[str, Any] = {}

            if metadata:
                extra_args["ContentType"] = metadata.content_type
                if metadata.content_encoding:
                    extra_args["ContentEncoding"] = metadata.content_encoding
                if metadata.content_language:
                    extra_args["ContentLanguage"] = metadata.content_language
                if metadata.cache_control:
                    extra_args["CacheControl"] = metadata.cache_control
                if metadata.content_disposition:
                    extra_args["ContentDisposition"] = metadata.content_disposition
                if metadata.custom_metadata:
                    extra_args["Metadata"] = metadata.custom_metadata

            if access_control:
                acl_map = {
                    AccessLevel.PRIVATE: "private",
                    AccessLevel.PUBLIC_READ: "public-read",
                    AccessLevel.PUBLIC_READ_WRITE: "public-read-write",
                }
                extra_args["ACL"] = acl_map[access_control.access_level]

            if self.config.enable_encryption:
                extra_args["ServerSideEncryption"] = "AES256"

            # Upload to S3
            response = await self._client.put_object(
                Bucket=self.config.bucket_name,
                Key=key,
                Body=content,
                **extra_args,
            )

            upload_time_ms = int((time.time() - start_time) * 1000)

            # Generate URL if public or signed
            url = None
            if (
                access_control
                and access_control.access_level != AccessLevel.PRIVATE
            ):
                if access_control.access_level == AccessLevel.PUBLIC_READ:
                    url = f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{key}"
                else:
                    url = await self.generate_signed_url(
                        key, expiry_seconds=access_control.signed_url_expiry
                    )

            result = UploadResult(
                key=key,
                etag=response["ETag"].strip('"'),
                version_id=response.get("VersionId"),
                size=len(content),
                upload_time_ms=upload_time_ms,
                url=url,
            )

            logger.info(
                "s3_file_uploaded",
                key=key,
                size=len(content),
                upload_time_ms=upload_time_ms,
                version_id=result.version_id,
            )

            return result

        except Exception as e:
            logger.error(
                "s3_upload_failed",
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
        """Upload large file using multipart upload.

        Args:
            key: Object key in S3
            content: Async iterator of file chunks
            size: Total file size
            metadata: Optional metadata
            access_control: Optional access control

        Returns:
            Upload result with ETag and version
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        start_time = time.time()

        try:
            # Initiate multipart upload
            extra_args: dict[str, Any] = {}

            if metadata:
                extra_args["ContentType"] = metadata.content_type
                if metadata.custom_metadata:
                    extra_args["Metadata"] = metadata.custom_metadata

            if access_control:
                acl_map = {
                    AccessLevel.PRIVATE: "private",
                    AccessLevel.PUBLIC_READ: "public-read",
                    AccessLevel.PUBLIC_READ_WRITE: "public-read-write",
                }
                extra_args["ACL"] = acl_map[access_control.access_level]

            if self.config.enable_encryption:
                extra_args["ServerSideEncryption"] = "AES256"

            multipart_upload = await self._client.create_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=key,
                **extra_args,
            )

            upload_id = multipart_upload["UploadId"]
            parts = []
            part_number = 1

            try:
                # Upload parts
                async for chunk in content:
                    response = await self._client.upload_part(
                        Bucket=self.config.bucket_name,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk,
                    )

                    parts.append(
                        {
                            "PartNumber": part_number,
                            "ETag": response["ETag"],
                        }
                    )

                    part_number += 1

                # Complete multipart upload
                response = await self._client.complete_multipart_upload(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )

            except Exception:
                # Abort multipart upload on error
                await self._client.abort_multipart_upload(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    UploadId=upload_id,
                )
                raise

            upload_time_ms = int((time.time() - start_time) * 1000)

            result = UploadResult(
                key=key,
                etag=response["ETag"].strip('"'),
                version_id=response.get("VersionId"),
                size=size,
                upload_time_ms=upload_time_ms,
                url=None,
            )

            logger.info(
                "s3_file_uploaded_streaming",
                key=key,
                size=size,
                parts=len(parts),
                upload_time_ms=upload_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "s3_streaming_upload_failed",
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
        """Download file from S3.

        Args:
            key: Object key in S3
            version_id: Optional version to download
            byte_range: Optional byte range (start, end)

        Returns:
            Download result with content and metadata
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        start_time = time.time()

        try:
            # Prepare download parameters
            get_params: dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Key": key,
            }

            if version_id:
                get_params["VersionId"] = version_id

            if byte_range:
                get_params["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"

            # Download from S3
            response = await self._client.get_object(**get_params)

            # Read content
            content = await response["Body"].read()

            download_time_ms = int((time.time() - start_time) * 1000)

            # Extract metadata
            metadata = StorageMetadata(
                content_type=response.get("ContentType", "application/octet-stream"),
                content_encoding=response.get("ContentEncoding"),
                content_language=response.get("ContentLanguage"),
                cache_control=response.get("CacheControl"),
                content_disposition=response.get("ContentDisposition"),
                custom_metadata=response.get("Metadata", {}),
            )

            result = DownloadResult(
                key=key,
                content=content,
                metadata=metadata,
                version_id=response.get("VersionId"),
                etag=response["ETag"].strip('"'),
                size=response["ContentLength"],
                download_time_ms=download_time_ms,
            )

            logger.info(
                "s3_file_downloaded",
                key=key,
                size=len(content),
                download_time_ms=download_time_ms,
                version_id=result.version_id,
            )

            return result

        except Exception as e:
            logger.error(
                "s3_download_failed",
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
            key: Object key in S3
            version_id: Optional version
            byte_range: Optional byte range

        Yields:
            File content chunks
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        try:
            get_params: dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Key": key,
            }

            if version_id:
                get_params["VersionId"] = version_id

            if byte_range:
                get_params["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"

            response = await self._client.get_object(**get_params)

            # Stream content in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            async for chunk in response["Body"].iter_chunks(chunk_size):
                yield chunk

            logger.info(
                "s3_file_downloaded_streaming",
                key=key,
                version_id=version_id,
            )

        except Exception as e:
            logger.error(
                "s3_streaming_download_failed",
                error=str(e),
                key=key,
            )
            raise

    async def delete_file(
        self,
        key: str,
        version_id: str | None = None,
    ) -> bool:
        """Delete file from S3.

        Args:
            key: Object key
            version_id: Optional version to delete

        Returns:
            True if deletion was successful
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        try:
            delete_params: dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Key": key,
            }

            if version_id:
                delete_params["VersionId"] = version_id

            await self._client.delete_object(**delete_params)

            logger.info(
                "s3_file_deleted",
                key=key,
                version_id=version_id,
            )

            return True

        except Exception as e:
            logger.error(
                "s3_delete_failed",
                error=str(e),
                key=key,
            )
            raise

    async def list_files(
        self,
        prefix: str | None = None,
        max_results: int = 1000,
    ) -> list[StorageObject]:
        """List files in S3 bucket.

        Args:
            prefix: Optional key prefix filter
            max_results: Maximum results

        Returns:
            List of storage objects
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        try:
            list_params: dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "MaxKeys": max_results,
            }

            if prefix:
                list_params["Prefix"] = prefix

            response = await self._client.list_objects_v2(**list_params)

            objects = []
            for obj in response.get("Contents", []):
                # Get metadata for each object (lightweight HEAD request)
                head_response = await self._client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=obj["Key"],
                )

                metadata = StorageMetadata(
                    content_type=head_response.get("ContentType", "application/octet-stream"),
                    custom_metadata=head_response.get("Metadata", {}),
                )

                storage_obj = StorageObject(
                    key=obj["Key"],
                    size=obj["Size"],
                    etag=obj["ETag"].strip('"'),
                    last_modified=obj["LastModified"],
                    metadata=metadata,
                    version_id=obj.get("VersionId"),
                    storage_class=obj.get("StorageClass"),
                )

                objects.append(storage_obj)

            logger.info(
                "s3_files_listed",
                count=len(objects),
                prefix=prefix,
            )

            return objects

        except Exception as e:
            logger.error(
                "s3_list_failed",
                error=str(e),
                prefix=prefix,
            )
            raise

    async def get_metadata(
        self,
        key: str,
        version_id: str | None = None,
    ) -> StorageObject:
        """Get object metadata from S3.

        Args:
            key: Object key
            version_id: Optional version

        Returns:
            Storage object with metadata
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        try:
            head_params: dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Key": key,
            }

            if version_id:
                head_params["VersionId"] = version_id

            response = await self._client.head_object(**head_params)

            metadata = StorageMetadata(
                content_type=response.get("ContentType", "application/octet-stream"),
                content_encoding=response.get("ContentEncoding"),
                content_language=response.get("ContentLanguage"),
                cache_control=response.get("CacheControl"),
                content_disposition=response.get("ContentDisposition"),
                custom_metadata=response.get("Metadata", {}),
            )

            storage_obj = StorageObject(
                key=key,
                size=response["ContentLength"],
                etag=response["ETag"].strip('"'),
                last_modified=response["LastModified"],
                metadata=metadata,
                version_id=response.get("VersionId"),
                storage_class=response.get("StorageClass"),
            )

            logger.info(
                "s3_metadata_retrieved",
                key=key,
                version_id=version_id,
            )

            return storage_obj

        except Exception as e:
            logger.error(
                "s3_get_metadata_failed",
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
        """Update object metadata in S3.

        Args:
            key: Object key
            metadata: New metadata
            version_id: Optional version

        Returns:
            True if update was successful
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        try:
            # S3 requires copy-in-place to update metadata
            copy_source = {
                "Bucket": self.config.bucket_name,
                "Key": key,
            }

            if version_id:
                copy_source["VersionId"] = version_id

            copy_params: dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Key": key,
                "CopySource": copy_source,
                "MetadataDirective": "REPLACE",
                "ContentType": metadata.content_type,
            }

            if metadata.content_encoding:
                copy_params["ContentEncoding"] = metadata.content_encoding
            if metadata.cache_control:
                copy_params["CacheControl"] = metadata.cache_control
            if metadata.custom_metadata:
                copy_params["Metadata"] = metadata.custom_metadata

            await self._client.copy_object(**copy_params)

            logger.info(
                "s3_metadata_updated",
                key=key,
                version_id=version_id,
            )

            return True

        except Exception as e:
            logger.error(
                "s3_update_metadata_failed",
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
            key: Object key
            expiry_seconds: URL expiration time
            method: HTTP method (GET, PUT, DELETE)

        Returns:
            Signed URL
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        try:
            # Map method to S3 operation
            operation_map = {
                "GET": "get_object",
                "PUT": "put_object",
                "DELETE": "delete_object",
            }

            operation = operation_map.get(method.upper(), "get_object")

            url = await self._client.generate_presigned_url(
                ClientMethod=operation,
                Params={
                    "Bucket": self.config.bucket_name,
                    "Key": key,
                },
                ExpiresIn=expiry_seconds,
            )

            logger.info(
                "s3_signed_url_generated",
                key=key,
                expiry_seconds=expiry_seconds,
                method=method,
            )

            return url

        except Exception as e:
            logger.error(
                "s3_generate_signed_url_failed",
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
        """Copy file within S3.

        Args:
            source_key: Source object key
            destination_key: Destination object key
            source_version_id: Optional source version

        Returns:
            Upload result for copied object
        """
        if not self._connected or self._client is None:
            raise RuntimeError("S3 not connected. Call connect() first.")

        start_time = time.time()

        try:
            copy_source = {
                "Bucket": self.config.bucket_name,
                "Key": source_key,
            }

            if source_version_id:
                copy_source["VersionId"] = source_version_id

            response = await self._client.copy_object(
                Bucket=self.config.bucket_name,
                Key=destination_key,
                CopySource=copy_source,
            )

            upload_time_ms = int((time.time() - start_time) * 1000)

            # Get size from source
            head_response = await self._client.head_object(
                Bucket=self.config.bucket_name,
                Key=destination_key,
            )

            result = UploadResult(
                key=destination_key,
                etag=response["CopyObjectResult"]["ETag"].strip('"'),
                version_id=response.get("VersionId"),
                size=head_response["ContentLength"],
                upload_time_ms=upload_time_ms,
                url=None,
            )

            logger.info(
                "s3_file_copied",
                source_key=source_key,
                destination_key=destination_key,
                upload_time_ms=upload_time_ms,
            )

            return result

        except Exception as e:
            logger.error(
                "s3_copy_failed",
                error=str(e),
                source_key=source_key,
                destination_key=destination_key,
            )
            raise

    async def health_check(self) -> bool:
        """Check S3 connection health.

        Returns:
            True if connection is healthy
        """
        if not self._connected or self._client is None:
            return False

        try:
            await self._client.head_bucket(Bucket=self.config.bucket_name)
            return True

        except Exception as e:
            logger.warning(
                "s3_health_check_failed",
                error=str(e),
            )
            return False
