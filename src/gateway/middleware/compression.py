"""
Response Compression Middleware

Implements gzip compression for response payloads to reduce bandwidth.
"""

from __future__ import annotations

import gzip
from io import BytesIO
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from gateway.config import settings


class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Compress response bodies using gzip when appropriate.

    Only compresses responses when:
    - Client supports gzip (Accept-Encoding header)
    - Response size exceeds minimum threshold
    - Content-Type is compressible
    - Response not already compressed
    """

    COMPRESSIBLE_TYPES = {
        "text/html",
        "text/plain",
        "text/css",
        "text/javascript",
        "application/javascript",
        "application/json",
        "application/xml",
        "application/x-javascript",
        "text/xml",
    }

    def __init__(self, app, min_size: int = 1024, compression_level: int = 6):
        """
        Initialize compression middleware.

        Args:
            app: FastAPI application
            min_size: Minimum response size in bytes to compress (default 1KB)
            compression_level: gzip compression level 1-9 (default 6)
        """
        super().__init__(app)
        self.min_size = min_size
        self.compression_level = compression_level

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Compress response if appropriate."""
        response = await call_next(request)

        # Check if client accepts gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response

        # Skip if already compressed
        if response.headers.get("content-encoding"):
            return response

        # Get content type
        content_type = response.headers.get("content-type", "").split(";")[0].strip()

        # Skip if not compressible type
        if content_type not in self.COMPRESSIBLE_TYPES:
            return response

        # Get response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        # Skip if below minimum size
        if len(response_body) < self.min_size:
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type,
            )

        # Compress the response
        buffer = BytesIO()
        with gzip.GzipFile(
            fileobj=buffer, mode="wb", compresslevel=self.compression_level
        ) as gz_file:
            gz_file.write(response_body)

        compressed_body = buffer.getvalue()

        # Only use compressed if it's actually smaller
        if len(compressed_body) < len(response_body):
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed_body))
            response.headers["Vary"] = "Accept-Encoding"
            return Response(
                content=compressed_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=content_type,
            )

        # Return uncompressed if compression didn't help
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=content_type,
        )
