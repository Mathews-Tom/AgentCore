"""
Request/Response Transformation Middleware

Handles request preprocessing and response postprocessing.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from gateway.config import settings


class TransformationMiddleware(BaseHTTPMiddleware):
    """
    Transform requests and responses.

    Features:
    - Request header normalization
    - Response header standardization
    - JSON response wrapping
    - Error response formatting
    - Request ID injection
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Transform request and response."""
        # Add trace ID to request state if not present
        if "X-Trace-ID" not in request.headers:
            import uuid

            request.state.trace_id = str(uuid.uuid4())
        else:
            request.state.trace_id = request.headers["X-Trace-ID"]

        # Add request ID if not present
        if "X-Request-ID" not in request.headers:
            import uuid

            request.state.request_id = str(uuid.uuid4())
        else:
            request.state.request_id = request.headers["X-Request-ID"]

        # Process request
        response = await call_next(request)

        # Add trace and request IDs to response headers
        response.headers["X-Trace-ID"] = request.state.trace_id
        response.headers["X-Request-ID"] = request.state.request_id

        # Add timing header if available
        if hasattr(request.state, "start_time"):
            import time

            duration_ms = (time.time() - request.state.start_time) * 1000
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Add cache control headers to responses.

    Configures appropriate caching strategies based on endpoint and response type.
    """

    # Default cache durations by content type (in seconds)
    DEFAULT_CACHE_DURATIONS = {
        "application/json": 0,  # No cache for JSON APIs
        "text/html": 3600,  # 1 hour for HTML
        "text/css": 86400,  # 1 day for CSS
        "application/javascript": 86400,  # 1 day for JS
        "image/png": 604800,  # 1 week for images
        "image/jpeg": 604800,
        "image/svg+xml": 604800,
    }

    # Paths that should never be cached
    NO_CACHE_PATHS = {
        "/api/v1/auth",
        "/api/v1/oauth",
        "/api/v1/health",
        "/metrics",
    }

    def __init__(
        self,
        app,
        enable_etag: bool = True,
        default_max_age: int = 0,
        cache_durations: dict[str, int] | None = None,
    ):
        """
        Initialize cache control middleware.

        Args:
            app: FastAPI application
            enable_etag: Generate and validate ETags
            default_max_age: Default max-age for cacheable responses
            cache_durations: Custom cache durations by content type
        """
        super().__init__(app)
        self.enable_etag = enable_etag
        self.default_max_age = default_max_age
        self.cache_durations = cache_durations or self.DEFAULT_CACHE_DURATIONS

    def _should_not_cache(self, request: Request) -> bool:
        """Check if request path should not be cached."""
        for path in self.NO_CACHE_PATHS:
            if request.url.path.startswith(path):
                return True
        return False

    def _generate_etag(self, content: bytes) -> str:
        """Generate ETag from response content."""
        import hashlib

        return hashlib.md5(content).hexdigest()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add cache control headers."""
        response = await call_next(request)

        # Skip caching for certain paths
        if self._should_not_cache(request):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        # Get content type
        content_type = response.headers.get("content-type", "").split(";")[0].strip()

        # Get cache duration
        max_age = self.cache_durations.get(content_type, self.default_max_age)

        # Set cache headers
        if max_age > 0:
            response.headers["Cache-Control"] = f"public, max-age={max_age}"

            # Generate ETag if enabled
            if self.enable_etag:
                # Get response body
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk

                etag = self._generate_etag(response_body)
                response.headers["ETag"] = f'"{etag}"'

                # Check If-None-Match header
                if_none_match = request.headers.get("if-none-match")
                if if_none_match and if_none_match.strip('"') == etag:
                    # Return 304 Not Modified
                    return Response(
                        content=b"",
                        status_code=304,
                        headers=dict(response.headers),
                    )

                # Return response with body
                return Response(
                    content=response_body,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=content_type,
                )
        else:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

        return response
