"""Async HTTP API client with authentication, rate limiting, and retry logic.

RESTful API client using httpx with comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

import httpx
import structlog
from redis.asyncio import Redis

from agentcore.integration.api.exceptions import (
    APIAuthenticationError,
    APIAuthorizationError,
    APIConnectionError,
    APIError,
    APINotFoundError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    APIValidationError,
)
from agentcore.integration.api.models import (
    APIConfig,
    APIRequest,
    APIResponse,
    AuthScheme,
    HTTPMethod,
)
from agentcore.integration.api.rate_limiter import RateLimiter
from agentcore.integration.api.transformer import ResponseTransformer

logger = structlog.get_logger(__name__)


class APIClient:
    """Async HTTP API client with authentication, rate limiting, and retry.

    Features:
    - Multiple authentication schemes (Bearer, Basic, API Key, OAuth2)
    - Automatic retry with exponential backoff
    - Rate limiting with token bucket algorithm
    - Request/response transformation
    - Connection pooling
    - Timeout handling
    - Comprehensive error handling
    """

    def __init__(
        self,
        config: APIConfig,
        redis_client: Redis[Any] | None = None,
        transformer: ResponseTransformer | None = None,
    ) -> None:
        """Initialize API client.

        Args:
            config: API configuration
            redis_client: Optional Redis client for distributed rate limiting
            transformer: Optional response transformer
        """
        self.config = config
        self._transformer = transformer or ResponseTransformer()
        self._closed = False

        # Initialize rate limiter
        self._rate_limiter = RateLimiter(config.rate_limit, redis_client)

        # Create httpx client
        limits = httpx.Limits(
            max_connections=config.pool_connections,
            max_keepalive_connections=config.pool_maxsize,
            keepalive_expiry=config.pool_keepalive,
        )

        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
            limits=limits,
            follow_redirects=config.follow_redirects,
            max_redirects=config.max_redirects,
            verify=config.verify_ssl,
        )

        # OAuth2 token cache
        self._oauth2_token: str | None = None
        self._oauth2_token_expires: float | None = None

        logger.info(
            "api_client_initialized",
            name=config.name,
            base_url=config.base_url,
            auth_scheme=config.auth.scheme,
        )

    async def request(
        self,
        request: APIRequest,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute an HTTP request with retry logic.

        Args:
            request: API request
            **kwargs: Additional httpx request parameters

        Returns:
            API response

        Raises:
            APIError: If request fails after all retries
        """
        if self._closed:
            raise APIError("API client is closed")

        # Apply rate limiting
        rate_limit_key = self._get_rate_limit_key(request)
        # Use None timeout to raise immediately when rate limit exceeded
        await self._rate_limiter.acquire(rate_limit_key, timeout=None)

        # Execute request with retry
        if self.config.retry.enabled and request.retry:
            return await self._request_with_retry(request, **kwargs)

        return await self._execute_request(request, 1, **kwargs)

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute GET request.

        Args:
            url: Request URL
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.GET,
            url=url,
            params=params or {},
            request_id=str(uuid.uuid4()),
        )
        return await self.request(request, **kwargs)

    async def post(
        self,
        url: str,
        body: dict[str, Any] | str | bytes | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute POST request.

        Args:
            url: Request URL
            body: Request body
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.POST,
            url=url,
            body=body,
            params=params or {},
            request_id=str(uuid.uuid4()),
        )
        return await self.request(request, **kwargs)

    async def put(
        self,
        url: str,
        body: dict[str, Any] | str | bytes | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute PUT request.

        Args:
            url: Request URL
            body: Request body
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.PUT,
            url=url,
            body=body,
            params=params or {},
            request_id=str(uuid.uuid4()),
        )
        return await self.request(request, **kwargs)

    async def patch(
        self,
        url: str,
        body: dict[str, Any] | str | bytes | None = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute PATCH request.

        Args:
            url: Request URL
            body: Request body
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.PATCH,
            url=url,
            body=body,
            params=params or {},
            request_id=str(uuid.uuid4()),
        )
        return await self.request(request, **kwargs)

    async def delete(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute DELETE request.

        Args:
            url: Request URL
            params: Query parameters
            **kwargs: Additional request parameters

        Returns:
            API response
        """
        request = APIRequest(
            method=HTTPMethod.DELETE,
            url=url,
            params=params or {},
            request_id=str(uuid.uuid4()),
        )
        return await self.request(request, **kwargs)

    async def _request_with_retry(
        self,
        request: APIRequest,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute request with retry logic.

        Args:
            request: API request
            **kwargs: Additional httpx request parameters

        Returns:
            API response

        Raises:
            APIError: If all retries fail
        """
        last_error: Exception | None = None
        backoff_ms = self.config.retry.initial_backoff_ms

        for attempt in range(1, self.config.retry.max_attempts + 1):
            try:
                return await self._execute_request(request, attempt, **kwargs)
            except (
                APIConnectionError,
                APITimeoutError,
                APIServerError,
                APIRateLimitError,
            ) as e:
                last_error = e

                # Check if we should retry
                if attempt >= self.config.retry.max_attempts:
                    break

                # Check if status code is retryable
                if (
                    hasattr(e, "status_code")
                    and e.status_code
                    and e.status_code not in self.config.retry.retry_on_status_codes
                ):
                    raise

                # Calculate backoff with jitter
                if self.config.retry.jitter:
                    jitter = random.uniform(0, backoff_ms * 0.1)
                    sleep_ms = backoff_ms + jitter
                else:
                    sleep_ms = backoff_ms

                logger.warning(
                    "request_retry",
                    attempt=attempt,
                    max_attempts=self.config.retry.max_attempts,
                    backoff_ms=sleep_ms,
                    error=str(e),
                )

                # Sleep before retry
                await asyncio.sleep(sleep_ms / 1000)

                # Increase backoff
                backoff_ms = min(
                    backoff_ms * self.config.retry.backoff_multiplier,
                    self.config.retry.max_backoff_ms,
                )

        # All retries failed
        raise last_error or APIError("Request failed after all retries")

    async def _execute_request(
        self,
        request: APIRequest,
        attempt: int,
        **kwargs: Any,
    ) -> APIResponse:
        """Execute a single HTTP request.

        Args:
            request: API request
            attempt: Attempt number
            **kwargs: Additional httpx request parameters

        Returns:
            API response

        Raises:
            APIError: If request fails
        """
        # Build request URL
        url = self._build_url(request.url)

        # Build headers
        headers = await self._build_headers(request)

        # Build request parameters
        timeout = request.timeout or self.config.timeout_seconds
        params = request.params

        # Prepare body
        body_data: dict[str, Any] | str | bytes | None = None
        json_data: dict[str, Any] | None = None

        if request.body:
            if isinstance(request.body, dict):
                json_data = request.body
            else:
                body_data = request.body

        # Execute request
        request_time = datetime.now(UTC)
        start_time = time.monotonic()

        try:
            response = await self._client.request(
                method=request.method.value,
                url=url,
                params=params,
                headers=headers,
                data=body_data,
                json=json_data,
                timeout=timeout,
                **kwargs,
            )
        except httpx.TimeoutException as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "request_timeout",
                url=url,
                method=request.method,
                duration_ms=duration_ms,
            )
            raise APITimeoutError(
                f"Request timeout after {timeout}s",
                status_code=None,
            ) from e
        except httpx.ConnectError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "request_connection_error",
                url=url,
                method=request.method,
                duration_ms=duration_ms,
            )
            raise APIConnectionError(
                f"Connection failed: {e}",
                status_code=None,
            ) from e
        except httpx.HTTPError as e:
            duration_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "request_http_error",
                url=url,
                method=request.method,
                duration_ms=duration_ms,
                error=str(e),
            )
            raise APIError(f"HTTP error: {e}") from e

        response_time = datetime.now(UTC)
        duration_ms = (time.monotonic() - start_time) * 1000

        # Check for errors
        self._check_response_errors(response)

        # Parse response
        raw_body = response.text
        content_type = response.headers.get("content-type")

        try:
            parsed_body = self._transformer.transform(raw_body, content_type)
        except Exception as e:
            logger.warning("response_transformation_failed", error=str(e))
            parsed_body = raw_body

        # Build API response
        api_response = APIResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            body=parsed_body,
            raw_body=raw_body,
            request_time=request_time,
            response_time=response_time,
            duration_ms=duration_ms,
            attempt_number=attempt,
            total_attempts=self.config.retry.max_attempts
            if self.config.retry.enabled
            else 1,
            request=request,
        )

        logger.info(
            "request_completed",
            url=url,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            attempt=attempt,
        )

        return api_response

    def _build_url(self, url: str) -> str:
        """Build full URL from base URL and request URL.

        Args:
            url: Request URL (can be relative or absolute)

        Returns:
            Full URL
        """
        if url.startswith(("http://", "https://")):
            return url

        return urljoin(self.config.base_url, url)

    async def _build_headers(self, request: APIRequest) -> dict[str, str]:
        """Build request headers with authentication.

        Args:
            request: API request

        Returns:
            Request headers
        """
        headers = dict(self.config.default_headers)
        headers.update(request.headers)

        # Add authentication headers
        if self.config.auth.scheme == AuthScheme.BEARER:
            if self.config.auth.token:
                headers["Authorization"] = (
                    f"Bearer {self.config.auth.token.get_secret_value()}"
                )

        elif self.config.auth.scheme == AuthScheme.BASIC:
            if self.config.auth.username and self.config.auth.password:
                import base64

                credentials = f"{self.config.auth.username}:{self.config.auth.password.get_secret_value()}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers["Authorization"] = f"Basic {encoded}"

        elif self.config.auth.scheme == AuthScheme.API_KEY:
            if self.config.auth.api_key and self.config.auth.api_key_header:
                headers[self.config.auth.api_key_header] = (
                    self.config.auth.api_key.get_secret_value()
                )

        elif self.config.auth.scheme == AuthScheme.OAUTH2:
            token = await self._get_oauth2_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        return headers

    async def _get_oauth2_token(self) -> str | None:
        """Get OAuth2 access token (with caching and refresh).

        Returns:
            Access token or None
        """
        # Check if token is still valid
        if self._oauth2_token and self._oauth2_token_expires:
            if time.time() < self._oauth2_token_expires - 60:  # 60s buffer
                return self._oauth2_token

        # Request new token
        if not all(
            [
                self.config.auth.oauth2_token_url,
                self.config.auth.oauth2_client_id,
                self.config.auth.oauth2_client_secret,
            ]
        ):
            return None

        try:
            response = await self._client.post(
                self.config.auth.oauth2_token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.config.auth.oauth2_client_id,
                    "client_secret": self.config.auth.oauth2_client_secret.get_secret_value(),
                    "scope": self.config.auth.oauth2_scope or "",
                },
            )
            response.raise_for_status()

            token_data = response.json()
            self._oauth2_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._oauth2_token_expires = time.time() + expires_in

            logger.info("oauth2_token_refreshed", expires_in=expires_in)
            return self._oauth2_token

        except Exception as e:
            logger.error("oauth2_token_refresh_failed", error=str(e))
            return None

    def _check_response_errors(self, response: httpx.Response) -> None:
        """Check response for HTTP errors.

        Args:
            response: httpx response

        Raises:
            APIError: If response contains an error
        """
        if response.status_code >= 200 and response.status_code < 300:
            return

        # Extract error message
        try:
            error_body = response.json()
            error_message = (
                error_body.get("error") or error_body.get("message") or response.text
            )
        except Exception:
            error_message = response.text

        # Map status codes to exceptions
        if response.status_code == 401:
            raise APIAuthenticationError(
                error_message,
                status_code=response.status_code,
                response_body=response.text,
            )

        if response.status_code == 403:
            raise APIAuthorizationError(
                error_message,
                status_code=response.status_code,
                response_body=response.text,
            )

        if response.status_code == 404:
            raise APINotFoundError(
                error_message,
                status_code=response.status_code,
                response_body=response.text,
            )

        if response.status_code == 422:
            raise APIValidationError(
                error_message,
                status_code=response.status_code,
                response_body=response.text,
            )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_after_seconds = int(retry_after) if retry_after else None
            raise APIRateLimitError(
                error_message,
                retry_after=retry_after_seconds,
                status_code=response.status_code,
                response_body=response.text,
            )

        if response.status_code >= 500:
            raise APIServerError(
                error_message,
                status_code=response.status_code,
                response_body=response.text,
            )

        # Generic error
        raise APIError(
            error_message,
            status_code=response.status_code,
            response_body=response.text,
        )

    def _get_rate_limit_key(self, request: APIRequest) -> str:
        """Get rate limit key for request.

        Args:
            request: API request

        Returns:
            Rate limit key
        """
        parts = [self.config.name]

        if self.config.rate_limit.per_endpoint:
            parts.append(request.url)

        if self.config.rate_limit.per_tenant and request.tenant_id:
            parts.append(request.tenant_id)

        return ":".join(parts)

    async def close(self) -> None:
        """Close the API client and release resources."""
        if not self._closed:
            await self._client.aclose()
            self._closed = True
            logger.info("api_client_closed", name=self.config.name)

    async def __aenter__(self) -> APIClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager."""
        await self.close()
