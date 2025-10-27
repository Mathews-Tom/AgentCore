"""Tests for API client with authentication, rate limiting, and retry logic."""

from __future__ import annotations

import pytest
import respx
from httpx import Response
from pydantic import SecretStr

from agentcore.integration.api import (
    APIAuthenticationError,
    APIClient,
    APIConfig,
    APINotFoundError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
    AuthConfig,
    AuthScheme,
    HTTPMethod,
    RateLimitConfig,
    RetryConfig)


@pytest.fixture
def base_config() -> APIConfig:
    """Create base API configuration for testing."""
    return APIConfig(
        name="test-api",
        base_url="https://api.example.com",
        timeout_seconds=5,
        rate_limit=RateLimitConfig(enabled=False),  # Disable for most tests
        retry=RetryConfig(enabled=False),  # Disable for most tests
    )


@pytest.fixture
def bearer_auth_config(base_config: APIConfig) -> APIConfig:
    """Create configuration with Bearer auth."""
    base_config.auth = AuthConfig(
        scheme=AuthScheme.BEARER,
        token=SecretStr("test-token-123"))
    return base_config


@pytest.fixture
def basic_auth_config(base_config: APIConfig) -> APIConfig:
    """Create configuration with Basic auth."""
    base_config.auth = AuthConfig(
        scheme=AuthScheme.BASIC,
        username="testuser",
        password=SecretStr("testpass"))
    return base_config


@pytest.fixture
def api_key_config(base_config: APIConfig) -> APIConfig:
    """Create configuration with API Key auth."""
    base_config.auth = AuthConfig(
        scheme=AuthScheme.API_KEY,
        api_key=SecretStr("test-api-key"),
        api_key_header="X-API-Key")
    return base_config


class TestAPIClientBasicOperations:
    """Test basic API client operations."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_get_request(self, base_config: APIConfig) -> None:
        """Test GET request."""
        route = respx.get("https://api.example.com/users").mock(
            return_value=Response(200, json={"users": [{"id": 1, "name": "John"}]})
        )

        async with APIClient(base_config) as client:
            response = await client.get("/users")

        assert route.called
        assert response.status_code == 200
        assert response.body == {"users": [{"id": 1, "name": "John"}]}

    @respx.mock
    @pytest.mark.asyncio
    async def test_post_request(self, base_config: APIConfig) -> None:
        """Test POST request."""
        route = respx.post("https://api.example.com/users").mock(
            return_value=Response(201, json={"id": 2, "name": "Jane"})
        )

        async with APIClient(base_config) as client:
            response = await client.post("/users", body={"name": "Jane"})

        assert route.called
        assert response.status_code == 201
        assert response.body == {"id": 2, "name": "Jane"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_put_request(self, base_config: APIConfig) -> None:
        """Test PUT request."""
        route = respx.put("https://api.example.com/users/1").mock(
            return_value=Response(200, json={"id": 1, "name": "John Updated"})
        )

        async with APIClient(base_config) as client:
            response = await client.put("/users/1", body={"name": "John Updated"})

        assert route.called
        assert response.status_code == 200
        assert response.body["name"] == "John Updated"

    @respx.mock
    @pytest.mark.asyncio
    async def test_patch_request(self, base_config: APIConfig) -> None:
        """Test PATCH request."""
        route = respx.patch("https://api.example.com/users/1").mock(
            return_value=Response(200, json={"id": 1, "name": "John Patched"})
        )

        async with APIClient(base_config) as client:
            response = await client.patch("/users/1", body={"name": "John Patched"})

        assert route.called
        assert response.status_code == 200

    @respx.mock
    @pytest.mark.asyncio
    async def test_delete_request(self, base_config: APIConfig) -> None:
        """Test DELETE request."""
        route = respx.delete("https://api.example.com/users/1").mock(
            return_value=Response(204)
        )

        async with APIClient(base_config) as client:
            response = await client.delete("/users/1")

        assert route.called
        assert response.status_code == 204


class TestAuthentication:
    """Test authentication schemes."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_bearer_authentication(self, bearer_auth_config: APIConfig) -> None:
        """Test Bearer token authentication."""
        route = respx.get("https://api.example.com/protected").mock(
            return_value=Response(200, json={"data": "secret"})
        )

        async with APIClient(bearer_auth_config) as client:
            response = await client.get("/protected")

        assert route.called
        request = route.calls.last.request
        assert request.headers["Authorization"] == "Bearer test-token-123"
        assert response.status_code == 200

    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_authentication(self, basic_auth_config: APIConfig) -> None:
        """Test Basic authentication."""
        route = respx.get("https://api.example.com/protected").mock(
            return_value=Response(200, json={"data": "secret"})
        )

        async with APIClient(basic_auth_config) as client:
            response = await client.get("/protected")

        assert route.called
        request = route.calls.last.request
        assert "Authorization" in request.headers
        assert request.headers["Authorization"].startswith("Basic ")

    @respx.mock
    @pytest.mark.asyncio
    async def test_api_key_authentication(self, api_key_config: APIConfig) -> None:
        """Test API Key authentication."""
        route = respx.get("https://api.example.com/protected").mock(
            return_value=Response(200, json={"data": "secret"})
        )

        async with APIClient(api_key_config) as client:
            response = await client.get("/protected")

        assert route.called
        request = route.calls.last.request
        assert request.headers["X-API-Key"] == "test-api-key"


class TestErrorHandling:
    """Test error handling."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_authentication_error(self, base_config: APIConfig) -> None:
        """Test 401 authentication error."""
        respx.get("https://api.example.com/protected").mock(
            return_value=Response(401, json={"error": "Unauthorized"})
        )

        async with APIClient(base_config) as client:
            with pytest.raises(APIAuthenticationError) as exc_info:
                await client.get("/protected")

        assert exc_info.value.status_code == 401

    @respx.mock
    @pytest.mark.asyncio
    async def test_404_not_found_error(self, base_config: APIConfig) -> None:
        """Test 404 not found error."""
        respx.get("https://api.example.com/missing").mock(
            return_value=Response(404, json={"error": "Not found"})
        )

        async with APIClient(base_config) as client:
            with pytest.raises(APINotFoundError) as exc_info:
                await client.get("/missing")

        assert exc_info.value.status_code == 404

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_rate_limit_error(self, base_config: APIConfig) -> None:
        """Test 429 rate limit error."""
        respx.get("https://api.example.com/limited").mock(
            return_value=Response(
                429,
                headers={"Retry-After": "60"},
                json={"error": "Rate limit exceeded"})
        )

        async with APIClient(base_config) as client:
            with pytest.raises(APIRateLimitError) as exc_info:
                await client.get("/limited")

        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 60

    @respx.mock
    @pytest.mark.asyncio
    async def test_500_server_error(self, base_config: APIConfig) -> None:
        """Test 500 server error."""
        respx.get("https://api.example.com/error").mock(
            return_value=Response(500, json={"error": "Internal server error"})
        )

        async with APIClient(base_config) as client:
            with pytest.raises(APIServerError) as exc_info:
                await client.get("/error")

        assert exc_info.value.status_code == 500


class TestRetryLogic:
    """Test retry logic."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, base_config: APIConfig) -> None:
        """Test retry on timeout error."""
        base_config.retry = RetryConfig(
            enabled=True,
            max_attempts=3,
            initial_backoff_ms=10,
            jitter=False)

        # First two attempts timeout, third succeeds
        route = respx.get("https://api.example.com/slow").mock(
            side_effect=[
                Response(504, json={"error": "Timeout"}),
                Response(504, json={"error": "Timeout"}),
                Response(200, json={"data": "success"}),
            ]
        )

        async with APIClient(base_config) as client:
            response = await client.get("/slow")

        assert route.call_count == 3
        assert response.status_code == 200
        assert response.attempt_number == 3

    @respx.mock
    @pytest.mark.asyncio
    async def test_retry_exhausted(self, base_config: APIConfig) -> None:
        """Test retry exhaustion."""
        base_config.retry = RetryConfig(
            enabled=True,
            max_attempts=2,
            initial_backoff_ms=10,
            jitter=False)

        route = respx.get("https://api.example.com/broken").mock(
            return_value=Response(500, json={"error": "Server error"})
        )

        async with APIClient(base_config) as client:
            with pytest.raises(APIServerError):
                await client.get("/broken")

        assert route.call_count == 2


class TestRateLimiting:
    """Test rate limiting."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limiting_local(self, base_config: APIConfig) -> None:
        """Test local rate limiting."""
        base_config.rate_limit = RateLimitConfig(
            enabled=True,
            requests_per_window=2,
            window_seconds=10,  # Longer window to ensure rate limit is hit
            burst_size=2)

        respx.get("https://api.example.com/data").mock(
            return_value=Response(200, json={"data": "ok"})
        )

        async with APIClient(base_config) as client:
            # First two requests should succeed
            response1 = await client.get("/data")
            response2 = await client.get("/data")

            assert response1.status_code == 200
            assert response2.status_code == 200

            # Third request should be rate limited (no timeout, should raise immediately)
            with pytest.raises(APIRateLimitError):
                await client.get("/data")


class TestResponseTransformation:
    """Test response transformation."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_json_response_parsing(self, base_config: APIConfig) -> None:
        """Test JSON response parsing."""
        respx.get("https://api.example.com/json").mock(
            return_value=Response(
                200,
                headers={"Content-Type": "application/json"},
                json={"key": "value", "nested": {"data": "test"}})
        )

        async with APIClient(base_config) as client:
            response = await client.get("/json")

        assert isinstance(response.body, dict)
        assert response.body["key"] == "value"
        assert response.body["nested"]["data"] == "test"

    @respx.mock
    @pytest.mark.asyncio
    async def test_text_response_parsing(self, base_config: APIConfig) -> None:
        """Test plain text response parsing."""
        respx.get("https://api.example.com/text").mock(
            return_value=Response(
                200,
                headers={"Content-Type": "text/plain"},
                text="Plain text response")
        )

        async with APIClient(base_config) as client:
            response = await client.get("/text")

        assert isinstance(response.body, str)
        assert response.body == "Plain text response"


class TestConnectionLifecycle:
    """Test connection lifecycle."""

    @pytest.mark.asyncio
    async def test_context_manager(self, base_config: APIConfig) -> None:
        """Test async context manager."""
        async with APIClient(base_config) as client:
            assert client is not None
            assert not client._closed

        # Client should be closed after context
        assert client._closed

    @pytest.mark.asyncio
    async def test_manual_close(self, base_config: APIConfig) -> None:
        """Test manual close."""
        client = APIClient(base_config)
        assert not client._closed

        await client.close()
        assert client._closed
