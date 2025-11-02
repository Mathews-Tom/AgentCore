"""Comprehensive tests for A2A Client with retry, pooling, and circuit breaker.

This module tests the enhanced A2A client features:
- Retry logic with exponential backoff
- Connection pooling configuration
- Circuit breaker pattern
- Rate limit handling
- Timeout handling
- Error recovery scenarios

Test Coverage:
- Retry scenarios (transient failures, permanent failures)
- Circuit breaker states (closed, open, half-open)
- Rate limit handling with Retry-After header
- Concurrent request handling with connection pooling
- Error type validation (TimeoutError, ConnectionError, RateLimitError, CircuitOpenError)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from agentcore.agent_runtime.services.a2a_client import (
    A2ACircuitOpenError,
    A2AClient,
    A2AClientError,
    A2AConnectionError,
    A2ARateLimitError,
    A2ATimeoutError,
)


class TestA2AClientRetryLogic:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry on connection errors with exponential backoff."""
        client = A2AClient(timeout=5.0, max_retries=3)

        with patch.object(client, "_client") as mock_client:
            # First 2 attempts fail, 3rd succeeds
            mock_client.post = AsyncMock(
                side_effect=[
                    httpx.ConnectError("Connection refused"),
                    httpx.ConnectError("Connection refused"),
                    Mock(
                        status_code=200,
                        json=lambda: {"jsonrpc": "2.0", "id": "123", "result": {"success": True}},
                    ),
                ]
            )

            async with client:
                result = await client._call_jsonrpc("test.method", {})

            assert result == {"success": True}
            assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises_error(self):
        """Test that exhausted retries raise appropriate error."""
        client = A2AClient(timeout=5.0, max_retries=3)

        with patch.object(client, "_client") as mock_client:
            # All attempts fail
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            async with client:
                with pytest.raises(A2AConnectionError, match="Connection failed after 3 attempts"):
                    await client._call_jsonrpc("test.method", {})

            assert mock_client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test exponential backoff delays (1s, 2s, 4s)."""
        client = A2AClient(timeout=5.0, max_retries=3)
        backoff_times = []

        async def mock_sleep(delay: float):
            backoff_times.append(delay)

        with patch.object(client, "_client") as mock_client:
            with patch("asyncio.sleep", side_effect=mock_sleep):
                mock_client.post = AsyncMock(
                    side_effect=httpx.TimeoutException("Timeout")
                )

                async with client:
                    with pytest.raises(A2ATimeoutError):
                        await client._call_jsonrpc("test.method", {})

        # Should have 2 backoffs (before 2nd and 3rd attempts)
        assert len(backoff_times) == 2
        assert backoff_times[0] == 1  # 2^0
        assert backoff_times[1] == 2  # 2^1

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx_errors(self):
        """Test that 4xx errors are not retried (except 429)."""
        client = A2AClient(timeout=5.0, max_retries=3)

        with patch.object(client, "_client") as mock_client:
            response_mock = Mock(status_code=400)
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Bad Request", request=Mock(), response=response_mock
                )
            )

            async with client:
                with pytest.raises(A2AClientError, match="HTTP 400"):
                    await client._call_jsonrpc("test.method", {})

            # Should only attempt once (no retries on 4xx)
            assert mock_client.post.call_count == 1


class TestA2AClientCircuitBreaker:
    """Test circuit breaker pattern."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold_failures(self):
        """Test circuit breaker opens after threshold failures."""
        client = A2AClient(
            timeout=5.0,
            max_retries=1,  # Fail fast
            circuit_breaker_threshold=3,
        )

        with patch.object(client, "_client") as mock_client:
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            async with client:
                # Fail 3 times to open circuit
                for _ in range(3):
                    with pytest.raises(A2AConnectionError):
                        await client._call_jsonrpc("test.method", {})

                # Circuit should now be open
                assert client._failure_count == 3
                assert client._circuit_open_until is not None

                # Next call should fail immediately with CircuitOpenError
                with pytest.raises(A2ACircuitOpenError, match="Circuit breaker is open"):
                    await client._call_jsonrpc("test.method", {})

    @pytest.mark.asyncio
    async def test_circuit_resets_after_timeout(self):
        """Test circuit breaker resets after timeout period."""
        client = A2AClient(
            timeout=5.0,
            max_retries=1,
            circuit_breaker_threshold=2,
            circuit_breaker_timeout=0.5,  # Short timeout for testing
        )

        with patch.object(client, "_client") as mock_client:
            # Fail to open circuit
            mock_client.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            async with client:
                # Open circuit
                for _ in range(2):
                    with pytest.raises(A2AConnectionError):
                        await client._call_jsonrpc("test.method", {})

                assert client._is_circuit_open()

                # Wait for circuit to reset
                await asyncio.sleep(0.6)

                # Circuit should be closed now
                assert not client._is_circuit_open()
                assert client._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_resets_on_success(self):
        """Test circuit breaker resets failure count on successful request."""
        client = A2AClient(
            timeout=5.0,
            max_retries=3,
            circuit_breaker_threshold=5,
        )

        with patch.object(client, "_client") as mock_client:
            # Fail once, then succeed
            mock_client.post = AsyncMock(
                side_effect=[
                    httpx.ConnectError("Connection refused"),
                    httpx.ConnectError("Connection refused"),
                    Mock(
                        status_code=200,
                        json=lambda: {"jsonrpc": "2.0", "id": "123", "result": {"success": True}},
                    ),
                ]
            )

            async with client:
                result = await client._call_jsonrpc("test.method", {})

            # Success should reset failure count
            assert client._failure_count == 0
            assert client._circuit_open_until is None


class TestA2AClientRateLimiting:
    """Test rate limit handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_retry_after_header(self):
        """Test rate limit retry respects Retry-After header."""
        client = A2AClient(timeout=5.0, max_retries=3)
        sleep_times = []

        async def mock_sleep(delay: float):
            sleep_times.append(delay)

        with patch.object(client, "_client") as mock_client:
            with patch("asyncio.sleep", side_effect=mock_sleep):
                # First request gets rate limited with Retry-After: 2
                # Second request succeeds
                mock_client.post = AsyncMock(
                    side_effect=[
                        Mock(
                            status_code=429,
                            headers={"Retry-After": "2"},
                            raise_for_status=Mock(side_effect=httpx.HTTPStatusError(
                                "Rate limited", request=Mock(), response=Mock(status_code=429)
                            )),
                        ),
                        Mock(
                            status_code=200,
                            json=lambda: {"jsonrpc": "2.0", "id": "123", "result": {"success": True}},
                        ),
                    ]
                )

                async with client:
                    result = await client._call_jsonrpc("test.method", {})

        assert result == {"success": True}
        assert 2 in sleep_times  # Should have slept for Retry-After duration

    @pytest.mark.asyncio
    async def test_rate_limit_exhausted_retries_raises_error(self):
        """Test that exhausted rate limit retries raise RateLimitError."""
        client = A2AClient(timeout=5.0, max_retries=2)

        with patch.object(client, "_client") as mock_client:
            # Always return 429
            mock_client.post = AsyncMock(
                return_value=Mock(
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
            )

            async with client:
                with pytest.raises(A2ARateLimitError, match="Rate limit exceeded"):
                    await client._call_jsonrpc("test.method", {})


class TestA2AClientTimeout:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_error_raised(self):
        """Test that timeout errors are properly raised."""
        client = A2AClient(timeout=5.0, max_retries=3)

        with patch.object(client, "_client") as mock_client:
            mock_client.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timeout")
            )

            async with client:
                with pytest.raises(A2ATimeoutError, match="Request timed out after 3 attempts"):
                    await client._call_jsonrpc("test.method", {})

    @pytest.mark.asyncio
    async def test_timeout_with_retry(self):
        """Test timeout is retried with exponential backoff."""
        client = A2AClient(timeout=5.0, max_retries=3)

        with patch.object(client, "_client") as mock_client:
            # First 2 timeout, 3rd succeeds
            mock_client.post = AsyncMock(
                side_effect=[
                    httpx.TimeoutException("Timeout"),
                    httpx.TimeoutException("Timeout"),
                    Mock(
                        status_code=200,
                        json=lambda: {"jsonrpc": "2.0", "id": "123", "result": {"success": True}},
                    ),
                ]
            )

            async with client:
                result = await client._call_jsonrpc("test.method", {})

            assert result == {"success": True}
            assert mock_client.post.call_count == 3


class TestA2AClientConnectionPooling:
    """Test connection pooling configuration."""

    @pytest.mark.asyncio
    async def test_connection_limits_configured(self):
        """Test connection pool limits are properly configured."""
        client = A2AClient(max_connections=15)

        async with client:
            assert client._client is not None
            assert client._client._limits.max_connections == 15
            assert client._client._limits.max_keepalive_connections == 15

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests with connection pooling."""
        client = A2AClient(max_connections=5)

        async def mock_post(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate network delay
            return Mock(
                status_code=200,
                json=lambda: {"jsonrpc": "2.0", "id": "123", "result": {"success": True}},
            )

        with patch.object(client, "_client") as mock_client:
            mock_client.post = AsyncMock(side_effect=mock_post)

            async with client:
                # Make 10 concurrent requests
                tasks = [
                    client._call_jsonrpc("test.method", {"index": i})
                    for i in range(10)
                ]
                results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r == {"success": True} for r in results)


class TestA2AClientErrorHandling:
    """Test comprehensive error handling."""

    @pytest.mark.asyncio
    async def test_jsonrpc_error_response(self):
        """Test handling of JSON-RPC error responses."""
        client = A2AClient(timeout=5.0)

        with patch.object(client, "_client") as mock_client:
            mock_client.post = AsyncMock(
                return_value=Mock(
                    status_code=200,
                    json=lambda: {
                        "jsonrpc": "2.0",
                        "id": "123",
                        "error": {"code": -32600, "message": "Invalid Request"},
                    },
                )
            )

            async with client:
                with pytest.raises(A2AClientError, match="JSON-RPC error: Invalid Request"):
                    await client._call_jsonrpc("test.method", {})

    @pytest.mark.asyncio
    async def test_client_not_initialized_error(self):
        """Test error when client is not initialized."""
        client = A2AClient()

        with pytest.raises(A2AConnectionError, match="Client not initialized"):
            await client._call_jsonrpc("test.method", {})


class TestA2AClientContextManager:
    """Test async context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_client(self):
        """Test context manager properly initializes HTTP client."""
        client = A2AClient()

        assert client._client is None

        async with client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)

        assert client._client is None  # Should be closed after exit

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_error(self):
        """Test context manager cleans up even if error occurs."""
        client = A2AClient()

        try:
            async with client:
                assert client._client is not None
                raise ValueError("Test error")
        except ValueError:
            pass

        assert client._client is None  # Should be cleaned up despite error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
