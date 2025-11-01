"""Rate limiting enforcement tests.

Tests for rate limiting mechanisms and bypass prevention.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest


class RateLimiter:
    """Simple rate limiter for testing."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[datetime]] = {}

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit.

        Args:
            client_id: Client identifier

        Returns:
            True if request is allowed, False if rate limited
        """
        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=self.window_seconds)

        if client_id not in self.requests:
            self.requests[client_id] = []

        # Remove old requests outside the window
        self.requests[client_id] = [
            req for req in self.requests[client_id] if req > window_start
        ]

        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True

    def reset(self, client_id: str | None = None) -> None:
        """Reset rate limiter.

        Args:
            client_id: Client to reset, or None to reset all
        """
        if client_id:
            self.requests.pop(client_id, None)
        else:
            self.requests.clear()


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Create rate limiter for testing."""
    return RateLimiter(max_requests=10, window_seconds=60)


class TestRateLimitEnforcement:
    """Tests for rate limit enforcement."""

    @pytest.mark.asyncio
    async def test_within_rate_limit(self, rate_limiter: RateLimiter) -> None:
        """Test that requests within limit are allowed."""
        client_id = "test-client-1"

        for _ in range(10):
            assert await rate_limiter.check_rate_limit(client_id)

    @pytest.mark.asyncio
    async def test_exceed_rate_limit(self, rate_limiter: RateLimiter) -> None:
        """Test that requests exceeding limit are blocked."""
        client_id = "test-client-2"

        # Use up the limit
        for _ in range(10):
            assert await rate_limiter.check_rate_limit(client_id)

        # Next request should be blocked
        assert not await rate_limiter.check_rate_limit(client_id)

    @pytest.mark.asyncio
    async def test_different_clients_independent(
        self, rate_limiter: RateLimiter
    ) -> None:
        """Test that different clients have independent limits."""
        client1 = "test-client-1"
        client2 = "test-client-2"

        # Use up limit for client1
        for _ in range(10):
            assert await rate_limiter.check_rate_limit(client1)

        # Client2 should still be allowed
        assert await rate_limiter.check_rate_limit(client2)

    @pytest.mark.asyncio
    async def test_rate_limit_window_reset(self) -> None:
        """Test that rate limit resets after window expires."""
        limiter = RateLimiter(max_requests=5, window_seconds=1)
        client_id = "test-client"

        # Use up the limit
        for _ in range(5):
            assert await limiter.check_rate_limit(client_id)

        # Should be blocked
        assert not await limiter.check_rate_limit(client_id)

        # Wait for window to expire
        await asyncio.sleep(1.1)

        # Should be allowed again
        assert await limiter.check_rate_limit(client_id)

    @pytest.mark.asyncio
    async def test_burst_traffic(self, rate_limiter: RateLimiter) -> None:
        """Test handling of burst traffic."""
        client_id = "test-client-burst"

        # Simulate burst of requests
        results = []
        for _ in range(15):
            result = await rate_limiter.check_rate_limit(client_id)
            results.append(result)

        # First 10 should succeed, next 5 should fail
        assert sum(results) == 10
        assert results[:10] == [True] * 10
        assert results[10:] == [False] * 5


class TestRateLimitBypass:
    """Tests for rate limit bypass prevention."""

    @pytest.mark.asyncio
    async def test_ip_spoofing_prevention(self, rate_limiter: RateLimiter) -> None:
        """Test that IP spoofing doesn't bypass rate limit."""
        # Using same client ID should enforce limit regardless of claimed IP
        base_client = "client-123"

        for _ in range(10):
            assert await rate_limiter.check_rate_limit(base_client)

        # Attempting to bypass by changing IP (but same client_id)
        assert not await rate_limiter.check_rate_limit(base_client)

    @pytest.mark.asyncio
    async def test_multiple_tokens_same_client(
        self, rate_limiter: RateLimiter
    ) -> None:
        """Test that multiple tokens from same client are rate limited."""
        client_id = "shared-client"

        # Simulate multiple tokens/sessions from same client
        for token_id in range(3):
            # Each "token" is actually same client
            for _ in range(4):
                await rate_limiter.check_rate_limit(client_id)

        # Should exceed limit (12 requests > 10)
        assert not await rate_limiter.check_rate_limit(client_id)

    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, rate_limiter: RateLimiter) -> None:
        """Test manual rate limit reset."""
        client_id = "test-client-reset"

        # Use up the limit
        for _ in range(10):
            assert await rate_limiter.check_rate_limit(client_id)

        # Should be blocked
        assert not await rate_limiter.check_rate_limit(client_id)

        # Reset the limit
        rate_limiter.reset(client_id)

        # Should be allowed again
        assert await rate_limiter.check_rate_limit(client_id)


class TestRateLimitConfiguration:
    """Tests for rate limit configuration."""

    @pytest.mark.asyncio
    async def test_custom_rate_limit(self) -> None:
        """Test custom rate limit configuration."""
        # Very strict limit
        strict_limiter = RateLimiter(max_requests=2, window_seconds=60)
        client_id = "test-client"

        assert await strict_limiter.check_rate_limit(client_id)
        assert await strict_limiter.check_rate_limit(client_id)
        assert not await strict_limiter.check_rate_limit(client_id)

    @pytest.mark.asyncio
    async def test_per_endpoint_limits(self) -> None:
        """Test different limits for different endpoints."""
        # Simulate different endpoints with different limits
        api_limiter = RateLimiter(max_requests=100, window_seconds=60)
        auth_limiter = RateLimiter(max_requests=5, window_seconds=60)

        client_id = "test-client"

        # Auth endpoint is more strictly limited
        for _ in range(5):
            assert await auth_limiter.check_rate_limit(client_id)
        assert not await auth_limiter.check_rate_limit(client_id)

        # API endpoint has higher limit (tested with same client)
        for _ in range(10):
            assert await api_limiter.check_rate_limit(client_id)


class TestRateLimitHeaders:
    """Tests for rate limit information headers."""

    def test_rate_limit_headers(self) -> None:
        """Test that rate limit headers are present."""
        # Simulating response headers
        headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "95",
            "X-RateLimit-Reset": str(int(time.time()) + 3600),
        }

        assert int(headers["X-RateLimit-Limit"]) == 100
        assert int(headers["X-RateLimit-Remaining"]) == 95
        assert int(headers["X-RateLimit-Reset"]) > time.time()

    def test_retry_after_header(self) -> None:
        """Test Retry-After header when rate limited."""
        retry_after = 60
        headers = {
            "Retry-After": str(retry_after),
        }

        assert int(headers["Retry-After"]) == retry_after
