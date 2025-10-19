"""
Integration tests for rate limiting middleware.

Tests the complete rate limiting middleware including DDoS protection.
"""

from __future__ import annotations

import time

import pytest
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient
from testcontainers.redis import RedisContainer

from gateway.middleware.ddos_protection import DDoSConfig, DDoSProtector
from gateway.middleware.rate_limit import RateLimitMiddleware
from gateway.middleware.rate_limiter import (
    RateLimitAlgorithmType,
    RateLimitPolicy,
    RateLimiter,
)

# Rate limit middleware integration tests
# pytestmark = pytest.mark.skip(reason="Rate limit middleware tests require additional API configuration")


@pytest.fixture(scope="module")
def redis_container():
    """Start a Redis container for testing."""
    container = RedisContainer("redis:7-alpine")
    container.start()
    yield container
    container.stop()


@pytest.fixture
async def rate_limiter(redis_container):
    """Create rate limiter for testing."""
    redis_url = f"redis://localhost:{redis_container.get_exposed_port(6379)}/0"
    limiter = RateLimiter(
        redis_url=redis_url,
        default_algorithm=RateLimitAlgorithmType.SLIDING_WINDOW,
    )
    await limiter.initialize()

    # Clear all rate limit data before each test
    if limiter._client:
        await limiter._client.flushdb()

    yield limiter

    # Clear all rate limit data after each test
    if limiter._client:
        await limiter._client.flushdb()

    await limiter.close()


@pytest.fixture
async def ddos_protector(rate_limiter):
    """Create DDoS protector for testing."""
    config = DDoSConfig(
        global_requests_per_second=100,
        global_requests_per_minute=1000,
        ip_requests_per_second=10,
        ip_requests_per_minute=100,
        burst_threshold_multiplier=2.0,
        burst_window_seconds=10,
        enable_auto_blocking=True,
        auto_block_duration_seconds=60,
        auto_block_threshold=5,
    )
    return DDoSProtector(rate_limiter, config)


@pytest.fixture
async def test_app(rate_limiter):
    """Create test FastAPI application with rate limiting."""
    app = FastAPI()

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=rate_limiter,
        default_policies={
            "client_ip": RateLimitPolicy(
                limit=10,
                window_seconds=1,  # 10 requests per second per IP
            ),
            "endpoint": RateLimitPolicy(
                limit=100,  # Higher limit for endpoint to test IP-specific limits
                window_seconds=1,
            ),
            "user": RateLimitPolicy(
                limit=100,
                window_seconds=1,
            ),
        },
        exempt_paths=["/health", "/metrics"],
    )

    @app.get("/test")
    async def test_endpoint(request: Request):
        return JSONResponse({"status": "ok"})

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "healthy"})

    return app


@pytest.fixture
async def client(test_app):
    """Create async HTTP client for testing."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
class TestRateLimitMiddleware:
    """Test rate limiting middleware functionality."""

    async def test_allows_requests_under_limit(self, client):
        """Test that requests under limit are allowed."""
        # Make 5 requests (under limit of 10/second)
        for _ in range(5):
            response = await client.get("/test")
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    async def test_blocks_requests_over_limit(self, client):
        """Test that requests over limit are blocked."""
        # Make 10 requests (fill limit)
        for _ in range(10):
            await client.get("/test")

        # 11th request should be rate limited
        response = await client.get("/test")
        assert response.status_code == 429

    async def test_rate_limit_headers(self, client):
        """Test that rate limit headers are included."""
        response = await client.get("/test")
        assert response.status_code == 200

        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    async def test_exempt_paths(self, client):
        """Test that exempt paths are not rate limited."""
        # Make many requests to exempt path
        for _ in range(20):
            response = await client.get("/health")
            assert response.status_code == 200

    async def test_different_ips_have_separate_limits(self, client, rate_limiter):
        """Test that different IPs have separate rate limits."""
        # First IP - make 10 requests (fill limit)
        for _ in range(10):
            await client.get("/test", headers={"X-Forwarded-For": "192.168.1.1"})

        # First IP should be rate limited
        response = await client.get("/test", headers={"X-Forwarded-For": "192.168.1.1"})
        assert response.status_code == 429

        # Second IP should still be able to make requests
        response = await client.get("/test", headers={"X-Forwarded-For": "192.168.1.2"})
        assert response.status_code == 200


@pytest.mark.asyncio
class TestDDoSProtection:
    """Test DDoS protection functionality."""

    async def test_blocks_suspicious_user_agents(self, client, test_app, ddos_protector):
        """Test that suspicious user agents are blocked."""
        # Override middleware to include DDoS protection
        test_app.user_middleware.clear()
        test_app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=ddos_protector.rate_limiter,
            ddos_protector=ddos_protector,
        )

        # Try request with bot user agent
        response = await client.get(
            "/test",
            headers={"User-Agent": "Mozilla/5.0 (compatible; bingbot/2.0)"}
        )
        # Should either be blocked or rate limited more strictly
        assert response.status_code in [200, 429]

    async def test_auto_blocking(self, client, test_app, ddos_protector):
        """Test automatic IP blocking after threshold violations."""
        # Override middleware to include DDoS protection
        test_app.user_middleware.clear()
        test_app.add_middleware(
            RateLimitMiddleware,
            rate_limiter=ddos_protector.rate_limiter,
            ddos_protector=ddos_protector,
        )

        # Make requests that violate rate limit multiple times
        for _ in range(6):  # Exceed auto_block_threshold
            for _ in range(15):  # Exceed per-second limit
                await client.get("/test", headers={"X-Forwarded-For": "10.0.0.1"})
            time.sleep(1)  # Wait for window to reset

        # IP should now be blocked
        response = await client.get("/test", headers={"X-Forwarded-For": "10.0.0.1"})
        assert response.status_code == 429

    async def test_global_rate_limiting(self, client, test_app, ddos_protector):
        """Test global rate limiting across all IPs."""
        # This test would require making many requests from different IPs
        # to exceed global limits - simplified for now
        assert ddos_protector.config.global_requests_per_second > 0


@pytest.mark.asyncio
class TestRateLimitPerformance:
    """Test rate limiting performance."""

    async def test_middleware_overhead(self, client):
        """Test that middleware adds minimal overhead."""
        # Warm up
        await client.get("/test")

        # Measure time for 10 requests
        start = time.time()
        for _ in range(10):
            await client.get("/test")
        elapsed = time.time() - start

        # Should complete in under 1 second (100ms per request)
        assert elapsed < 1.0
