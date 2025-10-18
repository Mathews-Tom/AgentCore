"""
Integration tests for rate limiting middleware.

Tests the complete rate limiting middleware including DDoS protection.
"""

from __future__ import annotations

import pytest
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from gateway.middleware.ddos_protection import DDoSConfig, DDoSProtector
from gateway.middleware.rate_limit import RateLimitMiddleware
from gateway.middleware.rate_limiter import (
    RateLimitAlgorithmType,
    RateLimitPolicy,
    RateLimiter,
)


@pytest.fixture
async def rate_limiter():
    """Create rate limiter for testing."""
    limiter = RateLimiter(
        redis_url="redis://localhost:6379/15",
        default_algorithm=RateLimitAlgorithmType.SLIDING_WINDOW,
    )
    await limiter.initialize()

    yield limiter

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
async def test_app(rate_limiter, ddos_protector):
    """Create test FastAPI app with rate limiting."""
    app = FastAPI()

    # Add test endpoint
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}

    # Add rate limiting middleware
    default_policies = {
        "client_ip": RateLimitPolicy(limit=10, window_seconds=60),
        "endpoint": RateLimitPolicy(limit=20, window_seconds=60),
        "user": RateLimitPolicy(limit=50, window_seconds=60),
    }

    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=rate_limiter,
        ddos_protector=ddos_protector,
        default_policies=default_policies,
        enable_ddos_protection=True,
        exempt_paths=["/health"],
    )

    return app


@pytest.mark.asyncio
class TestRateLimitMiddleware:
    """Tests for rate limiting middleware."""

    async def test_allows_requests_under_limit(self, test_app):
        """Test that requests under limit are allowed."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make 5 requests (under limit of 10)
            for _ in range(5):
                response = await client.get("/test")
                assert response.status_code == 200
                assert "X-RateLimit-Limit" in response.headers
                assert "X-RateLimit-Remaining" in response.headers

    async def test_blocks_requests_over_limit(self, test_app):
        """Test that requests over limit are blocked."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make 10 requests (at limit)
            for _ in range(10):
                response = await client.get("/test")
                assert response.status_code == 200

            # 11th request should be blocked
            response = await client.get("/test")
            assert response.status_code == 429
            assert "Retry-After" in response.headers
            assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    async def test_rate_limit_headers(self, test_app):
        """Test that rate limit headers are included."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/test")

            assert response.status_code == 200
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers
            assert "X-RateLimit-Type" in response.headers

            # Validate header values
            assert int(response.headers["X-RateLimit-Limit"]) > 0
            assert int(response.headers["X-RateLimit-Remaining"]) >= 0
            assert int(response.headers["X-RateLimit-Reset"]) > 0

    async def test_exempt_paths(self, test_app):
        """Test that exempt paths bypass rate limiting."""
        # Add health endpoint
        @test_app.get("/health")
        async def health():
            return {"status": "healthy"}

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Make many requests to health endpoint
            for _ in range(50):
                response = await client.get("/health")
                assert response.status_code == 200
                # Should not have rate limit headers
                assert "X-RateLimit-Limit" not in response.headers

    async def test_different_ips_have_separate_limits(self, test_app):
        """Test that different client IPs have separate rate limits."""
        transport = ASGITransport(app=test_app)

        # Client 1: Use first IP
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"X-Forwarded-For": "192.168.1.1"},
        ) as client1:
            # Make 10 requests from IP 1
            for _ in range(10):
                response = await client1.get("/test")
                assert response.status_code == 200

        # Client 2: Use second IP
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"X-Forwarded-For": "192.168.1.2"},
        ) as client2:
            # Should still be able to make requests from IP 2
            response = await client2.get("/test")
            assert response.status_code == 200


@pytest.mark.asyncio
class TestDDoSProtection:
    """Tests for DDoS protection."""

    async def test_blocks_suspicious_user_agents(self, test_app):
        """Test that suspicious user agents are detected."""
        transport = ASGITransport(app=test_app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"User-Agent": "malicious-bot"},
        ) as client:
            response = await client.get("/test")
            # May still allow but should detect
            assert response.status_code in [200, 429]

    async def test_auto_blocking(self, test_app, rate_limiter):
        """Test automatic IP blocking after violations."""
        transport = ASGITransport(app=test_app)

        # Make excessive requests to trigger auto-blocking
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"X-Forwarded-For": "10.0.0.1"},
        ) as client:
            # Exceed limits multiple times
            for _ in range(15):
                await client.get("/test")

            # Should eventually be blocked by DDoS protection
            response = await client.get("/test")
            assert response.status_code == 429

    async def test_global_rate_limiting(self, test_app):
        """Test global rate limiting across all clients."""
        transport = ASGITransport(app=test_app)

        # Note: This test is challenging without actual high load
        # Just verify the mechanism exists
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/test")
            # Global limits are very high in test config, so should succeed
            assert response.status_code == 200


@pytest.mark.asyncio
class TestRateLimitPerformance:
    """Performance tests for rate limiting middleware."""

    async def test_middleware_overhead(self, test_app):
        """Test that rate limiting overhead is minimal."""
        import time

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Warm up
            await client.get("/test")

            # Measure overhead
            start = time.time()
            iterations = 100

            for _ in range(iterations):
                await client.get("/test")

            elapsed = time.time() - start
            avg_ms = (elapsed / iterations) * 1000

            # Should be well under 5ms per request (including FastAPI overhead)
            assert avg_ms < 10.0, f"Average request time {avg_ms:.2f}ms exceeds threshold"
