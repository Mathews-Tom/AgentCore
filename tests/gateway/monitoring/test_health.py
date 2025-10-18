"""Tests for health check functionality."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.monitoring.health import HealthChecker


class TestHealthChecker:
    """Test HealthChecker functionality."""

    @pytest.fixture
    async def health_checker(self) -> HealthChecker:
        """Create a health checker instance."""
        checker = HealthChecker(
            redis_url="redis://localhost:6379",
            backend_services={
                "a2a_protocol": "http://localhost:8001",
                "agent_runtime": "http://localhost:8002",
            },
            check_timeout=2.0,
        )
        yield checker
        await checker.close()

    @pytest.mark.asyncio
    async def test_check_redis_healthy(
        self, health_checker: HealthChecker
    ) -> None:
        """Test Redis health check when healthy."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.aclose = AsyncMock()
            mock_redis.return_value = mock_client

            is_healthy, message = await health_checker.check_redis()

            assert is_healthy is True
            assert "healthy" in message.lower()
            mock_client.ping.assert_awaited_once()
            mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_redis_unhealthy(
        self, health_checker: HealthChecker
    ) -> None:
        """Test Redis health check when unhealthy."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis.side_effect = ConnectionError("Connection refused")

            is_healthy, message = await health_checker.check_redis()

            assert is_healthy is False
            assert "error" in message.lower()

    @pytest.mark.asyncio
    async def test_check_redis_timeout(
        self, health_checker: HealthChecker
    ) -> None:
        """Test Redis health check timeout."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(side_effect=TimeoutError())
            mock_client.aclose = AsyncMock()
            mock_redis.return_value = mock_client

            with patch("gateway.monitoring.health.asyncio.wait_for") as mock_wait:
                mock_wait.side_effect = TimeoutError()

                is_healthy, message = await health_checker.check_redis()

                assert is_healthy is False
                assert "timeout" in message.lower()

    @pytest.mark.asyncio
    async def test_check_backend_service_healthy(
        self, health_checker: HealthChecker
    ) -> None:
        """Test backend service health check when healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(
            health_checker._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_response

            is_healthy, message = await health_checker.check_backend_service(
                "test_service", "http://localhost:8000"
            )

            assert is_healthy is True
            assert "healthy" in message.lower()
            mock_get.assert_awaited_once_with("http://localhost:8000/health")

    @pytest.mark.asyncio
    async def test_check_backend_service_unhealthy(
        self, health_checker: HealthChecker
    ) -> None:
        """Test backend service health check when unhealthy."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch.object(
            health_checker._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_response

            is_healthy, message = await health_checker.check_backend_service(
                "test_service", "http://localhost:8000"
            )

            assert is_healthy is False
            assert "503" in message

    @pytest.mark.asyncio
    async def test_check_backend_service_timeout(
        self, health_checker: HealthChecker
    ) -> None:
        """Test backend service health check timeout."""
        with patch.object(
            health_checker._http_client, "get", new_callable=AsyncMock
        ) as mock_get:
            with patch("gateway.monitoring.health.asyncio.wait_for") as mock_wait:
                mock_wait.side_effect = TimeoutError()

                is_healthy, message = await health_checker.check_backend_service(
                    "test_service", "http://localhost:8000"
                )

                assert is_healthy is False
                assert ("timeout" in message.lower() or "timed out" in message.lower())

    @pytest.mark.asyncio
    async def test_check_all_healthy(
        self, health_checker: HealthChecker
    ) -> None:
        """Test comprehensive health check when all components are healthy."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.aclose = AsyncMock()
            mock_redis.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch.object(
                health_checker._http_client, "get", new_callable=AsyncMock
            ) as mock_get:
                mock_get.return_value = mock_response

                result = await health_checker.check_all()

                assert result["status"] == "healthy"
                assert "redis" in result["checks"]
                assert "a2a_protocol" in result["checks"]
                assert "agent_runtime" in result["checks"]
                assert result["checks"]["redis"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_check_all_unhealthy(
        self, health_checker: HealthChecker
    ) -> None:
        """Test comprehensive health check when components are unhealthy."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis.side_effect = ConnectionError("Connection refused")

            result = await health_checker.check_all()

            assert result["status"] == "unhealthy"
            assert "redis" in result["checks"]
            assert result["checks"]["redis"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_check_readiness_ready(
        self, health_checker: HealthChecker
    ) -> None:
        """Test readiness check when service is ready."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_client.aclose = AsyncMock()
            mock_redis.return_value = mock_client

            result = await health_checker.check_readiness()

            assert result["status"] == "ready"

    @pytest.mark.asyncio
    async def test_check_readiness_not_ready(
        self, health_checker: HealthChecker
    ) -> None:
        """Test readiness check when service is not ready."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis.side_effect = ConnectionError("Connection refused")

            result = await health_checker.check_readiness()

            assert result["status"] == "not_ready"
            assert "reason" in result

    @pytest.mark.asyncio
    async def test_check_readiness_no_redis(self) -> None:
        """Test readiness check when Redis is not configured."""
        checker = HealthChecker(
            redis_url=None,
            backend_services={},
            check_timeout=2.0,
        )

        result = await checker.check_readiness()

        assert result["status"] == "ready"

        await checker.close()

    @pytest.mark.asyncio
    async def test_close(self, health_checker: HealthChecker) -> None:
        """Test closing health checker resources."""
        await health_checker.close()
        # If no exception is raised, close succeeded
        assert True
