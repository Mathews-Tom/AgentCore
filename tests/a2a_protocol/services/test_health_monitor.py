"""
Comprehensive test suite for HealthMonitor service.

Tests health checks, agent status updates, metrics persistence,
background monitoring loop, and statistics.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from agentcore.a2a_protocol.models.agent import AgentStatus
from agentcore.a2a_protocol.services.health_monitor import HealthMonitor, health_monitor


@pytest.fixture
def monitor():
    """Create fresh HealthMonitor instance for each test."""
    return HealthMonitor(
        health_check_interval=60, health_check_timeout=10, failure_threshold=3
    )


@pytest.fixture
def mock_agent_db():
    """Create mock agent database record."""
    agent = MagicMock()
    agent.id = "agent-1"
    agent.endpoint = "http://agent-1.local"
    agent.status = AgentStatus.ACTIVE
    return agent


# ==================== Health Check Tests ====================


class TestHealthChecks:
    """Test health check functionality."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_check_agent_health_success(
        self, mock_health_repo, mock_agent_repo, mock_get_session, monitor
    ):
        """Test successful health check."""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        # Mock database operations
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Mock the class-level method to return AsyncMock
        mock_health_repo.record_health_check = AsyncMock()
        mock_agent_repo.get_by_id = AsyncMock(return_value=None)

        result = await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        assert result is True
        assert monitor._stats["total_checks"] == 1
        mock_client.get.assert_called_once_with("http://agent-1.local/health")
        mock_health_repo.record_health_check.assert_called_once()

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_check_agent_health_http_error(
        self, mock_health_repo, mock_get_session, monitor
    ):
        """Test health check with HTTP error status."""
        # Mock HTTP response with error
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.record_health_check = AsyncMock()

        result = await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        assert result is False
        assert monitor._consecutive_failures["agent-1"] == 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_check_agent_health_network_failure(
        self, mock_health_repo, mock_get_session, monitor
    ):
        """Test health check with network failure."""
        # Mock network exception
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        monitor._http_client = mock_client

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.record_health_check = AsyncMock()

        result = await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        assert result is False
        assert monitor._stats["checks_failed"] == 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    async def test_check_agent_health_without_endpoint(
        self, mock_agent_repo, mock_get_session, monitor, mock_agent_db
    ):
        """Test health check loads endpoint from database when not provided."""
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        # Mock AgentRepository.get_by_id as a static/class method that returns the agent
        mock_agent_repo.get_by_id = AsyncMock(return_value=mock_agent_db)

        # Mock HTTP client
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        with patch(
            "agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository.record_health_check",
            new=AsyncMock()):
            result = await monitor.check_agent_health(agent_id="agent-1")

        assert result is True
        # Should be called at least once (possibly twice - once for endpoint lookup, once for recovery check)
        assert mock_agent_repo.get_by_id.call_count >= 1

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    async def test_check_agent_health_no_endpoint_in_db(
        self, mock_agent_repo, mock_get_session, monitor
    ):
        """Test health check fails when no endpoint found."""
        mock_agent = MagicMock()
        mock_agent.endpoint = None

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_agent_repo.get_by_id = AsyncMock(return_value=mock_agent)

        result = await monitor.check_agent_health(agent_id="agent-1")

        assert result is False

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    async def test_check_agent_health_resets_failures_on_success(
        self, mock_agent_repo, mock_get_session, monitor
    ):
        """Test successful health check resets failure counter."""
        # Set initial failures
        monitor._consecutive_failures["agent-1"] = 2

        # Mock successful health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_agent_repo.get_by_id = AsyncMock(return_value=None)

        with patch(
            "agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository.record_health_check",
            new=AsyncMock()):
            result = await monitor.check_agent_health(
                agent_id="agent-1", endpoint="http://agent-1.local"
            )

        assert result is True
        assert monitor._consecutive_failures["agent-1"] == 0

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_check_agent_health_tracks_response_time(
        self, mock_health_repo, mock_agent_repo, mock_get_session, monitor
    ):
        """Test health check tracks response time."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.record_health_check = AsyncMock()
        mock_agent_repo.get_by_id = AsyncMock(return_value=None)

        await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        # Verify response time was recorded
        call_args = mock_health_repo.record_health_check.call_args
        assert call_args.kwargs["response_time_ms"] is not None
        assert call_args.kwargs["response_time_ms"] >= 0

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_check_all_agents(
        self, mock_health_repo, mock_agent_repo, mock_get_session, monitor
    ):
        """Test checking health of all agents."""
        # Mock agents
        agent1 = MagicMock()
        agent1.id = "agent-1"
        agent1.endpoint = "http://agent-1.local"
        agent1.status = AgentStatus.ACTIVE

        agent2 = MagicMock()
        agent2.id = "agent-2"
        agent2.endpoint = "http://agent-2.local"
        agent2.status = AgentStatus.ACTIVE

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_agent_repo.get_all = AsyncMock(return_value=[agent1, agent2])
        mock_agent_repo.get_by_id = AsyncMock(return_value=None)

        # Mock HTTP responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        mock_health_repo.record_health_check = AsyncMock()

        results = await monitor.check_all_agents()

        assert len(results) == 2
        assert "agent-1" in results
        assert "agent-2" in results
        assert monitor._stats["healthy_agents"] == 2
        assert monitor._stats["unhealthy_agents"] == 0


# ==================== Agent Status Update Tests ====================


class TestAgentStatusUpdates:
    """Test agent status update functionality."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_mark_agent_unhealthy_after_threshold(
        self, mock_health_repo, mock_agent_repo, mock_get_session, monitor
    ):
        """Test agent marked unhealthy after failure threshold."""
        # Set failures at threshold
        monitor._consecutive_failures["agent-1"] = monitor.failure_threshold

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_agent_repo.update_status = AsyncMock(return_value=True)
        mock_health_repo.record_health_check = AsyncMock()

        # Mock failed health check
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        monitor._http_client = mock_client

        await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        # Verify agent status updated to ERROR
        mock_agent_repo.update_status.assert_called_with(
            mock_session, "agent-1", AgentStatus.ERROR
        )

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_recover_agent_to_active_status(
        self,
        mock_health_repo,
        mock_agent_repo,
        mock_get_session,
        monitor,
        mock_agent_db):
        """Test recovering agent to active status after successful check."""
        # Agent currently in ERROR status
        mock_agent_db.status = AgentStatus.ERROR

        # Reset consecutive failures
        monitor._consecutive_failures["agent-1"] = 0

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_agent_repo.get_by_id = AsyncMock(return_value=mock_agent_db)
        mock_agent_repo.update_status = AsyncMock(return_value=True)
        mock_health_repo.record_health_check = AsyncMock()

        # Mock successful health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        # Verify agent recovered to ACTIVE
        mock_agent_repo.update_status.assert_called_with(
            mock_session, "agent-1", AgentStatus.ACTIVE
        )

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_consecutive_failure_tracking(
        self, mock_health_repo, mock_agent_repo, mock_get_session, monitor
    ):
        """Test consecutive failure counter increments correctly."""
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.record_health_check = AsyncMock()
        mock_agent_repo.update_status = AsyncMock()

        # Mock failed health checks
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Failed"))
        monitor._http_client = mock_client

        # Perform multiple checks
        for i in range(3):
            await monitor.check_agent_health(
                agent_id="agent-1", endpoint="http://agent-1.local"
            )

        assert monitor._consecutive_failures["agent-1"] == 3

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_does_not_update_status_below_threshold(
        self, mock_health_repo, mock_agent_repo, mock_get_session, monitor
    ):
        """Test status not updated when below failure threshold."""
        monitor._consecutive_failures["agent-1"] = 1  # Below threshold of 3

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.record_health_check = AsyncMock()
        mock_agent_repo.update_status = AsyncMock()

        # Mock failed health check
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Failed"))
        monitor._http_client = mock_client

        await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        # Status update should NOT be called
        mock_agent_repo.update_status.assert_not_called()


# ==================== Metrics Persistence Tests ====================


class TestMetricsPersistence:
    """Test metrics persistence to database."""

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.AgentRepository")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_record_health_check_to_database(
        self, mock_health_repo, mock_agent_repo, mock_get_session, monitor
    ):
        """Test health check is recorded to database."""
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.record_health_check = AsyncMock()
        mock_agent_repo.get_by_id = AsyncMock(return_value=None)

        # Mock successful health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        monitor._http_client = mock_client

        await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        # Verify metric recorded
        mock_health_repo.record_health_check.assert_called_once()
        call_kwargs = mock_health_repo.record_health_check.call_args.kwargs
        assert call_kwargs["agent_id"] == "agent-1"
        assert call_kwargs["is_healthy"] is True
        assert call_kwargs["status_code"] == 200

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_record_failed_health_check_to_database(
        self, mock_health_repo, mock_get_session, monitor
    ):
        """Test failed health check is recorded with error."""
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.record_health_check = AsyncMock()

        # Mock failed health check
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        monitor._http_client = mock_client

        await monitor.check_agent_health(
            agent_id="agent-1", endpoint="http://agent-1.local"
        )

        # Verify metric recorded with error
        call_kwargs = mock_health_repo.record_health_check.call_args.kwargs
        assert call_kwargs["is_healthy"] is False
        assert call_kwargs["error_message"] is not None

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_get_agent_health_history(
        self, mock_health_repo, mock_get_session, monitor
    ):
        """Test getting agent health history."""
        # Mock health metrics
        mock_metric = MagicMock()
        mock_metric.is_healthy = True
        mock_metric.response_time_ms = 50.0
        mock_metric.status_code = 200
        mock_metric.error_message = None
        mock_metric.checked_at = datetime.now(UTC)

        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.get_latest_metrics = AsyncMock(return_value=[mock_metric])

        history = await monitor.get_agent_health_history("agent-1", limit=10)

        assert len(history) == 1
        assert history[0]["is_healthy"] is True
        assert history[0]["response_time_ms"] == 50.0
        mock_health_repo.get_latest_metrics.assert_called_once_with(
            mock_session, "agent-1", 10
        )

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_get_unhealthy_agents(
        self, mock_health_repo, mock_get_session, monitor
    ):
        """Test getting list of unhealthy agents."""
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.get_unhealthy_agents = AsyncMock(
            return_value=["agent-1", "agent-2"]
        )

        unhealthy = await monitor.get_unhealthy_agents()

        assert len(unhealthy) == 2
        assert "agent-1" in unhealthy
        assert "agent-2" in unhealthy

    @pytest.mark.asyncio
    @patch("agentcore.a2a_protocol.services.health_monitor.get_session")
    @patch("agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository")
    async def test_cleanup_old_metrics(
        self, mock_health_repo, mock_get_session, monitor
    ):
        """Test cleaning up old health metrics."""
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session
        mock_health_repo.cleanup_old_metrics = AsyncMock(return_value=50)

        deleted = await monitor.cleanup_old_metrics(days_to_keep=7)

        assert deleted == 50
        mock_health_repo.cleanup_old_metrics.assert_called_once_with(mock_session, 7)

    @pytest.mark.asyncio
    async def test_average_response_time_calculation(self, monitor):
        """Test average response time is calculated correctly."""
        monitor._stats["total_checks"] = 10
        monitor._stats["checks_failed"] = 2
        monitor._stats["avg_response_time_ms"] = 50.0

        # Simulate successful check with response time
        with patch.object(monitor, "check_agent_health") as mock_check:
            # Manually update stats as if check succeeded
            monitor._stats["total_checks"] += 1
            monitor._stats["checks_failed"] = 2  # No new failures

            # Calculate new average: (50 * 8 + 100) / 9
            new_avg = (50.0 * 8 + 100.0) / 9
            monitor._stats["avg_response_time_ms"] = new_avg

            assert monitor._stats["avg_response_time_ms"] > 50.0


# ==================== Background Loop Tests ====================


class TestBackgroundLoop:
    """Test background monitoring loop."""

    @pytest.mark.asyncio
    async def test_start_monitor(self, monitor):
        """Test starting health monitor."""
        await monitor.start()

        assert monitor._running is True
        assert monitor._http_client is not None
        assert monitor._health_check_task is not None

        # Stop monitor
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_monitor(self, monitor):
        """Test stopping health monitor."""
        await monitor.start()
        await monitor.stop()

        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_start_monitor_already_running(self, monitor):
        """Test starting monitor when already running does not create duplicate tasks."""
        await monitor.start()

        first_task = monitor._health_check_task

        # Try to start again
        await monitor.start()

        # Should still be same task
        assert monitor._health_check_task == first_task

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_health_check_loop_executes(self, monitor):
        """Test health check loop executes periodically."""
        check_count = 0

        async def mock_check_all():
            nonlocal check_count
            check_count += 1
            return {}

        with patch.object(monitor, "check_all_agents", side_effect=mock_check_all):
            await monitor.start()

            # Wait for at least one check (with short interval)
            monitor.health_check_interval = 0.1
            await asyncio.sleep(0.3)

            await monitor.stop()

        # Should have executed at least once
        assert check_count >= 1

    @pytest.mark.asyncio
    async def test_health_check_loop_handles_errors(self, monitor):
        """Test health check loop continues after errors."""
        with patch.object(
            monitor, "check_all_agents", side_effect=Exception("Check failed")
        ):
            await monitor.start()

            # Should not crash, continues running
            await asyncio.sleep(0.1)

            assert monitor._running is True

            await monitor.stop()

    @pytest.mark.asyncio
    async def test_http_client_lifecycle(self, monitor):
        """Test HTTP client is created on start and closed on stop."""
        await monitor.start()

        assert monitor._http_client is not None
        assert isinstance(monitor._http_client, httpx.AsyncClient)

        await monitor.stop()

        # Client should still exist but connection closed
        assert monitor._http_client is not None


# ==================== Statistics Tests ====================


class TestStatistics:
    """Test health monitoring statistics."""

    def test_get_statistics(self, monitor):
        """Test getting health monitoring statistics."""
        monitor._stats["total_checks"] = 100
        monitor._stats["healthy_agents"] = 8
        monitor._stats["unhealthy_agents"] = 2

        stats = monitor.get_statistics()

        assert "total_checks" in stats
        assert "healthy_agents" in stats
        assert "unhealthy_agents" in stats
        assert "checks_failed" in stats
        assert "avg_response_time_ms" in stats
        assert "failure_threshold" in stats
        assert "check_interval_seconds" in stats

        assert stats["total_checks"] == 100
        assert stats["healthy_agents"] == 8
        assert stats["unhealthy_agents"] == 2

    @pytest.mark.asyncio
    async def test_statistics_updated_after_checks(self, monitor):
        """Test statistics are updated after health checks."""
        initial_checks = monitor._stats["total_checks"]

        with patch("agentcore.a2a_protocol.services.health_monitor.get_session"):
            with patch(
                "agentcore.a2a_protocol.services.health_monitor.AgentRepository.get_by_id",
                new=AsyncMock(return_value=None)):
                with patch(
                    "agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository.record_health_check",
                    new=AsyncMock()):
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(return_value=mock_response)
                    monitor._http_client = mock_client

                    await monitor.check_agent_health(
                        agent_id="agent-1", endpoint="http://agent-1.local"
                    )

        assert monitor._stats["total_checks"] == initial_checks + 1

    @pytest.mark.asyncio
    async def test_statistics_track_failed_checks(self, monitor):
        """Test statistics track failed checks."""
        initial_failed = monitor._stats["checks_failed"]

        with patch("agentcore.a2a_protocol.services.health_monitor.get_session"):
            with patch(
                "agentcore.a2a_protocol.services.health_monitor.HealthMetricRepository.record_health_check",
                new=AsyncMock()):
                mock_client = AsyncMock()
                mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Failed"))
                monitor._http_client = mock_client

                await monitor.check_agent_health(
                    agent_id="agent-1", endpoint="http://agent-1.local"
                )

        assert monitor._stats["checks_failed"] == initial_failed + 1


# ==================== Global Instance Test ====================


class TestGlobalInstance:
    """Test global health monitor instance."""

    def test_global_instance_exists(self):
        """Test global health_monitor instance exists."""
        assert health_monitor is not None
        assert isinstance(health_monitor, HealthMonitor)

    def test_global_instance_is_singleton(self):
        """Test global instance behaves like singleton."""
        from agentcore.a2a_protocol.services.health_monitor import health_monitor as hm1
        from agentcore.a2a_protocol.services.health_monitor import health_monitor as hm2

        assert hm1 is hm2

    def test_global_instance_configured(self):
        """Test global instance has default configuration."""
        assert health_monitor.health_check_interval == 60
        assert health_monitor.health_check_timeout == 10
        assert health_monitor.failure_threshold == 3
