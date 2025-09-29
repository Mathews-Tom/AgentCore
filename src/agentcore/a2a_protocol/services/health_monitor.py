"""
Health Monitoring Service

Agent health checks, metrics collection, and service discovery for A2A-008.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

import structlog
import httpx

from agentcore.a2a_protocol.database import get_session
from agentcore.a2a_protocol.database.repositories import AgentRepository, HealthMetricRepository
from agentcore.a2a_protocol.models.agent import AgentStatus

logger = structlog.get_logger()


class HealthMonitor:
    """
    Health monitoring service for agents.

    Features:
    - Agent health checks with configurable intervals
    - Response time tracking
    - Consecutive failure tracking
    - Automatic agent status updates
    - Health metrics persistence
    """

    def __init__(
        self,
        health_check_interval: int = 60,  # seconds
        health_check_timeout: int = 10,  # seconds
        failure_threshold: int = 3,  # consecutive failures before marking unhealthy
    ):
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.failure_threshold = failure_threshold

        # In-memory failure tracking
        self._consecutive_failures: Dict[str, int] = defaultdict(int)

        # Health check task
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        # HTTP client for health checks
        self._http_client: Optional[httpx.AsyncClient] = None

        # Statistics
        self._stats = {
            "total_checks": 0,
            "healthy_agents": 0,
            "unhealthy_agents": 0,
            "checks_failed": 0,
            "avg_response_time_ms": 0.0,
        }

    async def start(self) -> None:
        """Start health monitoring background task."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._http_client = httpx.AsyncClient(timeout=self.health_check_timeout)
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(
            "Health monitor started",
            interval_seconds=self.health_check_interval,
            timeout_seconds=self.health_check_timeout
        )

    async def stop(self) -> None:
        """Stop health monitoring background task."""
        if not self._running:
            return

        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._http_client:
            await self._http_client.aclose()

        logger.info("Health monitor stopped")

    async def _health_check_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self.check_all_agents()
            except Exception as e:
                logger.error("Health check loop error", error=str(e))

            await asyncio.sleep(self.health_check_interval)

    async def check_all_agents(self) -> Dict[str, bool]:
        """
        Check health of all active agents.

        Returns:
            Dictionary mapping agent_id to health status
        """
        async with get_session() as session:
            # Get all active and maintenance agents
            agents = await AgentRepository.get_all(session)
            agents = [a for a in agents if a.status in (AgentStatus.ACTIVE, AgentStatus.MAINTENANCE)]

        results = {}
        for agent in agents:
            is_healthy = await self.check_agent_health(agent.id, agent.endpoint)
            results[agent.id] = is_healthy

        # Update statistics
        self._stats["healthy_agents"] = sum(1 for h in results.values() if h)
        self._stats["unhealthy_agents"] = sum(1 for h in results.values() if not h)

        logger.info(
            "Health check completed",
            total_agents=len(results),
            healthy=self._stats["healthy_agents"],
            unhealthy=self._stats["unhealthy_agents"]
        )

        return results

    async def check_agent_health(
        self,
        agent_id: str,
        endpoint: Optional[str] = None
    ) -> bool:
        """
        Check health of a specific agent.

        Args:
            agent_id: Agent identifier
            endpoint: Agent endpoint URL (if None, load from database)

        Returns:
            True if agent is healthy, False otherwise
        """
        self._stats["total_checks"] += 1

        # Get agent from database if endpoint not provided
        if not endpoint:
            async with get_session() as session:
                agent_db = await AgentRepository.get_by_id(session, agent_id)
                if not agent_db or not agent_db.endpoint:
                    logger.warning("Agent endpoint not found", agent_id=agent_id)
                    return False
                endpoint = agent_db.endpoint

        # Perform health check
        start_time = datetime.utcnow()
        is_healthy = False
        status_code: Optional[int] = None
        error_message: Optional[str] = None

        try:
            # Try health check endpoint
            health_url = f"{endpoint}/health"
            if not self._http_client:
                self._http_client = httpx.AsyncClient(timeout=self.health_check_timeout)

            response = await self._http_client.get(health_url)
            status_code = response.status_code
            is_healthy = status_code == 200

            if is_healthy:
                self._consecutive_failures[agent_id] = 0
            else:
                self._consecutive_failures[agent_id] += 1
                error_message = f"Health check returned status {status_code}"

        except Exception as e:
            self._consecutive_failures[agent_id] += 1
            error_message = str(e)
            logger.debug(
                "Agent health check failed",
                agent_id=agent_id,
                error=error_message,
                consecutive_failures=self._consecutive_failures[agent_id]
            )
            self._stats["checks_failed"] += 1

        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000

        # Update average response time
        if is_healthy:
            current_avg = self._stats["avg_response_time_ms"]
            total_successful = self._stats["total_checks"] - self._stats["checks_failed"]
            self._stats["avg_response_time_ms"] = (
                (current_avg * (total_successful - 1) + response_time_ms) / total_successful
                if total_successful > 0 else response_time_ms
            )

        # Record health metric in database
        async with get_session() as session:
            await HealthMetricRepository.record_health_check(
                session,
                agent_id=agent_id,
                is_healthy=is_healthy,
                response_time_ms=response_time_ms if is_healthy else None,
                status_code=status_code,
                error_message=error_message
            )

            # Update agent status if threshold exceeded
            if self._consecutive_failures[agent_id] >= self.failure_threshold:
                await AgentRepository.update_status(session, agent_id, AgentStatus.ERROR)
                logger.warning(
                    "Agent marked as unhealthy",
                    agent_id=agent_id,
                    consecutive_failures=self._consecutive_failures[agent_id]
                )
            elif is_healthy and self._consecutive_failures[agent_id] == 0:
                # Recover agent to active status
                agent = await AgentRepository.get_by_id(session, agent_id)
                if agent and agent.status == AgentStatus.ERROR:
                    await AgentRepository.update_status(session, agent_id, AgentStatus.ACTIVE)
                    logger.info("Agent recovered to active status", agent_id=agent_id)

        return is_healthy

    async def get_agent_health_history(
        self,
        agent_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent health check history for an agent."""
        async with get_session() as session:
            metrics = await HealthMetricRepository.get_latest_metrics(session, agent_id, limit)

        return [
            {
                "is_healthy": m.is_healthy,
                "response_time_ms": m.response_time_ms,
                "status_code": m.status_code,
                "error_message": m.error_message,
                "checked_at": m.checked_at.isoformat(),
            }
            for m in metrics
        ]

    async def get_unhealthy_agents(self) -> List[str]:
        """Get list of currently unhealthy agent IDs."""
        async with get_session() as session:
            return await HealthMetricRepository.get_unhealthy_agents(session)

    def get_statistics(self) -> Dict:
        """Get health monitoring statistics."""
        return {
            **self._stats,
            "failure_threshold": self.failure_threshold,
            "check_interval_seconds": self.health_check_interval,
        }

    async def cleanup_old_metrics(self, days_to_keep: int = 7) -> int:
        """Clean up old health metrics."""
        async with get_session() as session:
            rows_deleted = await HealthMetricRepository.cleanup_old_metrics(session, days_to_keep)
            logger.info("Old health metrics cleaned up", rows_deleted=rows_deleted)
            return rows_deleted


# Global health monitor instance
health_monitor = HealthMonitor()