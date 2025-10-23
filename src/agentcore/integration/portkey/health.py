"""Health monitoring service for LLM providers.

Monitors provider health, tracks metrics, and manages circuit breaker states
for automatic failover and resilience.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import structlog

from agentcore.integration.portkey.provider import (
    CircuitBreakerState,
    ProviderCircuitBreaker,
    ProviderHealthMetrics,
    ProviderStatus,
)
from agentcore.integration.portkey.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


class ProviderHealthMonitor:
    """Monitors provider health and manages circuit breaker states.

    Tracks request metrics, calculates health scores, and automatically
    manages circuit breaker states for resilient provider failover.
    """

    def __init__(
        self,
        registry: ProviderRegistry,
        monitoring_window_seconds: int = 300,
        health_check_interval_seconds: int = 30,
    ) -> None:
        """Initialize the health monitor.

        Args:
            registry: Provider registry to monitor
            monitoring_window_seconds: Time window for calculating metrics
            health_check_interval_seconds: Interval between health checks
        """
        self.registry = registry
        self.monitoring_window = timedelta(seconds=monitoring_window_seconds)
        self.health_check_interval = health_check_interval_seconds

        # Track request history for each provider
        self._request_history: dict[str, deque[RequestRecord]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Background task handle
        self._monitoring_task: asyncio.Task[Any] | None = None
        self._running = False

        logger.info(
            "health_monitor_initialized",
            monitoring_window_seconds=monitoring_window_seconds,
            health_check_interval_seconds=health_check_interval_seconds,
        )

    async def start(self) -> None:
        """Start background health monitoring."""
        if self._running:
            logger.warning("health_monitor_already_running")
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("health_monitor_started")

    async def stop(self) -> None:
        """Stop background health monitoring."""
        if not self._running:
            return

        self._running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("health_monitor_stopped")

    async def _monitoring_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "health_check_error",
                    error=str(e),
                )
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered providers."""
        providers = self.registry.list_providers(enabled_only=True)

        for provider in providers:
            try:
                # Calculate health metrics
                metrics = self._calculate_health_metrics(provider.provider_id)

                # Update provider health
                provider.health = metrics

                # Update circuit breaker state
                await self._update_circuit_breaker(provider.provider_id, metrics)

            except Exception as e:
                logger.error(
                    "provider_health_check_failed",
                    provider_id=provider.provider_id,
                    error=str(e),
                )

    def record_request(
        self,
        provider_id: str,
        success: bool,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        """Record a request for health tracking.

        Args:
            provider_id: Provider identifier
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            error: Error message if request failed
        """
        record = RequestRecord(
            timestamp=datetime.now(),
            success=success,
            latency_ms=latency_ms,
            error=error,
        )

        self._request_history[provider_id].append(record)

        # Update circuit breaker immediately on failure
        if not success:
            circuit_breaker = self.registry.get_circuit_breaker(provider_id)
            if circuit_breaker:
                # Try to create task if event loop is running, otherwise skip
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(
                        self._handle_request_failure(provider_id, circuit_breaker)
                    )
                except RuntimeError:
                    # No running event loop - handle synchronously
                    circuit_breaker.consecutive_failures += 1
                    circuit_breaker.consecutive_successes = 0
                    circuit_breaker.last_failure_time = datetime.now()

        logger.debug(
            "request_recorded",
            provider_id=provider_id,
            success=success,
            latency_ms=latency_ms,
        )

    def _calculate_health_metrics(self, provider_id: str) -> ProviderHealthMetrics:
        """Calculate current health metrics for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Calculated health metrics
        """
        now = datetime.now()
        cutoff_time = now - self.monitoring_window

        # Get recent requests within monitoring window
        all_records = self._request_history[provider_id]
        recent_records = [r for r in all_records if r.timestamp >= cutoff_time]

        if not recent_records:
            # No recent data - assume healthy
            return ProviderHealthMetrics(
                status=ProviderStatus.HEALTHY,
                last_check=now,
                success_rate=1.0,
                average_latency_ms=None,
                error_count=0,
                total_requests=0,
                consecutive_failures=0,
                last_error=None,
                availability_percent=100.0,
            )

        # Calculate metrics
        total_requests = len(recent_records)
        successful_requests = sum(1 for r in recent_records if r.success)
        failed_requests = total_requests - successful_requests

        success_rate = successful_requests / total_requests if total_requests > 0 else 1.0

        # Calculate average latency (only for successful requests)
        successful_latencies = [r.latency_ms for r in recent_records if r.success]
        average_latency_ms = (
            int(sum(successful_latencies) / len(successful_latencies))
            if successful_latencies
            else None
        )

        # Count consecutive failures (from most recent backwards)
        consecutive_failures = 0
        for record in reversed(list(all_records)):
            if record.success:
                break
            consecutive_failures += 1

        # Get last error
        last_error = None
        for record in reversed(list(all_records)):
            if record.error:
                last_error = record.error
                break

        # Calculate availability
        availability_percent = success_rate * 100

        # Determine status
        status = self._determine_status(
            success_rate=success_rate,
            consecutive_failures=consecutive_failures,
            average_latency_ms=average_latency_ms,
        )

        return ProviderHealthMetrics(
            status=status,
            last_check=now,
            success_rate=success_rate,
            average_latency_ms=average_latency_ms,
            error_count=failed_requests,
            total_requests=total_requests,
            consecutive_failures=consecutive_failures,
            last_error=last_error,
            availability_percent=availability_percent,
        )

    def _determine_status(
        self,
        success_rate: float,
        consecutive_failures: int,
        average_latency_ms: int | None,
    ) -> ProviderStatus:
        """Determine provider status based on metrics.

        Args:
            success_rate: Success rate (0.0-1.0)
            consecutive_failures: Number of consecutive failures
            average_latency_ms: Average latency in milliseconds

        Returns:
            Provider status
        """
        # Check for critical issues
        if consecutive_failures >= 5 or success_rate < 0.5:
            return ProviderStatus.UNHEALTHY

        # Check for degraded performance
        if success_rate < 0.9 or consecutive_failures >= 2:
            return ProviderStatus.DEGRADED

        # Check for high latency (>5 seconds)
        if average_latency_ms and average_latency_ms > 5000:
            return ProviderStatus.DEGRADED

        return ProviderStatus.HEALTHY

    async def _update_circuit_breaker(
        self,
        provider_id: str,
        metrics: ProviderHealthMetrics,
    ) -> None:
        """Update circuit breaker state based on health metrics.

        Args:
            provider_id: Provider identifier
            metrics: Current health metrics
        """
        circuit_breaker = self.registry.get_circuit_breaker(provider_id)
        if not circuit_breaker:
            return

        now = datetime.now()

        # Handle OPEN circuit breaker
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            # Check if timeout has elapsed
            if circuit_breaker.opened_at:
                timeout_elapsed = (
                    now - circuit_breaker.opened_at
                ).total_seconds() >= circuit_breaker.config.timeout_seconds

                if timeout_elapsed:
                    # Move to HALF_OPEN to test recovery
                    circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                    circuit_breaker.consecutive_successes = 0

                    logger.info(
                        "circuit_breaker_half_open",
                        provider_id=provider_id,
                    )

        # Handle HALF_OPEN circuit breaker
        elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            # Check if enough successes to close circuit
            if (
                circuit_breaker.consecutive_successes
                >= circuit_breaker.config.success_threshold
            ):
                circuit_breaker.state = CircuitBreakerState.CLOSED
                circuit_breaker.consecutive_failures = 0
                circuit_breaker.opened_at = None

                logger.info(
                    "circuit_breaker_closed",
                    provider_id=provider_id,
                )

        # Handle CLOSED circuit breaker
        elif circuit_breaker.state == CircuitBreakerState.CLOSED:
            # Circuit remains closed - normal operation
            pass

    async def _handle_request_failure(
        self,
        provider_id: str,
        circuit_breaker: ProviderCircuitBreaker,
    ) -> None:
        """Handle a request failure and update circuit breaker.

        Args:
            provider_id: Provider identifier
            circuit_breaker: Circuit breaker to update
        """
        now = datetime.now()

        circuit_breaker.consecutive_failures += 1
        circuit_breaker.consecutive_successes = 0
        circuit_breaker.last_failure_time = now

        # Check if we should open the circuit
        if (
            circuit_breaker.state == CircuitBreakerState.CLOSED
            and circuit_breaker.consecutive_failures
            >= circuit_breaker.config.failure_threshold
        ):
            circuit_breaker.state = CircuitBreakerState.OPEN
            circuit_breaker.opened_at = now

            logger.warning(
                "circuit_breaker_opened",
                provider_id=provider_id,
                consecutive_failures=circuit_breaker.consecutive_failures,
            )

        # In HALF_OPEN state, any failure reopens the circuit
        elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            circuit_breaker.state = CircuitBreakerState.OPEN
            circuit_breaker.opened_at = now

            logger.warning(
                "circuit_breaker_reopened",
                provider_id=provider_id,
            )

    def record_request_success(
        self,
        provider_id: str,
        latency_ms: int,
    ) -> None:
        """Record a successful request.

        Args:
            provider_id: Provider identifier
            latency_ms: Request latency in milliseconds
        """
        self.record_request(provider_id, success=True, latency_ms=latency_ms)

        # Update circuit breaker on success
        circuit_breaker = self.registry.get_circuit_breaker(provider_id)
        if circuit_breaker:
            circuit_breaker.consecutive_failures = 0
            circuit_breaker.consecutive_successes += 1

    def record_request_failure(
        self,
        provider_id: str,
        latency_ms: int,
        error: str,
    ) -> None:
        """Record a failed request.

        Args:
            provider_id: Provider identifier
            latency_ms: Request latency in milliseconds
            error: Error message
        """
        self.record_request(
            provider_id,
            success=False,
            latency_ms=latency_ms,
            error=error,
        )

    def get_provider_metrics(self, provider_id: str) -> ProviderHealthMetrics | None:
        """Get current health metrics for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Health metrics or None if provider not found
        """
        provider = self.registry.get_provider(provider_id)
        if not provider or not provider.health:
            return None

        return provider.health

    def is_provider_available(self, provider_id: str) -> bool:
        """Check if a provider is available for requests.

        A provider is available if:
        - It is enabled
        - Circuit breaker is not OPEN
        - Health status is HEALTHY or DEGRADED

        Args:
            provider_id: Provider identifier

        Returns:
            True if provider is available
        """
        provider = self.registry.get_provider(provider_id)
        if not provider or not provider.enabled:
            return False

        # Check circuit breaker
        circuit_breaker = self.registry.get_circuit_breaker(provider_id)
        if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
            return False

        # Check health status
        if provider.health:
            return provider.health.status in (
                ProviderStatus.HEALTHY,
                ProviderStatus.DEGRADED,
            )

        # No health data - assume available
        return True


class RequestRecord:
    """Record of a single request for health tracking."""

    def __init__(
        self,
        timestamp: datetime,
        success: bool,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        """Initialize request record.

        Args:
            timestamp: When the request occurred
            success: Whether request succeeded
            latency_ms: Request latency in milliseconds
            error: Error message if request failed
        """
        self.timestamp = timestamp
        self.success = success
        self.latency_ms = latency_ms
        self.error = error
