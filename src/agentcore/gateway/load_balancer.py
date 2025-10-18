"""
Load Balancing Algorithms

Implements various load balancing strategies for backend services.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import structlog

from agentcore.gateway.models import (
    LoadBalancingAlgorithm,
    RoutingMetrics,
    ServiceEndpoint,
)

logger = structlog.get_logger(__name__)


class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies."""

    @abstractmethod
    async def select_service(
        self, services: list[ServiceEndpoint], metrics: dict[str, RoutingMetrics]
    ) -> ServiceEndpoint | None:
        """Select a service based on the strategy.

        Args:
            services: Available services
            metrics: Current routing metrics

        Returns:
            Selected service or None if none available
        """


class RoundRobinStrategy(LoadBalancingStrategy):
    """Round-robin load balancing."""

    def __init__(self) -> None:
        """Initialize round-robin strategy."""
        self._counter = 0
        self._lock = asyncio.Lock()

    async def select_service(
        self, services: list[ServiceEndpoint], metrics: dict[str, RoutingMetrics]
    ) -> ServiceEndpoint | None:
        """Select next service in round-robin order.

        Args:
            services: Available services
            metrics: Current routing metrics

        Returns:
            Selected service or None if none available
        """
        if not services:
            return None

        async with self._lock:
            selected = services[self._counter % len(services)]
            self._counter += 1
            return selected


class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Least connections load balancing."""

    async def select_service(
        self, services: list[ServiceEndpoint], metrics: dict[str, RoutingMetrics]
    ) -> ServiceEndpoint | None:
        """Select service with least active connections.

        Args:
            services: Available services
            metrics: Current routing metrics

        Returns:
            Selected service or None if none available
        """
        if not services:
            return None

        # Find service with minimum active connections
        min_connections = float("inf")
        selected_service = services[0]

        for service in services:
            service_metrics = metrics.get(service.service_id)
            connections = (
                service_metrics.active_connections if service_metrics else 0
            )

            if connections < min_connections:
                min_connections = connections
                selected_service = service

        return selected_service


class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round-robin load balancing."""

    def __init__(self) -> None:
        """Initialize weighted round-robin strategy."""
        self._current_index = 0
        self._current_weight = 0
        self._lock = asyncio.Lock()

    async def select_service(
        self, services: list[ServiceEndpoint], metrics: dict[str, RoutingMetrics]
    ) -> ServiceEndpoint | None:
        """Select service based on weighted round-robin.

        Args:
            services: Available services
            metrics: Current routing metrics

        Returns:
            Selected service or None if none available
        """
        if not services:
            return None

        async with self._lock:
            max_weight = max(s.weight for s in services)

            while True:
                self._current_index = (self._current_index + 1) % len(services)

                if self._current_index == 0:
                    self._current_weight -= 1
                    if self._current_weight <= 0:
                        self._current_weight = max_weight

                service = services[self._current_index]
                if service.weight >= self._current_weight:
                    return service


class RandomStrategy(LoadBalancingStrategy):
    """Random load balancing."""

    async def select_service(
        self, services: list[ServiceEndpoint], metrics: dict[str, RoutingMetrics]
    ) -> ServiceEndpoint | None:
        """Select random service.

        Args:
            services: Available services
            metrics: Current routing metrics

        Returns:
            Selected service or None if none available
        """
        if not services:
            return None

        return random.choice(services)


class LoadBalancer:
    """Load balancer for backend services."""

    def __init__(
        self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    ) -> None:
        """Initialize load balancer.

        Args:
            algorithm: Load balancing algorithm to use
        """
        self.algorithm = algorithm
        self._strategy = self._create_strategy(algorithm)
        self._metrics: dict[str, RoutingMetrics] = {}
        self._lock = asyncio.Lock()

    def _create_strategy(self, algorithm: LoadBalancingAlgorithm) -> LoadBalancingStrategy:
        """Create strategy instance for algorithm.

        Args:
            algorithm: Load balancing algorithm

        Returns:
            Strategy instance
        """
        strategies = {
            LoadBalancingAlgorithm.ROUND_ROBIN: RoundRobinStrategy,
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: LeastConnectionsStrategy,
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy,
            LoadBalancingAlgorithm.RANDOM: RandomStrategy,
        }
        strategy_class = strategies[algorithm]
        return strategy_class()

    async def select_service(
        self, services: list[ServiceEndpoint]
    ) -> ServiceEndpoint | None:
        """Select service using configured strategy.

        Args:
            services: Available services

        Returns:
            Selected service or None if none available
        """
        # Filter enabled services
        enabled_services = [s for s in services if s.enabled]
        if not enabled_services:
            logger.warning("no_enabled_services")
            return None

        selected = await self._strategy.select_service(enabled_services, self._metrics)

        if selected:
            logger.debug(
                "service_selected",
                service_id=selected.service_id,
                algorithm=self.algorithm.value,
            )

        return selected

    async def record_request_start(self, service_id: str) -> None:
        """Record start of a request to a service.

        Args:
            service_id: Service identifier
        """
        async with self._lock:
            if service_id not in self._metrics:
                self._metrics[service_id] = RoutingMetrics(service_id=service_id)

            metrics = self._metrics[service_id]
            metrics.total_requests += 1
            metrics.active_connections += 1
            metrics.last_request = datetime.now(timezone.utc)

    async def record_request_end(
        self, service_id: str, success: bool, response_time_ms: float
    ) -> None:
        """Record end of a request to a service.

        Args:
            service_id: Service identifier
            success: Whether request was successful
            response_time_ms: Response time in milliseconds
        """
        async with self._lock:
            if service_id not in self._metrics:
                return

            metrics = self._metrics[service_id]
            metrics.active_connections = max(0, metrics.active_connections - 1)

            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

            # Update average response time (simple moving average)
            total_completed = metrics.successful_requests + metrics.failed_requests
            if total_completed > 0:
                metrics.avg_response_time_ms = (
                    metrics.avg_response_time_ms * (total_completed - 1)
                    + response_time_ms
                ) / total_completed

    async def get_metrics(self, service_id: str) -> RoutingMetrics | None:
        """Get routing metrics for a service.

        Args:
            service_id: Service identifier

        Returns:
            Routing metrics or None if not found
        """
        return self._metrics.get(service_id)

    async def get_all_metrics(self) -> dict[str, RoutingMetrics]:
        """Get routing metrics for all services.

        Returns:
            Dictionary mapping service IDs to metrics
        """
        return self._metrics.copy()

    async def reset_metrics(self, service_id: str | None = None) -> None:
        """Reset routing metrics.

        Args:
            service_id: Service to reset, or None to reset all
        """
        async with self._lock:
            if service_id:
                if service_id in self._metrics:
                    self._metrics[service_id] = RoutingMetrics(service_id=service_id)
            else:
                self._metrics.clear()

        logger.info("metrics_reset", service_id=service_id or "all")
