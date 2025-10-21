"""
Load Balancer

Intelligent load balancing algorithms for distributing requests across
backend service instances.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from enum import Enum

import structlog

from .discovery import ServiceDiscovery, ServiceInstance

logger = structlog.get_logger()


class LoadBalancingAlgorithm(str, Enum):
    """Load balancing algorithm types."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    IP_HASH = "ip_hash"


class LoadBalancingStrategy(ABC):
    """Base class for load balancing strategies."""

    @abstractmethod
    def select(
        self, instances: list[ServiceInstance], client_ip: str | None = None
    ) -> ServiceInstance | None:
        """
        Select a service instance based on the strategy.

        Args:
            instances: List of available service instances
            client_ip: Client IP address for IP hash strategy

        Returns:
            Selected service instance or None if no instances available
        """


class RoundRobinStrategy(LoadBalancingStrategy):
    """Round robin load balancing strategy."""

    def __init__(self) -> None:
        """Initialize round robin strategy."""
        self._current_index = 0

    def select(
        self, instances: list[ServiceInstance], client_ip: str | None = None
    ) -> ServiceInstance | None:
        """
        Select next instance in round robin order.

        Args:
            instances: List of available service instances
            client_ip: Not used for round robin

        Returns:
            Next service instance in rotation
        """
        if not instances:
            return None

        instance = instances[self._current_index % len(instances)]
        self._current_index += 1

        return instance


class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round robin load balancing strategy."""

    def __init__(self) -> None:
        """Initialize weighted round robin strategy."""
        self._current_index = 0
        self._current_weight = 0
        self._gcd_weight = 0
        self._max_weight = 0

    def select(
        self, instances: list[ServiceInstance], client_ip: str | None = None
    ) -> ServiceInstance | None:
        """
        Select next instance based on weights.

        Args:
            instances: List of available service instances
            client_ip: Not used for weighted round robin

        Returns:
            Next service instance based on weights
        """
        if not instances:
            return None

        # Calculate GCD and max weight if not set or instances changed
        if self._gcd_weight == 0 or self._max_weight == 0:
            self._calculate_weights(instances)

        while True:
            self._current_index = (self._current_index + 1) % len(instances)

            if self._current_index == 0:
                self._current_weight = self._current_weight - self._gcd_weight
                if self._current_weight <= 0:
                    self._current_weight = self._max_weight

            if instances[self._current_index].weight >= self._current_weight:
                return instances[self._current_index]

    def _calculate_weights(self, instances: list[ServiceInstance]) -> None:
        """Calculate GCD and max weight for instances."""
        import math

        weights = [i.weight for i in instances]
        self._max_weight = max(weights)
        self._gcd_weight = weights[0]

        for weight in weights[1:]:
            self._gcd_weight = math.gcd(self._gcd_weight, weight)


class RandomStrategy(LoadBalancingStrategy):
    """Random load balancing strategy."""

    def select(
        self, instances: list[ServiceInstance], client_ip: str | None = None
    ) -> ServiceInstance | None:
        """
        Select random instance.

        Args:
            instances: List of available service instances
            client_ip: Not used for random selection

        Returns:
            Randomly selected service instance
        """
        if not instances:
            return None

        return random.choice(instances)


class IPHashStrategy(LoadBalancingStrategy):
    """IP hash load balancing strategy for sticky sessions."""

    def select(
        self, instances: list[ServiceInstance], client_ip: str | None = None
    ) -> ServiceInstance | None:
        """
        Select instance based on client IP hash.

        Args:
            instances: List of available service instances
            client_ip: Client IP address for hashing

        Returns:
            Service instance selected based on IP hash
        """
        if not instances:
            return None

        if not client_ip:
            # Fallback to random if no client IP
            return random.choice(instances)

        # Hash client IP to select instance
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(instances)

        return instances[index]


class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Least connections load balancing strategy."""

    def __init__(self) -> None:
        """Initialize least connections strategy."""
        self._connection_counts: dict[str, int] = {}

    def select(
        self, instances: list[ServiceInstance], client_ip: str | None = None
    ) -> ServiceInstance | None:
        """
        Select instance with fewest active connections.

        Args:
            instances: List of available service instances
            client_ip: Not used for least connections

        Returns:
            Instance with fewest connections
        """
        if not instances:
            return None

        # Initialize connection counts for new instances
        for instance in instances:
            if instance.instance_id not in self._connection_counts:
                self._connection_counts[instance.instance_id] = 0

        # Find instance with fewest connections
        min_instance = min(
            instances, key=lambda i: self._connection_counts.get(i.instance_id, 0)
        )

        return min_instance

    def increment_connections(self, instance_id: str) -> None:
        """
        Increment connection count for instance.

        Args:
            instance_id: Instance ID
        """
        self._connection_counts[instance_id] = (
            self._connection_counts.get(instance_id, 0) + 1
        )

    def decrement_connections(self, instance_id: str) -> None:
        """
        Decrement connection count for instance.

        Args:
            instance_id: Instance ID
        """
        if instance_id in self._connection_counts:
            self._connection_counts[instance_id] = max(
                0, self._connection_counts[instance_id] - 1
            )


class LoadBalancer:
    """
    Load balancer for distributing requests across backend services.

    Integrates with service discovery and supports multiple load balancing algorithms.
    """

    def __init__(
        self,
        discovery: ServiceDiscovery,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN,
    ):
        """
        Initialize load balancer.

        Args:
            discovery: Service discovery instance
            algorithm: Load balancing algorithm to use
        """
        self.discovery = discovery
        self.algorithm = algorithm
        self._strategy = self._create_strategy(algorithm)

    def _create_strategy(
        self, algorithm: LoadBalancingAlgorithm
    ) -> LoadBalancingStrategy:
        """
        Create strategy instance based on algorithm.

        Args:
            algorithm: Load balancing algorithm

        Returns:
            Strategy instance
        """
        strategies = {
            LoadBalancingAlgorithm.ROUND_ROBIN: RoundRobinStrategy,
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy,
            LoadBalancingAlgorithm.RANDOM: RandomStrategy,
            LoadBalancingAlgorithm.IP_HASH: IPHashStrategy,
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: LeastConnectionsStrategy,
        }

        strategy_class = strategies.get(algorithm)
        if not strategy_class:
            logger.warning(
                "Unknown load balancing algorithm, using round robin",
                algorithm=algorithm,
            )
            strategy_class = RoundRobinStrategy

        return strategy_class()

    def select_instance(
        self, service_name: str, client_ip: str | None = None
    ) -> ServiceInstance | None:
        """
        Select a backend service instance for request.

        Args:
            service_name: Service name to select instance for
            client_ip: Client IP address (used for IP hash strategy)

        Returns:
            Selected service instance or None if no healthy instances
        """
        # Get healthy instances from discovery
        instances = self.discovery.get_instances(service_name, healthy_only=True)

        if not instances:
            logger.warning(
                "No healthy instances available",
                service=service_name,
                algorithm=self.algorithm,
            )
            return None

        # Select instance using strategy
        instance = self._strategy.select(instances, client_ip)

        if instance:
            logger.debug(
                "Instance selected",
                service=service_name,
                instance_id=instance.instance_id,
                algorithm=self.algorithm,
            )

        return instance

    def increment_connections(self, instance_id: str) -> None:
        """
        Increment active connections for instance (least connections only).

        Args:
            instance_id: Instance ID
        """
        if isinstance(self._strategy, LeastConnectionsStrategy):
            self._strategy.increment_connections(instance_id)

    def decrement_connections(self, instance_id: str) -> None:
        """
        Decrement active connections for instance (least connections only).

        Args:
            instance_id: Instance ID
        """
        if isinstance(self._strategy, LeastConnectionsStrategy):
            self._strategy.decrement_connections(instance_id)
