"""
Backend Service Routing

Intelligent routing to backend services with service discovery and proxying.
"""

from __future__ import annotations

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    CircuitState,
)
from .discovery import ServiceDiscovery, ServiceInstance, ServiceStatus
from .load_balancer import LoadBalancer, LoadBalancingAlgorithm
from .proxy import ServiceProxy

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitBreakerRegistry",
    "CircuitState",
    "LoadBalancer",
    "LoadBalancingAlgorithm",
    "ServiceDiscovery",
    "ServiceInstance",
    "ServiceProxy",
    "ServiceStatus",
]
