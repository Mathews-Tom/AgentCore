"""
Backend Service Routing

Intelligent routing to backend services with service discovery and proxying.
"""

from __future__ import annotations

from gateway.routing.discovery import ServiceDiscovery, ServiceInstance
from gateway.routing.load_balancer import LoadBalancer, LoadBalancerAlgorithm
from gateway.routing.proxy import BackendProxy
from gateway.routing.router import ServiceRouter

__all__ = [
    "ServiceDiscovery",
    "ServiceInstance",
    "LoadBalancer",
    "LoadBalancerAlgorithm",
    "BackendProxy",
    "ServiceRouter",
]
