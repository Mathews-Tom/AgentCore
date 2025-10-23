"""Portkey AI Gateway integration for LLM orchestration.

Provides access to 1600+ LLM providers through Portkey Gateway with:
- Intelligent routing and load balancing
- Automatic fallbacks and retries
- Cost optimization and tracking
- Comprehensive monitoring and observability
- Provider health monitoring and circuit breakers
"""

from __future__ import annotations

from agentcore.integration.portkey.client import PortkeyClient
from agentcore.integration.portkey.config import PortkeyConfig
from agentcore.integration.portkey.exceptions import (
    PortkeyAuthenticationError,
    PortkeyConfigurationError,
    PortkeyError,
    PortkeyProviderError,
    PortkeyRateLimitError,
    PortkeyTimeoutError,
    PortkeyValidationError,
)
from agentcore.integration.portkey.failover import FailoverManager
from agentcore.integration.portkey.health import ProviderHealthMonitor
from agentcore.integration.portkey.models import (
    LLMRequest,
    LLMResponse,
    ModelRequirements,
    ProviderConfig,
)
from agentcore.integration.portkey.provider import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    DataResidency,
    ProviderCapabilities,
    ProviderCapability,
    ProviderCircuitBreaker,
    ProviderConfiguration,
    ProviderHealthMetrics,
    ProviderMetadata,
    ProviderPricing,
    ProviderSelectionCriteria,
    ProviderSelectionResult,
    ProviderStatus,
)
from agentcore.integration.portkey.registry import ProviderRegistry, get_provider_registry

__all__ = [
    # Client and Config
    "PortkeyClient",
    "PortkeyConfig",
    # Exceptions
    "PortkeyError",
    "PortkeyAuthenticationError",
    "PortkeyConfigurationError",
    "PortkeyProviderError",
    "PortkeyRateLimitError",
    "PortkeyTimeoutError",
    "PortkeyValidationError",
    # Request/Response Models
    "LLMRequest",
    "LLMResponse",
    "ModelRequirements",
    "ProviderConfig",
    # Provider Management
    "ProviderRegistry",
    "get_provider_registry",
    "ProviderConfiguration",
    "ProviderCapabilities",
    "ProviderCapability",
    "ProviderStatus",
    "ProviderHealthMetrics",
    "ProviderMetadata",
    "ProviderPricing",
    "ProviderSelectionCriteria",
    "ProviderSelectionResult",
    "DataResidency",
    # Circuit Breaker
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "ProviderCircuitBreaker",
    # Health Monitoring
    "ProviderHealthMonitor",
    # Failover
    "FailoverManager",
]
