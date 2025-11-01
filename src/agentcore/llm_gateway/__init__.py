"""LLM Gateway integration for LLM orchestration.

Provides access to 1600+ LLM providers through unified gateway with:
- Intelligent routing and load balancing
- Automatic fallbacks and retries
- Cost optimization and tracking
- Comprehensive monitoring and observability
- Provider health monitoring and circuit breakers
- Performance monitoring with 50+ metrics per request
"""

from __future__ import annotations

from agentcore.llm_gateway.client import LLMGatewayClient
from agentcore.llm_gateway.config import LLMGatewayConfig
from agentcore.llm_gateway.exceptions import (
    LLMGatewayAuthenticationError,
    LLMGatewayConfigurationError,
    LLMGatewayError,
    LLMGatewayProviderError,
    LLMGatewayRateLimitError,
    LLMGatewayTimeoutError,
    LLMGatewayValidationError,
)
from agentcore.llm_gateway.failover import FailoverManager
from agentcore.llm_gateway.health import ProviderHealthMonitor
from agentcore.llm_gateway.metrics_collector import MetricsCollector, get_metrics_collector
from agentcore.llm_gateway.metrics_models import (
    AlertSeverity,
    DashboardData,
    MetricType,
    PerformanceAlert,
    PerformanceInsight,
    PerformanceLevel,
    PerformanceMetrics,
    PrometheusMetrics,
    ProviderPerformanceMetrics,
    RequestMetrics,
    SLAMetrics,
    SLAStatus,
)
from agentcore.llm_gateway.models import (
    LLMRequest,
    LLMResponse,
    ModelRequirements,
    ProviderConfig,
)
from agentcore.llm_gateway.performance_monitor import (
    PerformanceMonitor,
    get_performance_monitor,
)
from agentcore.llm_gateway.provider import (
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
from agentcore.llm_gateway.registry import ProviderRegistry, get_provider_registry

__all__ = [
    # Client and Config
    "LLMGatewayClient",
    "LLMGatewayConfig",
    # Exceptions
    "LLMGatewayError",
    "LLMGatewayAuthenticationError",
    "LLMGatewayConfigurationError",
    "LLMGatewayProviderError",
    "LLMGatewayRateLimitError",
    "LLMGatewayTimeoutError",
    "LLMGatewayValidationError",
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
    # Performance Monitoring (INT-005)
    "MetricsCollector",
    "get_metrics_collector",
    "PerformanceMonitor",
    "get_performance_monitor",
    # Performance Metrics Models
    "RequestMetrics",
    "PerformanceMetrics",
    "SLAMetrics",
    "SLAStatus",
    "ProviderPerformanceMetrics",
    "PerformanceAlert",
    "PerformanceInsight",
    "PrometheusMetrics",
    "DashboardData",
    "MetricType",
    "PerformanceLevel",
    "AlertSeverity",
]
