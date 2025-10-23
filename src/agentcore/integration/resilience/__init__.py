"""Resilience patterns for fault tolerance and graceful degradation.

Provides circuit breakers, bulkheads, timeouts, and resilience orchestration
for external service integrations.
"""

from agentcore.integration.resilience.bulkhead import Bulkhead, BulkheadRegistry
from agentcore.integration.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitBreakerState,
)
from agentcore.integration.resilience.config import (
    create_api_resilience,
    create_database_resilience,
    create_default_bulkhead,
    create_default_circuit_breaker,
    create_default_resilience,
    create_default_timeout,
    create_llm_resilience,
)
from agentcore.integration.resilience.exceptions import (
    BulkheadRejectedError,
    CircuitBreakerOpenError,
    ResilienceError,
    ResilienceTimeoutError,
)
from agentcore.integration.resilience.manager import (
    ResilienceManager,
    ResilienceRegistry,
)
from agentcore.integration.resilience.models import (
    BulkheadConfig,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    ResilienceConfig,
    TimeoutConfig,
)
from agentcore.integration.resilience.timeout import TimeoutManager, with_timeout

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitBreakerState",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    # Bulkhead
    "Bulkhead",
    "BulkheadRegistry",
    "BulkheadConfig",
    # Timeout
    "TimeoutManager",
    "TimeoutConfig",
    "with_timeout",
    # Manager
    "ResilienceManager",
    "ResilienceRegistry",
    "ResilienceConfig",
    # Config Helpers
    "create_default_circuit_breaker",
    "create_default_bulkhead",
    "create_default_timeout",
    "create_default_resilience",
    "create_llm_resilience",
    "create_api_resilience",
    "create_database_resilience",
    # Exceptions
    "ResilienceError",
    "CircuitBreakerOpenError",
    "BulkheadRejectedError",
    "ResilienceTimeoutError",
]
