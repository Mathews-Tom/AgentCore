"""Resilience configuration utilities.

Helper functions for creating common resilience configurations.
"""

from __future__ import annotations

from agentcore.integration.resilience.models import (
    BulkheadConfig,
    CircuitBreakerConfig,
    ResilienceConfig,
    TimeoutConfig,
)


def create_default_circuit_breaker(name: str) -> CircuitBreakerConfig:
    """Create default circuit breaker configuration.

    Args:
        name: Circuit breaker name

    Returns:
        Circuit breaker configuration with sensible defaults
    """
    return CircuitBreakerConfig(
        name=name,
        failure_threshold=5,
        success_threshold=2,
        timeout_seconds=60.0,
        half_open_max_requests=3,
    )


def create_default_bulkhead(name: str) -> BulkheadConfig:
    """Create default bulkhead configuration.

    Args:
        name: Bulkhead name

    Returns:
        Bulkhead configuration with sensible defaults
    """
    return BulkheadConfig(
        name=name,
        max_concurrent_requests=10,
        queue_size=10,
        queue_timeout_seconds=5.0,
    )


def create_default_timeout(name: str) -> TimeoutConfig:
    """Create default timeout configuration.

    Args:
        name: Timeout name

    Returns:
        Timeout configuration with sensible defaults
    """
    return TimeoutConfig(
        name=name,
        timeout_seconds=30.0,
    )


def create_default_resilience(name: str) -> ResilienceConfig:
    """Create default resilience configuration.

    Includes circuit breaker, bulkhead, and timeout with sensible defaults.

    Args:
        name: Base name for all components

    Returns:
        Resilience configuration with all patterns enabled
    """
    return ResilienceConfig(
        circuit_breaker=create_default_circuit_breaker(f"{name}_circuit"),
        bulkhead=create_default_bulkhead(f"{name}_bulkhead"),
        timeout=create_default_timeout(f"{name}_timeout"),
        enable_fallback=False,
    )


def create_llm_resilience(name: str) -> ResilienceConfig:
    """Create resilience configuration optimized for LLM providers.

    Args:
        name: Base name for all components

    Returns:
        Resilience configuration tuned for LLM workloads
    """
    return ResilienceConfig(
        circuit_breaker=CircuitBreakerConfig(
            name=f"{name}_circuit",
            failure_threshold=3,  # Lower threshold for expensive LLM calls
            success_threshold=2,
            timeout_seconds=120.0,  # Longer timeout for LLM recovery
            half_open_max_requests=2,
        ),
        bulkhead=BulkheadConfig(
            name=f"{name}_bulkhead",
            max_concurrent_requests=20,  # Higher concurrency for LLM
            queue_size=50,  # Larger queue for bursty traffic
            queue_timeout_seconds=10.0,
        ),
        timeout=TimeoutConfig(
            name=f"{name}_timeout",
            timeout_seconds=60.0,  # Longer timeout for LLM generation
        ),
        enable_fallback=True,  # Enable fallback for LLM provider switching
    )


def create_api_resilience(name: str) -> ResilienceConfig:
    """Create resilience configuration optimized for API calls.

    Args:
        name: Base name for all components

    Returns:
        Resilience configuration tuned for API workloads
    """
    return ResilienceConfig(
        circuit_breaker=CircuitBreakerConfig(
            name=f"{name}_circuit",
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_requests=3,
        ),
        bulkhead=BulkheadConfig(
            name=f"{name}_bulkhead",
            max_concurrent_requests=10,
            queue_size=10,
            queue_timeout_seconds=5.0,
        ),
        timeout=TimeoutConfig(
            name=f"{name}_timeout",
            timeout_seconds=10.0,  # Shorter timeout for API calls
        ),
        enable_fallback=False,
    )


def create_database_resilience(name: str) -> ResilienceConfig:
    """Create resilience configuration optimized for database operations.

    Args:
        name: Base name for all components

    Returns:
        Resilience configuration tuned for database workloads
    """
    return ResilienceConfig(
        circuit_breaker=CircuitBreakerConfig(
            name=f"{name}_circuit",
            failure_threshold=10,  # Higher threshold for transient DB errors
            success_threshold=3,
            timeout_seconds=60.0,
            half_open_max_requests=5,
        ),
        bulkhead=BulkheadConfig(
            name=f"{name}_bulkhead",
            max_concurrent_requests=50,  # Higher concurrency for DB
            queue_size=100,
            queue_timeout_seconds=30.0,  # Longer queue timeout
        ),
        timeout=TimeoutConfig(
            name=f"{name}_timeout",
            timeout_seconds=30.0,
        ),
        enable_fallback=False,
    )
