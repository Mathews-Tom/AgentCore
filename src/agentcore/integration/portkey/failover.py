"""Automatic failover management for LLM provider requests.

Implements intelligent failover logic with circuit breakers, health monitoring,
and automatic retry with fallback providers.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from agentcore.integration.portkey.client import PortkeyClient
from agentcore.integration.portkey.exceptions import (
    PortkeyError,
    PortkeyProviderError,
    PortkeyRateLimitError,
    PortkeyTimeoutError,
)
from agentcore.integration.portkey.health import ProviderHealthMonitor
from agentcore.integration.portkey.models import LLMRequest, LLMResponse
from agentcore.integration.portkey.provider import (
    CircuitBreakerState,
    ProviderSelectionCriteria,
)
from agentcore.integration.portkey.registry import ProviderRegistry

logger = structlog.get_logger(__name__)


class FailoverManager:
    """Manages automatic failover for LLM provider requests.

    Coordinates provider selection, request execution, health monitoring,
    and automatic failover to backup providers on failures.
    """

    def __init__(
        self,
        client: PortkeyClient,
        registry: ProviderRegistry,
        health_monitor: ProviderHealthMonitor,
        max_failover_attempts: int = 3,
    ) -> None:
        """Initialize the failover manager.

        Args:
            client: Portkey client for executing requests
            registry: Provider registry for selection
            health_monitor: Health monitor for tracking provider status
            max_failover_attempts: Maximum number of failover attempts
        """
        self.client = client
        self.registry = registry
        self.health_monitor = health_monitor
        self.max_failover_attempts = max_failover_attempts

        logger.info(
            "failover_manager_initialized",
            max_failover_attempts=max_failover_attempts,
        )

    async def execute_with_failover(
        self,
        request: LLMRequest,
        criteria: ProviderSelectionCriteria | None = None,
    ) -> LLMResponse:
        """Execute a request with automatic failover on failures.

        Attempts to execute the request using the best available provider.
        If the request fails, automatically fails over to backup providers
        based on health status and circuit breaker states.

        Args:
            request: LLM request to execute
            criteria: Optional provider selection criteria

        Returns:
            LLM response from successful provider

        Raises:
            PortkeyProviderError: If all providers fail
            PortkeyError: For other errors
        """
        # Use default criteria if not provided
        if criteria is None:
            criteria = self._create_default_criteria(request)

        # Select providers
        selection = self.registry.select_provider(criteria)

        # Track all providers to try
        providers_to_try = [selection.provider] + selection.fallback_providers
        providers_to_try = providers_to_try[: self.max_failover_attempts]

        # Track failures for reporting
        failures: list[dict[str, Any]] = []

        for attempt, provider in enumerate(providers_to_try, start=1):
            provider_id = provider.provider_id

            # Check if provider is available
            if not self.health_monitor.is_provider_available(provider_id):
                logger.info(
                    "provider_unavailable_skipping",
                    provider_id=provider_id,
                    attempt=attempt,
                )
                failures.append(
                    {
                        "provider_id": provider_id,
                        "reason": "Provider unavailable (circuit breaker or unhealthy)",
                        "attempt": attempt,
                    }
                )
                continue

            # Check circuit breaker state
            circuit_breaker = self.registry.get_circuit_breaker(provider_id)
            if circuit_breaker and circuit_breaker.state == CircuitBreakerState.OPEN:
                logger.info(
                    "circuit_breaker_open_skipping",
                    provider_id=provider_id,
                    attempt=attempt,
                )
                failures.append(
                    {
                        "provider_id": provider_id,
                        "reason": "Circuit breaker is open",
                        "attempt": attempt,
                    }
                )
                continue

            # Attempt request with this provider
            try:
                logger.info(
                    "attempting_provider",
                    provider_id=provider_id,
                    attempt=attempt,
                    total_attempts=len(providers_to_try),
                )

                start_time = time.time()

                # Update request with provider-specific configuration
                provider_request = self._configure_request_for_provider(
                    request,
                    provider,
                )

                # Execute request
                response = await self.client.complete(provider_request)

                # Calculate actual latency
                latency_ms = int((time.time() - start_time) * 1000)

                # Record success
                self.health_monitor.record_request_success(
                    provider_id=provider_id,
                    latency_ms=latency_ms,
                )

                # Update response with provider info
                response.provider = provider_id

                logger.info(
                    "provider_request_success",
                    provider_id=provider_id,
                    attempt=attempt,
                    latency_ms=latency_ms,
                )

                return response

            except (
                PortkeyRateLimitError,
                PortkeyTimeoutError,
                PortkeyProviderError,
            ) as e:
                # Retriable errors - record failure and try next provider
                latency_ms = int((time.time() - start_time) * 1000)

                self.health_monitor.record_request_failure(
                    provider_id=provider_id,
                    latency_ms=latency_ms,
                    error=str(e),
                )

                failures.append(
                    {
                        "provider_id": provider_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                    }
                )

                logger.warning(
                    "provider_request_failed",
                    provider_id=provider_id,
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__,
                )

                # Continue to next provider
                continue

            except Exception as e:
                # Non-retriable error - record and re-raise
                latency_ms = int((time.time() - start_time) * 1000)

                self.health_monitor.record_request_failure(
                    provider_id=provider_id,
                    latency_ms=latency_ms,
                    error=str(e),
                )

                logger.error(
                    "provider_request_error",
                    provider_id=provider_id,
                    attempt=attempt,
                    error=str(e),
                )

                raise

        # All providers failed
        error = PortkeyProviderError(
            f"All {len(providers_to_try)} provider(s) failed"
        )
        # Attach failure details as attribute
        error.details = {  # type: ignore[attr-defined]
            "failures": failures,
            "total_attempts": len(providers_to_try),
        }
        raise error

    def _create_default_criteria(
        self,
        request: LLMRequest,
    ) -> ProviderSelectionCriteria:
        """Create default selection criteria from request.

        Args:
            request: LLM request

        Returns:
            Provider selection criteria
        """
        criteria = ProviderSelectionCriteria()

        # Extract criteria from model requirements if provided
        if request.model_requirements:
            if request.model_requirements.capabilities:
                # Map string capabilities to enum
                from agentcore.integration.portkey.provider import ProviderCapability

                criteria.required_capabilities = [
                    ProviderCapability(cap)
                    for cap in request.model_requirements.capabilities
                    if cap in [c.value for c in ProviderCapability]
                ]

            if request.model_requirements.max_cost_per_token:
                # Convert per-token to per-1K-tokens
                criteria.max_cost_per_1k_tokens = (
                    request.model_requirements.max_cost_per_token * 1000
                )

            if request.model_requirements.max_latency_ms:
                criteria.max_latency_ms = request.model_requirements.max_latency_ms

            if request.model_requirements.data_residency:
                from agentcore.integration.portkey.provider import DataResidency

                # Try to map to DataResidency enum
                try:
                    criteria.data_residency = DataResidency(
                        request.model_requirements.data_residency
                    )
                except ValueError:
                    logger.warning(
                        "invalid_data_residency",
                        value=request.model_requirements.data_residency,
                    )

            if request.model_requirements.preferred_providers:
                criteria.preferred_providers = (
                    request.model_requirements.preferred_providers
                )

        return criteria

    def _configure_request_for_provider(
        self,
        request: LLMRequest,
        provider: Any,
    ) -> LLMRequest:
        """Configure request for a specific provider.

        Args:
            request: Original request
            provider: Provider configuration

        Returns:
            Request configured for provider
        """
        # Create a copy of the request
        provider_request = request.model_copy(deep=True)

        # Add provider ID to context
        provider_request.context["selected_provider"] = provider.provider_id

        # Apply provider-specific overrides if configured
        if provider.custom_config:
            # Apply any provider-specific parameter overrides
            for key, value in provider.custom_config.items():
                if hasattr(provider_request, key):
                    setattr(provider_request, key, value)

        return provider_request

    async def execute_with_specific_provider(
        self,
        request: LLMRequest,
        provider_id: str,
    ) -> LLMResponse:
        """Execute a request with a specific provider (no failover).

        Args:
            request: LLM request to execute
            provider_id: Specific provider ID to use

        Returns:
            LLM response from provider

        Raises:
            PortkeyProviderError: If provider not found or unavailable
            PortkeyError: For other errors
        """
        # Get provider
        provider = self.registry.get_provider(provider_id)
        if not provider:
            raise PortkeyProviderError(f"Provider not found: {provider_id}")

        if not provider.enabled:
            raise PortkeyProviderError(f"Provider is disabled: {provider_id}")

        # Check availability
        if not self.health_monitor.is_provider_available(provider_id):
            error = PortkeyProviderError(
                f"Provider is unavailable: {provider_id}"
            )
            error.details = {"health": provider.health}  # type: ignore[attr-defined]
            raise error

        # Execute request
        start_time = time.time()

        try:
            provider_request = self._configure_request_for_provider(
                request,
                provider,
            )

            response = await self.client.complete(provider_request)

            latency_ms = int((time.time() - start_time) * 1000)

            # Record success
            self.health_monitor.record_request_success(
                provider_id=provider_id,
                latency_ms=latency_ms,
            )

            response.provider = provider_id

            return response

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)

            # Record failure
            self.health_monitor.record_request_failure(
                provider_id=provider_id,
                latency_ms=latency_ms,
                error=str(e),
            )

            raise

    def get_available_providers_count(
        self,
        criteria: ProviderSelectionCriteria | None = None,
    ) -> int:
        """Get count of available providers matching criteria.

        Args:
            criteria: Optional selection criteria

        Returns:
            Number of available providers
        """
        if criteria is None:
            criteria = ProviderSelectionCriteria()

        try:
            selection = self.registry.select_provider(criteria)
            # Count primary + fallbacks
            return 1 + len(selection.fallback_providers)
        except PortkeyProviderError:
            return 0
