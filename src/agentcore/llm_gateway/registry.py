"""Provider registry for managing 1600+ LLM providers.

Provides centralized management of provider configurations, capabilities,
and selection logic for intelligent routing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from agentcore.llm_gateway.cost_models import OptimizationContext
from agentcore.llm_gateway.exceptions import (
    LLMGatewayConfigurationError,
    LLMGatewayProviderError,
)
from agentcore.llm_gateway.provider import (
    CircuitBreakerState,
    ProviderCapability,
    ProviderCircuitBreaker,
    ProviderConfiguration,
    ProviderSelectionCriteria,
    ProviderSelectionResult,
    ProviderStatus,
)

logger = structlog.get_logger(__name__)


class ProviderRegistry:
    """Registry for managing LLM provider configurations and selection.

    Maintains a centralized registry of 1600+ LLM providers with their
    capabilities, health status, and configuration. Provides intelligent
    provider selection based on requirements and real-time health data.
    """

    def __init__(self) -> None:
        """Initialize the provider registry."""
        self._providers: dict[str, ProviderConfiguration] = {}
        self._circuit_breakers: dict[str, ProviderCircuitBreaker] = {}
        self._initialized = False

        logger.info("provider_registry_initialized")

    def register_provider(self, provider: ProviderConfiguration) -> None:
        """Register a new provider in the registry.

        Args:
            provider: Provider configuration to register

        Raises:
            LLMGatewayConfigurationError: If provider is invalid or already registered
        """
        if provider.provider_id in self._providers:
            logger.warning(
                "provider_already_registered",
                provider_id=provider.provider_id,
            )
            # Update existing provider
            self._providers[provider.provider_id] = provider
        else:
            self._providers[provider.provider_id] = provider

            # Initialize circuit breaker for this provider
            self._circuit_breakers[provider.provider_id] = ProviderCircuitBreaker(
                provider_id=provider.provider_id,
                config=provider.circuit_breaker,
            )

            logger.info(
                "provider_registered",
                provider_id=provider.provider_id,
                capabilities=len(provider.capabilities.capabilities),
                enabled=provider.enabled,
            )

    def register_providers(self, providers: Sequence[ProviderConfiguration]) -> None:
        """Register multiple providers in bulk.

        Args:
            providers: Sequence of provider configurations to register
        """
        for provider in providers:
            self.register_provider(provider)

        logger.info(
            "providers_registered_bulk",
            count=len(providers),
            total_providers=len(self._providers),
        )

    def get_provider(self, provider_id: str) -> ProviderConfiguration | None:
        """Get a provider by ID.

        Args:
            provider_id: Provider identifier

        Returns:
            Provider configuration or None if not found
        """
        return self._providers.get(provider_id)

    def list_providers(
        self,
        enabled_only: bool = True,
        capability: ProviderCapability | None = None,
        status: ProviderStatus | None = None,
        tags: list[str] | None = None,
    ) -> list[ProviderConfiguration]:
        """List providers matching criteria.

        Args:
            enabled_only: Only return enabled providers
            capability: Filter by specific capability
            status: Filter by health status
            tags: Filter by tags (must have all tags)

        Returns:
            List of matching provider configurations
        """
        providers = list(self._providers.values())

        # Filter by enabled status
        if enabled_only:
            providers = [p for p in providers if p.enabled]

        # Filter by capability
        if capability is not None:
            providers = [
                p for p in providers if capability in p.capabilities.capabilities
            ]

        # Filter by health status
        if status is not None:
            providers = [p for p in providers if p.health and p.health.status == status]

        # Filter by tags
        if tags:
            providers = [p for p in providers if all(tag in p.tags for tag in tags)]

        return providers

    def select_provider(
        self,
        criteria: ProviderSelectionCriteria,
    ) -> ProviderSelectionResult:
        """Select the best provider based on criteria.

        Implements intelligent provider selection considering:
        - Required capabilities
        - Cost constraints
        - Latency requirements
        - Data residency
        - Provider health and availability
        - Circuit breaker state

        Args:
            criteria: Selection criteria

        Returns:
            Provider selection result with primary and fallback providers

        Raises:
            LLMGatewayProviderError: If no suitable provider found
        """
        # Get enabled providers
        candidates = self.list_providers(enabled_only=True)

        # Filter by required capabilities
        if criteria.required_capabilities:
            candidates = [
                p
                for p in candidates
                if all(
                    cap in p.capabilities.capabilities
                    for cap in criteria.required_capabilities
                )
            ]

        # Filter by data residency
        if criteria.data_residency:
            candidates = [
                p
                for p in candidates
                if criteria.data_residency in p.capabilities.data_residency
            ]

        # Filter by tags
        if criteria.tags:
            candidates = [
                p for p in candidates if all(tag in p.tags for tag in criteria.tags)
            ]

        # Exclude providers
        if criteria.excluded_providers:
            candidates = [
                p
                for p in candidates
                if p.provider_id not in criteria.excluded_providers
            ]

        # Filter by circuit breaker state (exclude open circuits)
        candidates = [
            p
            for p in candidates
            if self._circuit_breakers.get(p.provider_id, None) is None
            or self._circuit_breakers[p.provider_id].state != CircuitBreakerState.OPEN
        ]

        # Filter by health status
        if criteria.require_healthy:
            candidates = [
                p
                for p in candidates
                if p.health is None
                or p.health.status
                in (ProviderStatus.HEALTHY, ProviderStatus.DEGRADED)
            ]

        # Filter by success rate
        candidates = [
            p
            for p in candidates
            if p.health is None or p.health.success_rate >= criteria.min_success_rate
        ]

        # Filter by cost if specified
        if criteria.max_cost_per_1k_tokens is not None:
            candidates = [
                p
                for p in candidates
                if p.pricing is None
                or (
                    p.pricing.input_token_price + p.pricing.output_token_price
                )
                / 2
                <= criteria.max_cost_per_1k_tokens
            ]

        # Filter by latency if specified
        if criteria.max_latency_ms is not None:
            candidates = [
                p
                for p in candidates
                if p.health is None
                or p.health.average_latency_ms is None
                or p.health.average_latency_ms <= criteria.max_latency_ms
            ]

        if not candidates:
            error = LLMGatewayProviderError(
                "No suitable provider found matching criteria"
            )
            # Attach criteria details as attribute
            error.details = {  # type: ignore[attr-defined]
                "required_capabilities": criteria.required_capabilities,
                "data_residency": criteria.data_residency,
                "max_cost": criteria.max_cost_per_1k_tokens,
                "max_latency": criteria.max_latency_ms,
            }
            raise error

        # Sort candidates by selection priority
        ranked_candidates = self._rank_providers(candidates, criteria)

        # Select primary provider
        primary = ranked_candidates[0]

        # Select fallback providers (up to 3)
        fallbacks = ranked_candidates[1:4]

        # Calculate estimated cost and latency
        estimated_cost = None
        if primary.pricing:
            # Average of input and output token prices per 1K tokens
            estimated_cost = (
                primary.pricing.input_token_price + primary.pricing.output_token_price
            ) / 2

        expected_latency_ms = None
        if primary.health and primary.health.average_latency_ms:
            expected_latency_ms = primary.health.average_latency_ms

        result = ProviderSelectionResult(
            provider=primary,
            fallback_providers=fallbacks,
            selection_reason=self._get_selection_reason(primary, criteria),
            estimated_cost=estimated_cost,
            expected_latency_ms=expected_latency_ms,
        )

        logger.info(
            "provider_selected",
            provider_id=primary.provider_id,
            fallback_count=len(fallbacks),
            estimated_cost=estimated_cost,
            expected_latency_ms=expected_latency_ms,
        )

        return result

    def _rank_providers(
        self,
        providers: list[ProviderConfiguration],
        criteria: ProviderSelectionCriteria,
    ) -> list[ProviderConfiguration]:
        """Rank providers by preference based on criteria.

        Args:
            providers: List of candidate providers
            criteria: Selection criteria

        Returns:
            Ranked list of providers (best first)
        """
        # Start with preferred providers in order
        ranked: list[ProviderConfiguration] = []

        # Add explicitly preferred providers first
        for provider_id in criteria.preferred_providers:
            provider = next(
                (p for p in providers if p.provider_id == provider_id),
                None,
            )
            if provider:
                ranked.append(provider)
                providers = [p for p in providers if p.provider_id != provider_id]

        # Score remaining providers
        scored_providers: list[tuple[ProviderConfiguration, float]] = []
        for provider in providers:
            score = self._calculate_provider_score(provider, criteria)
            scored_providers.append((provider, score))

        # Sort by score (highest first)
        scored_providers.sort(key=lambda x: x[1], reverse=True)

        # Add scored providers to ranked list
        ranked.extend(p for p, _ in scored_providers)

        return ranked

    def _calculate_provider_score(
        self,
        provider: ProviderConfiguration,
        criteria: ProviderSelectionCriteria,
    ) -> float:
        """Calculate a score for provider selection.

        Higher score = better match for criteria.

        Args:
            provider: Provider to score
            criteria: Selection criteria

        Returns:
            Provider score (higher is better)
        """
        score = 0.0

        # Base priority score
        score += provider.priority * 10

        # Health score
        if provider.health:
            score += provider.health.success_rate * 100
            score += provider.health.availability_percent

            # Penalize for recent errors
            if provider.health.consecutive_failures > 0:
                score -= provider.health.consecutive_failures * 20

        # Cost score (lower cost = higher score)
        if provider.pricing and criteria.max_cost_per_1k_tokens:
            avg_cost = (
                provider.pricing.input_token_price + provider.pricing.output_token_price
            ) / 2
            if avg_cost > 0:
                # Normalize: providers at 50% of max cost get +50 points
                cost_ratio = avg_cost / criteria.max_cost_per_1k_tokens
                score += (1.0 - cost_ratio) * 50

        # Latency score (lower latency = higher score)
        if provider.health and provider.health.average_latency_ms:
            if criteria.max_latency_ms:
                latency_ratio = (
                    provider.health.average_latency_ms / criteria.max_latency_ms
                )
                score += (1.0 - latency_ratio) * 30

        # Circuit breaker score
        cb = self._circuit_breakers.get(provider.provider_id)
        if cb:
            if cb.state == CircuitBreakerState.CLOSED:
                score += 20
            elif cb.state == CircuitBreakerState.HALF_OPEN:
                score += 5  # Lower preference for half-open circuits

        return score

    def _get_selection_reason(
        self,
        provider: ProviderConfiguration,
        criteria: ProviderSelectionCriteria,
    ) -> str:
        """Generate human-readable selection reason.

        Args:
            provider: Selected provider
            criteria: Selection criteria

        Returns:
            Selection reason string
        """
        reasons: list[str] = []

        # Check if it's a preferred provider
        if provider.provider_id in criteria.preferred_providers:
            index = criteria.preferred_providers.index(provider.provider_id)
            reasons.append(f"preferred provider (#{index + 1})")

        # Add health status
        if provider.health:
            if provider.health.status == ProviderStatus.HEALTHY:
                reasons.append(
                    f"healthy ({provider.health.success_rate:.1%} success rate)"
                )
            elif provider.health.status == ProviderStatus.DEGRADED:
                reasons.append("degraded but operational")

        # Add cost advantage
        if provider.pricing and criteria.max_cost_per_1k_tokens:
            avg_cost = (
                provider.pricing.input_token_price + provider.pricing.output_token_price
            ) / 2
            cost_savings = (
                1.0 - (avg_cost / criteria.max_cost_per_1k_tokens)
            ) * 100
            if cost_savings > 20:
                reasons.append(f"cost-effective ({cost_savings:.0f}% under budget)")

        # Add latency advantage
        if provider.health and provider.health.average_latency_ms:
            if criteria.max_latency_ms:
                latency_pct = (
                    provider.health.average_latency_ms / criteria.max_latency_ms
                ) * 100
                if latency_pct < 50:
                    reasons.append(f"low latency ({latency_pct:.0f}% of max)")

        # Add priority
        if provider.priority > 100:
            reasons.append(f"high priority ({provider.priority})")

        return ", ".join(reasons) if reasons else "meets all requirements"

    def get_circuit_breaker(self, provider_id: str) -> ProviderCircuitBreaker | None:
        """Get circuit breaker state for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Circuit breaker state or None if provider not found
        """
        return self._circuit_breakers.get(provider_id)

    def load_from_file(self, file_path: str | Path) -> None:
        """Load provider configurations from JSON file.

        Args:
            file_path: Path to JSON file with provider configurations

        Raises:
            LLMGatewayConfigurationError: If file cannot be loaded or parsed
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise LLMGatewayConfigurationError(f"Provider config file not found: {file_path}")

            with path.open("r") as f:
                data = json.load(f)

            # Parse provider configurations
            providers = [
                ProviderConfiguration.model_validate(p) for p in data.get("providers", [])
            ]

            # Register all providers
            self.register_providers(providers)

            logger.info(
                "providers_loaded_from_file",
                file_path=str(file_path),
                count=len(providers),
            )

        except Exception as e:
            raise LLMGatewayConfigurationError(
                f"Failed to load provider configurations: {e}"
            ) from e

    def save_to_file(self, file_path: str | Path) -> None:
        """Save current provider configurations to JSON file.

        Args:
            file_path: Path to output JSON file
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "providers": [
                    p.model_dump(mode="json") for p in self._providers.values()
                ],
                "metadata": {
                    "total_providers": len(self._providers),
                    "enabled_providers": len(
                        [p for p in self._providers.values() if p.enabled]
                    ),
                },
            }

            with path.open("w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(
                "providers_saved_to_file",
                file_path=str(file_path),
                count=len(self._providers),
            )

        except Exception as e:
            logger.error(
                "failed_to_save_providers",
                file_path=str(file_path),
                error=str(e),
            )
            raise

    def select_cost_optimized_provider(
        self,
        criteria: ProviderSelectionCriteria,
        optimization_context: OptimizationContext,
    ) -> ProviderSelectionResult:
        """Select provider with cost optimization.

        Uses the cost optimizer for intelligent provider selection that
        balances cost, performance, and quality requirements.

        Args:
            criteria: Provider selection criteria
            optimization_context: Cost optimization context

        Returns:
            Provider selection result with cost-optimized selection

        Raises:
            LLMGatewayProviderError: If no suitable provider found
        """
        # Import here to avoid circular dependency
        from agentcore.llm_gateway.cost_optimizer import CostOptimizer
        from agentcore.llm_gateway.cost_tracker import get_cost_tracker

        # Create optimizer instance
        optimizer = CostOptimizer(
            registry=self,
            cost_tracker=get_cost_tracker(),
        )

        # Select optimal provider
        selected_provider = optimizer.select_optimal_provider(criteria, optimization_context)

        # Get fallback providers
        all_candidates = [
            p
            for p in self.list_providers(enabled_only=True)
            if p.provider_id != selected_provider.provider_id
        ]

        # Rank fallbacks by cost optimization score
        comparisons = optimizer._compare_providers(  # type: ignore[attr-defined]
            all_candidates[:10],  # Limit to top 10 for performance
            optimization_context,
        )

        fallback_providers = [
            self.get_provider(c.provider_id)
            for c in comparisons[:3]
            if self.get_provider(c.provider_id) is not None
        ]

        # Estimate cost
        estimated_cost = optimizer.estimate_request_cost(
            selected_provider,
            optimization_context.estimated_input_tokens,
            optimization_context.estimated_output_tokens,
        )

        # Get expected latency
        expected_latency_ms = (
            selected_provider.health.average_latency_ms
            if selected_provider.health
            else None
        )

        return ProviderSelectionResult(
            provider=selected_provider,
            fallback_providers=fallback_providers,
            selection_reason=f"Cost-optimized selection: {optimization_context.optimization_strategy}",
            estimated_cost=estimated_cost,
            expected_latency_ms=expected_latency_ms,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        providers = list(self._providers.values())
        enabled_providers = [p for p in providers if p.enabled]
        healthy_providers = [
            p
            for p in enabled_providers
            if p.health and p.health.status == ProviderStatus.HEALTHY
        ]

        # Count providers by capability
        capability_counts: dict[str, int] = {}
        for provider in enabled_providers:
            for cap in provider.capabilities.capabilities:
                capability_counts[cap] = capability_counts.get(cap, 0) + 1

        # Count circuit breaker states
        cb_states: dict[str, int] = {}
        for cb in self._circuit_breakers.values():
            cb_states[cb.state] = cb_states.get(cb.state, 0) + 1

        # Get cost statistics if available
        cost_stats: dict[str, Any] = {}
        try:
            from agentcore.llm_gateway.cost_tracker import get_cost_tracker

            cost_tracker = get_cost_tracker()
            cost_stats = cost_tracker.get_stats()
        except Exception:
            pass

        return {
            "total_providers": len(providers),
            "enabled_providers": len(enabled_providers),
            "healthy_providers": len(healthy_providers),
            "capability_counts": capability_counts,
            "circuit_breaker_states": cb_states,
            "cost_tracking": cost_stats,
        }


# Global registry instance
_registry: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry instance.

    Returns:
        Global ProviderRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry
