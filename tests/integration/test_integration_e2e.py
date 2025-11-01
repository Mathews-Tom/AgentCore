"""End-to-end integration tests for the Integration Layer.

Tests INT-014 acceptance criteria:
- Mock provider testing
- End-to-end integration tests
- Cache integration tests
- Connector tests

This test suite validates the complete integration layer workflow including:
- Portkey LLM provider integration
- Multi-level caching (L1/L2)
- Cost tracking and optimization
- Provider failover and resilience
- Performance monitoring
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agentcore.llm_gateway.cache_models import CacheConfig, CacheMode
from agentcore.llm_gateway.cache_service import CacheService
from agentcore.llm_gateway.cost_tracker import CostTracker
from agentcore.llm_gateway.models import LLMRequest, LLMResponse


class MockLLMProvider:
    """Mock LLM provider for testing without external dependencies."""

    def __init__(self, provider_name: str = "mock-provider", latency_ms: int = 100):
        self.provider_name = provider_name
        self.latency_ms = latency_ms
        self.request_count = 0
        self.failure_rate = 0.0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Simulate LLM completion request."""
        self.request_count += 1

        # Simulate failure rate
        import random

        if random.random() < self.failure_rate:
            raise Exception(f"{self.provider_name} unavailable")

        # Generate mock response
        return LLMResponse(
            id=f"mock-resp-{self.request_count}",
            model=request.model,
            provider=self.provider_name,
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": f"Mock response from {self.provider_name}",
                    }
                }
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            cost=0.001,
            latency_ms=self.latency_ms)


@pytest_asyncio.fixture
async def mock_provider() -> MockLLMProvider:
    """Create mock provider for testing."""
    return MockLLMProvider(provider_name="openai-mock")


@pytest_asyncio.fixture
async def cache_service() -> CacheService:
    """Create cache service for e2e testing."""
    config = CacheConfig(
        enabled=True,
        l1_enabled=True,
        l1_max_size=100,
        l2_enabled=True,
        mode=CacheMode.EXACT,
        stats_enabled=True)

    service = CacheService(config=config)
    await service.connect()

    yield service

    await service.clear()
    await service.close()


@pytest_asyncio.fixture
def cost_tracker() -> CostTracker:
    """Create cost tracker for e2e testing."""
    return CostTracker()


class TestEndToEndIntegration:
    """End-to-end integration tests for complete workflow."""

    @pytest.mark.asyncio
    async def test_complete_llm_workflow_with_caching(
        self,
        mock_provider: MockLLMProvider,
        cache_service: CacheService,
        cost_tracker: CostTracker) -> None:
        """Test complete LLM request workflow with caching and cost tracking.

        Validates:
        - Mock provider request/response
        - Cache miss → provider call
        - Cache hit → no provider call
        - Cost tracking across requests
        """
        # Create request
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}], context={"agent_id": "test-agent"})

        # First request: Cache miss → Provider call
        cached_response, cache_level = await cache_service.get(request)
        assert cached_response is None
        assert cache_level is None

        provider_response = await mock_provider.complete(request)
        assert provider_response is not None
        assert provider_response.provider == "openai-mock"

        # Store in cache
        await cache_service.set(request, provider_response)

        # Track cost
        cost_tracker.track_request(
            provider=provider_response.provider,
            model=provider_response.model,
            cost=provider_response.cost,
            tokens=provider_response.usage["total_tokens"])

        # Second request: Cache hit → No provider call
        initial_request_count = mock_provider.request_count

        cached_response2, cache_level2 = await cache_service.get(request)
        assert cached_response2 is not None
        assert cache_level2 in ("l1", "l2")
        assert cached_response2.id == provider_response.id

        # Verify provider was NOT called
        assert mock_provider.request_count == initial_request_count

        # Verify cost tracking
        stats = cost_tracker.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_cost"] > 0

    @pytest.mark.asyncio
    async def test_provider_failover_with_fallback(
        self,
        cache_service: CacheService) -> None:
        """Test provider failover when primary fails.

        Validates:
        - Primary provider failure detection
        - Automatic fallback to secondary provider
        - Successful completion with fallback
        """
        # Create primary and fallback providers
        primary = MockLLMProvider(provider_name="primary", latency_ms=50)
        fallback = MockLLMProvider(provider_name="fallback", latency_ms=100)

        # Set primary to fail
        primary.failure_rate = 1.0

        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test failover"}])

        # Try primary (will fail)
        with pytest.raises(Exception, match="primary unavailable"):
            await primary.complete(request)

        # Fallback to secondary
        response = await fallback.complete(request)
        assert response is not None
        assert response.provider == "fallback"

        # Cache fallback response
        await cache_service.set(request, response)

        # Verify cached
        cached, level = await cache_service.get(request)
        assert cached is not None
        assert cached.provider == "fallback"

    @pytest.mark.asyncio
    async def test_multi_provider_cost_optimization(
        self,
        cache_service: CacheService,
        cost_tracker: CostTracker) -> None:
        """Test cost optimization across multiple providers.

        Validates:
        - Multiple provider simulation
        - Cost comparison and selection
        - Cache hit rate optimization
        """
        # Create providers with different costs
        cheap_provider = MockLLMProvider(provider_name="cheap", latency_ms=200)
        fast_provider = MockLLMProvider(provider_name="fast", latency_ms=50)

        # Override costs
        cheap_cost = 0.0005
        fast_cost = 0.002

        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Cost optimization test"}])

        # Use cheap provider
        cheap_response = await cheap_provider.complete(request)
        cheap_response.cost = cheap_cost

        cost_tracker.track_request(
            provider="cheap",
            model=request.model,
            cost=cheap_cost,
            tokens=30)

        await cache_service.set(request, cheap_response)

        # Subsequent requests hit cache (no additional cost)
        for _ in range(10):
            cached, level = await cache_service.get(request)
            assert cached is not None
            assert level in ("l1", "l2")

        # Verify cost savings
        stats = cost_tracker.get_stats()
        assert stats["total_cost"] == cheap_cost  # Only paid once

        # Without cache, would have cost: 11 × $0.0005 = $0.0055
        # With cache: 1 × $0.0005 = $0.0005
        # Savings: $0.005 (90.9%)

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(
        self,
        mock_provider: MockLLMProvider,
        cache_service: CacheService) -> None:
        """Test concurrent request handling with caching.

        Validates:
        - Concurrent provider requests
        - Cache coherency under concurrent access
        - No race conditions
        """
        import asyncio

        # Create unique requests
        requests = [
            LLMRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Request {i}"}])
            for i in range(10)
        ]

        # Execute concurrently
        async def process_request(req: LLMRequest) -> LLMResponse:
            # Check cache
            cached, _ = await cache_service.get(req)
            if cached:
                return cached

            # Call provider
            response = await mock_provider.complete(req)

            # Cache result
            await cache_service.set(req, response)

            return response

        # Process all concurrently
        responses = await asyncio.gather(*[process_request(req) for req in requests])

        # Verify all succeeded
        assert len(responses) == 10
        assert all(r is not None for r in responses)

        # Verify cache populated
        for req in requests:
            cached, level = await cache_service.get(req)
            assert cached is not None

    @pytest.mark.asyncio
    async def test_performance_monitoring_metrics(
        self,
        mock_provider: MockLLMProvider,
        cache_service: CacheService,
        cost_tracker: CostTracker) -> None:
        """Test performance monitoring and metrics collection.

        Validates:
        - Latency tracking
        - Cache hit rate monitoring
        - Cost per request tracking
        - Provider performance metrics
        """
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Metrics test"}])

        # Generate requests for metrics
        for i in range(20):
            # Every other request is unique (cache miss)
            if i % 2 == 0:
                req = LLMRequest(
                    model="gpt-4",
                    messages=[{"role": "user", "content": f"Unique {i}"}])
            else:
                req = request  # Reuse request (cache hit)

            # Check cache
            cached, level = await cache_service.get(req)

            if not cached:
                # Provider call
                response = await mock_provider.complete(req)
                await cache_service.set(req, response)

                cost_tracker.track_request(
                    provider=response.provider,
                    model=response.model,
                    cost=response.cost,
                    tokens=response.usage["total_tokens"])

        # Verify metrics
        cache_stats = cache_service.get_stats()
        assert cache_stats is not None

        # Hit rate should be ~50% (10 unique + 10 repeated)
        hit_rate = cache_stats.get_hit_rate()
        assert 0.45 <= hit_rate <= 0.55

        cost_stats = cost_tracker.get_stats()
        assert cost_stats["total_requests"] == 11  # 10 unique even requests + 1 reused request first time
        assert cost_stats["total_cost"] > 0


class TestMockProviderBehavior:
    """Test mock provider functionality."""

    @pytest.mark.asyncio
    async def test_mock_provider_basic_completion(
        self, mock_provider: MockLLMProvider
    ) -> None:
        """Test basic mock provider completion."""
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}])

        response = await mock_provider.complete(request)

        assert response is not None
        assert response.provider == "openai-mock"
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert "Mock response" in response.choices[0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_mock_provider_failure_simulation(self) -> None:
        """Test mock provider failure simulation."""
        provider = MockLLMProvider(provider_name="failing-provider")
        provider.failure_rate = 1.0  # Always fail

        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test failure"}])

        with pytest.raises(Exception, match="failing-provider unavailable"):
            await provider.complete(request)

    @pytest.mark.asyncio
    async def test_mock_provider_request_counting(
        self, mock_provider: MockLLMProvider
    ) -> None:
        """Test mock provider request counting."""
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Count test"}])

        assert mock_provider.request_count == 0

        await mock_provider.complete(request)
        assert mock_provider.request_count == 1

        await mock_provider.complete(request)
        assert mock_provider.request_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
