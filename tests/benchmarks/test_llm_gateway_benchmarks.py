"""Performance benchmarks for LLM Gateway components.

This module provides pytest-benchmark tests for:
1. Cache service (L1/L2) performance
2. LLM Gateway client performance
3. Provider registry and selection performance

Run with:
    uv run pytest tests/benchmarks/test_llm_gateway_benchmarks.py --benchmark-only
    uv run pytest tests/benchmarks/test_llm_gateway_benchmarks.py --benchmark-compare
    uv run pytest tests/benchmarks/test_llm_gateway_benchmarks.py --benchmark-autosave

For comparison with previous runs:
    uv run pytest tests/benchmarks/test_llm_gateway_benchmarks.py --benchmark-compare=0001

Performance targets:
- L1 cache operations: <1ms (1000 microseconds)
- L2 cache operations: <10ms (10000 microseconds)
- Cache lookup (hit): <2ms
- Cache lookup (miss): <5ms
- Provider selection: <2ms
- Request validation: <1ms
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentcore.llm_gateway.cache_models import (
    CacheConfig,
    CacheEntry,
    CacheKey,
    CacheMode,
    EvictionPolicy,
)
from agentcore.llm_gateway.cache_service import CacheService, L1Cache, L2Cache
from agentcore.llm_gateway.models import LLMRequest, LLMResponse
from agentcore.llm_gateway.provider import (
    ProviderCapabilities,
    ProviderCapability,
    ProviderConfiguration,
    ProviderMetadata,
    ProviderSelectionCriteria,
    ProviderSelectionResult,
)
from agentcore.llm_gateway.registry import ProviderRegistry


@pytest.fixture
def sample_cache_entry() -> CacheEntry:
    """Create sample cache entry for benchmarking."""
    return CacheEntry(
        cache_key="bench:key:1",
        response_id="resp_bench_1",
        model="gpt-4",
        provider="openai",
        choices=[{"message": {"content": "Benchmark response"}}],
        usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        cost=0.015,
        latency_ms=1500,
        created_at=datetime.now(UTC),
        access_count=1,
        last_accessed=datetime.now(UTC),
        ttl_seconds=3600,
    )


@pytest.fixture
def provider_registry() -> ProviderRegistry:
    """Create provider registry for benchmarking."""
    return ProviderRegistry()


class TestL1CacheBenchmarks:
    """Benchmarks for L1 (in-memory) cache operations."""

    def test_benchmark_l1_cache_set(
        self, benchmark: pytest.BenchmarkFixture, sample_cache_entry: CacheEntry
    ) -> None:
        """Benchmark L1 cache write operations.

        Target: <1ms (1000 microseconds)
        """
        cache = L1Cache(max_size=1000, ttl_seconds=3600)

        def set_operation() -> None:
            cache.set("bench:key:1", sample_cache_entry)

        benchmark(set_operation)

    def test_benchmark_l1_cache_get_hit(
        self, benchmark: pytest.BenchmarkFixture, sample_cache_entry: CacheEntry
    ) -> None:
        """Benchmark L1 cache read operations (cache hit).

        Target: <1ms (1000 microseconds)
        """
        cache = L1Cache(max_size=1000, ttl_seconds=3600)
        cache.set("bench:key:1", sample_cache_entry)

        def get_operation() -> CacheEntry | None:
            return cache.get("bench:key:1")

        result = benchmark(get_operation)
        assert result is not None

    def test_benchmark_l1_cache_get_miss(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark L1 cache read operations (cache miss).

        Target: <0.5ms (500 microseconds)
        """
        cache = L1Cache(max_size=1000, ttl_seconds=3600)

        def get_operation() -> CacheEntry | None:
            return cache.get("nonexistent:key")

        result = benchmark(get_operation)
        assert result is None

    def test_benchmark_l1_cache_eviction(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark L1 cache LRU eviction.

        Target: <2ms (2000 microseconds) for eviction + insert
        """
        cache = L1Cache(max_size=100, eviction_policy=EvictionPolicy.LRU)

        # Pre-fill cache to capacity
        for i in range(100):
            entry = CacheEntry(
                cache_key=f"key:{i}",
                response_id=f"resp_{i}",
                model="gpt-4",
                provider="openai",
                choices=[{"message": {"content": f"Response {i}"}}],
                created_at=datetime.now(UTC),
                access_count=1,
                last_accessed=datetime.now(UTC),
            )
            cache.set(f"key:{i}", entry)

        def eviction_operation() -> None:
            # This will trigger eviction
            new_entry = CacheEntry(
                cache_key="key:new",
                response_id="resp_new",
                model="gpt-4",
                provider="openai",
                choices=[{"message": {"content": "New"}}],
                created_at=datetime.now(UTC),
                access_count=1,
                last_accessed=datetime.now(UTC),
            )
            cache.set("key:new", new_entry)

        benchmark(eviction_operation)

    def test_benchmark_l1_cache_concurrent_access(
        self, benchmark: pytest.BenchmarkFixture, sample_cache_entry: CacheEntry
    ) -> None:
        """Benchmark concurrent L1 cache access (simulated with sequential ops).

        Target: <5ms for 100 operations
        """
        cache = L1Cache(max_size=1000)

        # Pre-populate cache
        for i in range(50):
            entry = sample_cache_entry.model_copy(deep=True)
            entry.cache_key = f"key:{i}"
            cache.set(f"key:{i}", entry)

        def mixed_operations() -> None:
            # Mix of reads and writes
            for i in range(100):
                if i % 3 == 0:
                    # Write
                    entry = sample_cache_entry.model_copy(deep=True)
                    entry.cache_key = f"key:{i % 50}"
                    cache.set(f"key:{i % 50}", entry)
                else:
                    # Read
                    cache.get(f"key:{i % 50}")

        benchmark(mixed_operations)


class TestCacheServiceBenchmarks:
    """Benchmarks for multi-level CacheService."""

    def test_benchmark_cache_service_lookup_l1_hit(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark cache service lookup with L1 hit.

        Target: <2ms (includes key hashing + L1 lookup)
        """
        config = CacheConfig(
            enabled=True,
            l1_enabled=True,
            l1_max_size=1000,
            l2_enabled=False,  # Disable L2 for L1-only benchmark
            mode=CacheMode.EXACT,
        )
        service = CacheService(config=config)

        # Create request
        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Benchmark"}],
        )

        # Create response
        response = LLMResponse(
            id="resp_bench",
            model="gpt-4",
            provider="openai",
            choices=[{"message": {"role": "assistant", "content": "Response"}, "index": 0}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        # Pre-populate cache
        import asyncio
        asyncio.run(service.set(request, response))

        async def get_operation() -> tuple[LLMResponse | None, str | None]:
            return await service.get(request)

        result = benchmark(lambda: asyncio.run(get_operation()))
        assert result[0] is not None  # Response should be found

    def test_benchmark_cache_service_lookup_miss(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark cache service lookup with complete miss.

        Target: <5ms (includes L1 + L2 miss)
        """
        config = CacheConfig(
            enabled=True,
            l1_enabled=True,
            l2_enabled=False,  # Keep L2 disabled to avoid Redis dependency
            mode=CacheMode.EXACT,
        )
        service = CacheService(config=config)

        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Nonexistent"}],
        )

        async def get_operation() -> tuple[LLMResponse | None, str | None]:
            return await service.get(request)

        import asyncio
        result = benchmark(lambda: asyncio.run(get_operation()))
        assert result[0] is None  # Response should not be found

    def test_benchmark_cache_service_store(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark cache service store operation.

        Target: <3ms (L1 + L2 write, L2 disabled for benchmark)
        """
        config = CacheConfig(
            enabled=True,
            l1_enabled=True,
            l2_enabled=False,
            mode=CacheMode.EXACT,
        )
        service = CacheService(config=config)

        request = LLMRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Store benchmark"}],
        )
        response = LLMResponse(
            id="resp_store",
            model="gpt-4",
            provider="openai",
            choices=[{"message": {"role": "assistant", "content": "Stored"}, "index": 0}],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

        async def set_operation() -> None:
            await service.set(request, response)

        import asyncio
        benchmark(lambda: asyncio.run(set_operation()))


class TestProviderRegistryBenchmarks:
    """Benchmarks for provider registry operations."""

    def test_benchmark_provider_registration(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark provider registration.

        Target: <2ms
        """
        registry = ProviderRegistry()

        provider_config = ProviderConfiguration(
            provider_id="bench_provider",
            enabled=True,
            priority=100,
            metadata=ProviderMetadata(name="Benchmark Provider"),
            capabilities=ProviderCapabilities(
                capabilities=[
                    ProviderCapability.TEXT_GENERATION,
                    ProviderCapability.CHAT_COMPLETION,
                ],
                supports_function_calling=True,
                supports_streaming=True,
                context_window=128000,
            ),
        )

        def register_operation() -> None:
            # Create new registry each time to avoid conflicts
            reg = ProviderRegistry()
            reg.register_provider(provider_config)

        benchmark(register_operation)

    def test_benchmark_provider_lookup_by_id(
        self, benchmark: pytest.BenchmarkFixture, provider_registry: ProviderRegistry
    ) -> None:
        """Benchmark provider lookup by ID.

        Target: <0.5ms
        """
        # Pre-register a provider
        provider_config = ProviderConfiguration(
            provider_id="lookup_bench",
            enabled=True,
            priority=100,
            metadata=ProviderMetadata(name="Lookup Bench"),
            capabilities=ProviderCapabilities(
                capabilities=[ProviderCapability.TEXT_GENERATION],
                context_window=8000,
            ),
        )
        provider_registry.register_provider(provider_config)

        def lookup_operation() -> ProviderConfiguration | None:
            return provider_registry.get_provider("lookup_bench")

        result = benchmark(lookup_operation)
        assert result is not None

    def test_benchmark_provider_selection_by_criteria(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark provider selection by criteria.

        Target: <2ms
        """
        registry = ProviderRegistry()

        # Register multiple providers
        for i in range(10):
            config = ProviderConfiguration(
                provider_id=f"provider_{i}",
                enabled=True,
                priority=100 - i,  # Decreasing priority
                metadata=ProviderMetadata(name=f"Provider {i}"),
                capabilities=ProviderCapabilities(
                    capabilities=[
                        ProviderCapability.TEXT_GENERATION,
                        ProviderCapability.CHAT_COMPLETION,
                    ],
                    context_window=8000 + (i * 1000),
                ),
            )
            registry.register_provider(config)

        criteria = ProviderSelectionCriteria(
            required_capabilities=[ProviderCapability.CHAT_COMPLETION],
            min_context_window=10000,
        )

        def selection_operation() -> ProviderSelectionResult:
            return registry.select_provider(criteria)

        result = benchmark(selection_operation)
        assert result.provider is not None

    def test_benchmark_list_all_providers(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark listing all providers.

        Target: <1ms for 50 providers
        """
        registry = ProviderRegistry()

        # Register many providers
        for i in range(50):
            config = ProviderConfiguration(
                provider_id=f"provider_{i}",
                enabled=i % 2 == 0,  # Half enabled, half disabled
                priority=100,
                metadata=ProviderMetadata(name=f"Provider {i}"),
                capabilities=ProviderCapabilities(
                    capabilities=[ProviderCapability.TEXT_GENERATION],
                    context_window=8000,
                ),
            )
            registry.register_provider(config)

        def list_operation() -> list[ProviderConfiguration]:
            return registry.list_providers(enabled_only=False)

        result = benchmark(list_operation)
        assert len(result) == 50


class TestCacheKeyBenchmarks:
    """Benchmarks for cache key generation and hashing."""

    def test_benchmark_cache_key_creation(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark cache key creation from request data.

        Target: <0.5ms
        """
        def create_key() -> CacheKey:
            return CacheKey(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

        result = benchmark(create_key)
        assert result is not None

    def test_benchmark_cache_key_hashing(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark cache key hashing for lookup.

        Target: <0.5ms
        """
        cache_key = CacheKey(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        def hash_operation() -> str:
            return cache_key.to_hash()

        result = benchmark(hash_operation)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_benchmark_cache_key_comparison(
        self, benchmark: pytest.BenchmarkFixture
    ) -> None:
        """Benchmark cache key equality comparison.

        Target: <0.2ms
        """
        key1 = CacheKey(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
        )
        key2 = CacheKey(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
        )

        def comparison_operation() -> bool:
            return key1.to_hash() == key2.to_hash()

        result = benchmark(comparison_operation)
        assert result is True


# Benchmark groups for easy filtering
pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.performance,
]
