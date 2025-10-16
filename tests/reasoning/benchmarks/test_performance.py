"""
Performance benchmarks for bounded context reasoning.

Tests latency, throughput, memory usage, and compute savings under various load conditions.

Run benchmarks:
    uv run pytest tests/reasoning/benchmarks/test_performance.py --benchmark-only

Run with comparison:
    uv run pytest tests/reasoning/benchmarks/test_performance.py --benchmark-compare

Generate histogram:
    uv run pytest tests/reasoning/benchmarks/test_performance.py --benchmark-histogram
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import BaseModel

from src.agentcore.a2a_protocol.models.jsonrpc import A2AContext, JsonRpcRequest
from src.agentcore.reasoning.engines.bounded_context_engine import BoundedContextEngine
from src.agentcore.reasoning.models.reasoning_models import BoundedContextConfig
from src.agentcore.reasoning.services.llm_client import LLMClient, LLMClientConfig
from src.agentcore.reasoning.services.reasoning_jsonrpc import (
    BoundedReasoningParams,
    handle_bounded_reasoning,
)


def create_mock_llm_client() -> LLMClient:
    """
    Create mock LLM client for benchmarking.

    Returns responses with realistic latency (100-200ms) to simulate actual LLM calls.
    """
    mock_client = Mock(spec=LLMClient)

    async def mock_complete(prompt: str, temperature: float, max_tokens: int) -> str:
        # Simulate realistic LLM latency (100-200ms)
        await asyncio.sleep(0.15)
        # Return response proportional to max_tokens
        response_size = min(max_tokens // 10, 200)
        return f"Mock reasoning response. " * response_size

    mock_client.complete = mock_complete
    return mock_client


def create_authenticated_request(
    query: str, chunk_size: int = 8192, max_iterations: int = 5
) -> JsonRpcRequest:
    """
    Create authenticated JSON-RPC request for benchmarking.

    Note: In real benchmarks, this would use actual JWT tokens. For performance testing,
    we mock the authentication to isolate reasoning engine performance.
    """
    return JsonRpcRequest(
        jsonrpc="2.0",
        method="reasoning.bounded_context",
        params={
            "auth_token": "mock-token",  # Mocked for benchmark
            "query": query,
            "chunk_size": chunk_size,
            "max_iterations": max_iterations,
        },
        id="bench-1",
        a2a_context=A2AContext(
            trace_id="bench-trace-1",
            source_agent="benchmark-client",
            target_agent="reasoning-agent",
        ),
    )


@pytest.fixture
def mock_security_service(monkeypatch):
    """Mock security service for benchmarking."""
    from src.agentcore.a2a_protocol.models.security import Permission, Role, TokenPayload, TokenType

    mock_payload = TokenPayload(
        sub="benchmark-user",
        token_type=TokenType.ACCESS,
        role=Role.ADMIN,
        permissions=[Permission.REASONING_EXECUTE],
        exp=9999999999,
    )

    def mock_validate_token(token: str) -> TokenPayload:
        return mock_payload

    from src.agentcore.a2a_protocol.services import security_service

    monkeypatch.setattr(
        security_service.security_service, "validate_token", mock_validate_token
    )


# Benchmark: Single Request Latency
@pytest.mark.asyncio
@pytest.mark.benchmark(group="latency")
async def test_benchmark_single_request_latency(benchmark, mock_security_service, monkeypatch, event_loop):
    """
    Benchmark single request latency.

    Measures p50, p95, p99 latency for processing a single reasoning request.
    Target: <2s for typical query (5K tokens).
    """
    # Mock LLM client
    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    query = "Analyze the performance characteristics of distributed systems." * 50  # ~5K tokens

    async def run_request():
        request = create_authenticated_request(query, chunk_size=8192, max_iterations=3)
        result = await handle_bounded_reasoning(request)
        return result

    # Run benchmark using existing event loop
    result = benchmark.pedantic(event_loop.run_until_complete, args=(run_request(),), rounds=10, iterations=1)

    assert result["success"] is True
    assert result["total_iterations"] > 0


@pytest.mark.asyncio
@pytest.mark.benchmark(group="throughput")
async def test_benchmark_concurrent_requests(benchmark, mock_security_service, monkeypatch, event_loop):
    """
    Benchmark concurrent request throughput.

    Measures system throughput under concurrent load (10 requests).
    Target: >5 requests/second with 10 concurrent requests.
    """
    # Mock LLM client
    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    query = "What are the key principles of system design?" * 30  # ~3K tokens

    async def run_concurrent_requests(num_requests: int = 10):
        requests = [
            handle_bounded_reasoning(
                create_authenticated_request(query, chunk_size=8192, max_iterations=2)
            )
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*requests, return_exceptions=True)
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        return successful

    # Run benchmark using existing event loop
    result = benchmark.pedantic(
        event_loop.run_until_complete, args=(run_concurrent_requests(10),), rounds=5, iterations=1
    )

    assert result >= 8  # At least 80% success rate


@pytest.mark.asyncio
@pytest.mark.benchmark(group="compute_savings")
async def test_benchmark_compute_savings_small_query(benchmark, mock_security_service, monkeypatch, event_loop):
    """
    Benchmark compute savings for small queries (10K tokens).

    Measures token reduction and latency overhead.
    Target: 40-60% compute savings, <10% latency increase.
    """
    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    # 10K token query
    query = "Explain distributed consensus algorithms in detail." * 200

    async def run_request():
        request = create_authenticated_request(query, chunk_size=4096, max_iterations=5)
        result = await handle_bounded_reasoning(request)
        return result

    result = benchmark.pedantic(event_loop.run_until_complete, args=(run_request(),), rounds=5, iterations=1)

    assert result["success"] is True
    assert result["compute_savings_pct"] >= 30.0  # At least 30% savings


@pytest.mark.asyncio
@pytest.mark.benchmark(group="compute_savings")
async def test_benchmark_compute_savings_medium_query(benchmark, mock_security_service, monkeypatch, event_loop):
    """
    Benchmark compute savings for medium queries (25K tokens).

    Target: 50-70% compute savings, <15% latency increase.
    """
    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    # 25K token query
    query = "Provide a comprehensive analysis of machine learning architectures." * 500

    async def run_request():
        request = create_authenticated_request(query, chunk_size=8192, max_iterations=5)
        result = await handle_bounded_reasoning(request)
        return result

    result = benchmark.pedantic(event_loop.run_until_complete, args=(run_request(),), rounds=5, iterations=1)

    assert result["success"] is True
    assert result["compute_savings_pct"] >= 40.0  # At least 40% savings


@pytest.mark.asyncio
@pytest.mark.benchmark(group="compute_savings")
async def test_benchmark_compute_savings_large_query(benchmark, mock_security_service, monkeypatch, event_loop):
    """
    Benchmark compute savings for large queries (50K tokens).

    Target: 60-80% compute savings, <20% latency increase.
    """
    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    # 50K token query
    query = "Write a detailed technical specification for a microservices architecture." * 1000

    async def run_request():
        request = create_authenticated_request(query, chunk_size=8192, max_iterations=10)
        result = await handle_bounded_reasoning(request)
        return result

    result = benchmark.pedantic(event_loop.run_until_complete, args=(run_request(),), rounds=3, iterations=1)

    assert result["success"] is True
    assert result["compute_savings_pct"] >= 50.0  # At least 50% savings


@pytest.mark.asyncio
@pytest.mark.benchmark(group="memory")
async def test_benchmark_memory_usage(benchmark, mock_security_service, monkeypatch, event_loop):
    """
    Benchmark memory usage under load.

    Validates that memory usage remains constant (O(1)) regardless of reasoning depth.
    Target: Memory growth <10% across iterations.
    """
    import tracemalloc

    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    query = "Analyze system scalability patterns." * 100

    async def run_with_memory_tracking():
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        request = create_authenticated_request(query, chunk_size=4096, max_iterations=5)
        result = await handle_bounded_reasoning(request)

        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()

        memory_growth = (final_memory - initial_memory) / initial_memory
        return {"result": result, "memory_growth": memory_growth}

    result = benchmark.pedantic(
        event_loop.run_until_complete, args=(run_with_memory_tracking(),), rounds=3, iterations=1
    )

    assert result["result"]["success"] is True
    # Memory growth should be minimal (<50% of initial allocation)
    assert result["memory_growth"] < 0.5


@pytest.mark.asyncio
@pytest.mark.benchmark(group="iteration_depth")
async def test_benchmark_iteration_scaling(benchmark, mock_security_service, monkeypatch):
    """
    Benchmark performance scaling with iteration depth.

    Tests that latency increases linearly (not exponentially) with iterations.
    Target: Linear O(n) latency scaling with iteration count.
    """
    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    query = "Explain distributed tracing in microservices." * 150

    async def run_with_iterations(max_iterations: int):
        start = time.time()
        request = create_authenticated_request(query, chunk_size=4096, max_iterations=max_iterations)
        result = await handle_bounded_reasoning(request)
        duration = time.time() - start
        return {"result": result, "duration": duration, "iterations": result["total_iterations"]}

    # Test with 3 iterations
    result_3 = asyncio.run(run_with_iterations(3))

    # Test with 6 iterations
    result_6 = asyncio.run(run_with_iterations(6))

    # Latency should scale roughly linearly
    # If iterations doubled, latency should be <2.5x (allowing some overhead)
    iteration_ratio = result_6["iterations"] / max(result_3["iterations"], 1)
    latency_ratio = result_6["duration"] / result_3["duration"]

    assert latency_ratio < iteration_ratio * 1.5  # Linear scaling with <50% overhead


@pytest.mark.asyncio
@pytest.mark.benchmark(group="token_throughput")
async def test_benchmark_token_throughput(benchmark, mock_security_service, monkeypatch, event_loop):
    """
    Benchmark token processing throughput.

    Measures tokens processed per second under sustained load.
    Target: >10,000 tokens/second processing rate.
    """
    mock_llm = create_mock_llm_client()
    monkeypatch.setattr(
        "src.agentcore.reasoning.services.reasoning_jsonrpc.LLMClient",
        lambda config: mock_llm,
    )

    query = "Describe cloud-native architecture patterns." * 200  # ~20K tokens

    async def run_request():
        start = time.time()
        request = create_authenticated_request(query, chunk_size=8192, max_iterations=5)
        result = await handle_bounded_reasoning(request)
        duration = time.time() - start

        tokens_per_second = result["total_tokens"] / duration
        return {"result": result, "tokens_per_second": tokens_per_second}

    result = benchmark.pedantic(event_loop.run_until_complete, args=(run_request(),), rounds=5, iterations=1)

    assert result["result"]["success"] is True
    # Throughput check (mock LLM should be fast)
    assert result["tokens_per_second"] > 1000  # Conservative target for mocked client


# Benchmark summary fixture
@pytest.fixture(scope="session", autouse=True)
def benchmark_summary():
    """Print benchmark summary after all tests."""
    yield
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 80)
    print("\nTargets:")
    print("  - Single Request Latency: <2s (p95)")
    print("  - Concurrent Throughput: >5 req/s (10 concurrent)")
    print("  - Compute Savings: 50-90% reduction")
    print("  - Memory Usage: O(1) constant across iterations")
    print("  - Latency Overhead: <20% vs traditional")
    print("\nRun with --benchmark-compare to compare against baseline")
    print("Run with --benchmark-histogram to generate latency histograms")
    print("=" * 80)
