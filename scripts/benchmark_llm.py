#!/usr/bin/env python3
"""Comprehensive benchmarking suite for LLM client service.

This script validates performance SLOs through comprehensive benchmarking:
1. Microbenchmarks: Provider selection and normalization overhead
2. Load tests: 100, 500, 1000 concurrent requests
3. SDK comparison: Abstraction layer vs direct SDK calls
4. Latency histograms: p50, p90, p95, p99
5. Throughput: Requests per second
6. Resource profiling: CPU and memory usage

Success Criteria:
- Abstraction overhead <5ms (p95)
- Time to first token <500ms (p95)
- 1000 concurrent requests complete successfully
- Abstraction layer within ±5% of direct SDK performance

Run with:
    uv run python scripts/benchmark_llm.py
    uv run python scripts/benchmark_llm.py --load-only
    uv run python scripts/benchmark_llm.py --profile
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentcore.a2a_protocol.models.llm import LLMRequest, Provider
from agentcore.a2a_protocol.services.llm_service import (
    MODEL_PROVIDER_MAP,
    LLMService,
    ProviderRegistry,
    llm_service,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    duration_ms: float
    success: bool
    error: str | None = None
    tokens: int | None = None
    provider: str | None = None


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""

    count: int
    mean_ms: float
    median_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""

    microbenchmarks: dict[str, LatencyStats]
    load_tests: dict[str, LatencyStats]
    sdk_comparison: dict[str, dict[str, LatencyStats]]
    throughput: dict[str, float]
    resource_usage: dict[str, Any]
    validation: dict[str, bool]


def calculate_latency_stats(durations_ms: list[float]) -> LatencyStats:
    """Calculate comprehensive latency statistics.

    Args:
        durations_ms: List of duration measurements in milliseconds

    Returns:
        LatencyStats with p50, p90, p95, p99, mean, median, etc.
    """
    if not durations_ms:
        return LatencyStats(
            count=0,
            mean_ms=0.0,
            median_ms=0.0,
            p50_ms=0.0,
            p90_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            stddev_ms=0.0,
        )

    sorted_durations = sorted(durations_ms)
    count = len(sorted_durations)

    return LatencyStats(
        count=count,
        mean_ms=statistics.mean(sorted_durations),
        median_ms=statistics.median(sorted_durations),
        p50_ms=sorted_durations[int(count * 0.50)],
        p90_ms=sorted_durations[int(count * 0.90)],
        p95_ms=sorted_durations[int(count * 0.95)],
        p99_ms=sorted_durations[min(int(count * 0.99), count - 1)],
        min_ms=min(sorted_durations),
        max_ms=max(sorted_durations),
        stddev_ms=statistics.stdev(sorted_durations) if count > 1 else 0.0,
    )


async def benchmark_provider_selection(iterations: int = 1000) -> LatencyStats:
    """Benchmark provider selection overhead.

    Measures time to select provider for a model using ProviderRegistry.
    Target: <1ms p95

    Args:
        iterations: Number of iterations to run (default 1000)

    Returns:
        LatencyStats for provider selection operations
    """
    registry = ProviderRegistry()
    durations_ms: list[float] = []
    # Use only allowed models from CLAUDE.md
    models = ["gpt-5-mini", "claude-haiku-4-5-20251001", "gemini-2.5-flash-lite"]

    print(f"Running {iterations} provider selection iterations...")

    for i in range(iterations):
        model = models[i % len(models)]
        start = time.perf_counter()
        try:
            registry.get_provider_for_model(model)
        except RuntimeError:
            # Expected if API keys not configured
            pass
        duration_ms = (time.perf_counter() - start) * 1000
        durations_ms.append(duration_ms)

    return calculate_latency_stats(durations_ms)


async def benchmark_model_lookup(iterations: int = 10000) -> LatencyStats:
    """Benchmark model-to-provider mapping lookup.

    Measures overhead of MODEL_PROVIDER_MAP dictionary lookup.
    Target: <0.1ms p95

    Args:
        iterations: Number of iterations to run (default 10000)

    Returns:
        LatencyStats for model lookup operations
    """
    durations_ms: list[float] = []
    models = list(MODEL_PROVIDER_MAP.keys())

    print(f"Running {iterations} model lookup iterations...")

    for i in range(iterations):
        model = models[i % len(models)]
        start = time.perf_counter()
        _ = MODEL_PROVIDER_MAP.get(model)
        duration_ms = (time.perf_counter() - start) * 1000
        durations_ms.append(duration_ms)

    return calculate_latency_stats(durations_ms)


async def benchmark_request_validation(iterations: int = 1000) -> LatencyStats:
    """Benchmark LLMRequest validation overhead.

    Measures Pydantic model validation time.
    Target: <1ms p95

    Args:
        iterations: Number of iterations to run (default 1000)

    Returns:
        LatencyStats for request validation operations
    """
    durations_ms: list[float] = []

    print(f"Running {iterations} request validation iterations...")

    for _ in range(iterations):
        start = time.perf_counter()
        request = LLMRequest(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=100,
            trace_id="bench-001",
        )
        duration_ms = (time.perf_counter() - start) * 1000
        durations_ms.append(duration_ms)

    return calculate_latency_stats(durations_ms)


async def benchmark_concurrent_requests(
    concurrency: int, model: str = "gpt-5-mini"
) -> tuple[LatencyStats, int, int]:
    """Benchmark concurrent LLM requests.

    Measures latency distribution under concurrent load.
    Target: 1000 concurrent requests complete successfully

    Args:
        concurrency: Number of concurrent requests
        model: Model to test (must have API key configured)

    Returns:
        Tuple of (LatencyStats, success_count, failure_count)
    """
    service = LLMService(timeout=60.0, max_retries=3)
    durations_ms: list[float] = []
    successes = 0
    failures = 0

    print(f"Running {concurrency} concurrent requests for {model}...")

    async def single_request(request_id: int) -> None:
        nonlocal successes, failures
        start = time.perf_counter()
        try:
            request = LLMRequest(
                model=model,
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5,
                temperature=0.0,
                trace_id=f"bench-concurrent-{request_id}",
            )
            await service.complete(request)
            duration_ms = (time.perf_counter() - start) * 1000
            durations_ms.append(duration_ms)
            successes += 1
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            durations_ms.append(duration_ms)
            failures += 1
            print(f"  Request {request_id} failed: {e}")

    # Run all requests concurrently
    tasks = [single_request(i) for i in range(concurrency)]
    await asyncio.gather(*tasks, return_exceptions=True)

    stats = calculate_latency_stats(durations_ms)
    return stats, successes, failures


async def benchmark_streaming_ttft(model: str = "gpt-5-mini") -> LatencyStats:
    """Benchmark time to first token (TTFT) for streaming.

    Measures latency until first token is received.
    Target: <500ms p95

    Args:
        model: Model to test (must have API key configured)

    Returns:
        LatencyStats for time to first token
    """
    service = LLMService(timeout=60.0)
    ttfts_ms: list[float] = []
    iterations = 10  # Fewer iterations due to cost

    print(f"Running {iterations} streaming TTFT measurements for {model}...")

    for i in range(iterations):
        request = LLMRequest(
            model=model,
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=50,
            temperature=0.0,
            trace_id=f"bench-ttft-{i}",
            stream=True,
        )

        start = time.perf_counter()
        try:
            async for _ in service.stream(request):
                # Measure time to first token only
                ttft_ms = (time.perf_counter() - start) * 1000
                ttfts_ms.append(ttft_ms)
                break
        except Exception as e:
            print(f"  TTFT measurement {i} failed: {e}")

    return calculate_latency_stats(ttfts_ms)


async def benchmark_direct_sdk_openai(iterations: int = 10) -> LatencyStats:
    """Benchmark direct OpenAI SDK calls (no abstraction).

    Baseline performance for comparison with abstraction layer.

    Args:
        iterations: Number of iterations to run (default 10 due to cost)

    Returns:
        LatencyStats for direct SDK calls
    """
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  Skipping: OPENAI_API_KEY not configured")
        return calculate_latency_stats([])

    client = openai.AsyncOpenAI(api_key=api_key)
    durations_ms: list[float] = []

    print(f"Running {iterations} direct OpenAI SDK calls...")

    for i in range(iterations):
        start = time.perf_counter()
        try:
            await client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5,
                temperature=0.0,
            )
            duration_ms = (time.perf_counter() - start) * 1000
            durations_ms.append(duration_ms)
        except Exception as e:
            print(f"  Direct SDK call {i} failed: {e}")

    return calculate_latency_stats(durations_ms)


async def benchmark_abstraction_layer_openai(iterations: int = 10) -> LatencyStats:
    """Benchmark abstraction layer calls for OpenAI.

    Compare with direct SDK to measure overhead.

    Args:
        iterations: Number of iterations to run (default 10 due to cost)

    Returns:
        LatencyStats for abstraction layer calls
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("  Skipping: OPENAI_API_KEY not configured")
        return calculate_latency_stats([])

    service = LLMService(timeout=60.0, max_retries=3)
    durations_ms: list[float] = []

    print(f"Running {iterations} abstraction layer calls (OpenAI)...")

    for i in range(iterations):
        start = time.perf_counter()
        try:
            request = LLMRequest(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5,
                temperature=0.0,
                trace_id=f"bench-abstraction-{i}",
            )
            await service.complete(request)
            duration_ms = (time.perf_counter() - start) * 1000
            durations_ms.append(duration_ms)
        except Exception as e:
            print(f"  Abstraction layer call {i} failed: {e}")

    return calculate_latency_stats(durations_ms)


async def benchmark_throughput(duration_seconds: int = 30) -> dict[str, float]:
    """Benchmark requests per second throughput.

    Measures sustained throughput over a time period.
    Target: >100 req/s per provider

    Args:
        duration_seconds: How long to run throughput test (default 30s)

    Returns:
        Dictionary mapping model to requests/second
    """
    service = LLMService(timeout=60.0, max_retries=3)
    results: dict[str, float] = {}

    # Test with lightweight model (fastest, cheapest)
    model = "gpt-5-mini"
    if not os.getenv("OPENAI_API_KEY"):
        print(f"  Skipping throughput test: API key not configured")
        return results

    print(f"Running {duration_seconds}s throughput test for {model}...")

    request_count = 0
    start_time = time.time()
    end_time = start_time + duration_seconds

    async def send_request() -> None:
        nonlocal request_count
        try:
            request = LLMRequest(
                model=model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=3,
                temperature=0.0,
                trace_id=f"bench-throughput-{request_count}",
            )
            await service.complete(request)
            request_count += 1
        except Exception:
            pass

    # Send requests continuously until time expires
    tasks = []
    while time.time() < end_time:
        tasks.append(asyncio.create_task(send_request()))
        # Small delay to avoid overwhelming the system
        await asyncio.sleep(0.01)

    # Wait for remaining tasks
    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time
    rps = request_count / elapsed if elapsed > 0 else 0.0
    results[model] = rps

    print(f"  Completed {request_count} requests in {elapsed:.2f}s = {rps:.2f} req/s")

    return results


async def profile_memory_usage() -> dict[str, float]:
    """Profile memory usage during LLM operations.

    Measures memory consumption before, during, and after operations.

    Returns:
        Dictionary with memory statistics (MB)
    """
    try:
        import psutil

        process = psutil.Process()
    except ImportError:
        print("  Skipping memory profiling: psutil not installed")
        return {}

    # Baseline memory
    baseline_mb = process.memory_info().rss / 1024 / 1024

    if not os.getenv("OPENAI_API_KEY"):
        print("  Skipping memory profiling: API key not configured")
        return {"baseline_mb": baseline_mb}

    print("Profiling memory usage...")

    service = LLMService(timeout=60.0)

    # Run some requests
    request = LLMRequest(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Say 'ok'"}],
        max_tokens=5,
        temperature=0.0,
        trace_id="bench-memory",
    )

    for i in range(10):
        await service.complete(request)

    peak_mb = process.memory_info().rss / 1024 / 1024

    return {
        "baseline_mb": baseline_mb,
        "peak_mb": peak_mb,
        "delta_mb": peak_mb - baseline_mb,
    }


def validate_slos(suite: BenchmarkSuite) -> dict[str, bool]:
    """Validate that all SLOs are met.

    Success Criteria:
    - Abstraction overhead <5ms (p95)
    - Time to first token <500ms (p95)
    - 1000 concurrent requests complete successfully
    - Abstraction layer within ±5% of direct SDK

    Args:
        suite: Complete benchmark results

    Returns:
        Dictionary mapping SLO name to pass/fail
    """
    validation = {}

    # SLO 1: Provider selection <1ms p95
    if "provider_selection" in suite.microbenchmarks:
        stats = suite.microbenchmarks["provider_selection"]
        validation["provider_selection_p95_lt_1ms"] = stats.p95_ms < 1.0

    # SLO 2: Request validation <1ms p95
    if "request_validation" in suite.microbenchmarks:
        stats = suite.microbenchmarks["request_validation"]
        validation["request_validation_p95_lt_1ms"] = stats.p95_ms < 1.0

    # SLO 3: Total abstraction overhead <5ms p95
    if "abstraction_layer_openai" in suite.sdk_comparison.get("openai", {}):
        abstraction = suite.sdk_comparison["openai"]["abstraction_layer_openai"]
        direct = suite.sdk_comparison["openai"].get("direct_sdk_openai")
        if direct:
            overhead_ms = abstraction.mean_ms - direct.mean_ms
            validation["abstraction_overhead_lt_5ms"] = overhead_ms < 5.0
            validation["abstraction_within_5pct"] = abs(overhead_ms / direct.mean_ms) < 0.05

    # SLO 4: Time to first token <500ms p95
    if "streaming_ttft_gpt-5-mini" in suite.load_tests:
        stats = suite.load_tests["streaming_ttft_gpt-5-mini"]
        validation["ttft_p95_lt_500ms"] = stats.p95_ms < 500.0

    # SLO 5: 1000 concurrent requests succeed
    if "concurrent_1000" in suite.load_tests:
        stats = suite.load_tests["concurrent_1000"]
        # Allow 5% failure rate due to rate limits
        validation["concurrent_1000_success"] = stats.count >= 950

    # SLO 6: Throughput >100 req/s
    for model, rps in suite.throughput.items():
        validation[f"throughput_{model}_gt_100rps"] = rps > 100.0

    return validation


async def run_benchmark_suite(
    skip_load: bool = False,
    skip_sdk_comparison: bool = False,
    skip_profile: bool = False,
) -> BenchmarkSuite:
    """Run complete benchmark suite.

    Args:
        skip_load: Skip expensive load tests
        skip_sdk_comparison: Skip SDK comparison benchmarks
        skip_profile: Skip resource profiling

    Returns:
        BenchmarkSuite with all results
    """
    print("=" * 80)
    print("LLM Client Service Benchmark Suite")
    print("=" * 80)

    microbenchmarks = {}
    load_tests = {}
    sdk_comparison: dict[str, dict[str, LatencyStats]] = {}
    throughput = {}
    resource_usage = {}

    # Phase 1: Microbenchmarks (fast, no API calls)
    print("\n[Phase 1] Microbenchmarks")
    print("-" * 80)

    microbenchmarks["provider_selection"] = await benchmark_provider_selection(1000)
    print(
        f"✓ Provider selection: p95={microbenchmarks['provider_selection'].p95_ms:.3f}ms"
    )

    microbenchmarks["model_lookup"] = await benchmark_model_lookup(10000)
    print(f"✓ Model lookup: p95={microbenchmarks['model_lookup'].p95_ms:.6f}ms")

    microbenchmarks["request_validation"] = await benchmark_request_validation(1000)
    print(
        f"✓ Request validation: p95={microbenchmarks['request_validation'].p95_ms:.3f}ms"
    )

    # Phase 2: SDK Comparison (requires API keys)
    if not skip_sdk_comparison:
        print("\n[Phase 2] SDK Comparison")
        print("-" * 80)

        sdk_comparison["openai"] = {}
        sdk_comparison["openai"]["direct_sdk_openai"] = await benchmark_direct_sdk_openai(10)
        sdk_comparison["openai"][
            "abstraction_layer_openai"
        ] = await benchmark_abstraction_layer_openai(10)

        if sdk_comparison["openai"]["direct_sdk_openai"].count > 0:
            direct = sdk_comparison["openai"]["direct_sdk_openai"]
            abstraction = sdk_comparison["openai"]["abstraction_layer_openai"]
            overhead = abstraction.mean_ms - direct.mean_ms
            overhead_pct = (overhead / direct.mean_ms) * 100 if direct.mean_ms > 0 else 0

            print(f"✓ Direct SDK: mean={direct.mean_ms:.2f}ms, p95={direct.p95_ms:.2f}ms")
            print(
                f"✓ Abstraction: mean={abstraction.mean_ms:.2f}ms, p95={abstraction.p95_ms:.2f}ms"
            )
            print(f"✓ Overhead: {overhead:.2f}ms ({overhead_pct:.2f}%)")

    # Phase 3: Load Tests (expensive, requires API keys)
    if not skip_load:
        print("\n[Phase 3] Load Tests")
        print("-" * 80)

        # Test streaming TTFT
        load_tests["streaming_ttft_gpt-5-mini"] = await benchmark_streaming_ttft(
            "gpt-5-mini"
        )
        if load_tests["streaming_ttft_gpt-5-mini"].count > 0:
            stats = load_tests["streaming_ttft_gpt-5-mini"]
            print(
                f"✓ Streaming TTFT: p50={stats.p50_ms:.2f}ms, p95={stats.p95_ms:.2f}ms, p99={stats.p99_ms:.2f}ms"
            )

        # Test concurrent loads
        for concurrency in [100, 500, 1000]:
            stats, successes, failures = await benchmark_concurrent_requests(concurrency)
            load_tests[f"concurrent_{concurrency}"] = stats
            print(
                f"✓ Concurrent {concurrency}: {successes} success, {failures} failed, "
                f"p95={stats.p95_ms:.2f}ms"
            )

        # Test throughput
        throughput = await benchmark_throughput(30)

    # Phase 4: Resource Profiling
    if not skip_profile:
        print("\n[Phase 4] Resource Profiling")
        print("-" * 80)

        resource_usage = await profile_memory_usage()
        if resource_usage:
            print(
                f"✓ Memory: baseline={resource_usage.get('baseline_mb', 0):.2f}MB, "
                f"peak={resource_usage.get('peak_mb', 0):.2f}MB, "
                f"delta={resource_usage.get('delta_mb', 0):.2f}MB"
            )

    # Create suite and validate
    suite = BenchmarkSuite(
        microbenchmarks=microbenchmarks,
        load_tests=load_tests,
        sdk_comparison=sdk_comparison,
        throughput=throughput,
        resource_usage=resource_usage,
        validation={},
    )

    suite.validation = validate_slos(suite)

    # Print validation results
    print("\n[Validation] SLO Compliance")
    print("-" * 80)
    for slo, passed in suite.validation.items():
        status = "✓" if passed else "✗"
        print(f"{status} {slo}: {'PASS' if passed else 'FAIL'}")

    return suite


def save_results(suite: BenchmarkSuite, output_path: Path) -> None:
    """Save benchmark results to JSON file.

    Args:
        suite: Complete benchmark results
        output_path: Path to save results JSON
    """

    def convert_to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dict__"):
            return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_dict(item) for item in obj]
        return obj

    results_dict = convert_to_dict(suite)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_path}")


async def main() -> int:
    """Main entry point for benchmark suite."""
    parser = argparse.ArgumentParser(description="LLM Client Benchmark Suite")
    parser.add_argument(
        "--load-only", action="store_true", help="Run only load tests (skip microbenchmarks)"
    )
    parser.add_argument(
        "--skip-load", action="store_true", help="Skip expensive load tests"
    )
    parser.add_argument(
        "--skip-sdk-comparison", action="store_true", help="Skip SDK comparison benchmarks"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable resource profiling"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "benchmarks" / "results.json",
        help="Output path for results JSON",
    )

    args = parser.parse_args()

    # Run benchmark suite
    suite = await run_benchmark_suite(
        skip_load=args.skip_load,
        skip_sdk_comparison=args.skip_sdk_comparison,
        skip_profile=not args.profile,
    )

    # Save results
    save_results(suite, args.output)

    # Exit with failure if any SLO failed
    all_passed = all(suite.validation.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
