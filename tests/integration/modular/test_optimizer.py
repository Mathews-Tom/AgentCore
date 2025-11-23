"""
Integration tests for Module Performance Optimizer

Tests profiling, caching, parallel execution, and optimization metrics
to validate performance improvements and NFR compliance.
"""

import asyncio
from datetime import datetime, timezone
from typing import Callable

import pytest

from agentcore.modular.interfaces import ExecutionPlan, ExecutionResult, PlanStep
from agentcore.modular.models import ModuleType
from agentcore.modular.optimizer import (
    ModuleProfiler,
    ParallelExecutor,
    PerformanceOptimizer,
    PromptOptimizer,
    ResponseCache,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def profiler() -> ModuleProfiler:
    """Create module profiler instance."""
    return ModuleProfiler()


@pytest.fixture
def prompt_optimizer() -> PromptOptimizer:
    """Create prompt optimizer instance."""
    return PromptOptimizer()


@pytest.fixture
def response_cache() -> ResponseCache:
    """Create response cache instance."""
    return ResponseCache(default_ttl_seconds=60.0)


@pytest.fixture
def parallel_executor() -> ParallelExecutor:
    """Create parallel executor instance."""
    return ParallelExecutor(max_concurrent=5)


@pytest.fixture
def performance_optimizer() -> PerformanceOptimizer:
    """Create integrated performance optimizer."""
    return PerformanceOptimizer(
        cache_ttl_seconds=60.0,
        max_concurrent=5,
        enable_caching=True,
        enable_prompt_optimization=True,
    )


@pytest.fixture
def sample_plan() -> ExecutionPlan:
    """Create sample execution plan with dependencies."""
    steps = [
        PlanStep(
            step_id="step1",
            action="search",
            parameters={"query": "test"},
            dependencies=[],
        ),
        PlanStep(
            step_id="step2",
            action="analyze",
            parameters={"data": "result1"},
            dependencies=["step1"],
        ),
        PlanStep(
            step_id="step3",
            action="search",
            parameters={"query": "more"},
            dependencies=[],
        ),
        PlanStep(
            step_id="step4",
            action="synthesize",
            parameters={"data": ["result1", "result2", "result3"]},
            dependencies=["step1", "step2", "step3"],
        ),
    ]

    return ExecutionPlan(
        plan_id="test-plan-001",
        steps=steps,
        total_estimated_cost=0.05,
    )


# ============================================================================
# Module Profiler Tests
# ============================================================================


@pytest.mark.asyncio
async def test_profiler_basic_execution(profiler: ModuleProfiler) -> None:
    """Test basic profiling of async function execution."""

    async def sample_operation() -> str:
        await asyncio.sleep(0.01)  # 10ms operation
        return "result"

    result, metrics = await profiler.profile_execution(
        ModuleType.PLANNER, "test_operation", sample_operation
    )

    assert result == "result"
    assert metrics.module_type == ModuleType.PLANNER
    assert metrics.operation == "test_operation"
    assert metrics.duration_ms >= 10.0  # At least 10ms
    assert metrics.metadata["success"] is True


@pytest.mark.asyncio
async def test_profiler_multiple_calls(profiler: ModuleProfiler) -> None:
    """Test profiling accumulates statistics across multiple calls."""

    async def sample_operation() -> str:
        await asyncio.sleep(0.005)
        return "result"

    # Execute multiple times
    for i in range(5):
        await profiler.profile_execution(
            ModuleType.EXECUTOR, "repeated_op", sample_operation
        )

    # Get profile
    profile = profiler.get_profile(ModuleType.EXECUTOR, "repeated_op")

    assert profile is not None
    assert profile.total_calls == 5
    assert profile.min_duration_ms > 0
    assert profile.max_duration_ms >= profile.min_duration_ms
    assert len(profile.durations) == 5


@pytest.mark.asyncio
async def test_profiler_error_handling(profiler: ModuleProfiler) -> None:
    """Test profiling handles errors correctly."""

    async def failing_operation() -> None:
        await asyncio.sleep(0.005)
        raise ValueError("Intentional test error")

    with pytest.raises(ValueError, match="Intentional test error"):
        await profiler.profile_execution(
            ModuleType.VERIFIER, "failing_op", failing_operation
        )

    # Check metrics were still recorded
    assert len(profiler._metrics_history) == 1
    metrics = profiler._metrics_history[0]
    assert metrics.metadata["success"] is False
    assert "error" in metrics.metadata


@pytest.mark.asyncio
async def test_profiler_bottleneck_identification(profiler: ModuleProfiler) -> None:
    """Test bottleneck identification for slow operations."""

    # Fast operation
    async def fast_op() -> str:
        await asyncio.sleep(0.001)
        return "fast"

    # Slow operation (bottleneck)
    async def slow_op() -> str:
        await asyncio.sleep(0.6)  # 600ms - exceeds 500ms threshold
        return "slow"

    await profiler.profile_execution(ModuleType.PLANNER, "fast_operation", fast_op)
    await profiler.profile_execution(ModuleType.EXECUTOR, "slow_operation", slow_op)

    # Identify bottlenecks (>500ms threshold)
    bottlenecks = profiler.get_bottlenecks(threshold_ms=500.0)

    assert len(bottlenecks) == 1
    assert bottlenecks[0][0] == "executor:slow_operation"
    assert bottlenecks[0][1].get_avg_duration_ms() >= 600.0


def test_profiler_summary_report(profiler: ModuleProfiler) -> None:
    """Test summary report generation."""
    # Add some fake measurements
    profile = profiler._profiles["planner:analyze"]
    profile.name = "planner:analyze"
    profile.add_measurement(100.0)
    profile.add_measurement(150.0)
    profile.add_measurement(200.0)

    report = profiler.get_summary_report()

    assert report["total_operations"] == 3
    assert report["total_duration_ms"] == 450.0
    assert report["avg_duration_ms"] == 150.0
    assert len(report["profiles"]) == 1
    assert report["profiles"][0]["avg_duration_ms"] == 150.0


# ============================================================================
# Prompt Optimizer Tests
# ============================================================================


def test_prompt_optimizer_basic(prompt_optimizer: PromptOptimizer) -> None:
    """Test basic prompt optimization."""
    original = "Please make sure to analyze this in a clear and concise manner"

    optimized, tokens_saved = prompt_optimizer.optimize_prompt(original)

    assert len(optimized) < len(original)
    assert tokens_saved > 0
    assert "Please" not in optimized
    assert "make sure to" not in optimized


def test_prompt_optimizer_whitespace_removal(
    prompt_optimizer: PromptOptimizer,
) -> None:
    """Test whitespace compression."""
    original = "This  has   excessive     whitespace"

    optimized, tokens_saved = prompt_optimizer.optimize_prompt(original)

    assert "  " not in optimized  # No double spaces
    assert optimized == "This has excessive whitespace"


def test_prompt_optimizer_phrase_replacement(
    prompt_optimizer: PromptOptimizer,
) -> None:
    """Test redundant phrase replacement."""
    original = "In order to complete this task, make sure that you proceed step by step"

    optimized, tokens_saved = prompt_optimizer.optimize_prompt(original)

    assert "in order to" not in optimized.lower()
    assert "make sure that" not in optimized.lower()
    assert "step by step" not in optimized.lower()
    assert "to complete" in optimized.lower()
    assert "sequentially" in optimized.lower()


def test_prompt_optimizer_total_savings(prompt_optimizer: PromptOptimizer) -> None:
    """Test total token savings tracking."""
    prompts = [
        "Please analyze this in order to understand the problem",
        "Make sure to provide as much detail as possible",
        "Step by step, can you please explain this",
    ]

    for prompt in prompts:
        prompt_optimizer.optimize_prompt(prompt)

    total_saved = prompt_optimizer.get_total_tokens_saved()
    assert total_saved > 0


# ============================================================================
# Response Cache Tests
# ============================================================================


@pytest.mark.asyncio
async def test_cache_basic_hit(response_cache: ResponseCache) -> None:
    """Test basic cache hit scenario."""
    call_count = 0

    async def expensive_operation() -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return 5 * 2

    # First call - cache miss
    result1, hit1 = await response_cache.get_or_compute(
        expensive_operation, cache_key_args=(5,), ttl_seconds=10.0
    )
    assert result1 == 10
    assert hit1 is False
    assert call_count == 1

    # Second call - cache hit
    result2, hit2 = await response_cache.get_or_compute(
        expensive_operation, cache_key_args=(5,), ttl_seconds=10.0
    )
    assert result2 == 10
    assert hit2 is True
    assert call_count == 1  # Not called again


@pytest.mark.asyncio
async def test_cache_different_args(response_cache: ResponseCache) -> None:
    """Test cache correctly differentiates arguments."""
    results_map = {
        (1, 2): 3,
        (2, 3): 5,
    }

    async def operation_factory(x: int, y: int) -> Callable[[], int]:
        async def operation() -> int:
            return results_map[(x, y)]

        return operation

    # Different arguments should be cache misses
    result1, hit1 = await response_cache.get_or_compute(
        await operation_factory(1, 2), cache_key_args=(1, 2)
    )
    result2, hit2 = await response_cache.get_or_compute(
        await operation_factory(2, 3), cache_key_args=(2, 3)
    )
    result3, hit3 = await response_cache.get_or_compute(
        await operation_factory(1, 2), cache_key_args=(1, 2)
    )

    assert result1 == 3
    assert result2 == 5
    assert result3 == 3
    assert hit1 is False
    assert hit2 is False
    assert hit3 is True  # Same as first call


@pytest.mark.asyncio
async def test_cache_expiration(response_cache: ResponseCache) -> None:
    """Test cache entry expiration."""

    async def operation() -> int:
        return 5 * 2

    # Short TTL
    result1, hit1 = await response_cache.get_or_compute(
        operation, cache_key_args=(5,), ttl_seconds=0.05  # 50ms TTL
    )
    assert hit1 is False

    # Immediate read - cache hit
    result2, hit2 = await response_cache.get_or_compute(
        operation, cache_key_args=(5,), ttl_seconds=0.05
    )
    assert hit2 is True

    # Wait for expiration
    await asyncio.sleep(0.1)  # 100ms - exceeds TTL

    # Should be cache miss after expiration
    result3, hit3 = await response_cache.get_or_compute(
        operation, cache_key_args=(5,), ttl_seconds=0.05
    )
    assert hit3 is False


def test_cache_invalidation(response_cache: ResponseCache) -> None:
    """Test manual cache invalidation."""

    # Add entry to cache manually
    cache_key = response_cache._generate_cache_key(10, 20)
    from agentcore.modular.optimizer import CacheEntry

    entry = CacheEntry(
        key=cache_key,
        value=30,
        created_at=datetime.now(timezone.utc),
        ttl_seconds=60.0,
    )
    response_cache._cache[cache_key] = entry

    # Invalidate
    invalidated = response_cache.invalidate(10, 20)
    assert invalidated is True
    assert cache_key not in response_cache._cache


def test_cache_stats(response_cache: ResponseCache) -> None:
    """Test cache statistics."""
    # Simulate hits and misses
    response_cache._hits = 10
    response_cache._misses = 5

    stats = response_cache.get_stats()

    assert stats["hits"] == 10
    assert stats["misses"] == 5
    assert stats["hit_rate"] == 10 / 15  # 0.666...


# ============================================================================
# Parallel Executor Tests
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_executor_basic(
    parallel_executor: ParallelExecutor, sample_plan: ExecutionPlan
) -> None:
    """Test basic parallel execution of independent steps."""
    execution_order: list[str] = []

    async def mock_executor(step: PlanStep) -> ExecutionResult:
        execution_order.append(step.step_id)
        await asyncio.sleep(0.01)
        return ExecutionResult(
            step_id=step.step_id,
            success=True,
            result=f"result_{step.step_id}",
            error=None,
            execution_time=0.01,
        )

    results = await parallel_executor.execute_steps_parallel(
        sample_plan.steps, mock_executor, respect_dependencies=True
    )

    # All steps should complete
    assert len(results) == 4
    assert all(r.success for r in results)

    # Step 1 and 3 should execute before step 2 (dependencies)
    assert execution_order.index("step1") < execution_order.index("step2")
    assert execution_order.index("step1") < execution_order.index("step4")
    assert execution_order.index("step2") < execution_order.index("step4")
    assert execution_order.index("step3") < execution_order.index("step4")


@pytest.mark.asyncio
async def test_parallel_executor_no_dependencies(
    parallel_executor: ParallelExecutor,
) -> None:
    """Test parallel execution without respecting dependencies."""
    steps = [
        PlanStep(step_id=f"step{i}", action="test", parameters={}, dependencies=[])
        for i in range(5)
    ]

    concurrent_count = 0
    max_concurrent = 0

    async def mock_executor(step: PlanStep) -> ExecutionResult:
        nonlocal concurrent_count, max_concurrent
        concurrent_count += 1
        max_concurrent = max(max_concurrent, concurrent_count)
        await asyncio.sleep(0.05)
        concurrent_count -= 1
        return ExecutionResult(
            step_id=step.step_id,
            success=True,
            result="done",
            error=None,
            execution_time=0.05,
        )

    results = await parallel_executor.execute_steps_parallel(
        steps, mock_executor, respect_dependencies=False
    )

    assert len(results) == 5
    # Should have executed multiple steps concurrently (up to semaphore limit)
    assert max_concurrent >= 2


@pytest.mark.asyncio
async def test_parallel_executor_error_handling(
    parallel_executor: ParallelExecutor,
) -> None:
    """Test parallel executor handles step errors."""
    steps = [
        PlanStep(step_id="step1", action="test", parameters={}, dependencies=[]),
        PlanStep(step_id="step2", action="fail", parameters={}, dependencies=[]),
        PlanStep(step_id="step3", action="test", parameters={}, dependencies=[]),
    ]

    async def mock_executor(step: PlanStep) -> ExecutionResult:
        if step.action == "fail":
            raise ValueError("Step failed intentionally")

        return ExecutionResult(
            step_id=step.step_id,
            success=True,
            result="done",
            error=None,
            execution_time=0.01,
        )

    results = await parallel_executor.execute_steps_parallel(
        steps, mock_executor, respect_dependencies=False
    )

    assert len(results) == 3
    # Check step2 failed
    step2_result = next(r for r in results if r.step_id == "step2")
    assert step2_result.success is False
    assert "Step failed intentionally" in step2_result.error


# ============================================================================
# Integrated Performance Optimizer Tests
# ============================================================================


@pytest.mark.asyncio
async def test_optimizer_module_call_with_cache(
    performance_optimizer: PerformanceOptimizer,
) -> None:
    """Test optimized module call with caching."""
    call_count = 0

    async def module_operation() -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return 5 * 2

    # First call - cache miss (but we don't profile cached calls)
    result1, metrics1, hit1 = await performance_optimizer.optimize_module_call(
        ModuleType.PLANNER,
        "test_operation",
        module_operation,
        use_cache=True,
    )

    assert result1 == 10
    assert hit1 is False
    assert call_count == 1

    # Second call - should hit cache
    result2, metrics2, hit2 = await performance_optimizer.optimize_module_call(
        ModuleType.PLANNER,
        "test_operation",
        module_operation,
        use_cache=True,
    )

    assert result2 == 10
    assert hit2 is True
    assert call_count == 1  # Not called again


@pytest.mark.asyncio
async def test_optimizer_prompt_optimization_integration(
    performance_optimizer: PerformanceOptimizer,
) -> None:
    """Test prompt optimization integration."""
    original_prompt = (
        "Please make sure to analyze this step by step in a clear manner"
    )

    optimized = performance_optimizer.optimize_prompt_text(original_prompt)

    assert len(optimized) < len(original_prompt)
    assert "Please" not in optimized


@pytest.mark.asyncio
async def test_optimizer_parallel_execution_integration(
    performance_optimizer: PerformanceOptimizer, sample_plan: ExecutionPlan
) -> None:
    """Test parallel execution through optimizer."""

    async def mock_executor(step: PlanStep) -> ExecutionResult:
        await asyncio.sleep(0.01)
        return ExecutionResult(
            step_id=step.step_id,
            success=True,
            result=f"result_{step.step_id}",
            error=None,
            execution_time=0.01,
        )

    results = await performance_optimizer.execute_plan_parallel(
        sample_plan, mock_executor, respect_dependencies=True
    )

    assert len(results) == 4
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_optimizer_metrics_collection(
    performance_optimizer: PerformanceOptimizer,
) -> None:
    """Test comprehensive metrics collection."""

    async def sample_op(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    # Execute some operations
    for i in range(5):
        await performance_optimizer.optimize_module_call(
            ModuleType.PLANNER, "test_op", sample_op, i, use_cache=True
        )

    # Get metrics
    metrics = performance_optimizer.get_optimization_metrics()

    assert metrics.cache_hits >= 0
    assert metrics.cache_misses >= 0
    assert 0.0 <= metrics.cache_hit_rate <= 1.0


@pytest.mark.asyncio
async def test_optimizer_performance_report(
    performance_optimizer: PerformanceOptimizer,
) -> None:
    """Test performance report generation."""

    async def sample_op(x: int) -> int:
        await asyncio.sleep(0.01)
        return x

    # Execute operations
    await performance_optimizer.optimize_module_call(
        ModuleType.EXECUTOR, "operation1", sample_op, 1
    )
    await performance_optimizer.optimize_module_call(
        ModuleType.VERIFIER, "operation2", sample_op, 2
    )

    report = performance_optimizer.get_performance_report()

    assert "optimization_metrics" in report
    assert "profiling_summary" in report
    assert "bottlenecks" in report
    assert "cache_stats" in report
    assert "nfr_compliance" in report

    # Check NFR compliance section
    nfr = report["nfr_compliance"]
    assert nfr["module_transition_target_ms"] == 500.0
    assert isinstance(nfr["meets_target"], bool)


@pytest.mark.asyncio
async def test_optimizer_nfr_compliance_validation(
    performance_optimizer: PerformanceOptimizer,
) -> None:
    """Test NFR compliance target (<500ms module transition)."""

    async def fast_transition() -> str:
        await asyncio.sleep(0.1)  # 100ms - well under target
        return "complete"

    # Execute multiple transitions
    for _ in range(10):
        await performance_optimizer.optimize_module_call(
            ModuleType.PLANNER, "module_transition", fast_transition
        )

    report = performance_optimizer.get_performance_report()

    # Validate against NFR target
    nfr = report["nfr_compliance"]
    assert nfr["p95_module_transition_ms"] < 500.0
    assert nfr["meets_target"] is True
