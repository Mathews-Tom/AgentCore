"""
Module Performance Optimizer for Modular Agent Core

Provides profiling, caching, parallel execution, and LLM prompt optimization
to reduce module response times and meet performance targets (<500ms module
transition time, <2x baseline latency).

Performance Targets (NFR-1.3):
- Module transition time: <500ms (excluding tool execution)
- Overall latency: <2x baseline
- Throughput: 100+ concurrent executions per instance
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field

from agentcore.modular.interfaces import (
    ExecutionPlan,
    ExecutionResult,
    PlanStep,
)
from agentcore.modular.models import ModuleType

logger = structlog.get_logger()


# ============================================================================
# Performance Metrics Models
# ============================================================================


class ProfileMetrics(BaseModel):
    """Profiling metrics for module execution."""

    module_type: ModuleType = Field(..., description="Type of module profiled")
    operation: str = Field(..., description="Operation being profiled")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="When metric was recorded",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional profiling metadata"
    )


class OptimizationMetrics(BaseModel):
    """Metrics for optimization effectiveness."""

    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    cache_hit_rate: float = Field(default=0.0, description="Cache hit rate (0.0-1.0)")
    parallel_executions: int = Field(
        default=0, description="Number of parallel executions"
    )
    sequential_executions: int = Field(
        default=0, description="Number of sequential executions"
    )
    prompt_tokens_saved: int = Field(
        default=0, description="Tokens saved through optimization"
    )
    avg_module_transition_ms: float = Field(
        default=0.0, description="Average module transition time"
    )
    p95_module_transition_ms: float = Field(
        default=0.0, description="95th percentile module transition time"
    )


@dataclass
class PerformanceProfile:
    """Performance profile for a specific module or operation."""

    name: str
    total_calls: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    durations: list[float] = field(default_factory=list)

    def add_measurement(self, duration_ms: float) -> None:
        """Add a duration measurement."""
        self.total_calls += 1
        self.total_duration_ms += duration_ms
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.durations.append(duration_ms)

    def get_avg_duration_ms(self) -> float:
        """Get average duration."""
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls

    def get_p50_duration_ms(self) -> float:
        """Get 50th percentile (median) duration."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        return sorted_durations[len(sorted_durations) // 2]

    def get_p95_duration_ms(self) -> float:
        """Get 95th percentile duration."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.95)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]

    def get_p99_duration_ms(self) -> float:
        """Get 99th percentile duration."""
        if not self.durations:
            return 0.0
        sorted_durations = sorted(self.durations)
        idx = int(len(sorted_durations) * 0.99)
        return sorted_durations[min(idx, len(sorted_durations) - 1)]


# ============================================================================
# Module Performance Profiler
# ============================================================================


class ModuleProfiler:
    """
    Profiles module execution to identify bottlenecks.

    Tracks execution times per module type and operation, providing
    detailed statistics (avg, min, max, p50, p95, p99) for optimization.
    """

    def __init__(self) -> None:
        """Initialize the profiler."""
        self._profiles: dict[str, PerformanceProfile] = defaultdict(
            lambda: PerformanceProfile(name="unknown")
        )
        self._metrics_history: list[ProfileMetrics] = []
        logger.info("ModuleProfiler initialized")

    def profile_key(self, module_type: ModuleType, operation: str) -> str:
        """Generate unique key for profiling."""
        return f"{module_type.value}:{operation}"

    async def profile_execution(
        self,
        module_type: ModuleType,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ProfileMetrics]:
        """
        Profile execution of an async function.

        Args:
            module_type: Type of module being profiled
            operation: Operation name
            func: Async function to profile
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Tuple of (function result, profile metrics)
        """
        start_time = time.perf_counter()

        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Record metrics
            profile_key = self.profile_key(module_type, operation)
            profile = self._profiles[profile_key]
            profile.name = profile_key
            profile.add_measurement(duration_ms)

            metrics = ProfileMetrics(
                module_type=module_type,
                operation=operation,
                duration_ms=duration_ms,
                metadata={
                    "success": True,
                    "total_calls": profile.total_calls,
                    "avg_duration_ms": profile.get_avg_duration_ms(),
                },
            )

            self._metrics_history.append(metrics)

            logger.debug(
                "operation_profiled",
                module_type=module_type.value,
                operation=operation,
                duration_ms=duration_ms,
                avg_duration_ms=profile.get_avg_duration_ms(),
            )

            return result, metrics

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000

            metrics = ProfileMetrics(
                module_type=module_type,
                operation=operation,
                duration_ms=duration_ms,
                metadata={"success": False, "error": str(e)},
            )

            self._metrics_history.append(metrics)

            logger.error(
                "operation_profiled_error",
                module_type=module_type.value,
                operation=operation,
                duration_ms=duration_ms,
                error=str(e),
            )

            raise

    def get_profile(
        self, module_type: ModuleType, operation: str
    ) -> PerformanceProfile | None:
        """Get performance profile for specific module/operation."""
        profile_key = self.profile_key(module_type, operation)
        return self._profiles.get(profile_key)

    def get_all_profiles(self) -> dict[str, PerformanceProfile]:
        """Get all performance profiles."""
        return dict(self._profiles)

    def get_bottlenecks(
        self, threshold_ms: float = 500.0
    ) -> list[tuple[str, PerformanceProfile]]:
        """
        Identify bottlenecks exceeding threshold.

        Args:
            threshold_ms: Threshold for bottleneck identification (default: 500ms)

        Returns:
            List of (profile_key, profile) tuples sorted by avg duration
        """
        bottlenecks = [
            (key, profile)
            for key, profile in self._profiles.items()
            if profile.get_avg_duration_ms() > threshold_ms
        ]

        # Sort by average duration (descending)
        bottlenecks.sort(key=lambda x: x[1].get_avg_duration_ms(), reverse=True)

        logger.info(
            "bottlenecks_identified",
            count=len(bottlenecks),
            threshold_ms=threshold_ms,
        )

        return bottlenecks

    def get_summary_report(self) -> dict[str, Any]:
        """Generate summary profiling report."""
        total_operations = sum(p.total_calls for p in self._profiles.values())
        total_duration_ms = sum(p.total_duration_ms for p in self._profiles.values())

        profiles_summary = [
            {
                "name": key,
                "total_calls": profile.total_calls,
                "avg_duration_ms": profile.get_avg_duration_ms(),
                "p50_duration_ms": profile.get_p50_duration_ms(),
                "p95_duration_ms": profile.get_p95_duration_ms(),
                "p99_duration_ms": profile.get_p99_duration_ms(),
                "min_duration_ms": profile.min_duration_ms,
                "max_duration_ms": profile.max_duration_ms,
            }
            for key, profile in self._profiles.items()
        ]

        # Sort by total time spent (impact)
        profiles_summary.sort(
            key=lambda x: x["total_calls"] * x["avg_duration_ms"], reverse=True
        )

        return {
            "total_operations": total_operations,
            "total_duration_ms": total_duration_ms,
            "avg_duration_ms": total_duration_ms / total_operations
            if total_operations > 0
            else 0.0,
            "profiles": profiles_summary,
            "metrics_collected": len(self._metrics_history),
        }


# ============================================================================
# LLM Prompt Optimizer
# ============================================================================


class PromptOptimizer:
    """
    Optimizes LLM prompts to reduce token usage while maintaining quality.

    Strategies:
    - Remove redundant instructions
    - Compress examples
    - Use more concise phrasing
    - Cache common prompt fragments
    """

    def __init__(self) -> None:
        """Initialize the prompt optimizer."""
        self._original_token_counts: dict[str, int] = {}
        self._optimized_token_counts: dict[str, int] = {}
        logger.info("PromptOptimizer initialized")

    def optimize_prompt(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> tuple[str, int]:
        """
        Optimize an LLM prompt to reduce token usage.

        Args:
            prompt: Original prompt text
            context: Optional context for optimization decisions

        Returns:
            Tuple of (optimized prompt, estimated tokens saved)
        """
        if not prompt:
            return prompt, 0

        original_length = len(prompt)
        optimized = prompt

        # Strategy 1: Remove excessive whitespace
        optimized = " ".join(optimized.split())

        # Strategy 2: Remove redundant phrases (case-insensitive)
        import re

        redundant_phrases = [
            r"\bPlease\s+",
            r"\bI would like you to\s+",
            r"\bCan you please\s+",
            r"\bYou should\s+",
            r"\bMake sure to\s+",
            r"\bBe sure to\s+",
            r"\bRemember to\s+",
        ]
        for pattern in redundant_phrases:
            optimized = re.sub(pattern, "", optimized, flags=re.IGNORECASE)

        # Strategy 3: Compress common patterns (case-insensitive)
        replacements = {
            r"\bas much detail as possible\b": "detailed",
            r"\bin a clear and concise manner\b": "clearly",
            r"\bstep by step\b": "sequentially",
            r"\bone after another\b": "sequentially",
            r"\bmake sure that\b": "ensure",
            r"\bin order to\b": "to",
            r"\bdue to the fact that\b": "because",
            r"\bat this point in time\b": "now",
        }
        for pattern, replacement in replacements.items():
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)

        # Estimate tokens saved (rough approximation: 1 token ~= 4 characters)
        chars_saved = original_length - len(optimized)
        tokens_saved = chars_saved // 4

        # Track savings
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        self._original_token_counts[prompt_hash] = original_length // 4
        self._optimized_token_counts[prompt_hash] = len(optimized) // 4

        logger.debug(
            "prompt_optimized",
            original_length=original_length,
            optimized_length=len(optimized),
            tokens_saved=tokens_saved,
            reduction_pct=round((chars_saved / original_length) * 100, 1)
            if original_length > 0
            else 0,
        )

        return optimized, tokens_saved

    def get_total_tokens_saved(self) -> int:
        """Calculate total tokens saved through optimization."""
        total_original = sum(self._original_token_counts.values())
        total_optimized = sum(self._optimized_token_counts.values())
        return total_original - total_optimized


# ============================================================================
# Response Cache
# ============================================================================


@dataclass
class CacheEntry:
    """Cache entry with TTL support."""

    key: str
    value: Any
    created_at: datetime
    ttl_seconds: float
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds


class ResponseCache:
    """
    Caches module responses to avoid redundant LLM calls.

    Uses content-based hashing to identify identical queries and
    returns cached results for repeated requests within TTL window.
    """

    def __init__(self, default_ttl_seconds: float = 3600.0) -> None:
        """
        Initialize the response cache.

        Args:
            default_ttl_seconds: Default TTL for cache entries (default: 1 hour)
        """
        self._cache: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl_seconds
        self._hits = 0
        self._misses = 0
        logger.info(
            "ResponseCache initialized", default_ttl_seconds=default_ttl_seconds
        )

    def _generate_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""
        # Serialize args and kwargs to string
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)

        # Hash for consistent key
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get_or_compute(
        self,
        func: Callable[[], Any],
        cache_key_args: tuple[Any, ...] = (),
        cache_key_kwargs: dict[str, Any] | None = None,
        ttl_seconds: float | None = None,
    ) -> tuple[Any, bool]:
        """
        Get cached result or compute and cache.

        Args:
            func: Async function to call if cache miss (no args)
            cache_key_args: Tuple of args to use for cache key generation
            cache_key_kwargs: Dict of kwargs to use for cache key generation
            ttl_seconds: TTL for this entry (None uses default)

        Returns:
            Tuple of (result, cache_hit: bool)
        """
        if cache_key_kwargs is None:
            cache_key_kwargs = {}

        cache_key = self._generate_cache_key(*cache_key_args, **cache_key_kwargs)

        # Check cache
        entry = self._cache.get(cache_key)

        if entry and not entry.is_expired():
            # Cache hit
            entry.hits += 1
            self._hits += 1

            logger.debug(
                "cache_hit",
                cache_key=cache_key[:16],
                hits=entry.hits,
                age_seconds=round(
                    (datetime.now(timezone.utc) - entry.created_at).total_seconds(), 2
                ),
            )

            return entry.value, True

        # Cache miss - compute value
        self._misses += 1
        result = await func()

        # Store in cache
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        entry = CacheEntry(
            key=cache_key,
            value=result,
            created_at=datetime.now(timezone.utc),
            ttl_seconds=ttl,
        )
        self._cache[cache_key] = entry

        logger.debug(
            "cache_miss_stored",
            cache_key=cache_key[:16],
            ttl_seconds=ttl,
            cache_size=len(self._cache),
        )

        return result, False

    def invalidate(self, *args: Any, **kwargs: Any) -> bool:
        """
        Invalidate cache entry.

        Args:
            *args: Arguments to identify cache entry
            **kwargs: Keyword arguments to identify cache entry

        Returns:
            True if entry was found and removed
        """
        cache_key = self._generate_cache_key(*args, **kwargs)
        if cache_key in self._cache:
            del self._cache[cache_key]
            logger.debug("cache_invalidated", cache_key=cache_key[:16])
            return True
        return False

    def clear_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.info("cache_expired_cleared", count=len(expired_keys))

        return len(expired_keys)

    def clear_all(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info("cache_cleared", entries_removed=count)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_entries": len(self._cache),
            "expired_entries": sum(
                1 for entry in self._cache.values() if entry.is_expired()
            ),
        }


# ============================================================================
# Parallel Execution Coordinator
# ============================================================================


class ParallelExecutor:
    """
    Coordinates parallel execution of independent plan steps.

    Analyzes step dependencies and executes steps concurrently when
    dependencies allow, reducing overall execution time.
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        """
        Initialize the parallel executor.

        Args:
            max_concurrent: Maximum concurrent executions (default: 10)
        """
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(
            "ParallelExecutor initialized", max_concurrent=max_concurrent
        )

    async def execute_steps_parallel(
        self,
        steps: list[PlanStep],
        executor_func: Callable[[PlanStep], Any],
        respect_dependencies: bool = True,
    ) -> list[ExecutionResult]:
        """
        Execute plan steps in parallel where dependencies allow.

        Args:
            steps: List of plan steps to execute
            executor_func: Async function to execute each step
            respect_dependencies: Whether to respect step dependencies

        Returns:
            List of execution results in original step order
        """
        if not steps:
            return []

        # Build dependency graph
        completed_steps: set[str] = set()
        results: dict[str, ExecutionResult] = {}
        pending_steps = list(steps)

        logger.info(
            "parallel_execution_started",
            total_steps=len(steps),
            max_concurrent=self._max_concurrent,
            respect_dependencies=respect_dependencies,
        )

        start_time = time.perf_counter()

        while pending_steps:
            # Find executable steps (no pending dependencies)
            executable = []

            for step in pending_steps:
                if not respect_dependencies:
                    executable.append(step)
                else:
                    deps_satisfied = all(
                        dep_id in completed_steps for dep_id in step.dependencies
                    )
                    if deps_satisfied:
                        executable.append(step)

            if not executable:
                # Circular dependency or all steps waiting
                logger.error(
                    "parallel_execution_deadlock",
                    pending_steps=len(pending_steps),
                    completed_steps=len(completed_steps),
                )
                break

            # Execute batch in parallel
            batch_tasks = [
                self._execute_with_semaphore(step, executor_func)
                for step in executable
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Process results
            for step, result in zip(executable, batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "parallel_step_error",
                        step_id=step.step_id,
                        error=str(result),
                    )
                    # Create error result
                    results[step.step_id] = ExecutionResult(
                        step_id=step.step_id,
                        success=False,
                        result=None,
                        error=str(result),
                        execution_time=0.0,
                    )
                else:
                    results[step.step_id] = result

                completed_steps.add(step.step_id)
                pending_steps.remove(step)

            logger.debug(
                "parallel_batch_complete",
                batch_size=len(executable),
                pending=len(pending_steps),
                completed=len(completed_steps),
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "parallel_execution_complete",
            total_steps=len(steps),
            completed=len(completed_steps),
            duration_ms=duration_ms,
        )

        # Return results in original order
        return [results[step.step_id] for step in steps if step.step_id in results]

    async def _execute_with_semaphore(
        self, step: PlanStep, executor_func: Callable[[PlanStep], Any]
    ) -> ExecutionResult:
        """Execute step with semaphore for concurrency control."""
        async with self._semaphore:
            return await executor_func(step)


# ============================================================================
# Integrated Performance Optimizer
# ============================================================================


class PerformanceOptimizer:
    """
    Integrated performance optimizer combining all optimization strategies.

    Provides:
    - Module profiling
    - Prompt optimization
    - Response caching
    - Parallel execution
    - Performance metrics collection
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 3600.0,
        max_concurrent: int = 10,
        enable_caching: bool = True,
        enable_prompt_optimization: bool = True,
    ) -> None:
        """
        Initialize the performance optimizer.

        Args:
            cache_ttl_seconds: Default cache TTL (default: 1 hour)
            max_concurrent: Max concurrent executions (default: 10)
            enable_caching: Enable response caching (default: True)
            enable_prompt_optimization: Enable prompt optimization (default: True)
        """
        self.profiler = ModuleProfiler()
        self.prompt_optimizer = PromptOptimizer()
        self.cache = ResponseCache(default_ttl_seconds=cache_ttl_seconds)
        self.parallel_executor = ParallelExecutor(max_concurrent=max_concurrent)

        self._enable_caching = enable_caching
        self._enable_prompt_optimization = enable_prompt_optimization

        logger.info(
            "PerformanceOptimizer initialized",
            caching_enabled=enable_caching,
            prompt_optimization_enabled=enable_prompt_optimization,
            cache_ttl_seconds=cache_ttl_seconds,
            max_concurrent=max_concurrent,
        )

    async def optimize_module_call(
        self,
        module_type: ModuleType,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        use_cache: bool = True,
        cache_ttl: float | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ProfileMetrics, bool]:
        """
        Optimize a module call with profiling and optional caching.

        Args:
            module_type: Type of module being called
            operation: Operation name
            func: Async function to call
            *args: Positional arguments
            use_cache: Whether to use caching (default: True)
            cache_ttl: Cache TTL override
            **kwargs: Keyword arguments

        Returns:
            Tuple of (result, profile metrics, cache hit)
        """
        cache_hit = False

        # Apply caching if enabled
        if self._enable_caching and use_cache:
            # Wrap function execution with profiling for cache misses
            async def _execute_with_profiling() -> Any:
                result, _ = await self.profiler.profile_execution(
                    module_type, operation, func, *args, **kwargs
                )
                return result

            result, cache_hit = await self.cache.get_or_compute(
                _execute_with_profiling,
                cache_key_args=args,
                cache_key_kwargs=kwargs,
                ttl_seconds=cache_ttl,
            )

            # Get the most recent profile metrics if cache miss, or create synthetic for cache hit
            if cache_hit:
                metrics = ProfileMetrics(
                    module_type=module_type,
                    operation=operation,
                    duration_ms=0.0,
                    metadata={"cache_hit": True},
                )
            else:
                # Get last profiled metrics
                profile = self.profiler.get_profile(module_type, operation)
                if profile and self.profiler._metrics_history:
                    metrics = self.profiler._metrics_history[-1]
                else:
                    metrics = ProfileMetrics(
                        module_type=module_type,
                        operation=operation,
                        duration_ms=0.0,
                        metadata={"no_profile": True},
                    )
        else:
            # No caching, just profile
            result, metrics = await self.profiler.profile_execution(
                module_type, operation, func, *args, **kwargs
            )

        return result, metrics, cache_hit

    def optimize_prompt_text(
        self, prompt: str, context: dict[str, Any] | None = None
    ) -> str:
        """
        Optimize prompt text if optimization enabled.

        Args:
            prompt: Original prompt
            context: Optional context

        Returns:
            Optimized prompt (or original if optimization disabled)
        """
        if not self._enable_prompt_optimization:
            return prompt

        optimized, tokens_saved = self.prompt_optimizer.optimize_prompt(prompt, context)
        return optimized

    async def execute_plan_parallel(
        self,
        plan: ExecutionPlan,
        executor_func: Callable[[PlanStep], Any],
        respect_dependencies: bool = True,
    ) -> list[ExecutionResult]:
        """
        Execute plan steps in parallel.

        Args:
            plan: Execution plan
            executor_func: Function to execute each step
            respect_dependencies: Respect step dependencies

        Returns:
            List of execution results
        """
        return await self.parallel_executor.execute_steps_parallel(
            plan.steps, executor_func, respect_dependencies
        )

    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get comprehensive optimization metrics."""
        cache_stats = self.cache.get_stats()
        profiling_report = self.profiler.get_summary_report()

        # Calculate module transition metrics
        transition_profiles = [
            profile
            for key, profile in self.profiler.get_all_profiles().items()
            if "transition" in key.lower()
        ]

        avg_transition_ms = 0.0
        p95_transition_ms = 0.0
        if transition_profiles:
            all_durations = []
            for profile in transition_profiles:
                all_durations.extend(profile.durations)

            if all_durations:
                avg_transition_ms = sum(all_durations) / len(all_durations)
                sorted_durations = sorted(all_durations)
                p95_idx = int(len(sorted_durations) * 0.95)
                p95_transition_ms = sorted_durations[
                    min(p95_idx, len(sorted_durations) - 1)
                ]

        return OptimizationMetrics(
            cache_hits=cache_stats["hits"],
            cache_misses=cache_stats["misses"],
            cache_hit_rate=cache_stats["hit_rate"],
            parallel_executions=0,  # TODO: Track parallel vs sequential
            sequential_executions=profiling_report["total_operations"],
            prompt_tokens_saved=self.prompt_optimizer.get_total_tokens_saved(),
            avg_module_transition_ms=avg_transition_ms,
            p95_module_transition_ms=p95_transition_ms,
        )

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        optimization_metrics = self.get_optimization_metrics()
        profiling_report = self.profiler.get_summary_report()
        bottlenecks = self.profiler.get_bottlenecks(threshold_ms=500.0)

        return {
            "optimization_metrics": optimization_metrics.model_dump(),
            "profiling_summary": profiling_report,
            "bottlenecks": [
                {
                    "name": name,
                    "avg_duration_ms": profile.get_avg_duration_ms(),
                    "p95_duration_ms": profile.get_p95_duration_ms(),
                    "total_calls": profile.total_calls,
                }
                for name, profile in bottlenecks
            ],
            "cache_stats": self.cache.get_stats(),
            "nfr_compliance": {
                "module_transition_target_ms": 500.0,
                "avg_module_transition_ms": optimization_metrics.avg_module_transition_ms,
                "p95_module_transition_ms": optimization_metrics.p95_module_transition_ms,
                "meets_target": optimization_metrics.p95_module_transition_ms < 500.0,
            },
        }
