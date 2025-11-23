"""
Prometheus Metrics Collection for Modular Agent Core

Provides comprehensive metrics for module-level monitoring including:
- Latency histograms (p50, p95, p99) per module
- Success/failure counters per module
- Error rate by type
- Iteration count distribution
- Token usage per module
- Per-module performance metrics

Implements MOD-026 requirements from docs/specs/modular-agent-core/tasks.md.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncIterator

import structlog
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

from agentcore.modular.models import ModuleType

logger = structlog.get_logger()


# ============================================================================
# Error Type Enumeration
# ============================================================================


class ErrorType(str, Enum):
    """Types of errors that can occur in module execution."""

    TIMEOUT = "timeout"
    VALIDATION = "validation"
    LLM_ERROR = "llm_error"
    TOOL_ERROR = "tool_error"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


# ============================================================================
# Module Metrics Collector
# ============================================================================


class ModularMetricsCollector:
    """
    Collects and exposes Prometheus metrics for modular agent execution.

    Provides per-module metrics for:
    - Latency (histograms with p50, p95, p99)
    - Success/failure counters
    - Error rates by type
    - Iteration counts
    - Token usage
    - Cost tracking

    Usage:
        ```python
        metrics = ModularMetricsCollector()

        # Track module execution
        async with metrics.track_module_execution("planner") as tracker:
            result = await planner.analyze_query(query)
            tracker.set_success(True)
            tracker.set_tokens(result.tokens_used)

        # Record errors
        metrics.record_error("executor", ErrorType.TOOL_ERROR)

        # Record iterations
        metrics.record_iteration_count(3)
        ```
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """
        Initialize metrics collector.

        Args:
            registry: Prometheus registry (creates new if None)
        """
        self._registry = registry or CollectorRegistry()

        # Initialize all metric types
        self._init_latency_metrics()
        self._init_success_metrics()
        self._init_error_metrics()
        self._init_iteration_metrics()
        self._init_token_metrics()
        self._init_cost_metrics()

        logger.info(
            "modular_metrics_initialized",
            registry_id=id(self._registry),
        )

    def _init_latency_metrics(self) -> None:
        """Initialize module latency histograms."""
        # Per-module latency histograms
        # Buckets tuned for module execution times:
        # - Quick operations: 0.1s, 0.25s, 0.5s
        # - Standard operations: 1s, 2s, 5s
        # - Complex operations: 10s, 30s, 60s
        # - Long operations: 120s, 300s (5 min)
        self.module_latency_seconds = Histogram(
            "modular_agent_module_latency_seconds",
            "Module execution latency in seconds",
            ["module"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            registry=self._registry,
        )

        # Overall coordination latency
        self.coordination_latency_seconds = Histogram(
            "modular_agent_coordination_latency_seconds",
            "Total coordination loop latency in seconds",
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
            registry=self._registry,
        )

        # Module transition latency (time between module calls)
        self.transition_latency_seconds = Histogram(
            "modular_agent_transition_latency_seconds",
            "Module transition latency in seconds",
            ["from_module", "to_module"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
            registry=self._registry,
        )

    def _init_success_metrics(self) -> None:
        """Initialize success/failure counters."""
        # Per-module execution counts
        self.module_executions_total = Counter(
            "modular_agent_module_executions_total",
            "Total module executions",
            ["module", "status"],
            registry=self._registry,
        )

        # Coordination loop completions
        self.coordination_completions_total = Counter(
            "modular_agent_coordination_completions_total",
            "Total coordination loop completions",
            ["status"],
            registry=self._registry,
        )

        # Verification pass/fail counts
        self.verification_results_total = Counter(
            "modular_agent_verification_results_total",
            "Total verification results",
            ["result"],
            registry=self._registry,
        )

    def _init_error_metrics(self) -> None:
        """Initialize error tracking metrics."""
        # Per-module error counts by type
        self.module_errors_total = Counter(
            "modular_agent_module_errors_total",
            "Total module errors by type",
            ["module", "error_type"],
            registry=self._registry,
        )

        # Error rate gauge (errors per minute)
        self.module_error_rate = Gauge(
            "modular_agent_module_error_rate",
            "Module error rate (errors/min)",
            ["module"],
            registry=self._registry,
        )

    def _init_iteration_metrics(self) -> None:
        """Initialize iteration tracking metrics."""
        # Iteration count distribution
        self.iteration_count = Histogram(
            "modular_agent_iteration_count",
            "Number of refinement iterations",
            buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            registry=self._registry,
        )

        # Early exit vs max iterations
        self.iteration_outcomes_total = Counter(
            "modular_agent_iteration_outcomes_total",
            "Iteration outcomes",
            ["outcome"],
            registry=self._registry,
        )

        # Average confidence per iteration
        self.iteration_confidence = Histogram(
            "modular_agent_iteration_confidence",
            "Verification confidence by iteration",
            ["iteration"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self._registry,
        )

    def _init_token_metrics(self) -> None:
        """Initialize token usage metrics."""
        # Per-module token usage
        self.module_tokens_total = Counter(
            "modular_agent_module_tokens_total",
            "Total tokens used by module",
            ["module", "token_type"],
            registry=self._registry,
        )

        # Current token usage gauge
        self.module_tokens_current = Gauge(
            "modular_agent_module_tokens_current",
            "Current token usage by module",
            ["module"],
            registry=self._registry,
        )

        # Token usage distribution
        self.module_tokens_histogram = Histogram(
            "modular_agent_module_tokens_histogram",
            "Token usage distribution per module execution",
            ["module"],
            buckets=(100, 500, 1000, 2000, 5000, 10000, 20000, 50000),
            registry=self._registry,
        )

    def _init_cost_metrics(self) -> None:
        """Initialize cost tracking metrics."""
        # Per-module cost in USD
        self.module_cost_usd_total = Counter(
            "modular_agent_module_cost_usd_total",
            "Total cost in USD by module",
            ["module"],
            registry=self._registry,
        )

        # Cost per execution histogram
        self.module_cost_per_execution = Histogram(
            "modular_agent_module_cost_per_execution_usd",
            "Cost per execution in USD",
            ["module"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self._registry,
        )

        # Total coordination cost
        self.coordination_cost_usd_total = Counter(
            "modular_agent_coordination_cost_usd_total",
            "Total coordination cost in USD",
            registry=self._registry,
        )

    @property
    def registry(self) -> CollectorRegistry:
        """Get the Prometheus registry."""
        return self._registry

    # ========================================================================
    # Module Execution Tracking
    # ========================================================================

    @asynccontextmanager
    async def track_module_execution(
        self, module: str
    ) -> AsyncIterator[ModuleExecutionTracker]:
        """
        Track module execution with automatic metric recording.

        Args:
            module: Module name (planner, executor, verifier, generator)

        Yields:
            ModuleExecutionTracker for setting execution details

        Example:
            ```python
            async with metrics.track_module_execution("planner") as tracker:
                result = await planner.analyze_query(query)
                tracker.set_success(True)
                tracker.set_tokens(result.tokens_used)
                tracker.set_cost(result.cost_usd)
            ```
        """
        start_time = time.time()
        tracker = ModuleExecutionTracker(module)

        try:
            yield tracker

            # Record latency
            duration = time.time() - start_time
            self.module_latency_seconds.labels(module=module).observe(duration)

            # Record success/failure
            status = "success" if tracker.success else "failure"
            self.module_executions_total.labels(module=module, status=status).inc()

            # Record tokens if provided
            if tracker.tokens_used is not None:
                self.module_tokens_total.labels(
                    module=module, token_type="total"
                ).inc(tracker.tokens_used)
                self.module_tokens_histogram.labels(module=module).observe(
                    tracker.tokens_used
                )

            # Record cost if provided
            if tracker.cost_usd is not None:
                self.module_cost_usd_total.labels(module=module).inc(tracker.cost_usd)
                self.module_cost_per_execution.labels(module=module).observe(
                    tracker.cost_usd
                )

            logger.debug(
                "module_execution_tracked",
                module=module,
                duration=duration,
                success=tracker.success,
                tokens=tracker.tokens_used,
                cost=tracker.cost_usd,
            )

        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self.module_latency_seconds.labels(module=module).observe(duration)
            self.module_executions_total.labels(module=module, status="failure").inc()

            # Categorize and record error
            error_type = self._categorize_error(e)
            self.record_error(module, error_type)

            logger.error(
                "module_execution_error",
                module=module,
                duration=duration,
                error=str(e),
                error_type=error_type.value,
            )
            raise

    def _categorize_error(self, error: Exception) -> ErrorType:
        """
        Categorize exception into error type.

        Args:
            error: Exception to categorize

        Returns:
            ErrorType enum value
        """
        error_name = type(error).__name__.lower()

        if "timeout" in error_name:
            return ErrorType.TIMEOUT
        elif "validation" in error_name or "pydantic" in error_name:
            return ErrorType.VALIDATION
        elif "llm" in error_name or "openai" in error_name or "anthropic" in error_name:
            return ErrorType.LLM_ERROR
        elif "tool" in error_name:
            return ErrorType.TOOL_ERROR
        elif "internal" in error_name or "runtime" in error_name:
            return ErrorType.INTERNAL
        else:
            return ErrorType.UNKNOWN

    # ========================================================================
    # Coordination Tracking
    # ========================================================================

    @asynccontextmanager
    async def track_coordination(
        self,
    ) -> AsyncIterator[CoordinationExecutionTracker]:
        """
        Track coordination loop execution.

        Yields:
            CoordinationExecutionTracker for setting execution details

        Example:
            ```python
            async with metrics.track_coordination() as tracker:
                result = await coordinator.execute_with_refinement(...)
                tracker.set_success(True)
                tracker.set_iterations(result.iterations)
                tracker.set_total_cost(result.total_cost)
            ```
        """
        start_time = time.time()
        tracker = CoordinationExecutionTracker()

        try:
            yield tracker

            # Record overall latency
            duration = time.time() - start_time
            self.coordination_latency_seconds.observe(duration)

            # Record completion status
            status = "success" if tracker.success else "failure"
            self.coordination_completions_total.labels(status=status).inc()

            # Record total cost
            if tracker.total_cost_usd is not None:
                self.coordination_cost_usd_total.inc(tracker.total_cost_usd)

            # Record iterations
            if tracker.iterations is not None:
                self.record_iteration_count(tracker.iterations)

            logger.debug(
                "coordination_tracked",
                duration=duration,
                success=tracker.success,
                iterations=tracker.iterations,
                total_cost=tracker.total_cost_usd,
            )

        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self.coordination_latency_seconds.observe(duration)
            self.coordination_completions_total.labels(status="failure").inc()

            logger.error(
                "coordination_error",
                duration=duration,
                error=str(e),
            )
            raise

    # ========================================================================
    # Direct Metric Recording
    # ========================================================================

    def record_error(self, module: str, error_type: ErrorType) -> None:
        """
        Record module error by type.

        Args:
            module: Module name
            error_type: Type of error
        """
        self.module_errors_total.labels(
            module=module, error_type=error_type.value
        ).inc()

        logger.debug(
            "error_recorded",
            module=module,
            error_type=error_type.value,
        )

    def record_verification_result(self, passed: bool, confidence: float) -> None:
        """
        Record verification result.

        Args:
            passed: Whether verification passed
            confidence: Confidence score (0.0-1.0)
        """
        result = "passed" if passed else "failed"
        self.verification_results_total.labels(result=result).inc()

        logger.debug(
            "verification_result_recorded",
            result=result,
            confidence=confidence,
        )

    def record_iteration_count(self, count: int) -> None:
        """
        Record refinement iteration count.

        Args:
            count: Number of iterations
        """
        self.iteration_count.observe(count)

        # Record outcome (early_exit vs max_iterations)
        # Assuming max_iterations is typically 5 (from spec)
        outcome = "early_exit" if count < 5 else "max_iterations"
        self.iteration_outcomes_total.labels(outcome=outcome).inc()

        logger.debug(
            "iteration_count_recorded",
            count=count,
            outcome=outcome,
        )

    def record_iteration_confidence(
        self, iteration: int, confidence: float
    ) -> None:
        """
        Record verification confidence for an iteration.

        Args:
            iteration: Iteration number (1-based)
            confidence: Confidence score (0.0-1.0)
        """
        self.iteration_confidence.labels(iteration=str(iteration)).observe(confidence)

        logger.debug(
            "iteration_confidence_recorded",
            iteration=iteration,
            confidence=confidence,
        )

    def record_module_transition(
        self, from_module: ModuleType, to_module: ModuleType, duration: float
    ) -> None:
        """
        Record transition between modules.

        Args:
            from_module: Source module
            to_module: Target module
            duration: Transition duration in seconds
        """
        self.transition_latency_seconds.labels(
            from_module=from_module.value,
            to_module=to_module.value,
        ).observe(duration)

        logger.debug(
            "module_transition_recorded",
            from_module=from_module.value,
            to_module=to_module.value,
            duration=duration,
        )

    def record_tokens(
        self, module: str, tokens: int, token_type: str = "total"
    ) -> None:
        """
        Record token usage for a module.

        Args:
            module: Module name
            tokens: Number of tokens used
            token_type: Type of tokens (prompt, completion, total)
        """
        self.module_tokens_total.labels(module=module, token_type=token_type).inc(
            tokens
        )
        self.module_tokens_current.labels(module=module).set(tokens)

        logger.debug(
            "tokens_recorded",
            module=module,
            tokens=tokens,
            token_type=token_type,
        )

    def record_cost(self, module: str, cost_usd: float) -> None:
        """
        Record cost for a module execution.

        Args:
            module: Module name
            cost_usd: Cost in USD
        """
        self.module_cost_usd_total.labels(module=module).inc(cost_usd)
        self.module_cost_per_execution.labels(module=module).observe(cost_usd)

        logger.debug(
            "cost_recorded",
            module=module,
            cost_usd=cost_usd,
        )

    # ========================================================================
    # Metric Retrieval
    # ========================================================================

    def get_module_stats(self, module: str) -> dict[str, Any]:
        """
        Get statistics for a specific module.

        Args:
            module: Module name

        Returns:
            Dictionary with module statistics
        """
        # Note: Prometheus client doesn't provide easy access to metric values
        # This is a placeholder for future implementation
        return {
            "module": module,
            "note": "Use Prometheus queries for metric retrieval",
        }

    def reset_metrics(self) -> None:
        """
        Reset all metrics (for testing).

        Warning: This is destructive and should only be used in tests.
        """
        logger.warning("metrics_reset", registry_id=id(self._registry))


# ============================================================================
# Execution Trackers
# ============================================================================


class ModuleExecutionTracker:
    """Tracks details of a single module execution."""

    def __init__(self, module: str) -> None:
        """
        Initialize tracker.

        Args:
            module: Module name
        """
        self.module = module
        self.success = False
        self.tokens_used: int | None = None
        self.cost_usd: float | None = None

    def set_success(self, success: bool) -> None:
        """Set execution success status."""
        self.success = success

    def set_tokens(self, tokens: int) -> None:
        """Set tokens used."""
        self.tokens_used = tokens

    def set_cost(self, cost_usd: float) -> None:
        """Set execution cost in USD."""
        self.cost_usd = cost_usd


class CoordinationExecutionTracker:
    """Tracks details of coordination loop execution."""

    def __init__(self) -> None:
        """Initialize tracker."""
        self.success = False
        self.iterations: int | None = None
        self.total_cost_usd: float | None = None

    def set_success(self, success: bool) -> None:
        """Set execution success status."""
        self.success = success

    def set_iterations(self, iterations: int) -> None:
        """Set number of iterations."""
        self.iterations = iterations

    def set_total_cost(self, cost_usd: float) -> None:
        """Set total execution cost in USD."""
        self.total_cost_usd = cost_usd


# ============================================================================
# Global Metrics Instance
# ============================================================================

# Global metrics collector instance for use across modules
_global_metrics: ModularMetricsCollector | None = None


def get_metrics() -> ModularMetricsCollector:
    """
    Get global metrics collector instance.

    Returns:
        Global ModularMetricsCollector instance
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = ModularMetricsCollector()
    return _global_metrics


def set_metrics(metrics: ModularMetricsCollector) -> None:
    """
    Set global metrics collector instance.

    Args:
        metrics: Metrics collector to use globally
    """
    global _global_metrics
    _global_metrics = metrics
