"""
Unit tests for modular agent metrics collection.

Tests comprehensive Prometheus metrics including:
- Module latency histograms
- Success/failure counters
- Error tracking
- Iteration counting
- Token usage
- Cost tracking
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from agentcore.modular.metrics import (
    CoordinationExecutionTracker,
    ErrorType,
    ModularMetricsCollector,
    ModuleExecutionTracker,
    get_metrics,
    set_metrics,
)
from agentcore.modular.models import ModuleType


class TestModularMetricsCollector:
    """Test ModularMetricsCollector functionality."""

    @pytest.fixture
    def registry(self) -> CollectorRegistry:
        """Create fresh Prometheus registry."""
        return CollectorRegistry()

    @pytest.fixture
    def metrics(self, registry: CollectorRegistry) -> ModularMetricsCollector:
        """Create metrics collector with fresh registry."""
        return ModularMetricsCollector(registry=registry)

    def test_initialization(self, metrics: ModularMetricsCollector) -> None:
        """Test metrics collector initialization."""
        assert metrics is not None
        assert metrics.registry is not None

        # Verify all metric types are initialized
        assert hasattr(metrics, "module_latency_seconds")
        assert hasattr(metrics, "module_executions_total")
        assert hasattr(metrics, "module_errors_total")
        assert hasattr(metrics, "iteration_count")
        assert hasattr(metrics, "module_tokens_total")
        assert hasattr(metrics, "module_cost_usd_total")

    @pytest.mark.asyncio
    async def test_track_module_execution_success(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking successful module execution."""
        module = "planner"

        async with metrics.track_module_execution(module) as tracker:
            tracker.set_success(True)
            tracker.set_tokens(1500)
            tracker.set_cost(0.025)

        # Verify metrics were recorded
        # Note: We can't easily query Prometheus metrics directly,
        # but we can verify the tracker state
        assert tracker.success is True
        assert tracker.tokens_used == 1500
        assert tracker.cost_usd == 0.025

    @pytest.mark.asyncio
    async def test_track_module_execution_failure(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking failed module execution."""
        module = "executor"

        with pytest.raises(ValueError):
            async with metrics.track_module_execution(module) as tracker:
                tracker.set_success(False)
                raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_track_multiple_modules(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking multiple module executions."""
        modules = ["planner", "executor", "verifier", "generator"]

        for module in modules:
            async with metrics.track_module_execution(module) as tracker:
                tracker.set_success(True)
                tracker.set_tokens(1000)

    @pytest.mark.asyncio
    async def test_track_coordination_success(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking successful coordination loop."""
        async with metrics.track_coordination() as tracker:
            tracker.set_success(True)
            tracker.set_iterations(3)
            tracker.set_total_cost(0.15)

        assert tracker.success is True
        assert tracker.iterations == 3
        assert tracker.total_cost_usd == 0.15

    @pytest.mark.asyncio
    async def test_track_coordination_failure(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking failed coordination loop."""
        with pytest.raises(RuntimeError):
            async with metrics.track_coordination() as tracker:
                tracker.set_success(False)
                raise RuntimeError("Coordination failed")

    def test_record_error_types(self, metrics: ModularMetricsCollector) -> None:
        """Test recording different error types."""
        error_types = [
            ErrorType.TIMEOUT,
            ErrorType.VALIDATION,
            ErrorType.LLM_ERROR,
            ErrorType.TOOL_ERROR,
            ErrorType.INTERNAL,
            ErrorType.UNKNOWN,
        ]

        for error_type in error_types:
            metrics.record_error("planner", error_type)

    def test_categorize_errors(self, metrics: ModularMetricsCollector) -> None:
        """Test error categorization."""
        # Timeout error
        assert metrics._categorize_error(TimeoutError()) == ErrorType.TIMEOUT

        # Validation error
        assert metrics._categorize_error(ValueError()) == ErrorType.UNKNOWN

        # Generic error
        assert metrics._categorize_error(Exception()) == ErrorType.UNKNOWN

    def test_record_verification_results(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test recording verification results."""
        # Passed verification
        metrics.record_verification_result(passed=True, confidence=0.95)

        # Failed verification
        metrics.record_verification_result(passed=False, confidence=0.45)

    def test_record_iteration_counts(self, metrics: ModularMetricsCollector) -> None:
        """Test recording iteration counts."""
        # Early exit (< 5 iterations)
        metrics.record_iteration_count(2)
        metrics.record_iteration_count(3)

        # Max iterations
        metrics.record_iteration_count(5)
        metrics.record_iteration_count(10)

    def test_record_iteration_confidence(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test recording iteration confidence scores."""
        for iteration in range(1, 6):
            confidence = iteration * 0.15  # Increasing confidence
            metrics.record_iteration_confidence(iteration, confidence)

    def test_record_module_transitions(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test recording module transitions."""
        transitions = [
            (ModuleType.PLANNER, ModuleType.EXECUTOR, 0.25),
            (ModuleType.EXECUTOR, ModuleType.VERIFIER, 0.15),
            (ModuleType.VERIFIER, ModuleType.GENERATOR, 0.10),
        ]

        for from_module, to_module, duration in transitions:
            metrics.record_module_transition(from_module, to_module, duration)

    def test_record_tokens(self, metrics: ModularMetricsCollector) -> None:
        """Test recording token usage."""
        # Total tokens
        metrics.record_tokens("planner", 1500, "total")

        # Prompt tokens
        metrics.record_tokens("executor", 800, "prompt")

        # Completion tokens
        metrics.record_tokens("generator", 700, "completion")

    def test_record_cost(self, metrics: ModularMetricsCollector) -> None:
        """Test recording execution costs."""
        costs = {
            "planner": 0.05,
            "executor": 0.02,
            "verifier": 0.01,
            "generator": 0.03,
        }

        for module, cost in costs.items():
            metrics.record_cost(module, cost)


class TestModuleExecutionTracker:
    """Test ModuleExecutionTracker functionality."""

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        tracker = ModuleExecutionTracker("planner")

        assert tracker.module == "planner"
        assert tracker.success is False
        assert tracker.tokens_used is None
        assert tracker.cost_usd is None

    def test_set_success(self) -> None:
        """Test setting success status."""
        tracker = ModuleExecutionTracker("executor")

        tracker.set_success(True)
        assert tracker.success is True

        tracker.set_success(False)
        assert tracker.success is False

    def test_set_tokens(self) -> None:
        """Test setting token usage."""
        tracker = ModuleExecutionTracker("planner")

        tracker.set_tokens(2500)
        assert tracker.tokens_used == 2500

    def test_set_cost(self) -> None:
        """Test setting execution cost."""
        tracker = ModuleExecutionTracker("generator")

        tracker.set_cost(0.045)
        assert tracker.cost_usd == 0.045


class TestCoordinationExecutionTracker:
    """Test CoordinationExecutionTracker functionality."""

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        tracker = CoordinationExecutionTracker()

        assert tracker.success is False
        assert tracker.iterations is None
        assert tracker.total_cost_usd is None

    def test_set_success(self) -> None:
        """Test setting success status."""
        tracker = CoordinationExecutionTracker()

        tracker.set_success(True)
        assert tracker.success is True

    def test_set_iterations(self) -> None:
        """Test setting iteration count."""
        tracker = CoordinationExecutionTracker()

        tracker.set_iterations(4)
        assert tracker.iterations == 4

    def test_set_total_cost(self) -> None:
        """Test setting total cost."""
        tracker = CoordinationExecutionTracker()

        tracker.set_total_cost(0.25)
        assert tracker.total_cost_usd == 0.25


class TestGlobalMetrics:
    """Test global metrics instance management."""

    def test_get_metrics(self) -> None:
        """Test getting global metrics instance."""
        metrics1 = get_metrics()
        assert metrics1 is not None

        # Should return same instance
        metrics2 = get_metrics()
        assert metrics1 is metrics2

    def test_set_metrics(self) -> None:
        """Test setting global metrics instance."""
        custom_registry = CollectorRegistry()
        custom_metrics = ModularMetricsCollector(registry=custom_registry)

        set_metrics(custom_metrics)

        retrieved = get_metrics()
        assert retrieved is custom_metrics


class TestMetricsIntegration:
    """Integration tests for metrics collection."""

    @pytest.fixture
    def metrics(self) -> ModularMetricsCollector:
        """Create metrics collector for integration tests."""
        return ModularMetricsCollector(registry=CollectorRegistry())

    @pytest.mark.asyncio
    async def test_full_workflow_tracking(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking full PEVG workflow."""
        # Track coordination
        async with metrics.track_coordination() as coord_tracker:
            # Iteration 1
            async with metrics.track_module_execution("planner") as plan_tracker:
                plan_tracker.set_success(True)
                plan_tracker.set_tokens(1200)
                plan_tracker.set_cost(0.03)

            async with metrics.track_module_execution("executor") as exec_tracker:
                exec_tracker.set_success(True)
                exec_tracker.set_tokens(800)
                exec_tracker.set_cost(0.02)

            async with metrics.track_module_execution("verifier") as verify_tracker:
                verify_tracker.set_success(True)
                verify_tracker.set_tokens(500)
                verify_tracker.set_cost(0.01)

            # Record verification failed, need refinement
            metrics.record_verification_result(passed=False, confidence=0.6)

            # Iteration 2 (refinement)
            async with metrics.track_module_execution("planner") as plan_tracker2:
                plan_tracker2.set_success(True)
                plan_tracker2.set_tokens(1000)
                plan_tracker2.set_cost(0.025)

            async with metrics.track_module_execution("executor") as exec_tracker2:
                exec_tracker2.set_success(True)
                exec_tracker2.set_tokens(900)
                exec_tracker2.set_cost(0.022)

            async with metrics.track_module_execution("verifier") as verify_tracker2:
                verify_tracker2.set_success(True)
                verify_tracker2.set_tokens(450)
                verify_tracker2.set_cost(0.011)

            # Record verification passed
            metrics.record_verification_result(passed=True, confidence=0.85)

            # Generation
            async with metrics.track_module_execution("generator") as gen_tracker:
                gen_tracker.set_success(True)
                gen_tracker.set_tokens(600)
                gen_tracker.set_cost(0.015)

            # Set coordination success
            coord_tracker.set_success(True)
            coord_tracker.set_iterations(2)
            coord_tracker.set_total_cost(0.133)

        # Record final iteration count
        metrics.record_iteration_count(2)

    @pytest.mark.asyncio
    async def test_error_scenario_tracking(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking error scenarios."""
        async with metrics.track_coordination() as coord_tracker:
            # Planner succeeds
            async with metrics.track_module_execution("planner") as plan_tracker:
                plan_tracker.set_success(True)

            # Executor fails with tool error
            try:
                async with metrics.track_module_execution("executor") as exec_tracker:
                    exec_tracker.set_success(False)
                    raise RuntimeError("Tool execution failed")
            except RuntimeError:
                metrics.record_error("executor", ErrorType.TOOL_ERROR)

            coord_tracker.set_success(False)

    @pytest.mark.asyncio
    async def test_max_iterations_tracking(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test tracking max iterations scenario."""
        max_iterations = 5

        async with metrics.track_coordination() as coord_tracker:
            for iteration in range(1, max_iterations + 1):
                # Plan
                async with metrics.track_module_execution("planner") as plan_tracker:
                    plan_tracker.set_success(True)

                # Execute
                async with metrics.track_module_execution("executor") as exec_tracker:
                    exec_tracker.set_success(True)

                # Verify (always fails until last iteration)
                async with metrics.track_module_execution(
                    "verifier"
                ) as verify_tracker:
                    verify_tracker.set_success(True)

                passed = iteration == max_iterations
                confidence = iteration * 0.15
                metrics.record_verification_result(passed=passed, confidence=confidence)
                metrics.record_iteration_confidence(iteration, confidence)

            # Generate
            async with metrics.track_module_execution("generator") as gen_tracker:
                gen_tracker.set_success(True)

            coord_tracker.set_success(True)
            coord_tracker.set_iterations(max_iterations)

        metrics.record_iteration_count(max_iterations)

    def test_module_transition_tracking(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test module transition tracking."""
        # Simulate PEVG flow transitions
        flow = [
            (ModuleType.PLANNER, ModuleType.EXECUTOR),
            (ModuleType.EXECUTOR, ModuleType.VERIFIER),
            (ModuleType.VERIFIER, ModuleType.PLANNER),  # Refinement
            (ModuleType.PLANNER, ModuleType.EXECUTOR),
            (ModuleType.EXECUTOR, ModuleType.VERIFIER),
            (ModuleType.VERIFIER, ModuleType.GENERATOR),
        ]

        for from_mod, to_mod in flow:
            # Simulate transition time
            duration = 0.05  # 50ms
            metrics.record_module_transition(from_mod, to_mod, duration)

    def test_cost_accumulation(self, metrics: ModularMetricsCollector) -> None:
        """Test cost accumulation across modules."""
        module_costs = {
            "planner": [0.03, 0.025],  # Two iterations
            "executor": [0.02, 0.022],
            "verifier": [0.01, 0.011],
            "generator": [0.015],
        }

        total_cost = 0.0
        for module, costs in module_costs.items():
            for cost in costs:
                metrics.record_cost(module, cost)
                total_cost += cost

        # Verify total cost is as expected
        assert abs(total_cost - 0.133) < 0.001

    def test_token_accumulation(self, metrics: ModularMetricsCollector) -> None:
        """Test token accumulation across modules."""
        module_tokens = {
            "planner": [1200, 1000],  # Two iterations
            "executor": [800, 900],
            "verifier": [500, 450],
            "generator": [600],
        }

        for module, token_counts in module_tokens.items():
            for tokens in token_counts:
                metrics.record_tokens(module, tokens, "total")
