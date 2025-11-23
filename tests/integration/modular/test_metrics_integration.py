"""
Integration tests for modular agent metrics with coordinator.

Tests metrics integration with the coordination loop and actual module execution.
"""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from agentcore.modular.metrics import ModularMetricsCollector, get_metrics
from agentcore.modular.models import ModuleType


class TestMetricsCoordinatorIntegration:
    """Test metrics integration with module coordinator."""

    @pytest.fixture
    def metrics(self) -> ModularMetricsCollector:
        """Create metrics collector for tests."""
        return ModularMetricsCollector(registry=CollectorRegistry())

    @pytest.mark.asyncio
    async def test_metrics_context_managers(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test metrics context managers work correctly."""
        # Simulate a full PEVG workflow
        async with metrics.track_coordination() as coord_tracker:
            # Planning
            async with metrics.track_module_execution("planner") as plan_tracker:
                plan_tracker.set_success(True)
                plan_tracker.set_tokens(1500)
                plan_tracker.set_cost(0.03)

            # Execution
            async with metrics.track_module_execution("executor") as exec_tracker:
                exec_tracker.set_success(True)
                exec_tracker.set_tokens(1000)
                exec_tracker.set_cost(0.02)

            # Verification
            async with metrics.track_module_execution("verifier") as verify_tracker:
                verify_tracker.set_success(True)
                verify_tracker.set_tokens(500)
                verify_tracker.set_cost(0.01)

            # Generation
            async with metrics.track_module_execution("generator") as gen_tracker:
                gen_tracker.set_success(True)
                gen_tracker.set_tokens(800)
                gen_tracker.set_cost(0.02)

            coord_tracker.set_success(True)
            coord_tracker.set_iterations(1)
            coord_tracker.set_total_cost(0.08)

    @pytest.mark.asyncio
    async def test_metrics_with_refinement_loop(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test metrics tracking with plan refinement."""
        async with metrics.track_coordination() as coord_tracker:
            for iteration in range(1, 4):  # 3 iterations
                # Planning (or refinement after iteration 1)
                async with metrics.track_module_execution("planner") as plan_tracker:
                    plan_tracker.set_success(True)
                    plan_tracker.set_tokens(1200)

                # Execution
                async with metrics.track_module_execution("executor") as exec_tracker:
                    exec_tracker.set_success(True)
                    exec_tracker.set_tokens(900)

                # Verification
                async with metrics.track_module_execution("verifier") as verify_tracker:
                    verify_tracker.set_success(True)
                    verify_tracker.set_tokens(450)

                # Record verification confidence
                confidence = iteration * 0.25  # Increasing confidence
                passed = confidence >= 0.7
                metrics.record_verification_result(passed, confidence)
                metrics.record_iteration_confidence(iteration, confidence)

                if passed:
                    break

            # Generation
            async with metrics.track_module_execution("generator") as gen_tracker:
                gen_tracker.set_success(True)

            coord_tracker.set_success(True)
            coord_tracker.set_iterations(3)

        metrics.record_iteration_count(3)

    @pytest.mark.asyncio
    async def test_metrics_with_module_failures(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test metrics tracking when modules fail."""
        from agentcore.modular.metrics import ErrorType

        async with metrics.track_coordination() as coord_tracker:
            # Planning succeeds
            async with metrics.track_module_execution("planner") as plan_tracker:
                plan_tracker.set_success(True)

            # Execution fails
            try:
                async with metrics.track_module_execution("executor") as exec_tracker:
                    exec_tracker.set_success(False)
                    raise RuntimeError("Tool invocation failed")
            except RuntimeError:
                metrics.record_error("executor", ErrorType.TOOL_ERROR)

            coord_tracker.set_success(False)

    def test_module_transition_recording(
        self, metrics: ModularMetricsCollector
    ) -> None:
        """Test recording module transitions."""
        # Record transitions in PEVG flow
        metrics.record_module_transition(
            ModuleType.PLANNER, ModuleType.EXECUTOR, 0.05
        )
        metrics.record_module_transition(
            ModuleType.EXECUTOR, ModuleType.VERIFIER, 0.03
        )
        metrics.record_module_transition(
            ModuleType.VERIFIER, ModuleType.GENERATOR, 0.02
        )


class TestGlobalMetricsUsage:
    """Test global metrics instance usage."""

    def test_get_global_metrics(self) -> None:
        """Test getting global metrics instance."""
        metrics = get_metrics()
        assert metrics is not None
        assert isinstance(metrics, ModularMetricsCollector)

    @pytest.mark.asyncio
    async def test_global_metrics_in_workflow(self) -> None:
        """Test using global metrics in a workflow."""
        metrics = get_metrics()

        async with metrics.track_module_execution("planner") as tracker:
            tracker.set_success(True)
            tracker.set_tokens(1000)
