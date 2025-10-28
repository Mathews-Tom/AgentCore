"""
Tests for plugin interface
"""

from __future__ import annotations

from typing import Any

import dspy
import pytest

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.plugins.interface import OptimizerPlugin
from agentcore.dspy_optimization.plugins.models import (
    PluginCapability,
    PluginConfig,
    PluginMetadata,
    PluginValidationResult,
)


class MockOptimizer(BaseOptimizer):
    """Mock optimizer for testing"""

    def __init__(self, llm: dspy.LM | None = None) -> None:
        super().__init__(llm)
        self.optimize_called = False

    async def optimize(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> OptimizationResult:
        self.optimize_called = True
        return OptimizationResult(
            status=OptimizationStatus.COMPLETED,
            baseline_performance=baseline_metrics,
            optimized_performance=PerformanceMetrics(
                success_rate=0.95,
                avg_cost_per_task=0.08,
                avg_latency_ms=1800,
                quality_score=0.92,
            ),
            improvement_percentage=10.0,
            optimization_details=OptimizationDetails(
                algorithm_used="mock",
                iterations=5,
            ),
        )

    def get_algorithm_name(self) -> str:
        return "mock_optimizer"


class TestOptimizerPlugin(OptimizerPlugin):
    """Test implementation of OptimizerPlugin"""

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="test_optimizer",
            version="1.0.0",
            author="Test Author",
            description="Test optimizer plugin",
            capabilities=[PluginCapability.GRADIENT_FREE],
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        return MockOptimizer(**kwargs)

    def validate(self) -> PluginValidationResult:
        return PluginValidationResult(
            plugin_name="test_optimizer",
            is_valid=True,
            checks_passed=5,
            checks_total=5,
        )


class TestOptimizerPluginInterface:
    """Tests for OptimizerPlugin interface"""

    def test_get_metadata(self) -> None:
        """Test get_metadata returns valid metadata"""
        plugin = TestOptimizerPlugin()
        metadata = plugin.get_metadata()

        assert metadata.name == "test_optimizer"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert PluginCapability.GRADIENT_FREE in metadata.capabilities

    def test_create_optimizer(self) -> None:
        """Test create_optimizer returns BaseOptimizer"""
        plugin = TestOptimizerPlugin()
        config = PluginConfig(plugin_name="test_optimizer")

        optimizer = plugin.create_optimizer(config)

        assert isinstance(optimizer, BaseOptimizer)
        assert optimizer.get_algorithm_name() == "mock_optimizer"

    def test_create_optimizer_with_llm(self) -> None:
        """Test create_optimizer with custom LLM"""
        plugin = TestOptimizerPlugin()
        config = PluginConfig(plugin_name="test_optimizer")
        custom_llm = dspy.LM("openai/gpt-4.1")

        optimizer = plugin.create_optimizer(config, llm=custom_llm)

        assert optimizer.llm == custom_llm

    def test_validate(self) -> None:
        """Test validate returns validation result"""
        plugin = TestOptimizerPlugin()
        result = plugin.validate()

        assert result.plugin_name == "test_optimizer"
        assert result.is_valid is True
        assert result.checks_passed == 5
        assert result.checks_total == 5

    def test_get_default_config(self) -> None:
        """Test get_default_config returns valid config"""
        plugin = TestOptimizerPlugin()
        config = plugin.get_default_config()

        assert config.plugin_name == "test_optimizer"
        assert config.enabled is True
        assert config.priority == 100

    def test_lifecycle_hooks(self) -> None:
        """Test on_load and on_unload hooks"""
        plugin = TestOptimizerPlugin()

        # Should not raise
        plugin.on_load()
        plugin.on_unload()

    @pytest.mark.asyncio
    async def test_optimizer_functionality(self) -> None:
        """Test that created optimizer works correctly"""
        plugin = TestOptimizerPlugin()
        config = PluginConfig(plugin_name="test_optimizer")
        optimizer = plugin.create_optimizer(config)

        from agentcore.dspy_optimization.models import (
            MetricType,
            OptimizationObjective,
            OptimizationScope,
            OptimizationTarget,
            OptimizationTargetType,
        )

        request = OptimizationRequest(
            target=OptimizationTarget(
                type=OptimizationTargetType.AGENT,
                id="test-agent",
                scope=OptimizationScope.INDIVIDUAL,
            ),
            objectives=[
                OptimizationObjective(
                    metric=MetricType.SUCCESS_RATE, target_value=0.95
                )
            ],
        )

        baseline_metrics = PerformanceMetrics(
            success_rate=0.8,
            avg_cost_per_task=0.1,
            avg_latency_ms=2000,
            quality_score=0.85,
        )

        result = await optimizer.optimize(request, baseline_metrics, [])

        assert result.status == OptimizationStatus.COMPLETED
        assert result.optimized_performance is not None
        assert isinstance(optimizer, MockOptimizer)
        assert optimizer.optimize_called is True
