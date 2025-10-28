"""
Example custom optimizer plugin

This example demonstrates how to create a custom optimizer plugin
that can be registered with the AgentCore DSPy optimization engine.
"""

from __future__ import annotations

from typing import Any

import dspy

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.models import (
    OptimizationDetails,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.plugins import (
    OptimizerPlugin,
    PluginCapability,
    PluginConfig,
    PluginMetadata,
    PluginValidationResult,
)


class SimpleRandomSearchOptimizer(BaseOptimizer):
    """
    Simple random search optimizer

    This is a basic optimizer that performs random search over
    hyperparameters. It serves as an example of implementing
    the BaseOptimizer interface.
    """

    def __init__(
        self,
        llm: dspy.LM | None = None,
        num_trials: int = 10,
        random_seed: int = 42,
    ) -> None:
        super().__init__(llm)
        self.num_trials = num_trials
        self.random_seed = random_seed

    async def optimize(
        self,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> OptimizationResult:
        """
        Perform random search optimization

        Args:
            request: Optimization request
            baseline_metrics: Baseline performance
            training_data: Training examples

        Returns:
            Optimization result
        """
        # For this example, we'll simulate optimization
        # In a real implementation, you would:
        # 1. Sample random hyperparameters
        # 2. Train/optimize with each configuration
        # 3. Evaluate performance
        # 4. Return best result

        # Simulate improvement
        optimized_metrics = PerformanceMetrics(
            success_rate=min(1.0, baseline_metrics.success_rate * 1.1),
            avg_cost_per_task=baseline_metrics.avg_cost_per_task * 0.9,
            avg_latency_ms=int(baseline_metrics.avg_latency_ms * 0.95),
            quality_score=min(1.0, baseline_metrics.quality_score * 1.05),
        )

        improvement = self.calculate_improvement(
            baseline_metrics, optimized_metrics
        )

        return OptimizationResult(
            status=OptimizationStatus.COMPLETED,
            baseline_performance=baseline_metrics,
            optimized_performance=optimized_metrics,
            improvement_percentage=improvement,
            optimization_details=OptimizationDetails(
                algorithm_used=self.get_algorithm_name(),
                iterations=self.num_trials,
                key_improvements=[
                    "Random hyperparameter search",
                    f"Evaluated {self.num_trials} configurations",
                ],
                parameters={
                    "num_trials": self.num_trials,
                    "random_seed": self.random_seed,
                },
            ),
        )

    def get_algorithm_name(self) -> str:
        return "random_search"


class RandomSearchOptimizerPlugin(OptimizerPlugin):
    """
    Plugin wrapper for RandomSearchOptimizer

    This demonstrates how to wrap a custom optimizer in the
    plugin interface for registration with AgentCore.
    """

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        return PluginMetadata(
            name="random_search",
            version="1.0.0",
            author="AgentCore Team",
            description="Simple random search optimizer for hyperparameter tuning",
            capabilities=[
                PluginCapability.GRADIENT_FREE,
                PluginCapability.MULTI_OBJECTIVE,
            ],
            requires_python=">=3.12",
            dependencies=["dspy-ai>=2.5.0"],
            tags=["random-search", "hyperparameter-tuning", "simple"],
            documentation_url="https://example.com/docs/random-search",
            license="MIT",
        )

    def create_optimizer(
        self, config: PluginConfig, **kwargs: Any
    ) -> BaseOptimizer:
        """
        Create optimizer instance

        Args:
            config: Plugin configuration
            **kwargs: Additional optimizer arguments (e.g., llm)

        Returns:
            Configured optimizer instance
        """
        # Extract parameters from config
        num_trials = config.parameters.get("num_trials", 10)
        random_seed = config.parameters.get("random_seed", 42)

        return SimpleRandomSearchOptimizer(
            num_trials=num_trials,
            random_seed=random_seed,
            **kwargs,
        )

    def validate(self) -> PluginValidationResult:
        """
        Validate plugin implementation

        Returns:
            Validation result
        """
        # Use the built-in validator
        from agentcore.dspy_optimization.plugins import PluginValidator

        validator = PluginValidator()
        return validator.validate(self)

    def on_load(self) -> None:
        """Called when plugin is loaded"""
        print(f"Loading {self.get_metadata().name} plugin...")
        # Initialize any resources here
        # e.g., load models, connect to services, etc.

    def on_unload(self) -> None:
        """Called when plugin is unloaded"""
        print(f"Unloading {self.get_metadata().name} plugin...")
        # Clean up resources here
        # e.g., close connections, save state, etc.


async def main() -> None:
    """Example usage of custom optimizer plugin"""
    from agentcore.dspy_optimization.models import (
        MetricType,
        OptimizationObjective,
        OptimizationScope,
        OptimizationTarget,
        OptimizationTargetType,
    )
    from agentcore.dspy_optimization.plugins import get_plugin_registry

    # Create plugin instance
    plugin = RandomSearchOptimizerPlugin()

    # Get global registry
    registry = get_plugin_registry()

    # Register plugin with custom config
    config = PluginConfig(
        plugin_name="random_search",
        enabled=True,
        priority=150,
        parameters={
            "num_trials": 20,
            "random_seed": 123,
        },
    )

    registration = await registry.register(plugin, config=config)
    print(f"Registered plugin: {registration.metadata.name}")
    print(f"Status: {registration.status}")

    # Get optimizer from registry
    optimizer = await registry.get_optimizer("random_search")
    print(f"Created optimizer: {optimizer.get_algorithm_name()}")

    # Use optimizer
    request = OptimizationRequest(
        target=OptimizationTarget(
            type=OptimizationTargetType.AGENT,
            id="example-agent",
            scope=OptimizationScope.INDIVIDUAL,
        ),
        objectives=[
            OptimizationObjective(
                metric=MetricType.SUCCESS_RATE,
                target_value=0.95,
                weight=0.4,
            ),
            OptimizationObjective(
                metric=MetricType.COST_EFFICIENCY,
                target_value=0.8,
                weight=0.3,
            ),
            OptimizationObjective(
                metric=MetricType.LATENCY,
                target_value=0.85,
                weight=0.3,
            ),
        ],
    )

    baseline_metrics = PerformanceMetrics(
        success_rate=0.80,
        avg_cost_per_task=0.10,
        avg_latency_ms=2000,
        quality_score=0.85,
    )

    result = await optimizer.optimize(request, baseline_metrics, [])
    print(f"\nOptimization Result:")
    print(f"  Status: {result.status}")
    print(f"  Improvement: {result.improvement_percentage:.2f}%")

    # List all registered plugins
    plugins = registry.list_plugins()
    print(f"\nRegistered plugins: {[p.metadata.name for p in plugins]}")

    # Unregister plugin
    await registry.unregister("random_search")
    print("\nPlugin unregistered")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
