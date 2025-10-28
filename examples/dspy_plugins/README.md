# Custom DSPy Optimizer Plugins

This directory contains examples of custom optimizer plugins for the AgentCore DSPy optimization engine.

## Overview

The plugin system allows you to create custom optimization algorithms that integrate seamlessly with AgentCore's optimization pipeline. Plugins can be registered, validated, and used alongside built-in optimizers like MIPROv2 and GEPA.

## Plugin Architecture

The plugin system consists of several key components:

### 1. Plugin Interface (`OptimizerPlugin`)

All custom plugins must implement the `OptimizerPlugin` interface:

- `get_metadata()`: Returns plugin metadata (name, version, capabilities, etc.)
- `create_optimizer()`: Creates and returns a configured optimizer instance
- `validate()`: Validates the plugin implementation
- `on_load()`: Optional hook called when plugin is loaded
- `on_unload()`: Optional hook called when plugin is unloaded

### 2. Plugin Registry

The `PluginRegistry` manages plugin lifecycle:

- Register/unregister plugins
- Create optimizer instances
- Track usage statistics
- Discover plugins from directories

### 3. Plugin Validator

The `PluginValidator` performs comprehensive checks:

- Metadata validation
- Interface compliance
- Optimizer creation capability
- Configuration validation
- Documentation checks

### 4. Performance Comparator

The `PerformanceComparator` enables algorithm comparison:

- Compare metrics across multiple algorithms
- Rank algorithms by weighted performance
- Generate comparison reports (text/markdown)

## Creating a Custom Plugin

### Step 1: Implement BaseOptimizer

```python
from agentcore.dspy_optimization.algorithms.base import BaseOptimizer

class MyCustomOptimizer(BaseOptimizer):
    async def optimize(self, request, baseline_metrics, training_data):
        # Implement optimization logic
        pass

    def get_algorithm_name(self) -> str:
        return "my_custom_algorithm"
```

### Step 2: Create Plugin Wrapper

```python
from agentcore.dspy_optimization.plugins import OptimizerPlugin

class MyCustomOptimizerPlugin(OptimizerPlugin):
    def get_metadata(self):
        return PluginMetadata(
            name="my_custom_optimizer",
            version="1.0.0",
            author="Your Name",
            description="My custom optimizer",
            capabilities=[PluginCapability.GRADIENT_FREE],
        )

    def create_optimizer(self, config, **kwargs):
        return MyCustomOptimizer(**kwargs)

    def validate(self):
        from agentcore.dspy_optimization.plugins import PluginValidator
        validator = PluginValidator()
        return validator.validate(self)
```

### Step 3: Register and Use

```python
from agentcore.dspy_optimization.plugins import get_plugin_registry

# Get global registry
registry = get_plugin_registry()

# Register plugin
plugin = MyCustomOptimizerPlugin()
await registry.register(plugin)

# Get optimizer
optimizer = await registry.get_optimizer("my_custom_optimizer")

# Use optimizer
result = await optimizer.optimize(request, baseline_metrics, training_data)
```

## Plugin Capabilities

Plugins can declare the following capabilities:

- `MULTI_OBJECTIVE`: Supports multiple optimization objectives
- `EVOLUTIONARY`: Uses evolutionary/genetic algorithms
- `GRADIENT_FREE`: Does not require gradients
- `GRADIENT_BASED`: Uses gradient-based optimization
- `BAYESIAN`: Uses Bayesian optimization
- `REINFORCEMENT`: Uses reinforcement learning
- `HYBRID`: Combines multiple approaches

## Plugin Configuration

Plugins are configured via `PluginConfig`:

```python
config = PluginConfig(
    plugin_name="my_optimizer",
    enabled=True,
    priority=150,  # Higher priority = preferred selection
    timeout_seconds=3600,
    max_memory_mb=2048,
    parameters={
        "learning_rate": 0.01,
        "num_iterations": 100,
    }
)
```

## Plugin Discovery

Plugins can be auto-discovered from directories:

```python
plugin_dir = Path("/path/to/plugins")
registered = await registry.discover_plugins(plugin_dir)
print(f"Discovered {len(registered)} plugins")
```

## Performance Comparison

Compare multiple algorithms:

```python
from agentcore.dspy_optimization.plugins import PerformanceComparator

comparator = PerformanceComparator()
comparison = comparator.compare_results(baseline_metrics, results)

# Generate report
report = comparator.generate_comparison_report(comparison, format="markdown")
print(report)

# Rank algorithms
rankings = comparator.rank_algorithms(results)
for algorithm, score in rankings:
    print(f"{algorithm}: {score:.4f}")
```

## Examples

- `custom_optimizer_example.py`: Complete example of creating and using a custom optimizer plugin

## Best Practices

1. **Validation**: Always implement proper validation in your plugin
2. **Documentation**: Provide clear docstrings for all methods
3. **Error Handling**: Handle errors gracefully and provide meaningful messages
4. **Resource Management**: Clean up resources in `on_unload()`
5. **Testing**: Write comprehensive tests for your optimizer
6. **Versioning**: Use semantic versioning (e.g., 1.0.0)
7. **Dependencies**: Clearly declare all dependencies in metadata

## API Reference

See the main AgentCore documentation for complete API reference:

- Plugin Interface: `src/agentcore/dspy_optimization/plugins/interface.py`
- Plugin Models: `src/agentcore/dspy_optimization/plugins/models.py`
- Plugin Registry: `src/agentcore/dspy_optimization/plugins/registry.py`
- Plugin Validator: `src/agentcore/dspy_optimization/plugins/validator.py`
- Performance Comparator: `src/agentcore/dspy_optimization/plugins/comparison.py`
