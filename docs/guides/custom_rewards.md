# Custom Reward Functions Guide

**Version:** 1.0
**Last Updated:** 2025-10-17
**Component:** Flow-Based Optimization (Training Infrastructure)

---

## Overview

This guide explains how to create, register, and use custom reward functions with the AgentCore training infrastructure. The RewardRegistry enables domain-specific reward strategies for different agent types, enabling fine-tuned training optimization.

---

## Quick Start

### Basic Custom Reward Function

```python
from agentcore.training import Trajectory, RewardRegistry, get_global_registry

# Create custom reward function
def my_custom_reward(trajectory: Trajectory) -> float:
    """Custom reward based on task completion time."""
    reward = 0.5 if trajectory.success else 0.0

    # Bonus for fast completion
    if trajectory.execution_time_ms and trajectory.execution_time_ms < 5000:
        reward += 0.3

    return reward

# Register function
registry = get_global_registry()
registry.register("fast_completion", my_custom_reward)

# Use in training
from agentcore.training import RewardEngine

engine = RewardEngine(registry=registry)
reward = engine.compute_reward(
    trajectory=my_trajectory,
    custom_function="fast_completion",
    use_registry=True,
)
```

---

## RewardRegistry API

### Initialization

```python
from agentcore.training import RewardRegistry

# Create new registry
registry = RewardRegistry()

# Or use global registry
from agentcore.training import get_global_registry
registry = get_global_registry()
```

### Registering Functions

```python
def my_reward(trajectory: Trajectory) -> float:
    """Reward function must return float in [0, 1]."""
    return 0.8

# Register with validation (default)
registry.register("my_reward", my_reward, validate=True)

# Register without validation (advanced use only)
registry.register("risky_reward", untested_reward, validate=False)
```

**Validation checks:**
- Function is callable
- Returns numeric value (int or float)
- Return value is in range [0, 1]

### Agent-Type Strategies

```python
# Configure strategy for specific agent type
registry.set_agent_strategy("code_generator", "code_quality")
registry.set_agent_strategy("qa_agent", "response_accuracy")

# Set default strategy (fallback)
registry.set_default_strategy("task_efficiency")

# Resolve strategy for agent type
function_name = registry.resolve_strategy("code_generator")
# Returns "code_quality"

# Compute reward with agent type
reward = registry.compute_reward(
    trajectory=my_trajectory,
    agent_type="code_generator",  # Uses "code_quality" function
)
```

### Listing and Retrieving Functions

```python
# List all registered functions
functions = registry.list_functions()
# Returns: ["code_quality", "response_accuracy", "task_efficiency"]

# Get specific function
reward_func = registry.get("code_quality")

# Check agent strategy
strategy = registry.get_agent_strategy("qa_agent")
# Returns: "response_accuracy"

# Get default strategy
default = registry.get_default_strategy()
# Returns: "task_efficiency"
```

### Unregistering Functions

```python
# Remove function from registry
registry.unregister("my_reward")

# Automatically removes from agent strategies
```

---

## Built-in Example Functions

### 1. Code Quality Reward

Optimized for code generation agents.

```python
from agentcore.training import code_quality_reward

# Evaluates:
# - Successful execution (0.5 base)
# - Syntax errors (penalty: -0.1 per error)
# - Test coverage (bonus: up to +0.3 for 100% coverage)
# - Code complexity (penalty: up to -0.2 for high complexity)

# Use directly
reward = code_quality_reward(trajectory)

# Or register for agent type
registry.set_agent_strategy("code_generator", "code_quality")
```

**Example trajectory data:**

```python
step = TrajectoryStep(
    state={},
    action={"step_type": "run_tests"},
    result={
        "test_coverage": 0.95,  # 95% coverage
        "complexity": 8,  # Acceptable complexity
    },
    timestamp=datetime.now(timezone.utc),
    duration_ms=500,
)

trajectory = Trajectory(
    job_id=uuid4(),
    agent_id="code_agent",
    query="generate sorting function",
    steps=[step],
    success=True,
)

reward = code_quality_reward(trajectory)
# Returns: ~0.785 (0.5 base + 0.285 coverage - 0.0 complexity)
```

### 2. Response Accuracy Reward

Optimized for QA and response agents.

```python
from agentcore.training import response_accuracy_reward

# Evaluates:
# - Task success (0.6 base)
# - Source citations (bonus: +0.15)
# - Confidence scores (bonus: up to +0.2)
# - Clarification requests (penalty: -0.05 per request, max -0.15)

# Example trajectory
step = TrajectoryStep(
    state={},
    action={"step_type": "answer"},
    result={
        "sources": ["doc1.md", "doc2.md"],
        "citations": ["page 12", "section 3.2"],
        "confidence": 0.9,
    },
    timestamp=datetime.now(timezone.utc),
    duration_ms=300,
)

trajectory = Trajectory(
    job_id=uuid4(),
    agent_id="qa_agent",
    query="what is GRPO algorithm?",
    steps=[step],
    success=True,
)

reward = response_accuracy_reward(trajectory)
# Returns: ~0.93 (0.6 base + 0.15 sources + 0.18 confidence)
```

### 3. Task Efficiency Reward

Optimized for general task execution.

```python
from agentcore.training import task_efficiency_reward

# Evaluates:
# - Task success (0.5 base)
# - Number of steps (bonus: up to +0.3 for ≤3 steps)
# - Execution time (bonus: +0.1 if <10s, penalty: -0.1 if >60s)
# - Resource usage (penalty: up to -0.2 for high usage)

# Efficient execution example
steps = [
    TrajectoryStep(
        state={},
        action={"step_type": "fetch_data"},
        result={"status": "success"},
        timestamp=datetime.now(timezone.utc),
        duration_ms=200,
    ),
    TrajectoryStep(
        state={},
        action={"step_type": "process"},
        result={"status": "success"},
        timestamp=datetime.now(timezone.utc),
        duration_ms=150,
    ),
]

trajectory = Trajectory(
    job_id=uuid4(),
    agent_id="task_agent",
    query="fetch and process user data",
    steps=steps,
    success=True,
)

reward = task_efficiency_reward(trajectory)
# Returns: ~0.8 (0.5 base + 0.3 efficiency)
```

---

## Creating Custom Reward Functions

### Reward Function Signature

```python
from agentcore.training.models import Trajectory

def my_reward_function(trajectory: Trajectory) -> float:
    """
    Custom reward function.

    Args:
        trajectory: Complete agent execution trajectory

    Returns:
        Reward value in range [0, 1]

    Raises:
        ValueError: If trajectory is invalid (optional)
    """
    # Your reward computation logic
    reward = 0.0

    # Base reward for success
    if trajectory.success:
        reward += 0.5

    # Custom domain logic
    # ...

    # Clamp to [0, 1]
    return max(0.0, min(1.0, reward))
```

### Accessing Trajectory Data

```python
def analyze_trajectory(trajectory: Trajectory) -> float:
    """Example showing trajectory data access."""

    # Basic info
    job_id = trajectory.job_id  # UUID
    agent_id = trajectory.agent_id  # str
    query = trajectory.query  # str
    success = trajectory.success  # bool | None

    # Execution metadata
    execution_time = trajectory.execution_time_ms  # int | None
    step_count = len(trajectory.steps)

    # Iterate through steps
    for step in trajectory.steps:
        state = step.state  # dict[str, object]
        action = step.action  # dict[str, object]
        result = step.result  # dict[str, object]
        timestamp = step.timestamp  # datetime
        duration = step.duration_ms  # int

        # Extract custom data from result
        if "custom_metric" in result:
            metric_value = result["custom_metric"]
            # Use in reward computation

    return compute_final_reward()
```

### Domain-Specific Examples

#### Example 1: Data Processing Agent

```python
def data_processing_reward(trajectory: Trajectory) -> float:
    """Reward for data processing tasks."""
    reward = 0.0

    # Base reward for completion
    if trajectory.success:
        reward += 0.4

    # Analyze processing quality
    records_processed = 0
    errors_encountered = 0

    for step in trajectory.steps:
        result = step.result
        if "records_processed" in result:
            records_processed += int(result["records_processed"])
        if "errors" in result:
            errors_encountered += len(result["errors"])

    # Reward for throughput
    if records_processed > 1000:
        reward += 0.3
    elif records_processed > 500:
        reward += 0.2
    elif records_processed > 100:
        reward += 0.1

    # Penalty for errors
    error_rate = errors_encountered / max(records_processed, 1)
    reward -= min(error_rate * 0.5, 0.3)

    return max(0.0, min(1.0, reward))
```

#### Example 2: Customer Service Agent

```python
def customer_service_reward(trajectory: Trajectory) -> float:
    """Reward for customer service interactions."""
    reward = 0.0

    # Base reward for resolution
    if trajectory.success:
        reward += 0.5

    # Analyze customer satisfaction indicators
    sentiment_score = 0.0
    resolution_time_ms = trajectory.execution_time_ms or 0
    follow_up_needed = False

    for step in trajectory.steps:
        result = step.result

        # Check sentiment
        if "customer_sentiment" in result:
            sentiment = result["customer_sentiment"]
            if sentiment == "positive":
                sentiment_score += 0.1
            elif sentiment == "negative":
                sentiment_score -= 0.1

        # Check if follow-up required
        if result.get("requires_follow_up"):
            follow_up_needed = True

    # Reward for positive sentiment
    reward += min(sentiment_score, 0.3)

    # Bonus for fast resolution
    if resolution_time_ms < 60000:  # < 1 minute
        reward += 0.2

    # Penalty for follow-up
    if follow_up_needed:
        reward -= 0.1

    return max(0.0, min(1.0, reward))
```

---

## Integration with RewardEngine

### Basic Integration

```python
from agentcore.training import RewardEngine, RewardRegistry

# Create registry with custom functions
registry = RewardRegistry()
registry.register("my_reward", my_custom_reward)

# Create RewardEngine with registry
engine = RewardEngine(registry=registry)

# Compute reward using registry
reward = engine.compute_reward(
    trajectory=my_trajectory,
    custom_function="my_reward",
    use_registry=True,  # Enable registry mode
)
```

### Agent-Type Based Computation

```python
# Configure strategies
registry.set_agent_strategy("code_agent", "code_quality")
registry.set_agent_strategy("qa_agent", "response_accuracy")
registry.set_default_strategy("task_efficiency")

# Compute with agent type resolution
reward = engine.compute_reward(
    trajectory=my_trajectory,
    agent_type="code_agent",  # Resolves to "code_quality"
    use_registry=True,
)
```

### Fallback Behavior

```python
# Registry mode with fallback to default
try:
    reward = engine.compute_reward(
        trajectory=my_trajectory,
        agent_type="unknown_agent",
        use_registry=True,
    )
except ValueError:
    # No strategy configured, falls back to outcome+shaped rewards
    reward = engine.compute_reward(trajectory=my_trajectory)
```

---

## Best Practices

### 1. Reward Function Design

**DO:**
- Return values in [0, 1] range
- Use `max(0.0, min(1.0, reward))` to clamp values
- Provide clear documentation
- Handle edge cases gracefully
- Test with various trajectory types

**DON'T:**
- Return values outside [0, 1]
- Raise exceptions unless truly invalid
- Use blocking I/O operations
- Access external state or databases
- Modify trajectory data

### 2. Validation

```python
# Always validate during registration (default)
registry.register("my_reward", my_reward, validate=True)

# Only skip validation if function is proven correct
# (e.g., migrating from legacy system)
registry.register("legacy_reward", old_reward, validate=False)
```

### 3. Agent Strategy Configuration

```python
# Configure strategies at application startup
def setup_reward_strategies():
    registry = get_global_registry()

    # Code generation agents
    registry.set_agent_strategy("code_generator", "code_quality")
    registry.set_agent_strategy("code_reviewer", "code_quality")

    # QA agents
    registry.set_agent_strategy("qa_agent", "response_accuracy")
    registry.set_agent_strategy("chatbot", "response_accuracy")

    # Task execution agents
    registry.set_default_strategy("task_efficiency")

# Call once during app initialization
setup_reward_strategies()
```

### 4. Testing Custom Rewards

```python
import pytest
from uuid import uuid4
from datetime import datetime, timezone
from agentcore.training import Trajectory, TrajectoryStep

def test_my_custom_reward():
    """Test custom reward function."""
    # Create test trajectory
    trajectory = Trajectory(
        job_id=uuid4(),
        agent_id="test_agent",
        query="test query",
        steps=[],
        success=True,
    )

    # Compute reward
    reward = my_custom_reward(trajectory)

    # Validate output
    assert 0.0 <= reward <= 1.0
    assert isinstance(reward, (int, float))

def test_my_reward_with_registry():
    """Test reward via registry."""
    registry = RewardRegistry()

    # Should not raise
    registry.register("test", my_custom_reward, validate=True)

    # Test computation
    trajectory = create_test_trajectory()
    reward = registry.compute_reward(trajectory, function_name="test")
    assert 0.0 <= reward <= 1.0
```

---

## Troubleshooting

### Common Issues

**Issue: `RewardValidationError: must return value in [0, 1]`**

```python
# Problem
def bad_reward(trajectory: Trajectory) -> float:
    return 1.5  # Out of range!

# Solution
def good_reward(trajectory: Trajectory) -> float:
    raw_score = compute_score(trajectory)
    return max(0.0, min(1.0, raw_score))  # Clamp to [0, 1]
```

**Issue: `ValueError: No reward function specified or resolved`**

```python
# Problem
reward = registry.compute_reward(trajectory)  # No strategy configured

# Solution 1: Specify function explicitly
reward = registry.compute_reward(trajectory, function_name="code_quality")

# Solution 2: Configure default strategy
registry.set_default_strategy("task_efficiency")
reward = registry.compute_reward(trajectory)

# Solution 3: Use agent type
registry.set_agent_strategy("my_agent", "code_quality")
reward = registry.compute_reward(trajectory, agent_type="my_agent")
```

**Issue: `KeyError: Reward function 'xyz' not found`**

```python
# Problem
reward = registry.compute_reward(trajectory, function_name="xyz")  # Not registered

# Solution
registry.register("xyz", my_function)
reward = registry.compute_reward(trajectory, function_name="xyz")
```

---

## API Reference

### RewardRegistry

| Method | Description |
|--------|-------------|
| `register(name, func, validate=True)` | Register reward function |
| `unregister(name)` | Remove reward function |
| `get(name)` | Retrieve reward function |
| `list_functions()` | List all registered functions |
| `set_agent_strategy(agent_type, function_name)` | Configure agent-specific strategy |
| `get_agent_strategy(agent_type)` | Get strategy for agent type |
| `set_default_strategy(function_name)` | Set default fallback strategy |
| `get_default_strategy()` | Get default strategy |
| `resolve_strategy(agent_type)` | Resolve strategy for agent type |
| `compute_reward(trajectory, function_name, agent_type)` | Compute reward |

### Global Registry Functions

| Function | Description |
|----------|-------------|
| `get_global_registry()` | Get or create global registry |
| `reset_global_registry()` | Reset global registry (testing) |

### Example Reward Functions

| Function | Agent Type | Description |
|----------|-----------|-------------|
| `code_quality_reward` | Code generators | Evaluates code quality metrics |
| `response_accuracy_reward` | QA/chat agents | Evaluates response quality |
| `task_efficiency_reward` | Task agents | Evaluates execution efficiency |

---

## Further Reading

- [Training Infrastructure Overview](../api/training-api.md)
- [GRPO Algorithm Guide](../guides/training-agents.md)
- [Trajectory Collection](../ops/training-runbook.md)
- [Agent Configuration](../api/agent-config.md)

---

**Questions or Issues?**

- File bug reports: [GitHub Issues](https://github.com/your-org/agentcore/issues)
- Documentation feedback: [Docs Repo](https://github.com/your-org/agentcore-docs)
