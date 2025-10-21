"""
Custom Reward Function Example

Demonstrates how to create and register custom reward functions for agent training.
"""

from datetime import UTC, datetime
from uuid import uuid4

from agentcore.training import (
    RewardEngine,
    RewardRegistry,
    Trajectory,
    TrajectoryStep,
    get_global_registry,
)


def documentation_quality_reward(trajectory: Trajectory) -> float:
    """
    Custom reward function for documentation generation agents.

    Evaluates based on:
    - Successful completion (0.4 base)
    - Documentation completeness (presence of sections: +0.3)
    - Code examples included (+0.2)
    - Links/references provided (+0.1)

    Args:
        trajectory: Documentation generation trajectory

    Returns:
        Reward in [0, 1]
    """
    reward = 0.0

    # Base reward for success
    if trajectory.success:
        reward += 0.4

    # Check documentation completeness
    has_description = False
    has_examples = False
    has_references = False
    word_count = 0

    for step in trajectory.steps:
        result = step.result

        if isinstance(result, dict):
            # Check for required sections
            if result.get("has_description"):
                has_description = True

            if result.get("code_examples"):
                has_examples = True

            if result.get("references") or result.get("links"):
                has_references = True

            # Track word count
            if "word_count" in result:
                word_count = max(word_count, int(result["word_count"]))

    # Apply bonuses
    if has_description and word_count > 100:
        reward += 0.3  # Complete documentation

    if has_examples:
        reward += 0.2  # Code examples included

    if has_references:
        reward += 0.1  # Links/references provided

    # Clamp to [0, 1]
    return max(0.0, min(1.0, reward))


def test_coverage_reward(trajectory: Trajectory) -> float:
    """
    Custom reward function for test generation agents.

    Evaluates based on:
    - Tests pass (0.5 base)
    - Code coverage (up to +0.4)
    - Edge cases covered (+0.1)

    Args:
        trajectory: Test generation trajectory

    Returns:
        Reward in [0, 1]
    """
    reward = 0.0

    # Base reward for passing tests
    if trajectory.success:
        reward += 0.5

    # Analyze test coverage
    coverage = 0.0
    has_edge_cases = False

    for step in trajectory.steps:
        result = step.result

        if isinstance(result, dict):
            # Extract coverage
            if "coverage" in result:
                coverage = max(coverage, float(result["coverage"]))

            # Check for edge case testing
            if result.get("edge_cases_tested"):
                has_edge_cases = True

    # Coverage bonus (up to +0.4 for 100% coverage)
    reward += coverage * 0.4

    # Edge case bonus
    if has_edge_cases:
        reward += 0.1

    # Clamp to [0, 1]
    return max(0.0, min(1.0, reward))


def setup_custom_rewards() -> RewardRegistry:
    """
    Setup custom reward registry with domain-specific functions.

    Returns:
        Configured RewardRegistry
    """
    # Get global registry (includes built-in functions)
    registry = get_global_registry()

    # Register custom reward functions
    registry.register("documentation_quality", documentation_quality_reward)
    registry.register("test_coverage", test_coverage_reward)

    # Configure agent-type strategies
    registry.set_agent_strategy("doc_generator", "documentation_quality")
    registry.set_agent_strategy("test_generator", "test_coverage")
    registry.set_agent_strategy("code_generator", "code_quality")  # Built-in

    # Set default fallback
    registry.set_default_strategy("task_efficiency")  # Built-in

    print("Custom reward registry configured:")
    print(f"  Registered functions: {registry.list_functions()}")
    print(f"  Agent strategies:")
    print(f"    - doc_generator: {registry.get_agent_strategy('doc_generator')}")
    print(f"    - test_generator: {registry.get_agent_strategy('test_generator')}")
    print(f"    - code_generator: {registry.get_agent_strategy('code_generator')}")
    print(f"  Default strategy: {registry.get_default_strategy()}")

    return registry


def test_custom_reward() -> None:
    """Test custom reward function with sample trajectory."""

    # Create test trajectory for documentation generation
    steps = [
        TrajectoryStep(
            state={},
            action={"step_type": "generate_docs"},
            result={
                "has_description": True,
                "word_count": 500,
                "code_examples": 3,
                "references": ["api_ref.md", "guide.md"],
            },
            timestamp=datetime.now(UTC),
            duration_ms=2000,
        ),
    ]

    trajectory = Trajectory(
        job_id=uuid4(),
        agent_id="doc_generator",
        query="Generate API documentation for UserService",
        steps=steps,
        success=True,
    )

    # Compute reward using custom function
    reward = documentation_quality_reward(trajectory)

    print(f"\nTest Reward Computation:")
    print(f"  Agent: {trajectory.agent_id}")
    print(f"  Query: {trajectory.query}")
    print(f"  Success: {trajectory.success}")
    print(f"  Reward: {reward:.4f}")


def demo_reward_engine_integration() -> None:
    """Demonstrate RewardEngine integration with custom registry."""

    # Setup custom registry
    registry = setup_custom_rewards()

    # Create RewardEngine with custom registry
    engine = RewardEngine(registry=registry)

    # Create test trajectory
    trajectory = Trajectory(
        job_id=uuid4(),
        agent_id="test_generator",
        query="Generate unit tests for Calculator class",
        steps=[
            TrajectoryStep(
                state={},
                action={"step_type": "generate_tests"},
                result={
                    "coverage": 0.95,
                    "edge_cases_tested": True,
                },
                timestamp=datetime.now(UTC),
                duration_ms=1500,
            ),
        ],
        success=True,
    )

    # Compute reward using agent type resolution
    reward = engine.compute_reward(
        trajectory=trajectory,
        agent_type="test_generator",  # Resolves to "test_coverage" function
        use_registry=True,
    )

    print(f"\nRewardEngine Integration Demo:")
    print(f"  Agent Type: test_generator")
    print(f"  Resolved Function: test_coverage")
    print(f"  Computed Reward: {reward:.4f}")


def main() -> None:
    """Main execution."""
    print("=" * 60)
    print("Custom Reward Functions Example")
    print("=" * 60)

    # Test custom reward function
    test_custom_reward()

    print(f"\n{'=' * 60}\n")

    # Demo RewardEngine integration
    demo_reward_engine_integration()

    print(f"\n{'=' * 60}")
    print("✓ Examples completed successfully!")


if __name__ == "__main__":
    main()
    main()
