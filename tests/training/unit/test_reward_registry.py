"""
Unit tests for RewardRegistry and custom reward functions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from agentcore.training.models import Trajectory, TrajectoryStep
from agentcore.training.reward_registry import (
    RewardRegistry,
    RewardValidationError,
    code_quality_reward,
    get_global_registry,
    reset_global_registry,
    response_accuracy_reward,
    task_efficiency_reward)


def create_test_trajectory(
    success: bool = True, steps: list[TrajectoryStep] | None = None
) -> Trajectory:
    """Create test trajectory with valid UUIDs and required fields."""
    return Trajectory(
        job_id=uuid4(),
        agent_id="test_agent",
        query="test query",
        steps=steps or [],
        success=success)


def create_test_step(
    action: dict | None = None,
    result: dict | None = None,
    state: dict | None = None) -> TrajectoryStep:
    """Create test trajectory step with required fields."""
    return TrajectoryStep(
        state=state or {},
        action=action or {"step_type": "test"},
        result=result or {},
        timestamp=datetime.now(timezone.utc),
        duration_ms=100)


class TestRewardRegistry:
    """Test RewardRegistry functionality."""

    def test_registry_initialization(self) -> None:
        """Test registry initializes correctly."""
        registry = RewardRegistry()
        assert registry.list_functions() == []
        assert registry.get_default_strategy() is None

    def test_register_valid_function(self) -> None:
        """Test registering valid reward function."""
        registry = RewardRegistry()

        def simple_reward(traj: Trajectory) -> float:
            return 0.5

        registry.register("simple", simple_reward)
        assert "simple" in registry.list_functions()

    def test_register_duplicate_function_raises(self) -> None:
        """Test registering duplicate function name raises ValueError."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.5

        registry.register("test", reward_func)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", reward_func)

    def test_register_non_callable_raises(self) -> None:
        """Test registering non-callable raises RewardValidationError."""
        registry = RewardRegistry()

        with pytest.raises(RewardValidationError, match="must be callable"):
            registry.register("invalid", "not a function")  # type: ignore

    def test_register_function_returning_invalid_range_raises(self) -> None:
        """Test registering function with invalid output range raises."""
        registry = RewardRegistry()

        def invalid_reward(traj: Trajectory) -> float:
            return 1.5  # Out of [0, 1] range

        with pytest.raises(RewardValidationError, match="must return value in"):
            registry.register("invalid", invalid_reward, validate=True)

    def test_register_function_returning_non_numeric_raises(self) -> None:
        """Test registering function with non-numeric output raises."""
        registry = RewardRegistry()

        def invalid_reward(traj: Trajectory) -> float:
            return "not a number"  # type: ignore

        with pytest.raises(RewardValidationError, match="must return numeric value"):
            registry.register("invalid", invalid_reward, validate=True)

    def test_register_without_validation(self) -> None:
        """Test registering function without validation succeeds."""
        registry = RewardRegistry()

        def potentially_invalid(traj: Trajectory) -> float:
            return 2.0  # Would fail validation

        # Should not raise with validate=False
        registry.register("no_validate", potentially_invalid, validate=False)
        assert "no_validate" in registry.list_functions()

    def test_unregister_function(self) -> None:
        """Test unregistering function."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.5

        registry.register("test", reward_func)
        assert "test" in registry.list_functions()

        registry.unregister("test")
        assert "test" not in registry.list_functions()

    def test_unregister_nonexistent_function_raises(self) -> None:
        """Test unregistering nonexistent function raises KeyError."""
        registry = RewardRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.unregister("nonexistent")

    def test_get_function(self) -> None:
        """Test retrieving registered function."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.7

        registry.register("test", reward_func)
        retrieved = registry.get("test")
        assert retrieved == reward_func

        # Test function works
        test_traj = create_test_trajectory()
        assert retrieved(test_traj) == 0.7

    def test_get_nonexistent_function_raises(self) -> None:
        """Test retrieving nonexistent function raises KeyError."""
        registry = RewardRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_set_agent_strategy(self) -> None:
        """Test setting agent-specific reward strategy."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.5

        registry.register("test_reward", reward_func)
        registry.set_agent_strategy("code_agent", "test_reward")

        assert registry.get_agent_strategy("code_agent") == "test_reward"

    def test_set_agent_strategy_unregistered_function_raises(self) -> None:
        """Test setting agent strategy with unregistered function raises."""
        registry = RewardRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.set_agent_strategy("agent", "nonexistent")

    def test_get_agent_strategy_unset_returns_none(self) -> None:
        """Test getting unset agent strategy returns None."""
        registry = RewardRegistry()
        assert registry.get_agent_strategy("unknown_agent") is None

    def test_set_default_strategy(self) -> None:
        """Test setting default reward strategy."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.5

        registry.register("default_reward", reward_func)
        registry.set_default_strategy("default_reward")

        assert registry.get_default_strategy() == "default_reward"

    def test_set_default_strategy_unregistered_raises(self) -> None:
        """Test setting default strategy with unregistered function raises."""
        registry = RewardRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.set_default_strategy("nonexistent")

    def test_resolve_strategy_agent_specific(self) -> None:
        """Test strategy resolution with agent-specific strategy."""
        registry = RewardRegistry()

        def reward1(traj: Trajectory) -> float:
            return 0.5

        def reward2(traj: Trajectory) -> float:
            return 0.7

        registry.register("reward1", reward1)
        registry.register("reward2", reward2)
        registry.set_default_strategy("reward1")
        registry.set_agent_strategy("special_agent", "reward2")

        # Agent with specific strategy
        assert registry.resolve_strategy("special_agent") == "reward2"

        # Agent without specific strategy uses default
        assert registry.resolve_strategy("normal_agent") == "reward1"

        # No agent type uses default
        assert registry.resolve_strategy() == "reward1"

    def test_resolve_strategy_no_config(self) -> None:
        """Test strategy resolution with no configuration returns None."""
        registry = RewardRegistry()
        assert registry.resolve_strategy("any_agent") is None

    def test_compute_reward_explicit_function(self) -> None:
        """Test computing reward with explicit function name."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.8

        registry.register("test", reward_func)
        test_traj = create_test_trajectory()
        reward = registry.compute_reward(test_traj, function_name="test")
        assert reward == 0.8

    def test_compute_reward_resolved_strategy(self) -> None:
        """Test computing reward with resolved strategy."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.6

        registry.register("test", reward_func)
        registry.set_default_strategy("test")
        test_traj = create_test_trajectory()
        reward = registry.compute_reward(test_traj)
        assert reward == 0.6

    def test_compute_reward_agent_type_strategy(self) -> None:
        """Test computing reward with agent type strategy resolution."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.9

        registry.register("agent_reward", reward_func)
        registry.set_agent_strategy("qa_agent", "agent_reward")
        test_traj = create_test_trajectory()
        reward = registry.compute_reward(test_traj, agent_type="qa_agent")
        assert reward == 0.9

    def test_compute_reward_no_function_raises(self) -> None:
        """Test computing reward with no function raises ValueError."""
        registry = RewardRegistry()
        test_traj = create_test_trajectory()

        with pytest.raises(ValueError, match="No reward function"):
            registry.compute_reward(test_traj)

    def test_compute_reward_invalid_output_raises(self) -> None:
        """Test computing reward with invalid output raises."""
        registry = RewardRegistry()

        def bad_reward(traj: Trajectory) -> float:
            return 1.5  # Out of range

        registry.register("bad", bad_reward, validate=False)
        test_traj = create_test_trajectory()

        with pytest.raises(RewardValidationError, match="outside \\[0, 1\\]"):
            registry.compute_reward(test_traj, function_name="bad")

    def test_unregister_clears_agent_strategies(self) -> None:
        """Test unregistering function clears related agent strategies."""
        registry = RewardRegistry()

        def reward_func(traj: Trajectory) -> float:
            return 0.5

        registry.register("test", reward_func)
        registry.set_agent_strategy("agent1", "test")
        registry.set_agent_strategy("agent2", "test")
        registry.set_default_strategy("test")

        registry.unregister("test")

        assert registry.get_agent_strategy("agent1") is None
        assert registry.get_agent_strategy("agent2") is None
        assert registry.get_default_strategy() is None


class TestExampleRewardFunctions:
    """Test example domain-specific reward functions."""

    def test_code_quality_reward_success(self) -> None:
        """Test code quality reward for successful execution."""
        traj = create_test_trajectory(success=True)
        reward = code_quality_reward(traj)
        assert 0.0 <= reward <= 1.0
        assert reward >= 0.5  # Base reward for success

    def test_code_quality_reward_with_test_coverage(self) -> None:
        """Test code quality reward with high test coverage."""
        step = create_test_step(
            action={"step_type": "run_tests"},
            result={"test_coverage": 0.9})
        traj = create_test_trajectory(success=True, steps=[step])
        reward = code_quality_reward(traj)
        assert reward > 0.5  # Should have bonus from coverage
        assert reward <= 1.0

    def test_code_quality_reward_with_syntax_errors(self) -> None:
        """Test code quality reward with syntax errors."""
        step = create_test_step(
            action={"step_type": "validate"},
            result={"error_type": "syntax"})
        traj = create_test_trajectory(success=False, steps=[step])
        reward = code_quality_reward(traj)
        assert 0.0 <= reward <= 1.0

    def test_response_accuracy_reward_success(self) -> None:
        """Test response accuracy reward for successful task."""
        traj = create_test_trajectory(success=True)
        reward = response_accuracy_reward(traj)
        assert 0.0 <= reward <= 1.0
        assert reward >= 0.6  # Base reward for success

    def test_response_accuracy_reward_with_sources(self) -> None:
        """Test response accuracy reward with source citations."""
        step = create_test_step(
            action={"step_type": "answer"},
            result={"sources": ["source1", "source2"], "confidence": 0.9})
        traj = create_test_trajectory(success=True, steps=[step])
        reward = response_accuracy_reward(traj)
        assert reward > 0.6  # Bonus for sources and confidence
        assert reward <= 1.0

    def test_response_accuracy_reward_with_clarifications(self) -> None:
        """Test response accuracy reward with clarification requests."""
        step = create_test_step(
            action={"step_type": "clarify"},
            result={"needs_clarification": True})
        traj = create_test_trajectory(success=True, steps=[step])
        reward = response_accuracy_reward(traj)
        assert 0.0 <= reward <= 1.0

    def test_task_efficiency_reward_efficient(self) -> None:
        """Test task efficiency reward for efficient execution."""
        steps = [
            create_test_step(action={"step_type": "action1"}, result={"status": "done"}),
            create_test_step(action={"step_type": "action2"}, result={"status": "done"}),
        ]
        traj = create_test_trajectory(success=True, steps=steps)
        reward = task_efficiency_reward(traj)
        assert 0.0 <= reward <= 1.0
        assert reward > 0.5  # Bonus for efficiency (few steps)

    def test_task_efficiency_reward_many_steps(self) -> None:
        """Test task efficiency reward with many steps."""
        steps = [create_test_step(action={"step_type": f"step{i}"}) for i in range(15)]
        traj = create_test_trajectory(success=True, steps=steps)
        reward = task_efficiency_reward(traj)
        assert 0.0 <= reward <= 1.0

    def test_task_efficiency_reward_high_resource_usage(self) -> None:
        """Test task efficiency reward with high resource usage."""
        step = create_test_step(
            action={"step_type": "compute"},
            result={"resource_usage": 0.95})
        traj = create_test_trajectory(success=True, steps=[step])
        reward = task_efficiency_reward(traj)
        assert 0.0 <= reward <= 1.0


class TestGlobalRegistry:
    """Test global registry instance management."""

    def test_get_global_registry_creates_instance(self) -> None:
        """Test get_global_registry creates instance with default functions."""
        reset_global_registry()  # Start fresh
        registry = get_global_registry()

        assert isinstance(registry, RewardRegistry)
        assert "code_quality" in registry.list_functions()
        assert "response_accuracy" in registry.list_functions()
        assert "task_efficiency" in registry.list_functions()

    def test_get_global_registry_returns_same_instance(self) -> None:
        """Test get_global_registry returns same instance on subsequent calls."""
        reset_global_registry()
        registry1 = get_global_registry()
        registry2 = get_global_registry()

        assert registry1 is registry2

    def test_reset_global_registry(self) -> None:
        """Test reset_global_registry creates new instance."""
        reset_global_registry()
        registry1 = get_global_registry()

        reset_global_registry()
        registry2 = get_global_registry()

        assert registry1 is not registry2

    def test_global_registry_example_functions_work(self) -> None:
        """Test example functions registered in global registry work correctly."""
        reset_global_registry()
        registry = get_global_registry()
        test_traj = create_test_trajectory()

        # Test each example function
        reward1 = registry.compute_reward(test_traj, function_name="code_quality")
        assert 0.0 <= reward1 <= 1.0

        reward2 = registry.compute_reward(test_traj, function_name="response_accuracy")
        assert 0.0 <= reward2 <= 1.0

        reward3 = registry.compute_reward(test_traj, function_name="task_efficiency")
        assert 0.0 <= reward3 <= 1.0
