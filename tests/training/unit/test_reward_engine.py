"""Unit tests for reward computation system."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.training.models import Trajectory, TrajectoryStep
from agentcore.training.rewards import RewardConfig, RewardEngine


@pytest.fixture
def reward_engine():
    """Create reward engine with default config."""
    return RewardEngine()


@pytest.fixture
def custom_config():
    """Create custom reward configuration."""
    return RewardConfig(
        tool_usage_reward=0.2,
        verification_reward=0.1,
        length_penalty=-0.02,
        enable_shaping=True)


@pytest.fixture
def successful_trajectory():
    """Create successful trajectory with tool usage and verification."""
    now = datetime.now(UTC)
    steps = [
        TrajectoryStep(
            state={"step": 1},
            action={"step_type": "tool_call", "tool": "search"},
            result={"success": True},
            timestamp=now,
            duration_ms=100),
        TrajectoryStep(
            state={"step": 2},
            action={"step_type": "verify", "check": "result"},
            result={"valid": True},
            timestamp=now,
            duration_ms=50),
        TrajectoryStep(
            state={"step": 3},
            action={"step_type": "final_answer"},
            result={"answer": "correct"},
            timestamp=now,
            duration_ms=75),
    ]

    return Trajectory(
        job_id=uuid4(),
        agent_id="agent-test",
        query="test query",
        steps=steps,
        success=True)


@pytest.fixture
def failed_trajectory():
    """Create failed trajectory."""
    now = datetime.now(UTC)
    steps = [
        TrajectoryStep(
            state={"step": 1},
            action={"step_type": "think"},
            result={},
            timestamp=now,
            duration_ms=50),
    ]

    return Trajectory(
        job_id=uuid4(),
        agent_id="agent-test",
        query="test query",
        steps=steps,
        success=False)


def test_reward_engine_initialization():
    """Test reward engine initialization with default config."""
    engine = RewardEngine()

    assert engine.config.tool_usage_reward == 0.1
    assert engine.config.verification_reward == 0.05
    assert engine.config.length_penalty == -0.01
    assert engine.config.enable_shaping is True
    assert len(engine.custom_functions) == 0


def test_reward_engine_custom_config(custom_config):
    """Test reward engine with custom configuration."""
    engine = RewardEngine(config=custom_config)

    assert engine.config.tool_usage_reward == 0.2
    assert engine.config.verification_reward == 0.1
    assert engine.config.length_penalty == -0.02


def test_outcome_reward_success(reward_engine, successful_trajectory):
    """Test outcome reward for successful trajectory."""
    outcome = reward_engine._compute_outcome_reward(successful_trajectory)
    assert outcome == 1.0


def test_outcome_reward_failure(reward_engine, failed_trajectory):
    """Test outcome reward for failed trajectory."""
    outcome = reward_engine._compute_outcome_reward(failed_trajectory)
    assert outcome == 0.0


def test_shaped_rewards(reward_engine, successful_trajectory):
    """Test shaped reward computation."""
    shaped = reward_engine._compute_shaped_rewards(successful_trajectory)

    # Expected: 1 tool call (+0.1) + 1 verify (+0.05) + 3 steps (-0.03)
    expected = 0.1 + 0.05 + (3 * -0.01)
    assert shaped == pytest.approx(expected)


def test_total_reward_success(reward_engine, successful_trajectory):
    """Test total reward computation for successful trajectory."""
    total = reward_engine.compute_reward(successful_trajectory)

    # Expected: outcome (1.0) + shaped (0.12)
    expected = 1.0 + 0.12
    assert total == pytest.approx(expected)


def test_total_reward_failure(reward_engine, failed_trajectory):
    """Test total reward computation for failed trajectory."""
    total = reward_engine.compute_reward(failed_trajectory)

    # Expected: outcome (0.0) + shaped (1 * -0.01)
    expected = 0.0 + (1 * -0.01)
    assert total == pytest.approx(expected)


def test_shaped_rewards_disabled():
    """Test reward computation with shaping disabled."""
    config = RewardConfig(enable_shaping=False)
    engine = RewardEngine(config=config)

    now = datetime.now(UTC)
    steps = [
        TrajectoryStep(
            state={},
            action={"step_type": "tool_call"},
            result={},
            timestamp=now,
            duration_ms=100)
    ]
    trajectory = Trajectory(
        job_id=uuid4(),
        agent_id="agent-test",
        query="test",
        steps=steps,
        success=True)

    reward = engine.compute_reward(trajectory)

    # Only outcome reward (1.0), no shaping
    assert reward == 1.0


def test_batch_rewards(reward_engine, successful_trajectory, failed_trajectory):
    """Test batch reward computation."""
    trajectories = [successful_trajectory, failed_trajectory, successful_trajectory]

    rewards = reward_engine.compute_rewards(trajectories)

    assert len(rewards) == 3
    assert rewards[0] > rewards[1]  # Success > failure
    assert rewards[0] == rewards[2]  # Same trajectories = same reward


def test_normalize_rewards_basic():
    """Test reward normalization with basic case."""
    engine = RewardEngine()
    rewards = [1.0, 2.0, 3.0]

    normalized = engine.normalize_rewards(rewards)

    # Check zero mean
    mean = sum(normalized) / len(normalized)
    assert mean == pytest.approx(0.0, abs=1e-10)

    # Check unit variance
    variance = sum((r - mean) ** 2 for r in normalized) / len(normalized)
    assert variance == pytest.approx(1.0, abs=1e-10)


def test_normalize_rewards_empty():
    """Test normalization with empty list."""
    engine = RewardEngine()
    normalized = engine.normalize_rewards([])

    assert normalized == []


def test_normalize_rewards_single():
    """Test normalization with single reward."""
    engine = RewardEngine()
    normalized = engine.normalize_rewards([5.0])

    # Single value normalizes to 0
    assert normalized == [0.0]


def test_normalize_rewards_zero_std():
    """Test normalization when all rewards are identical (std=0)."""
    engine = RewardEngine()
    rewards = [1.0, 1.0, 1.0, 1.0]

    normalized = engine.normalize_rewards(rewards)

    # All zeros when std=0
    assert all(r == 0.0 for r in normalized)


def test_compute_advantages(reward_engine, successful_trajectory, failed_trajectory):
    """Test advantage computation (normalized rewards)."""
    trajectories = [successful_trajectory, failed_trajectory, successful_trajectory]

    advantages = reward_engine.compute_advantages(trajectories)

    assert len(advantages) == 3
    # Successful trajectories should have positive advantage
    assert advantages[0] > 0
    assert advantages[2] > 0
    # Failed trajectory should have negative advantage
    assert advantages[1] < 0
    # Mean should be ~0
    assert sum(advantages) / len(advantages) == pytest.approx(0.0, abs=1e-10)


def test_compute_advantages_no_normalization(
    reward_engine, successful_trajectory, failed_trajectory
):
    """Test advantage computation without normalization."""
    trajectories = [successful_trajectory, failed_trajectory]

    advantages = reward_engine.compute_advantages(trajectories, normalize=False)

    # Without normalization, advantages == rewards
    rewards = reward_engine.compute_rewards(trajectories)
    assert advantages == rewards


def test_custom_reward_function(reward_engine):
    """Test custom reward function registration and usage."""

    def constant_reward(trajectory: Trajectory) -> float:
        return 10.0

    # Register custom function
    reward_engine.register_custom_function("constant", constant_reward)

    trajectory = Trajectory(
        job_id=uuid4(),
        agent_id="agent-test",
        query="test",
        steps=[],
        success=True)

    # Use custom function
    reward = reward_engine.compute_reward(trajectory, custom_function="constant")

    assert reward == 10.0


def test_custom_reward_function_unknown(reward_engine, successful_trajectory):
    """Test error handling for unknown custom function."""
    with pytest.raises(ValueError, match="Unknown custom function"):
        reward_engine.compute_reward(
            successful_trajectory, custom_function="nonexistent"
        )


def test_tool_detection():
    """Test tool call detection logic."""
    engine = RewardEngine()

    # Tool call action
    assert engine._is_successful_tool_call({"step_type": "tool_call"}) is True
    assert engine._is_successful_tool_call({"step_type": "action"}) is True

    # Non-tool action
    assert engine._is_successful_tool_call({"step_type": "think"}) is False
    assert engine._is_successful_tool_call({}) is False


def test_verification_detection():
    """Test verification step detection logic."""
    engine = RewardEngine()

    # Verification actions
    assert engine._is_verification_step({"step_type": "verify"}) is True
    assert engine._is_verification_step({"step_type": "check"}) is True

    # Non-verification actions
    assert engine._is_verification_step({"step_type": "think"}) is False
    assert engine._is_verification_step({}) is False


def test_complex_trajectory():
    """Test reward computation for complex multi-step trajectory."""
    engine = RewardEngine()

    now = datetime.now(UTC)
    steps = [
        TrajectoryStep(state={}, action={"step_type": "think"}, result={}, timestamp=now, duration_ms=50),
        TrajectoryStep(state={}, action={"step_type": "tool_call"}, result={}, timestamp=now, duration_ms=100),
        TrajectoryStep(state={}, action={"step_type": "tool_call"}, result={}, timestamp=now, duration_ms=100),
        TrajectoryStep(state={}, action={"step_type": "verify"}, result={}, timestamp=now, duration_ms=75),
        TrajectoryStep(state={}, action={"step_type": "action"}, result={}, timestamp=now, duration_ms=100),
        TrajectoryStep(state={}, action={"step_type": "verify"}, result={}, timestamp=now, duration_ms=75),
        TrajectoryStep(state={}, action={"step_type": "final"}, result={}, timestamp=now, duration_ms=50),
    ]

    trajectory = Trajectory(
        job_id=uuid4(),
        agent_id="agent-test",
        query="complex task",
        steps=steps,
        success=True)

    reward = engine.compute_reward(trajectory)

    # Expected breakdown:
    # - Outcome: 1.0
    # - Tool calls: 3 * 0.1 = 0.3
    # - Verifications: 2 * 0.05 = 0.1
    # - Length penalty: 7 * -0.01 = -0.07
    # Total: 1.0 + 0.3 + 0.1 - 0.07 = 1.33

    expected = 1.0 + 0.3 + 0.1 - 0.07
    assert reward == pytest.approx(expected)


def test_reward_config_defaults():
    """Test reward configuration default values."""
    config = RewardConfig()

    assert config.tool_usage_reward == 0.1
    assert config.verification_reward == 0.05
    assert config.length_penalty == -0.01
    assert config.enable_shaping is True
