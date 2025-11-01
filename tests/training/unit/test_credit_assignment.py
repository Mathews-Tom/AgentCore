"""
Unit tests for multi-step credit assignment module.

Tests temporal difference rewards and per-step advantage computation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.training.credit_assignment import CreditAssignment, CreditAssignmentConfig
from agentcore.training.models import Trajectory, TrajectoryStep


def create_test_trajectory(num_steps: int, final_reward: float) -> Trajectory:
    """Create test trajectory with specified steps and reward."""
    steps = [
        TrajectoryStep(
            state={"step": i},
            action={"type": f"action_{i}"},
            result={"output": f"result_{i}"},
            timestamp=datetime.now(UTC),
            duration_ms=100)
        for i in range(num_steps)
    ]

    return Trajectory(
        job_id=uuid4(),
        agent_id="test-agent",
        query="test query",
        steps=steps,
        reward=final_reward,
        normalized_reward=0.0,
        advantage=0.0,
        execution_time_ms=1000,
        success=True)


# Test CreditAssignmentConfig


def test_credit_assignment_config_default():
    """Test default configuration."""
    config = CreditAssignmentConfig()

    assert config.gamma == 0.99
    assert config.enable_td_rewards is True


def test_credit_assignment_config_custom():
    """Test custom configuration."""
    config = CreditAssignmentConfig(gamma=0.95, enable_td_rewards=False)

    assert config.gamma == 0.95
    assert config.enable_td_rewards is False


def test_credit_assignment_config_invalid_gamma_zero():
    """Test invalid gamma (zero)."""
    with pytest.raises(ValueError, match="gamma must be in"):
        CreditAssignmentConfig(gamma=0.0)


def test_credit_assignment_config_invalid_gamma_negative():
    """Test invalid gamma (negative)."""
    with pytest.raises(ValueError, match="gamma must be in"):
        CreditAssignmentConfig(gamma=-0.5)


def test_credit_assignment_config_invalid_gamma_greater_than_one():
    """Test invalid gamma (> 1)."""
    with pytest.raises(ValueError, match="gamma must be in"):
        CreditAssignmentConfig(gamma=1.5)


# Test CreditAssignment initialization


def test_credit_assignment_initialization_default():
    """Test initialization with default config."""
    ca = CreditAssignment()

    assert ca.config.gamma == 0.99
    assert ca.config.enable_td_rewards is True


def test_credit_assignment_initialization_custom():
    """Test initialization with custom config."""
    config = CreditAssignmentConfig(gamma=0.95, enable_td_rewards=False)
    ca = CreditAssignment(config=config)

    assert ca.config.gamma == 0.95
    assert ca.config.enable_td_rewards is False


# Test compute_step_rewards


def test_compute_step_rewards_single_step():
    """Test step rewards for single-step trajectory."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=1, final_reward=1.0)

    step_rewards = ca.compute_step_rewards(traj)

    assert len(step_rewards) == 1
    # Single step: reward * gamma^0 = 1.0 * 1 = 1.0
    assert step_rewards[0] == pytest.approx(1.0)


def test_compute_step_rewards_three_steps():
    """Test step rewards for three-step trajectory."""
    ca = CreditAssignment(CreditAssignmentConfig(gamma=0.99))
    traj = create_test_trajectory(num_steps=3, final_reward=1.0)

    step_rewards = ca.compute_step_rewards(traj)

    assert len(step_rewards) == 3
    # Step 0: 1.0 * 0.99^2 = 0.9801
    assert step_rewards[0] == pytest.approx(1.0 * (0.99**2))
    # Step 1: 1.0 * 0.99^1 = 0.99
    assert step_rewards[1] == pytest.approx(1.0 * 0.99)
    # Step 2: 1.0 * 0.99^0 = 1.0
    assert step_rewards[2] == pytest.approx(1.0)


def test_compute_step_rewards_increasing_values():
    """Test that step rewards increase from early to late steps."""
    ca = CreditAssignment(CreditAssignmentConfig(gamma=0.99))
    traj = create_test_trajectory(num_steps=5, final_reward=1.0)

    step_rewards = ca.compute_step_rewards(traj)

    # Earlier steps should have lower rewards than later steps
    for i in range(len(step_rewards) - 1):
        assert step_rewards[i] < step_rewards[i + 1]


def test_compute_step_rewards_gamma_effect():
    """Test effect of different gamma values."""
    traj = create_test_trajectory(num_steps=5, final_reward=1.0)

    # Higher gamma = less discounting
    ca_high = CreditAssignment(CreditAssignmentConfig(gamma=0.99))
    rewards_high = ca_high.compute_step_rewards(traj)

    # Lower gamma = more discounting
    ca_low = CreditAssignment(CreditAssignmentConfig(gamma=0.9))
    rewards_low = ca_low.compute_step_rewards(traj)

    # For all steps except the last, high gamma should give higher rewards
    for i in range(len(rewards_high) - 1):
        assert rewards_high[i] > rewards_low[i]


def test_compute_step_rewards_uniform_when_td_disabled():
    """Test uniform rewards when TD is disabled."""
    config = CreditAssignmentConfig(gamma=0.99, enable_td_rewards=False)
    ca = CreditAssignment(config=config)
    traj = create_test_trajectory(num_steps=5, final_reward=1.0)

    step_rewards = ca.compute_step_rewards(traj)

    # All steps should have the same reward (final reward)
    assert all(reward == 1.0 for reward in step_rewards)
    assert len(step_rewards) == 5


def test_compute_step_rewards_negative_reward():
    """Test step rewards with negative final reward."""
    ca = CreditAssignment(CreditAssignmentConfig(gamma=0.99))
    traj = create_test_trajectory(num_steps=3, final_reward=-1.0)

    step_rewards = ca.compute_step_rewards(traj)

    assert len(step_rewards) == 3
    # All rewards should be negative
    assert all(reward < 0 for reward in step_rewards)
    # Still increasing in magnitude from early to late
    assert step_rewards[0] > step_rewards[1] > step_rewards[2]


# Test compute_step_advantages


def test_compute_step_advantages_single_trajectory():
    """Test step advantages for single trajectory."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=3, final_reward=1.0)

    advantages = ca.compute_step_advantages([traj], normalize=True)

    assert len(advantages) == 1
    assert len(advantages[0]) == 3
    # With single trajectory, normalized advantages should sum to ~0
    # (after normalization: (x - mean) / std)


def test_compute_step_advantages_multiple_trajectories():
    """Test step advantages for multiple trajectories."""
    ca = CreditAssignment()
    traj1 = create_test_trajectory(num_steps=3, final_reward=1.0)
    traj2 = create_test_trajectory(num_steps=2, final_reward=0.5)

    advantages = ca.compute_step_advantages([traj1, traj2], normalize=True)

    assert len(advantages) == 2
    assert len(advantages[0]) == 3  # traj1 has 3 steps
    assert len(advantages[1]) == 2  # traj2 has 2 steps


def test_compute_step_advantages_no_normalize():
    """Test step advantages without normalization."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=3, final_reward=1.0)

    advantages = ca.compute_step_advantages([traj], normalize=False)

    # Without normalization, advantages == step rewards
    step_rewards = ca.compute_step_rewards(traj)
    assert advantages[0] == step_rewards


def test_compute_step_advantages_empty_trajectories():
    """Test empty trajectory list."""
    ca = CreditAssignment()

    advantages = ca.compute_step_advantages([], normalize=True)

    assert advantages == []


# Test compute_trajectory_advantage


def test_compute_trajectory_advantage_positive_reward():
    """Test trajectory advantage with positive reward."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=3, final_reward=1.0)
    baseline = 0.5

    advantage = ca.compute_trajectory_advantage(traj, baseline=baseline)

    # Total discounted reward should be > baseline
    step_rewards = ca.compute_step_rewards(traj)
    expected_total = sum(step_rewards)
    assert advantage == pytest.approx(expected_total - baseline)
    assert advantage > 0


def test_compute_trajectory_advantage_zero_baseline():
    """Test trajectory advantage with zero baseline."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=3, final_reward=1.0)

    advantage = ca.compute_trajectory_advantage(traj, baseline=0.0)

    # Advantage should equal total discounted reward
    step_rewards = ca.compute_step_rewards(traj)
    expected_total = sum(step_rewards)
    assert advantage == pytest.approx(expected_total)


def test_compute_trajectory_advantage_below_baseline():
    """Test trajectory advantage below baseline (negative advantage)."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=2, final_reward=0.5)
    baseline = 2.0  # High baseline

    advantage = ca.compute_trajectory_advantage(traj, baseline=baseline)

    # Total reward < baseline, so advantage should be negative
    assert advantage < 0


# Test get_config


def test_get_config():
    """Test get_config method."""
    config = CreditAssignmentConfig(gamma=0.95, enable_td_rewards=False)
    ca = CreditAssignment(config=config)

    config_dict = ca.get_config()

    assert config_dict["gamma"] == 0.95
    assert config_dict["td_enabled"] is False


# Integration test: compare TD vs uniform rewards


def test_td_vs_uniform_rewards():
    """Test difference between TD and uniform rewards."""
    traj = create_test_trajectory(num_steps=5, final_reward=1.0)

    # TD rewards
    ca_td = CreditAssignment(CreditAssignmentConfig(gamma=0.99, enable_td_rewards=True))
    td_rewards = ca_td.compute_step_rewards(traj)

    # Uniform rewards
    ca_uniform = CreditAssignment(
        CreditAssignmentConfig(gamma=0.99, enable_td_rewards=False)
    )
    uniform_rewards = ca_uniform.compute_step_rewards(traj)

    # TD rewards should vary across steps
    assert len(set(td_rewards)) > 1

    # Uniform rewards should all be the same
    assert len(set(uniform_rewards)) == 1

    # Early steps in TD should have less reward than uniform
    assert td_rewards[0] < uniform_rewards[0]

    # Late steps in TD should equal uniform (or very close)
    assert td_rewards[-1] == pytest.approx(uniform_rewards[-1])


# Edge cases


def test_edge_case_single_step_no_discounting():
    """Test single-step trajectory has no discounting effect."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=1, final_reward=0.75)

    step_rewards = ca.compute_step_rewards(traj)

    # With one step, gamma^0 = 1, so reward = final_reward
    assert step_rewards[0] == pytest.approx(0.75)


def test_edge_case_zero_reward():
    """Test trajectory with zero reward."""
    ca = CreditAssignment()
    traj = create_test_trajectory(num_steps=3, final_reward=0.0)

    step_rewards = ca.compute_step_rewards(traj)

    # All step rewards should be zero
    assert all(reward == 0.0 for reward in step_rewards)


def test_edge_case_very_long_trajectory():
    """Test very long trajectory (100 steps)."""
    ca = CreditAssignment(CreditAssignmentConfig(gamma=0.99))
    traj = create_test_trajectory(num_steps=100, final_reward=1.0)

    step_rewards = ca.compute_step_rewards(traj)

    assert len(step_rewards) == 100
    # First step should have very small reward due to heavy discounting
    assert step_rewards[0] < 0.5
    # Last step should still be close to 1.0
    assert step_rewards[-1] == pytest.approx(1.0)
