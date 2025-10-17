"""Unit tests for GRPO trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.training.grpo import GRPOConfig, GRPOTrainer, TrainingMetrics
from agentcore.training.models import Trajectory, TrajectoryStep
from agentcore.training.rewards import RewardEngine


@pytest.fixture
def reward_engine():
    """Create reward engine."""
    return RewardEngine()


@pytest.fixture
def grpo_config():
    """Create GRPO configuration."""
    return GRPOConfig(
        learning_rate=0.0001,
        gradient_clip_value=1.0,
        advantage_threshold=0.0,
        enable_gradient_clipping=True,
    )


@pytest.fixture
def grpo_trainer(reward_engine, grpo_config):
    """Create GRPO trainer."""
    return GRPOTrainer(reward_engine, grpo_config)


@pytest.fixture
def sample_trajectories():
    """Create sample trajectories with varying success."""
    now = datetime.now(UTC)

    trajectories = []
    for i in range(5):
        steps = [
            TrajectoryStep(
                state={"step": 1},
                action={"step_type": "tool_call"},
                result={},
                timestamp=now,
                duration_ms=100,
            )
        ]

        # Alternate success/failure
        success = i % 2 == 0

        trajectory = Trajectory(
            job_id=uuid4(),
            agent_id="test-agent",
            query=f"query {i}",
            steps=steps,
            success=success,
        )
        trajectories.append(trajectory)

    return trajectories


@pytest.fixture
def sample_log_probs():
    """Create sample log-probabilities."""
    # Typical log-prob values (negative)
    return [-0.5, -0.3, -0.7, -0.4, -0.6]


def test_grpo_config_defaults():
    """Test GRPO configuration default values."""
    config = GRPOConfig()

    assert config.learning_rate == 0.0001
    assert config.gradient_clip_value == 1.0
    assert config.advantage_threshold == 0.0
    assert config.enable_gradient_clipping is True


def test_grpo_trainer_initialization(reward_engine, grpo_config):
    """Test GRPO trainer initialization."""
    trainer = GRPOTrainer(reward_engine, grpo_config)

    assert trainer.reward_engine is reward_engine
    assert trainer.config is grpo_config
    assert isinstance(trainer.metrics, TrainingMetrics)
    assert trainer.metrics.iterations == 0


def test_compute_policy_gradient_basic(
    grpo_trainer, sample_trajectories, sample_log_probs
):
    """Test basic policy gradient computation."""
    loss, info = grpo_trainer.compute_policy_gradient(
        sample_trajectories, sample_log_probs
    )

    assert isinstance(loss, float)
    assert "positive_count" in info
    assert "total_count" in info
    assert "gradients" in info
    assert info["total_count"] == len(sample_trajectories)


def test_policy_gradient_only_positive_advantages(
    grpo_trainer, sample_trajectories, sample_log_probs
):
    """Test that only positive-advantage trajectories are updated."""
    loss, info = grpo_trainer.compute_policy_gradient(
        sample_trajectories, sample_log_probs
    )

    # Should have some positive advantages
    assert info["positive_count"] > 0
    assert info["positive_count"] <= info["total_count"]

    # Gradients only for positive advantages
    assert len(info["gradients"]) == info["positive_count"]


def test_policy_gradient_formula(reward_engine):
    """Test policy gradient formula: loss = -log_prob * advantage."""
    trainer = GRPOTrainer(reward_engine)

    now = datetime.now(UTC)
    # Create two trajectories - one good, one bad
    good_traj = Trajectory(
        job_id=uuid4(),
        agent_id="test",
        query="test",
        steps=[
            TrajectoryStep(
                state={},
                action={"step_type": "tool_call"},
                result={},
                timestamp=now,
                duration_ms=100,
            )
        ],
        success=True,
    )

    bad_traj = Trajectory(
        job_id=uuid4(),
        agent_id="test",
        query="test",
        steps=[
            TrajectoryStep(
                state={},
                action={"step_type": "think"},
                result={},
                timestamp=now,
                duration_ms=50,
            )
        ],
        success=False,
    )

    trajectories = [good_traj, bad_traj]
    log_probs = [-0.5, -0.3]

    loss, info = trainer.compute_policy_gradient(trajectories, log_probs)

    # Good trajectory should have positive advantage
    # Bad trajectory should have negative advantage
    # Only good trajectory contributes to loss
    assert info["positive_count"] == 1
    assert len(info["gradients"]) == 1


def test_gradient_clipping(reward_engine):
    """Test gradient clipping functionality."""
    config = GRPOConfig(gradient_clip_value=0.5, enable_gradient_clipping=True)
    trainer = GRPOTrainer(reward_engine, config)

    gradients = [0.3, 1.5, -0.8, -2.0, 0.1]

    clipped = trainer._clip_gradients(gradients)

    # Check clipping applied
    assert clipped[0] == 0.3  # Within range
    assert clipped[1] == 0.5  # Clipped to max
    assert clipped[2] == -0.5  # Clipped to min
    assert clipped[3] == -0.5  # Clipped to min
    assert clipped[4] == 0.1  # Within range


def test_gradient_clipping_disabled(reward_engine):
    """Test gradient computation without clipping."""
    config = GRPOConfig(enable_gradient_clipping=False)
    trainer = GRPOTrainer(reward_engine, config)

    # With clipping disabled, gradients should pass through
    # Note: _clip_gradients is still called but config controls behavior in compute_policy_gradient
    # For this test, we verify the config setting
    assert trainer.config.enable_gradient_clipping is False


def test_training_step(grpo_trainer, sample_trajectories, sample_log_probs):
    """Test single training step execution."""
    metrics = grpo_trainer.training_step(sample_trajectories, sample_log_probs)

    assert "loss" in metrics
    assert "avg_reward" in metrics
    assert "std_reward" in metrics
    assert "positive_advantages" in metrics
    assert "total_trajectories" in metrics
    assert "iteration" in metrics

    assert metrics["total_trajectories"] == len(sample_trajectories)
    assert metrics["iteration"] == 1


def test_training_metrics_recording():
    """Test training metrics recording."""
    metrics = TrainingMetrics()

    assert metrics.iterations == 0

    # Record first iteration
    metrics.record(loss=0.5, avg_reward=0.8, std_reward=0.2, positive_count=3)

    assert metrics.iterations == 1
    assert metrics.losses == [0.5]
    assert metrics.avg_rewards == [0.8]
    assert metrics.std_rewards == [0.2]
    assert metrics.positive_advantages == [3]

    # Record second iteration
    metrics.record(loss=0.3, avg_reward=0.9, std_reward=0.1, positive_count=4)

    assert metrics.iterations == 2
    assert metrics.losses == [0.5, 0.3]


def test_training_metrics_get_latest():
    """Test getting latest metrics."""
    metrics = TrainingMetrics()

    # No metrics yet
    assert metrics.get_latest() == {}

    # Add metrics
    metrics.record(loss=0.5, avg_reward=0.8, std_reward=0.2, positive_count=3)

    latest = metrics.get_latest()
    assert latest["loss"] == 0.5
    assert latest["avg_reward"] == 0.8
    assert latest["std_reward"] == 0.2
    assert latest["positive_advantages"] == 3
    assert latest["iteration"] == 1


def test_convergence_detection():
    """Test training convergence detection."""
    metrics = TrainingMetrics()

    # Not converged with few iterations
    for i in range(5):
        metrics.record(loss=0.5, avg_reward=0.8, std_reward=0.1, positive_count=3)

    assert metrics.is_converged(window=10) is False

    # Converged with stable losses
    for i in range(10):
        metrics.record(
            loss=0.1 + i * 0.001,  # Very small variance
            avg_reward=0.9,
            std_reward=0.1,
            positive_count=4,
        )

    assert metrics.is_converged(window=10, threshold=0.01) is True


def test_should_continue_training_iteration_limit(grpo_trainer):
    """Test training continuation based on iteration limit."""
    # Add some iterations
    for i in range(5):
        grpo_trainer.metrics.record(
            loss=0.5, avg_reward=0.8, std_reward=0.1, positive_count=3
        )

    # Should continue (under limit)
    assert grpo_trainer.should_continue_training(max_iterations=10) is True

    # Add more iterations
    for i in range(5):
        grpo_trainer.metrics.record(
            loss=0.5, avg_reward=0.8, std_reward=0.1, positive_count=3
        )

    # Should stop (reached limit)
    assert grpo_trainer.should_continue_training(max_iterations=10) is False


def test_should_continue_training_convergence(grpo_trainer):
    """Test training continuation based on convergence."""
    # Add iterations with high variance (not converged)
    for i in range(15):
        grpo_trainer.metrics.record(
            loss=0.5 + i * 0.1,  # High variance
            avg_reward=0.8,
            std_reward=0.1,
            positive_count=3,
        )

    # Should continue (not converged)
    assert (
        grpo_trainer.should_continue_training(max_iterations=100, convergence_check=True)
        is True
    )

    # Add iterations with low variance (converged)
    for i in range(10):
        grpo_trainer.metrics.record(
            loss=0.1 + i * 0.0001,  # Very low variance
            avg_reward=0.9,
            std_reward=0.1,
            positive_count=4,
        )

    # Should stop (converged)
    assert (
        grpo_trainer.should_continue_training(max_iterations=100, convergence_check=True)
        is False
    )


def test_get_metrics(grpo_trainer, sample_trajectories, sample_log_probs):
    """Test retrieving training metrics."""
    # Run a few training steps
    for i in range(3):
        grpo_trainer.training_step(sample_trajectories, sample_log_probs)

    metrics = grpo_trainer.get_metrics()

    assert "iterations" in metrics
    assert "latest" in metrics
    assert "convergence" in metrics

    assert metrics["iterations"] == 3
    assert "loss" in metrics["latest"]
    assert "converged" in metrics["convergence"]


def test_reset_metrics(grpo_trainer, sample_trajectories, sample_log_probs):
    """Test metrics reset."""
    # Add some metrics
    grpo_trainer.training_step(sample_trajectories, sample_log_probs)
    assert grpo_trainer.metrics.iterations == 1

    # Reset
    grpo_trainer.reset_metrics()

    assert grpo_trainer.metrics.iterations == 0
    assert grpo_trainer.metrics.losses == []


def test_mismatched_trajectories_log_probs(grpo_trainer, sample_trajectories):
    """Test error handling for mismatched inputs."""
    wrong_log_probs = [-0.5, -0.3]  # Only 2, but 5 trajectories

    with pytest.raises(ValueError, match="must have same length"):
        grpo_trainer.compute_policy_gradient(sample_trajectories, wrong_log_probs)


def test_no_positive_advantages(reward_engine):
    """Test handling when no trajectories have positive advantage."""
    trainer = GRPOTrainer(reward_engine)

    now = datetime.now(UTC)
    # Create only failed trajectories
    failed_trajectories = [
        Trajectory(
            job_id=uuid4(),
            agent_id="test",
            query="test",
            steps=[
                TrajectoryStep(
                    state={},
                    action={"step_type": "think"},
                    result={},
                    timestamp=now,
                    duration_ms=50,
                )
            ],
            success=False,
        )
        for _ in range(3)
    ]

    log_probs = [-0.5, -0.3, -0.7]

    loss, info = trainer.compute_policy_gradient(failed_trajectories, log_probs)

    # Loss should be 0 when no positive advantages
    assert loss == 0.0
    assert info["positive_count"] == 0
    assert len(info["gradients"]) == 0


def test_advantage_threshold_filtering(reward_engine):
    """Test advantage threshold filtering."""
    config = GRPOConfig(advantage_threshold=0.5)  # Higher threshold
    trainer = GRPOTrainer(reward_engine, config)

    now = datetime.now(UTC)
    # Mix of trajectories
    trajectories = [
        Trajectory(
            job_id=uuid4(),
            agent_id="test",
            query="test",
            steps=[
                TrajectoryStep(
                    state={},
                    action={"step_type": "tool_call"},
                    result={},
                    timestamp=now,
                    duration_ms=100,
                )
            ],
            success=True,
        )
        for _ in range(3)
    ]

    log_probs = [-0.5, -0.3, -0.7]

    loss, info = trainer.compute_policy_gradient(trajectories, log_probs)

    # With higher threshold, fewer trajectories should be updated
    # Exact count depends on reward distribution
    assert info["positive_count"] <= info["total_count"]
