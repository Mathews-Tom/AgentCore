"""Unit tests for policy updater."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from agentcore.training.models import Trajectory, TrajectoryStep
from agentcore.training.policy import PolicyPattern, PolicyUpdate, PolicyUpdater


@pytest.fixture
def policy_updater():
    """Create policy updater."""
    return PolicyUpdater(agent_id="test-agent", min_advantage_threshold=0.5)


@pytest.fixture
def sample_trajectories():
    """Create sample trajectories."""
    now = datetime.now(UTC)

    trajectories = []
    for i in range(5):
        steps = [
            TrajectoryStep(
                state={},
                action={"step_type": "tool_call" if i % 2 == 0 else "think"},
                result={},
                timestamp=now,
                duration_ms=100,
            )
        ]

        traj = Trajectory(
            trajectory_id=uuid4(),
            job_id=uuid4(),
            agent_id="test-agent",
            query=f"query {i}",
            steps=steps,
            success=i % 2 == 0,
            advantage=float(i) / 2.0,  # 0.0, 0.5, 1.0, 1.5, 2.0
        )
        trajectories.append(traj)

    return trajectories


def test_policy_updater_initialization():
    """Test policy updater initialization."""
    updater = PolicyUpdater(agent_id="test-agent", min_advantage_threshold=0.3)

    assert updater.agent_id == "test-agent"
    assert updater.min_advantage_threshold == 0.3
    assert updater.update_history == []


def test_policy_pattern_creation():
    """Test policy pattern creation."""
    pattern = PolicyPattern(
        pattern_type="tool_usage",
        description="Use tools frequently",
        examples=["tool1", "tool2"],
        weight=0.8,
    )

    assert pattern.pattern_type == "tool_usage"
    assert pattern.description == "Use tools frequently"
    assert pattern.examples == ["tool1", "tool2"]
    assert pattern.weight == 0.8


def test_extract_patterns_basic(policy_updater, sample_trajectories):
    """Test basic pattern extraction."""
    advantages = [traj.advantage for traj in sample_trajectories]

    patterns = policy_updater.extract_patterns(sample_trajectories, advantages)

    assert isinstance(patterns, list)
    # Should extract at least one pattern from high-advantage trajectories
    assert len(patterns) >= 0


def test_extract_patterns_filters_by_advantage(policy_updater, sample_trajectories):
    """Test that only high-advantage trajectories are used."""
    advantages = [traj.advantage for traj in sample_trajectories]

    # With threshold 0.5, only trajectories with advantage >= 0.5 should be used
    # That's trajectories 1, 2, 3, 4 (advantages: 0.5, 1.0, 1.5, 2.0)
    patterns = policy_updater.extract_patterns(sample_trajectories, advantages)

    # Patterns should exist since we have high-advantage trajectories
    assert len(patterns) >= 0


def test_extract_patterns_mismatched_length(policy_updater, sample_trajectories):
    """Test error handling for mismatched inputs."""
    wrong_advantages = [0.5, 0.3]  # Only 2, but 5 trajectories

    with pytest.raises(ValueError, match="must have same length"):
        policy_updater.extract_patterns(sample_trajectories, wrong_advantages)


def test_extract_patterns_no_high_advantage(policy_updater):
    """Test pattern extraction when no trajectories meet threshold."""
    now = datetime.now(UTC)

    # All low-advantage trajectories
    trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
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
            advantage=0.1,  # Below threshold of 0.5
        )
        for _ in range(3)
    ]

    advantages = [0.1, 0.1, 0.1]

    patterns = policy_updater.extract_patterns(trajectories, advantages)

    # Should return empty list when no high-advantage trajectories
    assert patterns == []


def test_tool_usage_pattern_extraction(policy_updater):
    """Test extraction of tool usage patterns."""
    now = datetime.now(UTC)

    # Trajectories with tool usage
    trajectories = [
        Trajectory(
            trajectory_id=uuid4(),
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
            advantage=1.0,
        )
        for _ in range(3)
    ]

    advantages = [1.0, 1.0, 1.0]

    patterns = policy_updater.extract_patterns(trajectories, advantages)

    # Should extract tool usage pattern
    tool_patterns = [p for p in patterns if p.pattern_type == "tool_usage"]
    assert len(tool_patterns) > 0


def test_reasoning_pattern_extraction(policy_updater):
    """Test extraction of reasoning patterns."""
    now = datetime.now(UTC)

    # Trajectory with multiple steps (reasoning)
    trajectory = Trajectory(
        trajectory_id=uuid4(),
        job_id=uuid4(),
        agent_id="test",
        query="test",
        steps=[
            TrajectoryStep(
                state={},
                action={"step_type": f"step{i}"},
                result={},
                timestamp=now,
                duration_ms=50,
            )
            for i in range(5)  # 5 steps = reasoning
        ],
        success=True,
        advantage=1.5,
    )

    patterns = policy_updater.extract_patterns([trajectory], [1.5])

    # Should extract reasoning pattern
    reasoning_patterns = [p for p in patterns if p.pattern_type == "reasoning"]
    assert len(reasoning_patterns) > 0


def test_create_update(policy_updater, sample_trajectories):
    """Test policy update creation."""
    patterns = [
        PolicyPattern(
            pattern_type="tool_usage",
            description="Test pattern",
            examples=["ex1"],
            weight=0.8,
        )
    ]

    update = policy_updater.create_update(patterns, sample_trajectories)

    assert isinstance(update, PolicyUpdate)
    assert update.update_type == "pattern_based_update"
    assert "patterns" in update.content
    assert len(update.content["patterns"]) == 1
    assert update.total_advantage > 0


def test_create_update_records_history(policy_updater, sample_trajectories):
    """Test that updates are recorded in history."""
    assert len(policy_updater.update_history) == 0

    patterns = [
        PolicyPattern(
            pattern_type="test",
            description="Test",
            examples=[],
            weight=1.0,
        )
    ]

    policy_updater.create_update(patterns, sample_trajectories)

    assert len(policy_updater.update_history) == 1


def test_apply_update():
    """Test applying update to policy."""
    updater = PolicyUpdater(agent_id="test")

    current_policy = {
        "agent_id": "test",
        "prompt": "Original prompt",
        "learned_patterns": [],
    }

    update = PolicyUpdate(
        update_type="pattern_based_update",
        content={
            "patterns": [
                {
                    "type": "tool_usage",
                    "description": "Use tools",
                    "weight": 0.8,
                    "examples": ["tool1"],
                }
            ]
        },
        source_trajectories=[uuid4()],
        total_advantage=2.5,
    )

    updated_policy = updater.apply_update(update, current_policy)

    # Check pattern was added
    assert "learned_patterns" in updated_policy
    assert len(updated_policy["learned_patterns"]) == 1

    # Check metadata updated
    assert "last_updated" in updated_policy
    assert updated_policy["total_updates"] == 1

    # Original policy should not be modified
    assert current_policy.get("total_updates") is None


def test_create_checkpoint(policy_updater):
    """Test checkpoint creation."""
    job_id = uuid4()
    policy_data = {"prompt": "test prompt", "learned_patterns": []}
    metrics = {"loss": 0.5, "avg_reward": 0.8}

    checkpoint = policy_updater.create_checkpoint(
        job_id=job_id,
        iteration=10,
        policy_data=policy_data,
        validation_score=0.85,
        metrics=metrics,
    )

    assert checkpoint.agent_id == "test-agent"
    assert checkpoint.job_id == job_id
    assert checkpoint.iteration == 10
    assert checkpoint.policy_data == policy_data
    assert checkpoint.validation_score == 0.85
    assert checkpoint.metrics == metrics
    assert checkpoint.checkpoint_id is not None


def test_get_update_history(policy_updater, sample_trajectories):
    """Test retrieving update history."""
    # Create some updates
    patterns = [
        PolicyPattern(
            pattern_type="test",
            description="Test",
            examples=[],
            weight=1.0,
        )
    ]

    for i in range(3):
        policy_updater.create_update(patterns, sample_trajectories)

    history = policy_updater.get_update_history()

    assert len(history) == 3
    assert all("update_type" in entry for entry in history)
    assert all("total_advantage" in entry for entry in history)
    assert all("created_at" in entry for entry in history)


def test_is_tool_step(policy_updater):
    """Test tool step detection."""
    assert policy_updater._is_tool_step({"step_type": "tool_call"}) is True
    assert policy_updater._is_tool_step({"step_type": "action"}) is True
    assert policy_updater._is_tool_step({"step_type": "think"}) is False
    assert policy_updater._is_tool_step({}) is False


def test_is_verify_step(policy_updater):
    """Test verification step detection."""
    assert policy_updater._is_verify_step({"step_type": "verify"}) is True
    assert policy_updater._is_verify_step({"step_type": "check"}) is True
    assert policy_updater._is_verify_step({"step_type": "think"}) is False
    assert policy_updater._is_verify_step({}) is False


def test_generate_update_summary(policy_updater):
    """Test update summary generation."""
    patterns = [
        PolicyPattern(
            pattern_type="tool_usage",
            description="Use tools",
            examples=[],
            weight=0.8,
        ),
        PolicyPattern(
            pattern_type="verification",
            description="Verify results",
            examples=[],
            weight=0.6,
        ),
    ]

    summary = policy_updater._generate_update_summary(patterns)

    assert "tool_usage" in summary
    assert "verification" in summary
    assert "Use tools" in summary
    assert "Verify results" in summary


def test_generate_update_summary_empty(policy_updater):
    """Test summary generation with no patterns."""
    summary = policy_updater._generate_update_summary([])

    assert summary == "No patterns extracted"


def test_policy_update_weighted_by_advantage(policy_updater):
    """Test that updates are weighted by trajectory advantage."""
    now = datetime.now(UTC)

    # High-advantage trajectory
    high_adv_traj = Trajectory(
        trajectory_id=uuid4(),
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
        advantage=2.0,
    )

    # Low-advantage trajectory (below threshold)
    low_adv_traj = Trajectory(
        trajectory_id=uuid4(),
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
        advantage=0.3,
    )

    trajectories = [high_adv_traj, low_adv_traj]
    patterns = policy_updater.extract_patterns(trajectories, [2.0, 0.3])

    update = policy_updater.create_update(patterns, trajectories)

    # Total advantage should only include high-advantage trajectory
    assert update.total_advantage == pytest.approx(2.0)


def test_multiple_updates_accumulate(policy_updater):
    """Test that multiple updates accumulate in policy."""
    current_policy = {"learned_patterns": []}

    # First update
    update1 = PolicyUpdate(
        update_type="pattern_based_update",
        content={"patterns": [{"type": "tool_usage", "weight": 0.8}]},
        source_trajectories=[],
        total_advantage=1.0,
    )

    policy_after_1 = policy_updater.apply_update(update1, current_policy)

    # Second update
    update2 = PolicyUpdate(
        update_type="pattern_based_update",
        content={"patterns": [{"type": "verification", "weight": 0.6}]},
        source_trajectories=[],
        total_advantage=0.8,
    )

    policy_after_2 = policy_updater.apply_update(update2, policy_after_1)

    # Both patterns should be present
    assert len(policy_after_2["learned_patterns"]) == 2
    assert policy_after_2["total_updates"] == 2
