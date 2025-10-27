"""
Unit tests for checkpoint manager.

Tests save/restore checkpoints, best-checkpoint selection, and automatic cleanup.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from agentcore.training.checkpoint import CheckpointManager


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir: Path) -> CheckpointManager:
    """Create checkpoint manager instance."""
    return CheckpointManager(
        checkpoint_interval=10,
        max_checkpoints=5,
        storage_path=temp_checkpoint_dir)


# Test initialization


def test_checkpoint_manager_initialization(temp_checkpoint_dir: Path) -> None:
    """Test CheckpointManager initialization."""
    manager = CheckpointManager(
        checkpoint_interval=5,
        max_checkpoints=3,
        storage_path=temp_checkpoint_dir)

    assert manager.checkpoint_interval == 5
    assert manager.max_checkpoints == 3
    assert manager.storage_path == temp_checkpoint_dir
    assert temp_checkpoint_dir.exists()


def test_checkpoint_manager_default_storage_path() -> None:
    """Test CheckpointManager with default storage path."""
    manager = CheckpointManager()

    assert manager.checkpoint_interval == 10
    assert manager.max_checkpoints == 5
    assert manager.storage_path == Path(".checkpoints")


# Test checkpoint saving


def test_should_save_checkpoint(checkpoint_manager: CheckpointManager) -> None:
    """Test should_save_checkpoint logic."""
    assert not checkpoint_manager.should_save_checkpoint(0)
    assert not checkpoint_manager.should_save_checkpoint(5)
    assert checkpoint_manager.should_save_checkpoint(10)
    assert not checkpoint_manager.should_save_checkpoint(15)
    assert checkpoint_manager.should_save_checkpoint(20)


def test_should_save_checkpoint_custom_interval() -> None:
    """Test should_save_checkpoint with custom interval."""
    manager = CheckpointManager(checkpoint_interval=5)

    assert manager.should_save_checkpoint(5)
    assert manager.should_save_checkpoint(10)
    assert manager.should_save_checkpoint(15)
    assert not manager.should_save_checkpoint(7)


def test_save_checkpoint(checkpoint_manager: CheckpointManager) -> None:
    """Test checkpoint saving."""
    agent_id = "test-agent"
    job_id = uuid4()
    iteration = 10
    policy_data = {"prompt": "test prompt", "config": {"temperature": 0.7}}
    validation_score = 0.85
    metrics = {"loss": 0.15, "accuracy": 0.85}

    checkpoint = checkpoint_manager.save_checkpoint(
        agent_id=agent_id,
        job_id=job_id,
        iteration=iteration,
        policy_data=policy_data,
        validation_score=validation_score,
        metrics=metrics)

    assert checkpoint.checkpoint_id is not None
    assert checkpoint.agent_id == agent_id
    assert checkpoint.job_id == job_id
    assert checkpoint.iteration == iteration
    assert checkpoint.policy_data == policy_data
    assert checkpoint.validation_score == validation_score
    assert checkpoint.metrics == metrics
    assert checkpoint.created_at is not None


def test_save_checkpoint_without_metrics(checkpoint_manager: CheckpointManager) -> None:
    """Test checkpoint saving without metrics."""
    agent_id = "test-agent"
    job_id = uuid4()

    checkpoint = checkpoint_manager.save_checkpoint(
        agent_id=agent_id,
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "test"},
        validation_score=0.75)

    assert checkpoint.metrics == {}


def test_save_checkpoint_persists_to_disk(
    checkpoint_manager: CheckpointManager,
    temp_checkpoint_dir: Path) -> None:
    """Test checkpoint is persisted to disk."""
    job_id = uuid4()

    checkpoint = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "test"},
        validation_score=0.8)

    checkpoint_file = temp_checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
    assert checkpoint_file.exists()


# Test checkpoint loading


def test_load_checkpoint(checkpoint_manager: CheckpointManager) -> None:
    """Test checkpoint loading from memory."""
    job_id = uuid4()

    saved_checkpoint = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "test"},
        validation_score=0.8)

    loaded_checkpoint = checkpoint_manager.load_checkpoint(saved_checkpoint.checkpoint_id)  # type: ignore

    assert loaded_checkpoint is not None
    assert loaded_checkpoint.checkpoint_id == saved_checkpoint.checkpoint_id
    assert loaded_checkpoint.agent_id == saved_checkpoint.agent_id
    assert loaded_checkpoint.iteration == saved_checkpoint.iteration


def test_load_checkpoint_from_disk(
    checkpoint_manager: CheckpointManager,
    temp_checkpoint_dir: Path) -> None:
    """Test checkpoint loading from disk."""
    job_id = uuid4()

    saved_checkpoint = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "test"},
        validation_score=0.8)

    # Clear memory cache
    checkpoint_manager._checkpoints.clear()
    checkpoint_manager._checkpoint_order.clear()

    # Load from disk
    loaded_checkpoint = checkpoint_manager.load_checkpoint(saved_checkpoint.checkpoint_id)  # type: ignore

    assert loaded_checkpoint is not None
    assert loaded_checkpoint.checkpoint_id == saved_checkpoint.checkpoint_id
    assert loaded_checkpoint.iteration == saved_checkpoint.iteration


def test_load_checkpoint_not_found(checkpoint_manager: CheckpointManager) -> None:
    """Test loading non-existent checkpoint returns None."""
    nonexistent_id = uuid4()

    loaded_checkpoint = checkpoint_manager.load_checkpoint(nonexistent_id)

    assert loaded_checkpoint is None


# Test best checkpoint selection


def test_get_best_checkpoint(checkpoint_manager: CheckpointManager) -> None:
    """Test best checkpoint selection by validation score."""
    job_id = uuid4()

    # Save multiple checkpoints with different scores
    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "v1"},
        validation_score=0.7)

    best_checkpoint = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=20,
        policy_data={"prompt": "v2"},
        validation_score=0.9,  # Best
    )

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=30,
        policy_data={"prompt": "v3"},
        validation_score=0.75)

    loaded_best = checkpoint_manager.get_best_checkpoint(job_id)

    assert loaded_best is not None
    assert loaded_best.checkpoint_id == best_checkpoint.checkpoint_id
    assert loaded_best.validation_score == 0.9


def test_get_best_checkpoint_no_checkpoints(checkpoint_manager: CheckpointManager) -> None:
    """Test get_best_checkpoint with no checkpoints returns None."""
    job_id = uuid4()

    best_checkpoint = checkpoint_manager.get_best_checkpoint(job_id)

    assert best_checkpoint is None


# Test checkpoint retrieval


def test_get_checkpoints_for_job(checkpoint_manager: CheckpointManager) -> None:
    """Test retrieving all checkpoints for a job."""
    job_id = uuid4()
    other_job_id = uuid4()

    # Save checkpoints for job
    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "v1"},
        validation_score=0.7)

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=20,
        policy_data={"prompt": "v2"},
        validation_score=0.8)

    # Save checkpoint for different job
    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=other_job_id,
        iteration=10,
        policy_data={"prompt": "other"},
        validation_score=0.6)

    job_checkpoints = checkpoint_manager.get_checkpoints_for_job(job_id)

    assert len(job_checkpoints) == 2
    assert job_checkpoints[0].iteration == 10
    assert job_checkpoints[1].iteration == 20


def test_get_latest_checkpoint(checkpoint_manager: CheckpointManager) -> None:
    """Test getting latest checkpoint by iteration."""
    job_id = uuid4()

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "v1"},
        validation_score=0.7)

    latest_checkpoint = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=30,
        policy_data={"prompt": "v3"},
        validation_score=0.8)

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=20,
        policy_data={"prompt": "v2"},
        validation_score=0.75)

    loaded_latest = checkpoint_manager.get_latest_checkpoint(job_id)

    assert loaded_latest is not None
    assert loaded_latest.checkpoint_id == latest_checkpoint.checkpoint_id
    assert loaded_latest.iteration == 30


def test_get_latest_checkpoint_no_checkpoints(checkpoint_manager: CheckpointManager) -> None:
    """Test get_latest_checkpoint with no checkpoints returns None."""
    job_id = uuid4()

    latest_checkpoint = checkpoint_manager.get_latest_checkpoint(job_id)

    assert latest_checkpoint is None


# Test checkpoint cleanup


def test_automatic_cleanup(checkpoint_manager: CheckpointManager) -> None:
    """Test automatic cleanup keeps best N checkpoints."""
    job_id = uuid4()

    # Save 7 checkpoints (exceeds max of 5)
    scores = [0.6, 0.7, 0.5, 0.9, 0.8, 0.75, 0.65]

    for i, score in enumerate(scores):
        checkpoint_manager.save_checkpoint(
            agent_id="test-agent",
            job_id=job_id,
            iteration=(i + 1) * 10,
            policy_data={"prompt": f"v{i+1}"},
            validation_score=score)

    # Should keep best 5: 0.9, 0.8, 0.75, 0.7, 0.65
    remaining_checkpoints = checkpoint_manager.get_checkpoints_for_job(job_id)

    assert len(remaining_checkpoints) == 5

    remaining_scores = [cp.validation_score for cp in remaining_checkpoints]
    assert 0.9 in remaining_scores
    assert 0.8 in remaining_scores
    assert 0.75 in remaining_scores
    assert 0.7 in remaining_scores
    assert 0.65 in remaining_scores
    assert 0.6 not in remaining_scores
    assert 0.5 not in remaining_scores


def test_cleanup_respects_max_checkpoints(temp_checkpoint_dir: Path) -> None:
    """Test cleanup with custom max_checkpoints."""
    manager = CheckpointManager(
        checkpoint_interval=10,
        max_checkpoints=3,
        storage_path=temp_checkpoint_dir)

    job_id = uuid4()

    # Save 5 checkpoints
    for i in range(5):
        manager.save_checkpoint(
            agent_id="test-agent",
            job_id=job_id,
            iteration=(i + 1) * 10,
            policy_data={"prompt": f"v{i+1}"},
            validation_score=0.5 + i * 0.1)

    remaining_checkpoints = manager.get_checkpoints_for_job(job_id)

    assert len(remaining_checkpoints) == 3


# Test checkpoint resumption


def test_resume_from_checkpoint(checkpoint_manager: CheckpointManager) -> None:
    """Test resuming training from checkpoint."""
    job_id = uuid4()
    policy_data = {"prompt": "resume test", "config": {"temp": 0.8}}
    metrics = {"loss": 0.2, "accuracy": 0.8}

    checkpoint = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=50,
        policy_data=policy_data,
        validation_score=0.85,
        metrics=metrics)

    iteration, loaded_policy, loaded_metrics = checkpoint_manager.resume_from_checkpoint(
        checkpoint.checkpoint_id  # type: ignore
    )

    assert iteration == 50
    assert loaded_policy == policy_data
    assert loaded_metrics == metrics


def test_resume_from_checkpoint_not_found(checkpoint_manager: CheckpointManager) -> None:
    """Test resuming from non-existent checkpoint raises ValueError."""
    nonexistent_id = uuid4()

    with pytest.raises(ValueError, match="Checkpoint .* not found"):
        checkpoint_manager.resume_from_checkpoint(nonexistent_id)


# Test checkpoint management


def test_get_checkpoint_count(checkpoint_manager: CheckpointManager) -> None:
    """Test getting checkpoint count for a job."""
    job_id = uuid4()

    assert checkpoint_manager.get_checkpoint_count(job_id) == 0

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "v1"},
        validation_score=0.7)

    assert checkpoint_manager.get_checkpoint_count(job_id) == 1

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=20,
        policy_data={"prompt": "v2"},
        validation_score=0.8)

    assert checkpoint_manager.get_checkpoint_count(job_id) == 2


def test_clear_checkpoints(
    checkpoint_manager: CheckpointManager,
    temp_checkpoint_dir: Path) -> None:
    """Test clearing all checkpoints for a job."""
    job_id = uuid4()

    # Save checkpoints
    checkpoint_1 = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "v1"},
        validation_score=0.7)

    checkpoint_2 = checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id,
        iteration=20,
        policy_data={"prompt": "v2"},
        validation_score=0.8)

    # Clear checkpoints
    checkpoint_manager.clear_checkpoints(job_id)

    assert checkpoint_manager.get_checkpoint_count(job_id) == 0
    assert checkpoint_manager.load_checkpoint(checkpoint_1.checkpoint_id) is None  # type: ignore
    assert checkpoint_manager.load_checkpoint(checkpoint_2.checkpoint_id) is None  # type: ignore

    # Verify disk cleanup
    assert not (temp_checkpoint_dir / f"{checkpoint_1.checkpoint_id}.json").exists()
    assert not (temp_checkpoint_dir / f"{checkpoint_2.checkpoint_id}.json").exists()


def test_clear_checkpoints_does_not_affect_other_jobs(
    checkpoint_manager: CheckpointManager) -> None:
    """Test clearing checkpoints for one job doesn't affect others."""
    job_id_1 = uuid4()
    job_id_2 = uuid4()

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id_1,
        iteration=10,
        policy_data={"prompt": "v1"},
        validation_score=0.7)

    checkpoint_manager.save_checkpoint(
        agent_id="test-agent",
        job_id=job_id_2,
        iteration=10,
        policy_data={"prompt": "v2"},
        validation_score=0.8)

    # Clear job_id_1 checkpoints
    checkpoint_manager.clear_checkpoints(job_id_1)

    assert checkpoint_manager.get_checkpoint_count(job_id_1) == 0
    assert checkpoint_manager.get_checkpoint_count(job_id_2) == 1
