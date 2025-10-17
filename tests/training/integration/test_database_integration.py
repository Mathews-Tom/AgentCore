"""Integration tests for training database operations."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import select

from agentcore.a2a_protocol.database.connection import get_session
from agentcore.training.database_models import (
    PolicyCheckpointDB,
    TrainingJobDB,
    TrajectoryDB,
)
from agentcore.training.models import GRPOConfig, TrainingQuery
from agentcore.training.repositories import (
    CheckpointRepository,
    TrainingJobRepository,
    TrajectoryRepository,
)


@pytest.mark.asyncio
async def test_training_job_crud_operations(init_test_db) -> None:
    """Test CRUD operations for training jobs."""
    # Create training job
    config = GRPOConfig()
    queries = [
        TrainingQuery(query=f"Query {i}", expected_outcome={"answer": f"Answer {i}"})
        for i in range(100)
    ]

    from agentcore.training.models import TrainingJob

    job = TrainingJob(
        agent_id="test-agent-123",
        config=config,
        training_data=queries,
        total_iterations=100,
        budget_usd=Decimal("50.00"),
    )

    async with get_session() as session:
        # Create
        job_db = await TrainingJobRepository.create(session, job)
        await session.commit()

        assert job_db.job_id is not None
        assert job_db.agent_id == "test-agent-123"
        assert job_db.status == "queued"
        assert job_db.cost_usd == Decimal("0.00")

        job_id = job_db.job_id

    # Read
    async with get_session() as session:
        job_db = await TrainingJobRepository.get_by_id(session, job_id)
        assert job_db is not None
        assert job_db.agent_id == "test-agent-123"

    # Update status
    from agentcore.training.models import TrainingJobStatus

    async with get_session() as session:
        await TrainingJobRepository.update_status(
            session,
            job_id,
            TrainingJobStatus.RUNNING,
            started_at=datetime.now(UTC)(),  # Use naive datetime
        )
        await session.commit()

    async with get_session() as session:
        job_db = await TrainingJobRepository.get_by_id(session, job_id)
        assert job_db.status == "running"
        assert job_db.started_at is not None

    # Delete
    async with get_session() as session:
        await TrainingJobRepository.delete(session, job_id)
        await session.commit()

    async with get_session() as session:
        job_db = await TrainingJobRepository.get_by_id(session, job_id)
        assert job_db is None


@pytest.mark.asyncio
async def test_trajectory_crud_operations(init_test_db) -> None:
    """Test CRUD operations for trajectories."""
    # Setup: Create training job first
    config = GRPOConfig()
    queries = [
        TrainingQuery(query=f"Q{i}", expected_outcome={"a": True}) for i in range(100)
    ]

    from agentcore.training.models import TrainingJob, Trajectory, TrajectoryStep

    job = TrainingJob(
        agent_id="test-agent-456",
        config=config,
        training_data=queries,
        total_iterations=100,
        budget_usd=Decimal("50.00"),
    )

    async with get_session() as session:
        job_db = await TrainingJobRepository.create(session, job)
        await session.commit()
        job_id = job_db.job_id

    # Create trajectory
    step = TrajectoryStep(
        state={"context": "test"},
        action={"type": "tool_call"},
        result={"output": "success"},
        timestamp=datetime.now(UTC)(),
        duration_ms=100,
    )

    trajectory = Trajectory(
        job_id=job_id,
        agent_id="test-agent-456",
        query="What is 2+2?",
        steps=[step],
        reward=0.85,
        success=True,
        execution_time_ms=100,
    )

    async with get_session() as session:
        traj_db = await TrajectoryRepository.create(session, trajectory)
        await session.commit()

        assert traj_db.trajectory_id is not None
        assert traj_db.job_id == job_id
        assert traj_db.reward == 0.85
        assert traj_db.success is True

        traj_id = traj_db.trajectory_id

    # Read by job
    async with get_session() as session:
        trajectories = await TrajectoryRepository.get_by_job(session, job_id)
        assert len(trajectories) == 1
        assert trajectories[0].trajectory_id == traj_id

    # Cleanup
    async with get_session() as session:
        await TrainingJobRepository.delete(session, job_id)
        await session.commit()


@pytest.mark.asyncio
async def test_checkpoint_crud_operations(init_test_db) -> None:
    """Test CRUD operations for policy checkpoints."""
    # Setup: Create training job first
    config = GRPOConfig()
    queries = [
        TrainingQuery(query=f"Q{i}", expected_outcome={"a": True}) for i in range(100)
    ]

    from agentcore.training.models import PolicyCheckpoint, TrainingJob

    job = TrainingJob(
        agent_id="test-agent-789",
        config=config,
        training_data=queries,
        total_iterations=100,
        budget_usd=Decimal("50.00"),
    )

    async with get_session() as session:
        job_db = await TrainingJobRepository.create(session, job)
        await session.commit()
        job_id = job_db.job_id

    # Create checkpoint
    checkpoint = PolicyCheckpoint(
        agent_id="test-agent-789",
        job_id=job_id,
        iteration=10,
        policy_data={"prompt": "You are helpful"},
        validation_score=0.85,
        metrics={"accuracy": 0.9},
    )

    async with get_session() as session:
        cp_db = await CheckpointRepository.create(session, checkpoint)
        await session.commit()

        assert cp_db.checkpoint_id is not None
        assert cp_db.iteration == 10
        assert cp_db.validation_score == 0.85

        checkpoint_id = cp_db.checkpoint_id

    # Get best checkpoint
    async with get_session() as session:
        best_cp = await CheckpointRepository.get_best_by_job(session, job_id)
        assert best_cp is not None
        assert best_cp.checkpoint_id == checkpoint_id

    # Cleanup
    async with get_session() as session:
        await TrainingJobRepository.delete(session, job_id)
        await session.commit()


@pytest.mark.asyncio
async def test_foreign_key_cascade_delete(init_test_db) -> None:
    """Test that deleting training job cascades to trajectories and checkpoints."""
    # Setup
    config = GRPOConfig()
    queries = [
        TrainingQuery(query=f"Q{i}", expected_outcome={"a": True}) for i in range(100)
    ]

    from agentcore.training.models import (
        PolicyCheckpoint,
        TrainingJob,
        Trajectory,
        TrajectoryStep,
    )

    job = TrainingJob(
        agent_id="test-agent-cascade",
        config=config,
        training_data=queries,
        total_iterations=100,
        budget_usd=Decimal("50.00"),
    )

    async with get_session() as session:
        job_db = await TrainingJobRepository.create(session, job)
        await session.commit()
        job_id = job_db.job_id

    # Create trajectory
    step = TrajectoryStep(
        state={},
        action={},
        result={},
        timestamp=datetime.now(UTC)(),
        duration_ms=10,
    )
    trajectory = Trajectory(
        job_id=job_id,
        agent_id="test-agent-cascade",
        query="Test",
        steps=[step],
    )

    async with get_session() as session:
        traj_db = await TrajectoryRepository.create(session, trajectory)
        await session.commit()
        traj_id = traj_db.trajectory_id

    # Create checkpoint
    checkpoint = PolicyCheckpoint(
        agent_id="test-agent-cascade",
        job_id=job_id,
        iteration=5,
        validation_score=0.75,
    )

    async with get_session() as session:
        cp_db = await CheckpointRepository.create(session, checkpoint)
        await session.commit()
        cp_id = cp_db.checkpoint_id

    # Delete training job (should cascade)
    async with get_session() as session:
        await TrainingJobRepository.delete(session, job_id)
        await session.commit()

    # Verify cascade deletion
    async with get_session() as session:
        # Job should be deleted
        job_db = await TrainingJobRepository.get_by_id(session, job_id)
        assert job_db is None

        # Trajectory should be deleted
        traj_db = await TrajectoryRepository.get_by_id(session, traj_id)
        assert traj_db is None

        # Checkpoint should be deleted
        cp_db = await CheckpointRepository.get_by_id(session, cp_id)
        assert cp_db is None
