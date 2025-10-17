"""
Repository pattern for training infrastructure data access.

Provides async database operations for training jobs, trajectories, and checkpoints.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agentcore.training.database_models import (
    PolicyCheckpointDB,
    TrainingJobDB,
    TrajectoryDB,
)
from agentcore.training.models import (
    PolicyCheckpoint,
    TrainingJob,
    TrainingJobStatus,
    Trajectory,
)


class TrainingJobRepository:
    """Repository for training_jobs table operations."""

    @staticmethod
    async def create(session: AsyncSession, training_job: TrainingJob) -> TrainingJobDB:
        """Create new training job in database.

        Args:
            session: Async database session
            training_job: Training job Pydantic model

        Returns:
            TrainingJobDB: Created database record
        """
        job_db = TrainingJobDB(
            agent_id=training_job.agent_id,
            status=training_job.status.value,
            config=training_job.config.model_dump(mode="json"),
            training_data=[
                q.model_dump(mode="json") for q in training_job.training_data
            ],
            current_iteration=training_job.current_iteration,
            total_iterations=training_job.total_iterations,
            metrics=training_job.metrics,
            cost_usd=training_job.cost_usd,
            budget_usd=training_job.budget_usd,
            best_checkpoint_id=training_job.best_checkpoint_id,
        )
        session.add(job_db)
        await session.flush()
        await session.refresh(job_db)
        return job_db

    @staticmethod
    async def get_by_id(session: AsyncSession, job_id: UUID) -> TrainingJobDB | None:
        """Retrieve training job by ID.

        Args:
            session: Async database session
            job_id: Training job UUID

        Returns:
            TrainingJobDB | None: Database record or None if not found
        """
        result = await session.execute(
            select(TrainingJobDB).where(TrainingJobDB.job_id == job_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_agent(
        session: AsyncSession,
        agent_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TrainingJobDB]:
        """Retrieve training jobs for an agent.

        Args:
            session: Async database session
            agent_id: Agent identifier
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            list[TrainingJobDB]: List of training job records
        """
        result = await session.execute(
            select(TrainingJobDB)
            .where(TrainingJobDB.agent_id == agent_id)
            .order_by(TrainingJobDB.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_by_status(
        session: AsyncSession,
        status: TrainingJobStatus,
        limit: int = 100,
    ) -> list[TrainingJobDB]:
        """Retrieve training jobs by status.

        Args:
            session: Async database session
            status: Job status filter
            limit: Maximum number of results

        Returns:
            list[TrainingJobDB]: List of training job records
        """
        result = await session.execute(
            select(TrainingJobDB)
            .where(TrainingJobDB.status == status.value)
            .order_by(TrainingJobDB.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_status(
        session: AsyncSession,
        job_id: UUID,
        status: TrainingJobStatus,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update training job status.

        Args:
            session: Async database session
            job_id: Training job UUID
            status: New status
            started_at: Start timestamp (optional)
            completed_at: Completion timestamp (optional)
            error_message: Error message if failed (optional)
        """
        values = {"status": status.value}
        if started_at is not None:
            values["started_at"] = started_at
        if completed_at is not None:
            values["completed_at"] = completed_at
        if error_message is not None:
            values["error_message"] = error_message

        await session.execute(
            update(TrainingJobDB).where(TrainingJobDB.job_id == job_id).values(**values)
        )
        await session.flush()

    @staticmethod
    async def update_progress(
        session: AsyncSession,
        job_id: UUID,
        current_iteration: int,
        metrics: dict[str, float | int | str],
        cost_usd: Decimal,
    ) -> None:
        """Update training job progress metrics.

        Args:
            session: Async database session
            job_id: Training job UUID
            current_iteration: Current training iteration
            metrics: Updated metrics dictionary
            cost_usd: Accumulated cost
        """
        await session.execute(
            update(TrainingJobDB)
            .where(TrainingJobDB.job_id == job_id)
            .values(
                current_iteration=current_iteration,
                metrics=metrics,
                cost_usd=cost_usd,
            )
        )
        await session.flush()

    @staticmethod
    async def update_best_checkpoint(
        session: AsyncSession,
        job_id: UUID,
        checkpoint_id: UUID,
    ) -> None:
        """Update best checkpoint reference.

        Args:
            session: Async database session
            job_id: Training job UUID
            checkpoint_id: Best checkpoint UUID
        """
        await session.execute(
            update(TrainingJobDB)
            .where(TrainingJobDB.job_id == job_id)
            .values(best_checkpoint_id=checkpoint_id)
        )
        await session.flush()

    @staticmethod
    async def delete(session: AsyncSession, job_id: UUID) -> None:
        """Delete training job (cascade deletes trajectories and checkpoints).

        Args:
            session: Async database session
            job_id: Training job UUID
        """
        await session.execute(
            delete(TrainingJobDB).where(TrainingJobDB.job_id == job_id)
        )
        await session.flush()


class TrajectoryRepository:
    """Repository for trajectories table operations."""

    @staticmethod
    async def create(session: AsyncSession, trajectory: Trajectory) -> TrajectoryDB:
        """Create new trajectory in database.

        Args:
            session: Async database session
            trajectory: Trajectory Pydantic model

        Returns:
            TrajectoryDB: Created database record
        """
        trajectory_db = TrajectoryDB(
            job_id=trajectory.job_id,
            agent_id=trajectory.agent_id,
            query=trajectory.query,
            steps=[step.model_dump(mode="json") for step in trajectory.steps],
            reward=trajectory.reward,
            normalized_reward=trajectory.normalized_reward,
            advantage=trajectory.advantage,
            execution_time_ms=trajectory.execution_time_ms,
            success=trajectory.success,
        )
        session.add(trajectory_db)
        await session.flush()
        await session.refresh(trajectory_db)
        return trajectory_db

    @staticmethod
    async def bulk_create(
        session: AsyncSession, trajectories: list[Trajectory]
    ) -> list[TrajectoryDB]:
        """Bulk create multiple trajectories for performance.

        Args:
            session: Async database session
            trajectories: List of trajectory Pydantic models

        Returns:
            list[TrajectoryDB]: List of created database records
        """
        trajectory_dbs = [
            TrajectoryDB(
                job_id=t.job_id,
                agent_id=t.agent_id,
                query=t.query,
                steps=[step.model_dump(mode="json") for step in t.steps],
                reward=t.reward,
                normalized_reward=t.normalized_reward,
                advantage=t.advantage,
                execution_time_ms=t.execution_time_ms,
                success=t.success,
            )
            for t in trajectories
        ]
        session.add_all(trajectory_dbs)
        await session.flush()
        for tdb in trajectory_dbs:
            await session.refresh(tdb)
        return trajectory_dbs

    @staticmethod
    async def get_by_id(
        session: AsyncSession, trajectory_id: UUID
    ) -> TrajectoryDB | None:
        """Retrieve trajectory by ID.

        Args:
            session: Async database session
            trajectory_id: Trajectory UUID

        Returns:
            TrajectoryDB | None: Database record or None if not found
        """
        result = await session.execute(
            select(TrajectoryDB).where(TrajectoryDB.trajectory_id == trajectory_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_job(
        session: AsyncSession,
        job_id: UUID,
        success_only: bool = False,
        min_reward: float | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[TrajectoryDB]:
        """Retrieve trajectories for a training job.

        Args:
            session: Async database session
            job_id: Training job UUID
            success_only: Filter for successful trajectories only
            min_reward: Minimum reward threshold (optional)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            list[TrajectoryDB]: List of trajectory records
        """
        query = select(TrajectoryDB).where(TrajectoryDB.job_id == job_id)
        if success_only:
            query = query.where(TrajectoryDB.success == True)  # noqa: E712
        if min_reward is not None:
            query = query.where(TrajectoryDB.reward >= min_reward)
        query = (
            query.order_by(TrajectoryDB.created_at.desc()).limit(limit).offset(offset)
        )

        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_by_agent(
        session: AsyncSession,
        agent_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[TrajectoryDB]:
        """Retrieve trajectories for an agent.

        Args:
            session: Async database session
            agent_id: Agent identifier
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            list[TrajectoryDB]: List of trajectory records
        """
        result = await session.execute(
            select(TrajectoryDB)
            .where(TrajectoryDB.agent_id == agent_id)
            .order_by(TrajectoryDB.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update_rewards(
        session: AsyncSession,
        trajectory_id: UUID,
        normalized_reward: float,
        advantage: float,
    ) -> None:
        """Update trajectory reward values after normalization.

        Args:
            session: Async database session
            trajectory_id: Trajectory UUID
            normalized_reward: Normalized reward value
            advantage: Advantage value for policy gradient
        """
        await session.execute(
            update(TrajectoryDB)
            .where(TrajectoryDB.trajectory_id == trajectory_id)
            .values(
                normalized_reward=normalized_reward,
                advantage=advantage,
            )
        )
        await session.flush()

    @staticmethod
    async def delete_by_job(session: AsyncSession, job_id: UUID) -> None:
        """Delete all trajectories for a job.

        Args:
            session: Async database session
            job_id: Training job UUID
        """
        await session.execute(delete(TrajectoryDB).where(TrajectoryDB.job_id == job_id))
        await session.flush()


class CheckpointRepository:
    """Repository for policy_checkpoints table operations."""

    @staticmethod
    async def create(
        session: AsyncSession, checkpoint: PolicyCheckpoint
    ) -> PolicyCheckpointDB:
        """Create new policy checkpoint in database.

        Args:
            session: Async database session
            checkpoint: PolicyCheckpoint Pydantic model

        Returns:
            PolicyCheckpointDB: Created database record
        """
        checkpoint_db = PolicyCheckpointDB(
            agent_id=checkpoint.agent_id,
            job_id=checkpoint.job_id,
            iteration=checkpoint.iteration,
            policy_data=checkpoint.policy_data,
            policy_s3_path=checkpoint.policy_s3_path,
            validation_score=checkpoint.validation_score,
            metrics=checkpoint.metrics,
        )
        session.add(checkpoint_db)
        await session.flush()
        await session.refresh(checkpoint_db)
        return checkpoint_db

    @staticmethod
    async def get_by_id(
        session: AsyncSession, checkpoint_id: UUID
    ) -> PolicyCheckpointDB | None:
        """Retrieve checkpoint by ID.

        Args:
            session: Async database session
            checkpoint_id: Checkpoint UUID

        Returns:
            PolicyCheckpointDB | None: Database record or None if not found
        """
        result = await session.execute(
            select(PolicyCheckpointDB).where(
                PolicyCheckpointDB.checkpoint_id == checkpoint_id
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_job(
        session: AsyncSession,
        job_id: UUID,
        limit: int = 100,
    ) -> list[PolicyCheckpointDB]:
        """Retrieve checkpoints for a training job.

        Args:
            session: Async database session
            job_id: Training job UUID
            limit: Maximum number of results

        Returns:
            list[PolicyCheckpointDB]: List of checkpoint records
        """
        result = await session.execute(
            select(PolicyCheckpointDB)
            .where(PolicyCheckpointDB.job_id == job_id)
            .order_by(PolicyCheckpointDB.iteration.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_best_by_job(
        session: AsyncSession,
        job_id: UUID,
    ) -> PolicyCheckpointDB | None:
        """Retrieve best checkpoint for a training job.

        Args:
            session: Async database session
            job_id: Training job UUID

        Returns:
            PolicyCheckpointDB | None: Best checkpoint record or None
        """
        result = await session.execute(
            select(PolicyCheckpointDB)
            .where(PolicyCheckpointDB.job_id == job_id)
            .order_by(PolicyCheckpointDB.validation_score.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_latest_by_job(
        session: AsyncSession,
        job_id: UUID,
    ) -> PolicyCheckpointDB | None:
        """Retrieve latest checkpoint for a training job.

        Args:
            session: Async database session
            job_id: Training job UUID

        Returns:
            PolicyCheckpointDB | None: Latest checkpoint record or None
        """
        result = await session.execute(
            select(PolicyCheckpointDB)
            .where(PolicyCheckpointDB.job_id == job_id)
            .order_by(PolicyCheckpointDB.iteration.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def cleanup_old_checkpoints(
        session: AsyncSession,
        job_id: UUID,
        keep_best: int = 5,
    ) -> None:
        """Delete old checkpoints, keeping only the best N checkpoints.

        Args:
            session: Async database session
            job_id: Training job UUID
            keep_best: Number of best checkpoints to retain
        """
        # Get IDs of checkpoints to keep (top N by validation score)
        result = await session.execute(
            select(PolicyCheckpointDB.checkpoint_id)
            .where(PolicyCheckpointDB.job_id == job_id)
            .order_by(PolicyCheckpointDB.validation_score.desc())
            .limit(keep_best)
        )
        keep_ids = [row[0] for row in result.all()]

        if not keep_ids:
            return

        # Delete all checkpoints not in the keep list
        await session.execute(
            delete(PolicyCheckpointDB).where(
                PolicyCheckpointDB.job_id == job_id,
                PolicyCheckpointDB.checkpoint_id.not_in(keep_ids),
            )
        )
        await session.flush()

    @staticmethod
    async def delete_by_job(session: AsyncSession, job_id: UUID) -> None:
        """Delete all checkpoints for a job.

        Args:
            session: Async database session
            job_id: Training job UUID
        """
        await session.execute(
            delete(PolicyCheckpointDB).where(PolicyCheckpointDB.job_id == job_id)
        )
        await session.flush()
