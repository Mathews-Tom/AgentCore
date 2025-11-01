"""
Checkpoint manager for GRPO training.

Implements save/restore checkpoints with versioning, best-checkpoint selection,
and automatic cleanup.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from agentcore.training.models import PolicyCheckpoint

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Checkpoint manager for training continuity.

    Implements:
    - Save checkpoints every N iterations
    - Store policy parameters, iteration, metrics
    - Hybrid storage (memory for Phase 1, PostgreSQL + S3 for Phase 2)
    - Load checkpoint for resume after interruption
    - Best checkpoint selection by validation score
    - Automatic cleanup (keep best 5 checkpoints)
    """

    def __init__(
        self,
        checkpoint_interval: int = 10,
        max_checkpoints: int = 5,
        storage_path: Path | None = None,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_interval: Save checkpoint every N iterations (default: 10)
            max_checkpoints: Maximum checkpoints to keep (default: 5)
            storage_path: Path for checkpoint storage (default: .checkpoints/)
        """
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.storage_path = storage_path or Path(".checkpoints")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory checkpoint registry (Phase 1)
        self._checkpoints: dict[UUID, PolicyCheckpoint] = {}
        self._checkpoint_order: list[UUID] = []  # Ordered by creation time

        logger.info(
            f"CheckpointManager initialized: interval={checkpoint_interval}, "
            f"max={max_checkpoints}, path={self.storage_path}"
        )

    def should_save_checkpoint(self, iteration: int) -> bool:
        """
        Check if checkpoint should be saved at current iteration.

        Args:
            iteration: Current training iteration

        Returns:
            True if checkpoint should be saved, False otherwise
        """
        return iteration > 0 and iteration % self.checkpoint_interval == 0

    def save_checkpoint(
        self,
        agent_id: str,
        job_id: UUID,
        iteration: int,
        policy_data: dict[str, Any],
        validation_score: float,
        metrics: dict[str, float | int | str] | None = None,
    ) -> PolicyCheckpoint:
        """
        Save checkpoint with policy data and metrics.

        Args:
            agent_id: Agent identifier
            job_id: Training job ID
            iteration: Training iteration number
            policy_data: Policy parameters (prompts, config)
            validation_score: Validation performance score
            metrics: Training metrics at checkpoint

        Returns:
            PolicyCheckpoint with assigned checkpoint_id
        """
        checkpoint_id = uuid4()

        checkpoint = PolicyCheckpoint(
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            job_id=job_id,
            iteration=iteration,
            policy_data=policy_data,
            validation_score=validation_score,
            metrics=metrics or {},
            created_at=datetime.now(UTC),
        )

        # Store checkpoint
        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_order.append(checkpoint_id)

        # Persist to disk (Phase 1: JSON files)
        self._persist_checkpoint(checkpoint)

        logger.info(
            f"Saved checkpoint {checkpoint_id} for job {job_id} "
            f"at iteration {iteration} (score: {validation_score:.4f})"
        )

        # Automatic cleanup
        self._cleanup_old_checkpoints(job_id)

        return checkpoint

    def _persist_checkpoint(self, checkpoint: PolicyCheckpoint) -> None:
        """
        Persist checkpoint to disk.

        Phase 1: JSON files
        Phase 2: PostgreSQL metadata + S3 for large weights

        Args:
            checkpoint: Checkpoint to persist
        """
        checkpoint_file = self.storage_path / f"{checkpoint.checkpoint_id}.json"

        checkpoint_data = {
            "checkpoint_id": str(checkpoint.checkpoint_id),
            "agent_id": checkpoint.agent_id,
            "job_id": str(checkpoint.job_id),
            "iteration": checkpoint.iteration,
            "policy_data": checkpoint.policy_data,
            "validation_score": checkpoint.validation_score,
            "metrics": checkpoint.metrics,
            "created_at": checkpoint.created_at.isoformat()
            if checkpoint.created_at
            else None,
        }

        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))

    def load_checkpoint(self, checkpoint_id: UUID) -> PolicyCheckpoint | None:
        """
        Load checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            PolicyCheckpoint if found, None otherwise
        """
        # Check in-memory registry first
        if checkpoint_id in self._checkpoints:
            return self._checkpoints[checkpoint_id]

        # Load from disk if not in memory
        checkpoint_file = self.storage_path / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_data = json.loads(checkpoint_file.read_text())
            checkpoint = PolicyCheckpoint(
                checkpoint_id=UUID(checkpoint_data["checkpoint_id"]),
                agent_id=checkpoint_data["agent_id"],
                job_id=UUID(checkpoint_data["job_id"]),
                iteration=checkpoint_data["iteration"],
                policy_data=checkpoint_data["policy_data"],
                validation_score=checkpoint_data["validation_score"],
                metrics=checkpoint_data["metrics"],
                created_at=datetime.fromisoformat(checkpoint_data["created_at"])
                if checkpoint_data.get("created_at")
                else None,
            )

            # Cache in memory
            self._checkpoints[checkpoint_id] = checkpoint
            if checkpoint_id not in self._checkpoint_order:
                self._checkpoint_order.append(checkpoint_id)

            return checkpoint

        logger.warning(f"Checkpoint {checkpoint_id} not found")
        return None

    def get_best_checkpoint(self, job_id: UUID) -> PolicyCheckpoint | None:
        """
        Get best checkpoint for a job by validation score.

        Args:
            job_id: Training job ID

        Returns:
            Best PolicyCheckpoint if found, None otherwise
        """
        job_checkpoints = [
            cp for cp in self._checkpoints.values() if cp.job_id == job_id
        ]

        if not job_checkpoints:
            return None

        # Sort by validation score (descending)
        best_checkpoint = max(job_checkpoints, key=lambda cp: cp.validation_score)

        return best_checkpoint

    def get_checkpoints_for_job(self, job_id: UUID) -> list[PolicyCheckpoint]:
        """
        Get all checkpoints for a training job.

        Args:
            job_id: Training job ID

        Returns:
            List of PolicyCheckpoints for the job, sorted by iteration
        """
        job_checkpoints = [
            cp for cp in self._checkpoints.values() if cp.job_id == job_id
        ]

        # Sort by iteration (ascending)
        return sorted(job_checkpoints, key=lambda cp: cp.iteration)

    def _cleanup_old_checkpoints(self, job_id: UUID) -> None:
        """
        Clean up old checkpoints, keeping best N checkpoints.

        Keeps checkpoints with highest validation scores.

        Args:
            job_id: Training job ID
        """
        job_checkpoints = self.get_checkpoints_for_job(job_id)

        if len(job_checkpoints) <= self.max_checkpoints:
            return  # No cleanup needed

        # Sort by validation score (descending)
        sorted_checkpoints = sorted(
            job_checkpoints, key=lambda cp: cp.validation_score, reverse=True
        )

        # Keep best N checkpoints
        checkpoints_to_keep = sorted_checkpoints[: self.max_checkpoints]
        checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints :]

        for checkpoint in checkpoints_to_remove:
            self._remove_checkpoint(checkpoint.checkpoint_id)  # type: ignore

            logger.info(
                f"Removed checkpoint {checkpoint.checkpoint_id} for job {job_id} "
                f"(score: {checkpoint.validation_score:.4f})"
            )

    def _remove_checkpoint(self, checkpoint_id: UUID) -> None:
        """
        Remove checkpoint from storage.

        Args:
            checkpoint_id: Checkpoint identifier
        """
        # Remove from memory
        if checkpoint_id in self._checkpoints:
            del self._checkpoints[checkpoint_id]

        if checkpoint_id in self._checkpoint_order:
            self._checkpoint_order.remove(checkpoint_id)

        # Remove from disk
        checkpoint_file = self.storage_path / f"{checkpoint_id}.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()

    def get_latest_checkpoint(self, job_id: UUID) -> PolicyCheckpoint | None:
        """
        Get latest checkpoint for a job by iteration number.

        Args:
            job_id: Training job ID

        Returns:
            Latest PolicyCheckpoint if found, None otherwise
        """
        job_checkpoints = self.get_checkpoints_for_job(job_id)

        if not job_checkpoints:
            return None

        # Already sorted by iteration
        return job_checkpoints[-1]

    def resume_from_checkpoint(
        self, checkpoint_id: UUID
    ) -> tuple[int, dict[str, Any], dict[str, float | int | str]]:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Tuple of (iteration, policy_data, metrics)

        Raises:
            ValueError: If checkpoint not found
        """
        checkpoint = self.load_checkpoint(checkpoint_id)

        if checkpoint is None:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        logger.info(
            f"Resuming from checkpoint {checkpoint_id} at iteration {checkpoint.iteration}"
        )

        return (
            checkpoint.iteration,
            checkpoint.policy_data or {},
            checkpoint.metrics,
        )

    def get_checkpoint_count(self, job_id: UUID) -> int:
        """
        Get number of checkpoints for a job.

        Args:
            job_id: Training job ID

        Returns:
            Count of checkpoints
        """
        return len(self.get_checkpoints_for_job(job_id))

    def clear_checkpoints(self, job_id: UUID) -> None:
        """
        Clear all checkpoints for a job.

        Args:
            job_id: Training job ID
        """
        checkpoints = self.get_checkpoints_for_job(job_id)

        for checkpoint in checkpoints:
            self._remove_checkpoint(checkpoint.checkpoint_id)  # type: ignore

        logger.info(f"Cleared {len(checkpoints)} checkpoints for job {job_id}")
