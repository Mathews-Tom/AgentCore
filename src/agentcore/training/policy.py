"""
Policy updater for GRPO training.

Updates agent policies based on high-advantage trajectories.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from agentcore.training.models import PolicyCheckpoint, Trajectory

logger = structlog.get_logger()


class PolicyPattern:
    """Extracted pattern from successful trajectories."""

    def __init__(
        self,
        pattern_type: str,
        description: str,
        examples: list[str],
        weight: float,
    ) -> None:
        """
        Initialize policy pattern.

        Args:
            pattern_type: Type of pattern (tool_usage, reasoning, verification)
            description: Human-readable pattern description
            examples: Example trajectory snippets demonstrating pattern
            weight: Pattern importance weight (based on advantage)
        """
        self.pattern_type = pattern_type
        self.description = description
        self.examples = examples
        self.weight = weight


class PolicyUpdate:
    """Represents a policy update operation."""

    def __init__(
        self,
        update_type: str,
        content: dict[str, Any],
        source_trajectories: list[UUID],
        total_advantage: float,
    ) -> None:
        """
        Initialize policy update.

        Args:
            update_type: Type of update (prompt_addition, context_update, etc.)
            content: Update content/payload
            source_trajectories: Trajectory IDs that led to this update
            total_advantage: Sum of advantages from source trajectories
        """
        self.update_type = update_type
        self.content = content
        self.source_trajectories = source_trajectories
        self.total_advantage = total_advantage
        self.created_at = datetime.now(UTC)


class PolicyUpdater:
    """
    Updates agent policies based on trajectory performance.

    Extracts patterns from high-advantage trajectories and updates agent
    prompts/context accordingly.
    """

    def __init__(
        self,
        agent_id: str,
        min_advantage_threshold: float = 0.5,
    ) -> None:
        """
        Initialize policy updater.

        Args:
            agent_id: Agent identifier
            min_advantage_threshold: Minimum advantage for pattern extraction
        """
        self.agent_id = agent_id
        self.min_advantage_threshold = min_advantage_threshold
        self.update_history: list[PolicyUpdate] = []

        logger.info(
            "policy_updater_initialized",
            agent_id=agent_id,
            min_advantage_threshold=min_advantage_threshold,
        )

    def extract_patterns(
        self,
        trajectories: list[Trajectory],
        advantages: list[float],
    ) -> list[PolicyPattern]:
        """
        Extract successful patterns from high-advantage trajectories.

        Args:
            trajectories: List of trajectories
            advantages: Corresponding advantage values

        Returns:
            List of extracted patterns
        """
        if len(trajectories) != len(advantages):
            raise ValueError("Trajectories and advantages must have same length")

        # Filter high-advantage trajectories
        high_advantage_pairs = [
            (traj, adv)
            for traj, adv in zip(trajectories, advantages)
            if adv >= self.min_advantage_threshold
        ]

        if not high_advantage_pairs:
            logger.debug(
                "no_high_advantage_trajectories",
                total=len(trajectories),
                threshold=self.min_advantage_threshold,
            )
            return []

        patterns: list[PolicyPattern] = []

        # Extract tool usage patterns
        tool_pattern = self._extract_tool_usage_pattern(high_advantage_pairs)
        if tool_pattern:
            patterns.append(tool_pattern)

        # Extract reasoning patterns
        reasoning_pattern = self._extract_reasoning_pattern(high_advantage_pairs)
        if reasoning_pattern:
            patterns.append(reasoning_pattern)

        # Extract verification patterns
        verification_pattern = self._extract_verification_pattern(high_advantage_pairs)
        if verification_pattern:
            patterns.append(verification_pattern)

        logger.info(
            "patterns_extracted",
            agent_id=self.agent_id,
            pattern_count=len(patterns),
            high_advantage_count=len(high_advantage_pairs),
        )

        return patterns

    def create_update(
        self,
        patterns: list[PolicyPattern],
        source_trajectories: list[Trajectory],
    ) -> PolicyUpdate:
        """
        Create policy update from extracted patterns.

        Args:
            patterns: Extracted patterns
            source_trajectories: Trajectories that led to patterns

        Returns:
            Policy update operation
        """
        # Aggregate patterns into update content
        update_content: dict[str, Any] = {
            "patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "weight": p.weight,
                    "examples": p.examples,
                }
                for p in patterns
            ],
            "summary": self._generate_update_summary(patterns),
        }

        # Calculate total advantage
        total_advantage = sum(
            traj.advantage
            for traj in source_trajectories
            if traj.advantage > self.min_advantage_threshold
        )

        # Create update
        update = PolicyUpdate(
            update_type="pattern_based_update",
            content=update_content,
            source_trajectories=[
                traj.trajectory_id for traj in source_trajectories if traj.trajectory_id
            ],
            total_advantage=total_advantage,
        )

        # Record in history
        self.update_history.append(update)

        logger.info(
            "policy_update_created",
            agent_id=self.agent_id,
            pattern_count=len(patterns),
            total_advantage=total_advantage,
            source_count=len(source_trajectories),
        )

        return update

    def apply_update(
        self,
        update: PolicyUpdate,
        current_policy: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply update to current policy.

        Args:
            update: Policy update to apply
            current_policy: Current agent policy/prompt

        Returns:
            Updated policy
        """
        # Create copy of current policy
        updated_policy = dict(current_policy)

        # Add/update based on patterns
        if "patterns" in update.content:
            # Merge patterns into policy
            existing_patterns = updated_policy.get("learned_patterns", [])
            new_patterns = update.content["patterns"]

            # Combine patterns (in real implementation, would deduplicate)
            updated_policy["learned_patterns"] = existing_patterns + new_patterns

        # Update metadata
        updated_policy["last_updated"] = update.created_at.isoformat()
        updated_policy["total_updates"] = updated_policy.get("total_updates", 0) + 1

        logger.info(
            "policy_update_applied",
            agent_id=self.agent_id,
            update_type=update.update_type,
            pattern_count=len(update.content.get("patterns", [])),
        )

        return updated_policy

    def create_checkpoint(
        self,
        job_id: UUID,
        iteration: int,
        policy_data: dict[str, Any],
        validation_score: float = 0.0,
        metrics: dict[str, float | int | str] | None = None,
    ) -> PolicyCheckpoint:
        """
        Create policy checkpoint.

        Args:
            job_id: Training job ID
            iteration: Training iteration number
            policy_data: Policy state to checkpoint
            validation_score: Validation performance score
            metrics: Additional training metrics

        Returns:
            Policy checkpoint
        """
        checkpoint = PolicyCheckpoint(
            checkpoint_id=uuid4(),
            agent_id=self.agent_id,
            job_id=job_id,
            iteration=iteration,
            policy_data=policy_data,
            validation_score=validation_score,
            metrics=metrics or {},
            created_at=datetime.now(UTC),
        )

        logger.info(
            "checkpoint_created",
            agent_id=self.agent_id,
            checkpoint_id=str(checkpoint.checkpoint_id),
            iteration=iteration,
            validation_score=validation_score,
        )

        return checkpoint

    def _extract_tool_usage_pattern(
        self,
        trajectory_advantage_pairs: list[tuple[Trajectory, float]],
    ) -> PolicyPattern | None:
        """Extract tool usage patterns from successful trajectories."""
        tool_steps = []
        total_weight = 0.0

        for traj, adv in trajectory_advantage_pairs:
            for step in traj.steps:
                if self._is_tool_step(step.action):
                    tool_steps.append(step.action.get("step_type", "unknown"))
                    total_weight += adv

        if not tool_steps:
            return None

        # Find most common tool usage
        unique_tools = list(set(tool_steps))

        return PolicyPattern(
            pattern_type="tool_usage",
            description=f"Successful tool usage patterns: {', '.join(unique_tools[:3])}",
            examples=unique_tools[:5],
            weight=total_weight / len(trajectory_advantage_pairs),
        )

    def _extract_reasoning_pattern(
        self,
        trajectory_advantage_pairs: list[tuple[Trajectory, float]],
    ) -> PolicyPattern | None:
        """Extract reasoning patterns from successful trajectories."""
        reasoning_steps = []
        total_weight = 0.0

        for traj, adv in trajectory_advantage_pairs:
            # Count multi-step trajectories (indicates reasoning)
            if len(traj.steps) > 2:
                reasoning_steps.append(len(traj.steps))
                total_weight += adv

        if not reasoning_steps:
            return None

        avg_steps = sum(reasoning_steps) / len(reasoning_steps)

        return PolicyPattern(
            pattern_type="reasoning",
            description=f"Multi-step reasoning (avg {avg_steps:.1f} steps)",
            examples=[f"{steps} steps" for steps in reasoning_steps[:3]],
            weight=total_weight / len(trajectory_advantage_pairs),
        )

    def _extract_verification_pattern(
        self,
        trajectory_advantage_pairs: list[tuple[Trajectory, float]],
    ) -> PolicyPattern | None:
        """Extract verification patterns from successful trajectories."""
        verify_count = 0
        total_weight = 0.0

        for traj, adv in trajectory_advantage_pairs:
            for step in traj.steps:
                if self._is_verify_step(step.action):
                    verify_count += 1
                    total_weight += adv

        if verify_count == 0:
            return None

        return PolicyPattern(
            pattern_type="verification",
            description=f"Verification steps used ({verify_count} total)",
            examples=["verify", "check"],
            weight=total_weight / len(trajectory_advantage_pairs),
        )

    def _is_tool_step(self, action: dict[str, Any]) -> bool:
        """Check if action is a tool usage step."""
        step_type = action.get("step_type", "")
        return "tool" in step_type.lower() or "action" in step_type.lower()

    def _is_verify_step(self, action: dict[str, Any]) -> bool:
        """Check if action is a verification step."""
        step_type = action.get("step_type", "")
        return "verify" in step_type.lower() or "check" in step_type.lower()

    def _generate_update_summary(self, patterns: list[PolicyPattern]) -> str:
        """Generate human-readable update summary."""
        if not patterns:
            return "No patterns extracted"

        summary_parts = []
        for pattern in patterns:
            summary_parts.append(f"{pattern.pattern_type}: {pattern.description}")

        return "; ".join(summary_parts)

    def get_update_history(self) -> list[dict[str, Any]]:
        """
        Get policy update history.

        Returns:
            List of update summaries
        """
        return [
            {
                "update_type": update.update_type,
                "total_advantage": update.total_advantage,
                "pattern_count": len(update.content.get("patterns", [])),
                "created_at": update.created_at.isoformat(),
            }
            for update in self.update_history
        ]
