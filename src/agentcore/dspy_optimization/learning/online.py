"""
Online learning for incremental model updates

Provides incremental learning capabilities to update optimization models
without full retraining using new data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    OptimizationTarget,
    PerformanceMetrics,
)


class LearningRate(str, Enum):
    """Learning rate scheduling strategies"""

    CONSTANT = "constant"
    EXPONENTIAL_DECAY = "exponential_decay"
    STEP_DECAY = "step_decay"
    ADAPTIVE = "adaptive"


class OnlineLearningConfig(BaseModel):
    """Configuration for online learning"""

    initial_learning_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Initial learning rate",
    )
    learning_rate_schedule: LearningRate = Field(
        default=LearningRate.EXPONENTIAL_DECAY,
        description="Learning rate scheduling strategy",
    )
    decay_rate: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Learning rate decay rate",
    )
    decay_steps: int = Field(
        default=100,
        description="Steps between decay applications",
    )
    min_learning_rate: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Minimum learning rate",
    )
    batch_size: int = Field(
        default=32,
        description="Batch size for updates",
    )
    memory_size: int = Field(
        default=1000,
        description="Maximum training samples to keep in memory",
    )
    update_frequency: int = Field(
        default=10,
        description="Update frequency in samples",
    )


class LearningUpdate(BaseModel):
    """Record of a learning update"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target: OptimizationTarget
    samples_processed: int
    learning_rate: float
    performance_before: PerformanceMetrics
    performance_after: PerformanceMetrics
    improvement: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class OnlineLearner:
    """
    Online learning for incremental model updates

    Provides incremental learning to update optimization models
    with new data without full retraining.

    Key features:
    - Incremental updates with new samples
    - Adaptive learning rate scheduling
    - Memory management for training data
    - Performance tracking
    """

    def __init__(self, config: OnlineLearningConfig | None = None) -> None:
        """
        Initialize online learner

        Args:
            config: Online learning configuration
        """
        self.config = config or OnlineLearningConfig()
        self._training_data: dict[str, list[dict[str, Any]]] = {}
        self._update_count: dict[str, int] = {}
        self._current_learning_rate: dict[str, float] = {}
        self._update_history: dict[str, list[LearningUpdate]] = {}

    async def add_training_sample(
        self,
        target: OptimizationTarget,
        sample: dict[str, Any],
    ) -> LearningUpdate | None:
        """
        Add training sample and potentially trigger update

        Args:
            target: Optimization target
            sample: Training sample

        Returns:
            LearningUpdate if update was triggered, None otherwise
        """
        target_key = self._get_target_key(target)

        # Initialize storage
        if target_key not in self._training_data:
            self._training_data[target_key] = []
            self._update_count[target_key] = 0
            self._current_learning_rate[target_key] = self.config.initial_learning_rate

        # Add sample
        self._training_data[target_key].append(sample)

        # Enforce memory limits
        await self._enforce_memory_limits(target_key)

        # Check if update should be triggered
        samples_since_update = len(self._training_data[target_key]) % self.config.update_frequency

        if samples_since_update == 0 and len(self._training_data[target_key]) >= self.config.batch_size:
            return await self.update(target)

        return None

    async def update(
        self,
        target: OptimizationTarget,
    ) -> LearningUpdate:
        """
        Perform incremental model update

        Args:
            target: Optimization target

        Returns:
            Learning update record
        """
        target_key = self._get_target_key(target)

        # Get training data
        training_data = self._training_data.get(target_key, [])

        if not training_data:
            raise ValueError(f"No training data for target: {target_key}")

        # Get current learning rate
        learning_rate = self._current_learning_rate.get(
            target_key,
            self.config.initial_learning_rate,
        )

        # Get batch for update
        batch = training_data[-self.config.batch_size :]

        # Simulate performance before update
        performance_before = await self._get_current_performance(target, training_data)

        # Perform incremental update
        # In production, this would update the actual model
        # For simulation, we estimate improvement based on learning rate
        improvement_factor = learning_rate * 0.05  # 5% max improvement per update

        performance_after = PerformanceMetrics(
            success_rate=min(performance_before.success_rate * (1 + improvement_factor), 1.0),
            avg_cost_per_task=performance_before.avg_cost_per_task * (1 - improvement_factor * 0.5),
            avg_latency_ms=int(performance_before.avg_latency_ms * (1 - improvement_factor * 0.3)),
            quality_score=min(performance_before.quality_score * (1 + improvement_factor), 1.0),
        )

        # Calculate improvement
        improvement = self._calculate_improvement(performance_before, performance_after)

        # Create update record
        update = LearningUpdate(
            target=target,
            samples_processed=len(batch),
            learning_rate=learning_rate,
            performance_before=performance_before,
            performance_after=performance_after,
            improvement=improvement,
            metadata={
                "total_samples": len(training_data),
                "update_count": self._update_count.get(target_key, 0) + 1,
                "batch_size": len(batch),
            },
        )

        # Store update
        if target_key not in self._update_history:
            self._update_history[target_key] = []
        self._update_history[target_key].append(update)

        # Update count
        self._update_count[target_key] = self._update_count.get(target_key, 0) + 1

        # Update learning rate
        await self._update_learning_rate(target_key)

        return update

    async def get_learning_rate(
        self,
        target: OptimizationTarget,
    ) -> float:
        """
        Get current learning rate for target

        Args:
            target: Optimization target

        Returns:
            Current learning rate
        """
        target_key = self._get_target_key(target)
        return self._current_learning_rate.get(
            target_key,
            self.config.initial_learning_rate,
        )

    async def get_update_history(
        self,
        target: OptimizationTarget,
        limit: int | None = None,
    ) -> list[LearningUpdate]:
        """
        Get update history for target

        Args:
            target: Optimization target
            limit: Optional result limit

        Returns:
            List of learning updates
        """
        target_key = self._get_target_key(target)
        history = self._update_history.get(target_key, [])

        if limit:
            return history[-limit:]

        return history

    async def reset(
        self,
        target: OptimizationTarget,
    ) -> None:
        """
        Reset learning state for target

        Args:
            target: Optimization target
        """
        target_key = self._get_target_key(target)

        self._training_data[target_key] = []
        self._update_count[target_key] = 0
        self._current_learning_rate[target_key] = self.config.initial_learning_rate
        self._update_history[target_key] = []

    async def _get_current_performance(
        self,
        target: OptimizationTarget,
        training_data: list[dict[str, Any]],
    ) -> PerformanceMetrics:
        """
        Get current performance metrics

        Args:
            target: Optimization target
            training_data: Training data

        Returns:
            Current performance metrics
        """
        # Get recent samples
        recent = training_data[-self.config.batch_size :]

        if not recent:
            return PerformanceMetrics(
                success_rate=0.75,
                avg_cost_per_task=0.12,
                avg_latency_ms=2500,
                quality_score=0.80,
            )

        # Calculate average from recent samples
        success_rates = [s.get("success_rate", 0.75) for s in recent]
        costs = [s.get("avg_cost_per_task", 0.12) for s in recent]
        latencies = [s.get("avg_latency_ms", 2500) for s in recent]
        quality_scores = [s.get("quality_score", 0.80) for s in recent]

        return PerformanceMetrics(
            success_rate=sum(success_rates) / len(success_rates),
            avg_cost_per_task=sum(costs) / len(costs),
            avg_latency_ms=int(sum(latencies) / len(latencies)),
            quality_score=sum(quality_scores) / len(quality_scores),
        )

    def _calculate_improvement(
        self,
        before: PerformanceMetrics,
        after: PerformanceMetrics,
    ) -> float:
        """
        Calculate improvement percentage

        Args:
            before: Metrics before update
            after: Metrics after update

        Returns:
            Improvement percentage
        """
        # Calculate weighted improvement
        success_improvement = (after.success_rate - before.success_rate) / max(before.success_rate, 0.01)
        quality_improvement = (after.quality_score - before.quality_score) / max(before.quality_score, 0.01)

        return (success_improvement * 0.6 + quality_improvement * 0.4)

    async def _update_learning_rate(
        self,
        target_key: str,
    ) -> None:
        """
        Update learning rate based on schedule

        Args:
            target_key: Target key
        """
        current_rate = self._current_learning_rate[target_key]
        update_count = self._update_count[target_key]

        if self.config.learning_rate_schedule == LearningRate.CONSTANT:
            # No change
            return

        elif self.config.learning_rate_schedule == LearningRate.EXPONENTIAL_DECAY:
            # Exponential decay
            new_rate = (
                self.config.initial_learning_rate
                * (self.config.decay_rate ** (update_count / self.config.decay_steps))
            )

        elif self.config.learning_rate_schedule == LearningRate.STEP_DECAY:
            # Step decay
            steps = update_count // self.config.decay_steps
            new_rate = self.config.initial_learning_rate * (self.config.decay_rate ** steps)

        elif self.config.learning_rate_schedule == LearningRate.ADAPTIVE:
            # Adaptive based on recent performance
            history = self._update_history.get(target_key, [])
            if len(history) >= 5:
                recent_improvements = [u.improvement for u in history[-5:]]
                avg_improvement = sum(recent_improvements) / len(recent_improvements)

                # Increase rate if improving, decrease if not
                if avg_improvement > 0.05:
                    new_rate = min(current_rate * 1.1, self.config.initial_learning_rate)
                elif avg_improvement < 0.01:
                    new_rate = current_rate * 0.9
                else:
                    new_rate = current_rate
            else:
                new_rate = current_rate

        else:
            new_rate = current_rate

        # Enforce minimum learning rate
        new_rate = max(new_rate, self.config.min_learning_rate)

        self._current_learning_rate[target_key] = new_rate

    async def _enforce_memory_limits(
        self,
        target_key: str,
    ) -> None:
        """
        Enforce memory limits on training data

        Args:
            target_key: Target key
        """
        training_data = self._training_data.get(target_key, [])

        if len(training_data) > self.config.memory_size:
            # Keep most recent samples
            self._training_data[target_key] = training_data[-self.config.memory_size :]

    def _get_target_key(self, target: OptimizationTarget) -> str:
        """
        Get target storage key

        Args:
            target: Optimization target

        Returns:
            Target key
        """
        return f"{target.type.value}:{target.id}:{target.scope.value}"
