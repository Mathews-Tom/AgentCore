"""
Automated rollout management

Provides progressive rollout strategies with automatic decision-making
based on statistical validation and rollback mechanisms.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.testing.experiment import Experiment, ExperimentStatus
from agentcore.dspy_optimization.testing.validation import (
    ExperimentValidator,
    ValidationResult,
)


class RolloutStrategy(str, Enum):
    """Rollout strategy type"""

    IMMEDIATE = "immediate"
    PROGRESSIVE = "progressive"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


class RolloutPhase(str, Enum):
    """Phase in rollout process"""

    PENDING = "pending"
    CANARY = "canary"
    PROGRESSIVE = "progressive"
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"


class RolloutConfig(BaseModel):
    """Configuration for automated rollout"""

    strategy: RolloutStrategy = RolloutStrategy.PROGRESSIVE
    canary_percentage: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Initial canary rollout percentage",
    )
    progressive_steps: list[float] = Field(
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        description="Progressive rollout percentages",
    )
    step_duration_hours: int = Field(
        default=24,
        description="Duration for each progressive step",
    )
    auto_rollback_enabled: bool = Field(
        default=True,
        description="Enable automatic rollback on failure",
    )
    rollback_threshold_percentage: float = Field(
        default=-5.0,
        description="Performance degradation threshold for rollback",
    )
    min_samples_per_step: int = Field(
        default=100,
        description="Minimum samples before advancing step",
    )


class RolloutDecision(BaseModel):
    """Decision from rollout manager"""

    action: str  # "advance", "hold", "rollback"
    phase: RolloutPhase
    traffic_percentage: float
    validation: ValidationResult | None = None
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RolloutState(BaseModel):
    """State of active rollout"""

    experiment_id: str
    config: RolloutConfig
    phase: RolloutPhase = RolloutPhase.PENDING
    current_step: int = 0
    traffic_percentage: float = 0.0
    step_start_time: datetime | None = None
    decisions: list[RolloutDecision] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RolloutManager:
    """
    Manages automated rollouts

    Implements progressive rollout strategies with automatic
    decision-making based on validation results and configurable
    rollback mechanisms.
    """

    def __init__(
        self,
        validator: ExperimentValidator | None = None,
    ) -> None:
        """
        Initialize rollout manager

        Args:
            validator: Experiment validator instance
        """
        self.validator = validator or ExperimentValidator()
        self._rollouts: dict[str, RolloutState] = {}

    async def start_rollout(
        self,
        experiment: Experiment,
        config: RolloutConfig | None = None,
    ) -> RolloutState:
        """
        Start automated rollout

        Args:
            experiment: Validated experiment
            config: Rollout configuration

        Returns:
            Initial rollout state

        Raises:
            ValueError: If experiment invalid or rollout already exists
        """
        if experiment.id in self._rollouts:
            raise ValueError(f"Rollout already exists: {experiment.id}")

        # Validate experiment first
        validation = await self.validator.validate_experiment(experiment)
        if not validation.is_valid:
            raise ValueError(
                f"Experiment not valid for rollout: {validation.recommendation}"
            )

        # Create rollout state
        rollout_config = config or RolloutConfig()
        state = RolloutState(
            experiment_id=experiment.id,
            config=rollout_config,
        )

        # Determine initial phase based on strategy
        if rollout_config.strategy == RolloutStrategy.IMMEDIATE:
            state.phase = RolloutPhase.COMPLETE
            state.traffic_percentage = 1.0
        elif rollout_config.strategy == RolloutStrategy.CANARY:
            state.phase = RolloutPhase.CANARY
            state.traffic_percentage = rollout_config.canary_percentage
            state.step_start_time = datetime.now(UTC)
        else:  # PROGRESSIVE or BLUE_GREEN
            state.phase = RolloutPhase.PROGRESSIVE
            state.traffic_percentage = rollout_config.progressive_steps[0]
            state.step_start_time = datetime.now(UTC)

        # Record decision
        decision = RolloutDecision(
            action="start",
            phase=state.phase,
            traffic_percentage=state.traffic_percentage,
            validation=validation,
            reason=f"Starting {rollout_config.strategy.value} rollout",
        )
        state.decisions.append(decision)

        self._rollouts[experiment.id] = state
        return state

    async def evaluate_rollout(
        self,
        experiment: Experiment,
    ) -> RolloutDecision:
        """
        Evaluate rollout and make decision

        Args:
            experiment: Active experiment

        Returns:
            Rollout decision

        Raises:
            ValueError: If rollout not found
        """
        state = self._rollouts.get(experiment.id)
        if not state:
            raise ValueError(f"Rollout not found: {experiment.id}")

        # Check if rollout complete
        if state.phase == RolloutPhase.COMPLETE:
            return RolloutDecision(
                action="hold",
                phase=state.phase,
                traffic_percentage=state.traffic_percentage,
                reason="Rollout already complete",
            )

        # Check if rolled back
        if state.phase == RolloutPhase.ROLLED_BACK:
            return RolloutDecision(
                action="hold",
                phase=state.phase,
                traffic_percentage=0.0,
                reason="Rollout previously rolled back",
            )

        # Validate current performance
        validation = await self.validator.validate_experiment(experiment)

        # Check for rollback conditions
        if state.config.auto_rollback_enabled:
            if (
                validation.improvement_percentage
                < state.config.rollback_threshold_percentage
            ):
                return await self._rollback(
                    state, validation, "Performance below threshold"
                )

        # Check if sufficient samples collected
        if not experiment.has_minimum_samples():
            return RolloutDecision(
                action="hold",
                phase=state.phase,
                traffic_percentage=state.traffic_percentage,
                validation=validation,
                reason="Waiting for sufficient samples",
            )

        # Check step duration
        if state.step_start_time:
            elapsed_hours = (
                datetime.now(UTC) - state.step_start_time
            ).total_seconds() / 3600
            if elapsed_hours < state.config.step_duration_hours:
                return RolloutDecision(
                    action="hold",
                    phase=state.phase,
                    traffic_percentage=state.traffic_percentage,
                    validation=validation,
                    reason=f"Step duration not reached ({elapsed_hours:.1f}h/{state.config.step_duration_hours}h)",
                )

        # Advance to next step
        return await self._advance_step(state, validation)

    async def rollback(
        self,
        experiment_id: str,
        reason: str = "Manual rollback",
    ) -> RolloutDecision:
        """
        Manually rollback deployment

        Args:
            experiment_id: Experiment ID
            reason: Rollback reason

        Returns:
            Rollback decision

        Raises:
            ValueError: If rollout not found
        """
        state = self._rollouts.get(experiment_id)
        if not state:
            raise ValueError(f"Rollout not found: {experiment_id}")

        return await self._rollback(state, None, reason)

    async def get_rollout_state(
        self,
        experiment_id: str,
    ) -> RolloutState | None:
        """
        Get rollout state

        Args:
            experiment_id: Experiment ID

        Returns:
            Rollout state or None if not found
        """
        return self._rollouts.get(experiment_id)

    async def _advance_step(
        self,
        state: RolloutState,
        validation: ValidationResult,
    ) -> RolloutDecision:
        """
        Advance to next rollout step

        Args:
            state: Current rollout state
            validation: Validation result

        Returns:
            Advance decision
        """
        # Determine next step
        if state.phase == RolloutPhase.CANARY:
            # Move to progressive rollout
            state.phase = RolloutPhase.PROGRESSIVE
            state.current_step = 0
            state.traffic_percentage = state.config.progressive_steps[0]
            reason = "Canary successful, starting progressive rollout"

        elif state.phase == RolloutPhase.PROGRESSIVE:
            # Advance to next step
            state.current_step += 1

            if state.current_step >= len(state.config.progressive_steps):
                # Complete rollout
                state.phase = RolloutPhase.COMPLETE
                state.traffic_percentage = 1.0
                reason = "Progressive rollout complete"
            else:
                # Next progressive step
                state.traffic_percentage = state.config.progressive_steps[
                    state.current_step
                ]
                reason = f"Advancing to step {state.current_step + 1}/{len(state.config.progressive_steps)}"

        else:
            reason = "Invalid phase for advancement"

        state.step_start_time = datetime.now(UTC)
        state.updated_at = datetime.now(UTC)

        decision = RolloutDecision(
            action="advance",
            phase=state.phase,
            traffic_percentage=state.traffic_percentage,
            validation=validation,
            reason=reason,
            metadata={
                "step": state.current_step,
                "total_steps": len(state.config.progressive_steps),
            },
        )

        state.decisions.append(decision)
        return decision

    async def _rollback(
        self,
        state: RolloutState,
        validation: ValidationResult | None,
        reason: str,
    ) -> RolloutDecision:
        """
        Rollback deployment

        Args:
            state: Current rollout state
            validation: Validation result
            reason: Rollback reason

        Returns:
            Rollback decision
        """
        state.phase = RolloutPhase.ROLLED_BACK
        state.traffic_percentage = 0.0
        state.updated_at = datetime.now(UTC)

        decision = RolloutDecision(
            action="rollback",
            phase=state.phase,
            traffic_percentage=0.0,
            validation=validation,
            reason=reason,
        )

        state.decisions.append(decision)
        return decision
