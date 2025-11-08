"""
Intervention Outcome Tracking for COMPASS Meta-Thinker (COMPASS ACE-2 - ACE-023)

Measures intervention effectiveness to enable the COMPASS learning loop.
Captures before/after metrics, computes improvement deltas, and stores
outcomes for meta-learning and threshold updates.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import structlog

from agentcore.ace.integration.mem_interface import ACEMemoryInterface
from agentcore.ace.models.ace_models import (
    InterventionOutcome,
    InterventionRecord,
    InterventionType,
    PerformanceMetrics,
)

logger = structlog.get_logger()


class OutcomeTracker:
    """
    Tracks intervention outcomes for COMPASS learning loop (COMPASS ACE-2 - ACE-023).

    Captures before/after metrics, computes deltas, and stores outcomes for
    meta-learning and threshold updates.

    Features:
    - Delta computation (velocity, success rate, error rate)
    - Success determination (multi-criteria)
    - Learning data extraction for threshold updates
    - Intervention effectiveness tracking
    - Threshold adaptation based on historical performance

    Storage: In-memory for MVP (dict-based)
    TODO: Integrate with MEM Phase 5 for persistent storage
    """

    def __init__(self, mem_interface: ACEMemoryInterface | None = None) -> None:
        """
        Initialize OutcomeTracker.

        Args:
            mem_interface: Optional MEM interface for storage (currently unused, for future integration)
        """
        self.mem_interface = mem_interface
        self.outcomes: dict[UUID, InterventionOutcome] = {}
        self.logger = structlog.get_logger(__name__)

        # TODO: Replace with MEM Phase 5 integration
        self.logger.info(
            "OutcomeTracker initialized with in-memory storage",
            has_mem_interface=mem_interface is not None,
        )

    async def record_intervention_outcome(
        self,
        intervention_id: UUID,
        intervention_record: InterventionRecord,
        post_metrics: PerformanceMetrics,
    ) -> InterventionOutcome:
        """
        Record intervention outcome with before/after metrics.

        Computes improvement deltas, determines success, and extracts learning data
        for meta-learning and threshold updates.

        Args:
            intervention_id: Unique intervention identifier
            intervention_record: InterventionRecord with pre-metrics and intervention details
            post_metrics: Performance metrics after intervention

        Returns:
            InterventionOutcome with deltas, success determination, and learning data

        Raises:
            ValueError: If intervention_record is missing pre_metric_id or pre-metrics not found
        """
        if intervention_record.pre_metric_id is None:
            raise ValueError("intervention_record must have pre_metric_id set")

        # Get pre-metrics from intervention record
        # For MVP, we need to pass pre_metrics directly as we don't have database access
        # This is a limitation of the in-memory implementation
        # TODO: When MEM Phase 5 is available, fetch pre_metrics from storage
        pre_metrics_id = intervention_record.pre_metric_id

        # For now, we'll construct pre_metrics from post_metrics with adjusted values
        # This is a workaround for the MVP - in production, pre_metrics should be fetched
        # from the intervention_record's pre_metric_id
        # Since we don't have DB access in this layer, we'll require pre_metrics to be passed
        # via the trigger_metric_id being the actual pre_metrics object

        # WORKAROUND: For MVP, we need pre_metrics passed separately
        # We'll fetch it from outcomes if it exists as a referenced metric
        # Otherwise, raise an error
        if intervention_record.trigger_metric_id is None:
            raise ValueError(
                "intervention_record must have trigger_metric_id with pre-intervention metrics"
            )

        # For MVP: We'll need to modify the signature to accept pre_metrics directly
        # Let's add it as a parameter
        raise NotImplementedError(
            "Pre-metrics retrieval not implemented in MVP. Use record_intervention_outcome_with_pre_metrics instead."
        )

    async def record_intervention_outcome_with_pre_metrics(
        self,
        intervention_id: UUID,
        intervention_record: InterventionRecord,
        pre_metrics: PerformanceMetrics,
        post_metrics: PerformanceMetrics,
    ) -> InterventionOutcome:
        """
        Record intervention outcome with explicit pre/post metrics.

        This is the MVP version that accepts pre_metrics directly to avoid
        database dependencies. Use this method until MEM Phase 5 integration.

        Args:
            intervention_id: Unique intervention identifier
            intervention_record: InterventionRecord with intervention details
            pre_metrics: Performance metrics before intervention
            post_metrics: Performance metrics after intervention

        Returns:
            InterventionOutcome with deltas, success determination, and learning data

        Raises:
            ValueError: If metrics are from different tasks/agents or stages
        """
        # Validate metrics match (same task, agent, stage)
        if pre_metrics.task_id != post_metrics.task_id:
            raise ValueError(
                f"Pre and post metrics must be for same task: "
                f"{pre_metrics.task_id} != {post_metrics.task_id}"
            )
        if pre_metrics.agent_id != post_metrics.agent_id:
            raise ValueError(
                f"Pre and post metrics must be for same agent: "
                f"{pre_metrics.agent_id} != {post_metrics.agent_id}"
            )
        if pre_metrics.stage != post_metrics.stage:
            raise ValueError(
                f"Pre and post metrics must be for same stage: "
                f"{pre_metrics.stage} != {post_metrics.stage}"
            )

        self.logger.info(
            "Recording intervention outcome",
            intervention_id=str(intervention_id),
            intervention_type=intervention_record.intervention_type.value,
            trigger_type=intervention_record.trigger_type.value,
            task_id=str(pre_metrics.task_id),
            agent_id=pre_metrics.agent_id,
            stage=pre_metrics.stage,
        )

        # Compute deltas
        deltas = await self.compute_delta(pre_metrics, post_metrics)

        # Determine success
        success = self._determine_success(deltas)

        # Extract learning data
        learning_data = self._extract_learning_data(
            intervention_record=intervention_record,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            deltas=deltas,
            success=success,
        )

        # Create outcome
        outcome = InterventionOutcome(
            outcome_id=uuid4(),
            intervention_id=intervention_id,
            success=success,
            pre_metrics=pre_metrics,
            post_metrics=post_metrics,
            delta_velocity=deltas["delta_velocity"],
            delta_success_rate=deltas["delta_success_rate"],
            delta_error_rate=deltas["delta_error_rate"],
            overall_improvement=deltas["overall_improvement"],
            learning_data=learning_data,
            recorded_at=datetime.now(UTC),
        )

        # Store outcome (in-memory for MVP)
        self.outcomes[intervention_id] = outcome

        # TODO: Persist to MEM Phase 5 when available
        # await self.mem_interface.store_outcome(outcome)

        self.logger.info(
            "Intervention outcome recorded",
            outcome_id=str(outcome.outcome_id),
            intervention_id=str(intervention_id),
            success=success,
            overall_improvement=deltas["overall_improvement"],
            delta_velocity=deltas["delta_velocity"],
            delta_success_rate=deltas["delta_success_rate"],
            delta_error_rate=deltas["delta_error_rate"],
        )

        return outcome

    async def compute_delta(
        self,
        pre_metrics: PerformanceMetrics,
        post_metrics: PerformanceMetrics,
    ) -> dict[str, float]:
        """
        Compute improvement deltas for key metrics.

        Handles edge cases:
        - Zero pre-velocity: Use absolute change instead of percentage
        - Identical metrics: Return 0.0 for all deltas
        - Extreme values: Clamp to [-1, 1] range
        - Missing fields: Use 0.0 as fallback

        Delta formulas:
        - delta_velocity = (post.velocity - pre.velocity) / pre.velocity (percentage change)
        - delta_success_rate = post.success_rate - pre.success_rate (absolute change)
        - delta_error_rate = pre.error_rate - post.error_rate (absolute change, positive = improvement)
        - overall_improvement = (0.4 * delta_velocity) + (0.3 * delta_success_rate) + (0.3 * delta_error_rate)

        Args:
            pre_metrics: Performance metrics before intervention
            post_metrics: Performance metrics after intervention

        Returns:
            Dict with delta_velocity, delta_success_rate, delta_error_rate, overall_improvement
        """
        # Handle identical metrics (no change)
        if (
            pre_metrics.overall_progress_velocity == post_metrics.overall_progress_velocity
            and pre_metrics.stage_success_rate == post_metrics.stage_success_rate
            and pre_metrics.stage_error_rate == post_metrics.stage_error_rate
        ):
            return {
                "delta_velocity": 0.0,
                "delta_success_rate": 0.0,
                "delta_error_rate": 0.0,
                "overall_improvement": 0.0,
            }

        # Compute velocity delta (handle zero pre-velocity)
        if pre_metrics.overall_progress_velocity == 0.0:
            # Use absolute change if baseline is zero
            if post_metrics.overall_progress_velocity == 0.0:
                delta_velocity = 0.0
            else:
                # Positive absolute change indicates improvement
                delta_velocity = min(1.0, post_metrics.overall_progress_velocity / 10.0)
        else:
            # Percentage change
            velocity_change = (
                post_metrics.overall_progress_velocity - pre_metrics.overall_progress_velocity
            ) / pre_metrics.overall_progress_velocity
            delta_velocity = max(-1.0, min(1.0, velocity_change))

        # Compute success rate delta (absolute change)
        success_rate_change = post_metrics.stage_success_rate - pre_metrics.stage_success_rate
        delta_success_rate = max(-1.0, min(1.0, success_rate_change))

        # Compute error rate delta (positive = improvement)
        error_rate_change = pre_metrics.stage_error_rate - post_metrics.stage_error_rate
        delta_error_rate = max(-1.0, min(1.0, error_rate_change))

        # Compute overall improvement (weighted combination)
        overall_improvement = (
            (0.4 * delta_velocity) + (0.3 * delta_success_rate) + (0.3 * delta_error_rate)
        )
        overall_improvement = max(-1.0, min(1.0, overall_improvement))

        self.logger.debug(
            "Deltas computed",
            delta_velocity=delta_velocity,
            delta_success_rate=delta_success_rate,
            delta_error_rate=delta_error_rate,
            overall_improvement=overall_improvement,
        )

        return {
            "delta_velocity": delta_velocity,
            "delta_success_rate": delta_success_rate,
            "delta_error_rate": delta_error_rate,
            "overall_improvement": overall_improvement,
        }

    def _determine_success(self, deltas: dict[str, float]) -> bool:
        """
        Determine if intervention was successful using multi-criteria evaluation.

        Success criteria (any of):
        - overall_improvement >= 0.1 (10% improvement threshold)
        - delta_error_rate >= 0.2 (20% error reduction)
        - delta_success_rate >= 0.15 (15% success improvement)

        Args:
            deltas: Delta computation results

        Returns:
            True if intervention was successful, False otherwise
        """
        overall_improvement = deltas["overall_improvement"]
        delta_error_rate = deltas["delta_error_rate"]
        delta_success_rate = deltas["delta_success_rate"]

        # Check success criteria
        success = (
            overall_improvement >= 0.1
            or delta_error_rate >= 0.2
            or delta_success_rate >= 0.15
        )

        self.logger.debug(
            "Success determination",
            success=success,
            overall_improvement=overall_improvement,
            delta_error_rate=delta_error_rate,
            delta_success_rate=delta_success_rate,
            criteria_met=[
                "overall_improvement>=0.1" if overall_improvement >= 0.1 else None,
                "delta_error_rate>=0.2" if delta_error_rate >= 0.2 else None,
                "delta_success_rate>=0.15" if delta_success_rate >= 0.15 else None,
            ],
        )

        return success

    def _extract_learning_data(
        self,
        intervention_record: InterventionRecord,
        pre_metrics: PerformanceMetrics,
        post_metrics: PerformanceMetrics,
        deltas: dict[str, float],
        success: bool,
    ) -> dict[str, Any]:
        """
        Extract learning data for threshold updates.

        Captures:
        - trigger_type → intervention_type mapping
        - effectiveness score
        - context conditions (stage, task_type, agent_id)
        - time to improvement (post_metrics.timestamp - pre_metrics.timestamp)

        Args:
            intervention_record: InterventionRecord with intervention details
            pre_metrics: Pre-intervention metrics
            post_metrics: Post-intervention metrics
            deltas: Computed deltas
            success: Success determination

        Returns:
            Dict with learning data for threshold updates
        """
        # Compute time to improvement
        time_delta = post_metrics.recorded_at - pre_metrics.recorded_at
        time_to_improvement_ms = int(time_delta.total_seconds() * 1000)

        learning_data = {
            "trigger_type": intervention_record.trigger_type.value,
            "intervention_type": intervention_record.intervention_type.value,
            "effectiveness": deltas["overall_improvement"],
            "success": success,
            "context_conditions": {
                "stage": pre_metrics.stage,
                "agent_id": pre_metrics.agent_id,
                "task_id": str(pre_metrics.task_id),
            },
            "time_to_improvement_ms": time_to_improvement_ms,
            "trigger_confidence": intervention_record.decision_confidence,
            "trigger_signals": intervention_record.trigger_signals,
            "pre_velocity": pre_metrics.overall_progress_velocity,
            "post_velocity": post_metrics.overall_progress_velocity,
            "pre_success_rate": pre_metrics.stage_success_rate,
            "post_success_rate": post_metrics.stage_success_rate,
            "pre_error_rate": pre_metrics.stage_error_rate,
            "post_error_rate": post_metrics.stage_error_rate,
        }

        self.logger.debug(
            "Learning data extracted",
            trigger_type=learning_data["trigger_type"],
            intervention_type=learning_data["intervention_type"],
            effectiveness=learning_data["effectiveness"],
            time_to_improvement_ms=time_to_improvement_ms,
        )

        return learning_data

    async def get_intervention_effectiveness(
        self,
        intervention_type: InterventionType,
        window_days: int = 7,
    ) -> float:
        """
        Get average effectiveness for intervention type within time window.

        Queries historical outcomes and computes average effectiveness score.

        Args:
            intervention_type: Type of intervention to analyze
            window_days: Time window in days (default: 7)

        Returns:
            Average effectiveness score (0-1), or 0.0 if no outcomes found
        """
        # Calculate window start time
        window_start = datetime.now(UTC) - timedelta(days=window_days)

        # Filter outcomes by intervention type and time window
        relevant_outcomes = [
            outcome
            for outcome in self.outcomes.values()
            if outcome.learning_data.get("intervention_type") == intervention_type.value
            and outcome.recorded_at >= window_start
        ]

        if not relevant_outcomes:
            self.logger.debug(
                "No outcomes found for intervention type",
                intervention_type=intervention_type.value,
                window_days=window_days,
            )
            return 0.0

        # Compute average effectiveness
        total_effectiveness = sum(
            outcome.learning_data.get("effectiveness", 0.0) for outcome in relevant_outcomes
        )
        avg_effectiveness = total_effectiveness / len(relevant_outcomes)

        # Normalize to [0, 1] range (effectiveness is in [-1, 1])
        normalized_effectiveness = (avg_effectiveness + 1.0) / 2.0

        self.logger.info(
            "Intervention effectiveness computed",
            intervention_type=intervention_type.value,
            window_days=window_days,
            outcome_count=len(relevant_outcomes),
            avg_effectiveness=avg_effectiveness,
            normalized_effectiveness=normalized_effectiveness,
        )

        return normalized_effectiveness

    async def update_intervention_thresholds(
        self,
        intervention_type: InterventionType,
    ) -> dict[str, float]:
        """
        Update intervention thresholds based on historical effectiveness.

        Strategy:
        - If effectiveness < 0.7 (70%): increase threshold (require stronger signal)
        - If effectiveness > 0.9 (90%): decrease threshold (be more proactive)
        - Otherwise: keep current thresholds

        Args:
            intervention_type: Type of intervention to update thresholds for

        Returns:
            Dict with updated thresholds: {trigger_threshold: float, confidence_threshold: float}
        """
        # Get effectiveness from last 7 days
        effectiveness = await self.get_intervention_effectiveness(
            intervention_type=intervention_type,
            window_days=7,
        )

        # Base thresholds (conservative)
        base_trigger_threshold = 0.3
        base_confidence_threshold = 0.7

        # Adjust thresholds based on effectiveness
        if effectiveness < 0.7:
            # Low effectiveness: increase thresholds (be more cautious)
            adjustment_factor = 1.2
            self.logger.info(
                "Increasing intervention thresholds due to low effectiveness",
                intervention_type=intervention_type.value,
                effectiveness=effectiveness,
                adjustment_factor=adjustment_factor,
            )
        elif effectiveness > 0.9:
            # High effectiveness: decrease thresholds (be more proactive)
            adjustment_factor = 0.8
            self.logger.info(
                "Decreasing intervention thresholds due to high effectiveness",
                intervention_type=intervention_type.value,
                effectiveness=effectiveness,
                adjustment_factor=adjustment_factor,
            )
        else:
            # Moderate effectiveness: keep current thresholds
            adjustment_factor = 1.0
            self.logger.info(
                "Maintaining intervention thresholds",
                intervention_type=intervention_type.value,
                effectiveness=effectiveness,
            )

        # Apply adjustment
        trigger_threshold = min(1.0, base_trigger_threshold * adjustment_factor)
        confidence_threshold = min(1.0, base_confidence_threshold * adjustment_factor)

        thresholds = {
            "trigger_threshold": trigger_threshold,
            "confidence_threshold": confidence_threshold,
        }

        self.logger.info(
            "Intervention thresholds updated",
            intervention_type=intervention_type.value,
            effectiveness=effectiveness,
            trigger_threshold=trigger_threshold,
            confidence_threshold=confidence_threshold,
        )

        return thresholds

    def get_outcome_by_intervention_id(self, intervention_id: UUID) -> InterventionOutcome | None:
        """
        Get outcome by intervention ID.

        Args:
            intervention_id: Intervention identifier

        Returns:
            InterventionOutcome if found, None otherwise
        """
        return self.outcomes.get(intervention_id)

    def get_all_outcomes(self) -> list[InterventionOutcome]:
        """
        Get all recorded outcomes.

        Returns:
            List of all InterventionOutcome instances
        """
        return list(self.outcomes.values())
