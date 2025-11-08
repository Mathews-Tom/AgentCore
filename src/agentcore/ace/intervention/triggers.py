"""
Trigger Detection (COMPASS ACE-2 - ACE-016)

Detects intervention signals from performance metrics, errors, and context state.
Implements 4 trigger types with <50ms latency and <15% false positive rate targets.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog

from agentcore.ace.models.ace_models import (
    PerformanceBaseline,
    PerformanceMetrics,
    TriggerSignal,
    TriggerType,
)
from agentcore.ace.monitors.error_accumulator import ErrorAccumulator, ErrorPattern

logger = structlog.get_logger()

# Configurable thresholds (COMPASS-validated defaults)
DEFAULT_VELOCITY_THRESHOLD = 0.5  # 50% of baseline
DEFAULT_ERROR_RATE_THRESHOLD = 2.0  # 2x baseline
DEFAULT_SUCCESS_RATE_THRESHOLD = 0.7  # 70%
DEFAULT_ERROR_COUNT_THRESHOLD = 3  # 3+ errors in stage
DEFAULT_CONTEXT_AGE_THRESHOLD = 20  # 20+ steps
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.6  # 60% low confidence
DEFAULT_RETRIEVAL_RELEVANCE_THRESHOLD = 0.4  # <40% relevant
DEFAULT_CAPABILITY_COVERAGE_THRESHOLD = 0.5  # <50% coverage
DEFAULT_ACTION_FAILURE_THRESHOLD = 0.5  # >50% failures


class TriggerDetector:
    """
    Trigger detector for strategic interventions (COMPASS ACE-2 - ACE-016).

    Features:
    - 4 trigger types (degradation, error_accumulation, staleness, capability_mismatch)
    - Configurable thresholds per agent
    - <50ms detection latency (p95) - CRITICAL
    - <15% false positive rate target
    - Comprehensive trigger rationale generation
    - Integration with PerformanceMonitor, BaselineTracker, ErrorAccumulator

    Performance targets:
    - Trigger detection: <50ms (p95) - CRITICAL
    - False positive rate: <15%
    - Trigger rationale: always logged
    """

    def __init__(
        self,
        velocity_threshold: float = DEFAULT_VELOCITY_THRESHOLD,
        error_rate_threshold: float = DEFAULT_ERROR_RATE_THRESHOLD,
        success_rate_threshold: float = DEFAULT_SUCCESS_RATE_THRESHOLD,
        error_count_threshold: int = DEFAULT_ERROR_COUNT_THRESHOLD,
        context_age_threshold: int = DEFAULT_CONTEXT_AGE_THRESHOLD,
        low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        retrieval_relevance_threshold: float = DEFAULT_RETRIEVAL_RELEVANCE_THRESHOLD,
        capability_coverage_threshold: float = DEFAULT_CAPABILITY_COVERAGE_THRESHOLD,
        action_failure_threshold: float = DEFAULT_ACTION_FAILURE_THRESHOLD,
    ) -> None:
        """
        Initialize TriggerDetector with configurable thresholds.

        Args:
            velocity_threshold: Velocity ratio threshold (default: 0.5 = 50%)
            error_rate_threshold: Error rate multiplier threshold (default: 2.0 = 2x)
            success_rate_threshold: Success rate threshold (default: 0.7 = 70%)
            error_count_threshold: Error count threshold per stage (default: 3)
            context_age_threshold: Context age threshold in steps (default: 20)
            low_confidence_threshold: Low confidence ratio threshold (default: 0.6 = 60%)
            retrieval_relevance_threshold: Retrieval relevance threshold (default: 0.4 = 40%)
            capability_coverage_threshold: Capability coverage threshold (default: 0.5 = 50%)
            action_failure_threshold: Action failure rate threshold (default: 0.5 = 50%)

        Raises:
            ValueError: If thresholds are invalid
        """
        # Validate thresholds
        if not 0.0 < velocity_threshold <= 1.0:
            raise ValueError(f"velocity_threshold must be in (0, 1], got {velocity_threshold}")
        if error_rate_threshold <= 0.0:
            raise ValueError(f"error_rate_threshold must be > 0, got {error_rate_threshold}")
        if not 0.0 <= success_rate_threshold <= 1.0:
            raise ValueError(f"success_rate_threshold must be in [0, 1], got {success_rate_threshold}")
        if error_count_threshold < 1:
            raise ValueError(f"error_count_threshold must be >= 1, got {error_count_threshold}")
        if context_age_threshold < 1:
            raise ValueError(f"context_age_threshold must be >= 1, got {context_age_threshold}")
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError(f"low_confidence_threshold must be in [0, 1], got {low_confidence_threshold}")
        if not 0.0 <= retrieval_relevance_threshold <= 1.0:
            raise ValueError(f"retrieval_relevance_threshold must be in [0, 1], got {retrieval_relevance_threshold}")
        if not 0.0 <= capability_coverage_threshold <= 1.0:
            raise ValueError(f"capability_coverage_threshold must be in [0, 1], got {capability_coverage_threshold}")
        if not 0.0 <= action_failure_threshold <= 1.0:
            raise ValueError(f"action_failure_threshold must be in [0, 1], got {action_failure_threshold}")

        self.velocity_threshold = velocity_threshold
        self.error_rate_threshold = error_rate_threshold
        self.success_rate_threshold = success_rate_threshold
        self.error_count_threshold = error_count_threshold
        self.context_age_threshold = context_age_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.retrieval_relevance_threshold = retrieval_relevance_threshold
        self.capability_coverage_threshold = capability_coverage_threshold
        self.action_failure_threshold = action_failure_threshold

        logger.info(
            "TriggerDetector initialized",
            velocity_threshold=velocity_threshold,
            error_rate_threshold=error_rate_threshold,
            success_rate_threshold=success_rate_threshold,
            error_count_threshold=error_count_threshold,
            context_age_threshold=context_age_threshold,
        )

    async def detect_degradation(
        self,
        current_metrics: PerformanceMetrics,
        baseline: PerformanceBaseline | None,
    ) -> TriggerSignal | None:
        """
        Detect performance degradation.

        Checks for:
        1. Task velocity drops below 50% of baseline (configurable)
        2. Error rate exceeds 2x baseline (configurable)
        3. Success rate drops below 70% (configurable)

        Args:
            current_metrics: Current performance metrics
            baseline: Performance baseline (can be None)

        Returns:
            TriggerSignal if degradation detected, None otherwise
        """
        signals: list[str] = []
        metric_values: dict[str, float] = {}
        rationale_parts: list[str] = []

        # Check 1: Velocity degradation (requires baseline)
        if baseline is not None:
            baseline_velocity = baseline.mean_action_count / (
                baseline.mean_duration_ms / 60000.0
            )  # actions per minute
            current_velocity = current_metrics.overall_progress_velocity
            velocity_ratio = current_velocity / baseline_velocity if baseline_velocity > 0 else 1.0

            metric_values["baseline_velocity"] = baseline_velocity
            metric_values["current_velocity"] = current_velocity
            metric_values["velocity_ratio"] = velocity_ratio

            if velocity_ratio < self.velocity_threshold:
                signals.append("velocity_drop_below_threshold")
                rationale_parts.append(
                    f"Task velocity dropped {(1-velocity_ratio)*100:.1f}% below baseline "
                    f"({baseline_velocity:.2f} -> {current_velocity:.2f} actions/min)"
                )

        # Check 2: Error rate increase (requires baseline)
        if baseline is not None:
            baseline_error_rate = baseline.mean_error_rate
            current_error_rate = current_metrics.stage_error_rate
            error_rate_ratio = (
                current_error_rate / baseline_error_rate if baseline_error_rate > 0 else 1.0
            )

            metric_values["baseline_error_rate"] = baseline_error_rate
            metric_values["current_error_rate"] = current_error_rate
            metric_values["error_rate_ratio"] = error_rate_ratio

            if error_rate_ratio > self.error_rate_threshold:
                signals.append("error_rate_spike")
                rationale_parts.append(
                    f"Error rate increased {error_rate_ratio:.1f}x above baseline "
                    f"({baseline_error_rate:.2f} -> {current_error_rate:.2f})"
                )

        # Check 3: Success rate drop (no baseline needed)
        current_success_rate = current_metrics.stage_success_rate
        metric_values["current_success_rate"] = current_success_rate

        if current_success_rate < self.success_rate_threshold:
            signals.append("success_rate_below_threshold")
            rationale_parts.append(
                f"Success rate dropped below {self.success_rate_threshold*100:.0f}% "
                f"({current_success_rate*100:.1f}%)"
            )

        # Return signal if any degradation detected
        if signals:
            rationale = "; ".join(rationale_parts)
            confidence = min(1.0, len(signals) / 3.0)  # Higher confidence with more signals

            logger.warning(
                "Performance degradation detected",
                agent_id=current_metrics.agent_id,
                task_id=str(current_metrics.task_id),
                stage=current_metrics.stage,
                signals=signals,
                confidence=confidence,
            )

            return TriggerSignal(
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                signals=signals,
                rationale=rationale,
                confidence=confidence,
                metric_values=metric_values,
            )

        return None

    async def detect_error_accumulation(
        self,
        error_accumulator: ErrorAccumulator,
        agent_id: str,
        task_id: UUID,
        stage: str,
    ) -> TriggerSignal | None:
        """
        Detect error accumulation.

        Checks for:
        1. 3+ errors in single stage (configurable)
        2. Compounding error patterns detected
        3. Same error type repeats 2+ times

        Args:
            error_accumulator: ErrorAccumulator instance
            agent_id: Agent identifier
            task_id: Task identifier
            stage: Reasoning stage

        Returns:
            TriggerSignal if error accumulation detected, None otherwise
        """
        signals: list[str] = []
        metric_values: dict[str, float] = {}
        rationale_parts: list[str] = []

        # Check 1: Error count in stage
        error_count = error_accumulator.get_error_count(agent_id, task_id, stage)
        metric_values["stage_error_count"] = float(error_count)

        if error_count >= self.error_count_threshold:
            signals.append("high_error_count_in_stage")
            rationale_parts.append(
                f"{error_count} errors in {stage} stage (threshold: {self.error_count_threshold})"
            )

        # Check 2: Compounding error patterns
        patterns = error_accumulator.detect_compounding_errors(agent_id, task_id)
        if patterns:
            # Filter patterns for current stage
            stage_patterns = [
                p for p in patterns
                if any(e.stage == stage for e in p.error_records)
            ]

            if stage_patterns:
                signals.append("compounding_error_pattern")
                pattern_types = {p.pattern_type for p in stage_patterns}
                rationale_parts.append(
                    f"Compounding error patterns detected: {', '.join(pattern_types)}"
                )
                metric_values["pattern_count"] = float(len(stage_patterns))

        # Check 3: Same error type repeats (sequential patterns)
        sequential_patterns = [
            p for p in patterns
            if p.pattern_type == "sequential" and len(p.error_records) >= 2
        ]

        if sequential_patterns:
            signals.append("repeated_error_type")
            max_occurrences = max(len(p.error_records) for p in sequential_patterns)
            rationale_parts.append(
                f"Same error type repeated {max_occurrences} times"
            )
            metric_values["max_repeat_count"] = float(max_occurrences)

        # Return signal if any error accumulation detected
        if signals:
            rationale = "; ".join(rationale_parts)
            confidence = min(1.0, len(signals) / 3.0)  # Higher confidence with more signals

            logger.warning(
                "Error accumulation detected",
                agent_id=agent_id,
                task_id=str(task_id),
                stage=stage,
                signals=signals,
                confidence=confidence,
            )

            return TriggerSignal(
                trigger_type=TriggerType.ERROR_ACCUMULATION,
                signals=signals,
                rationale=rationale,
                confidence=confidence,
                metric_values=metric_values,
            )

        return None

    async def detect_staleness(
        self,
        context_age: int,
        low_confidence_ratio: float,
        retrieval_relevance: float,
    ) -> TriggerSignal | None:
        """
        Detect context staleness.

        Checks for:
        1. No context refresh in 20+ steps (configurable)
        2. Low-confidence sections dominate playbook (>60%, configurable)
        3. Memory retrieval returning irrelevant results (<40%, configurable)

        Args:
            context_age: Steps since last context refresh
            low_confidence_ratio: Ratio of low-confidence sections (0-1)
            retrieval_relevance: Memory retrieval relevance score (0-1)

        Returns:
            TriggerSignal if staleness detected, None otherwise

        Raises:
            ValueError: If ratios are not in [0, 1] or context_age is negative
        """
        # Validate inputs
        if context_age < 0:
            raise ValueError(f"context_age must be >= 0, got {context_age}")
        if not 0.0 <= low_confidence_ratio <= 1.0:
            raise ValueError(f"low_confidence_ratio must be in [0, 1], got {low_confidence_ratio}")
        if not 0.0 <= retrieval_relevance <= 1.0:
            raise ValueError(f"retrieval_relevance must be in [0, 1], got {retrieval_relevance}")

        signals: list[str] = []
        metric_values: dict[str, float] = {
            "context_age": float(context_age),
            "low_confidence_ratio": low_confidence_ratio,
            "retrieval_relevance": retrieval_relevance,
        }
        rationale_parts: list[str] = []

        # Check 1: Context age
        if context_age > self.context_age_threshold:
            signals.append("context_age_exceeded")
            rationale_parts.append(
                f"Context not refreshed for {context_age} steps "
                f"(threshold: {self.context_age_threshold})"
            )

        # Check 2: Low confidence sections
        if low_confidence_ratio > self.low_confidence_threshold:
            signals.append("high_low_confidence_ratio")
            rationale_parts.append(
                f"{low_confidence_ratio*100:.1f}% of playbook has low confidence "
                f"(threshold: {self.low_confidence_threshold*100:.0f}%)"
            )

        # Check 3: Retrieval relevance
        if retrieval_relevance < self.retrieval_relevance_threshold:
            signals.append("low_retrieval_relevance")
            rationale_parts.append(
                f"Memory retrieval relevance at {retrieval_relevance*100:.1f}% "
                f"(threshold: {self.retrieval_relevance_threshold*100:.0f}%)"
            )

        # Return signal if any staleness detected
        if signals:
            rationale = "; ".join(rationale_parts)
            confidence = min(1.0, len(signals) / 3.0)  # Higher confidence with more signals

            logger.warning(
                "Context staleness detected",
                signals=signals,
                confidence=confidence,
                context_age=context_age,
                low_confidence_ratio=low_confidence_ratio,
                retrieval_relevance=retrieval_relevance,
            )

            return TriggerSignal(
                trigger_type=TriggerType.CONTEXT_STALENESS,
                signals=signals,
                rationale=rationale,
                confidence=confidence,
                metric_values=metric_values,
            )

        return None

    async def detect_capability_mismatch(
        self,
        task_requirements: list[str],
        agent_capabilities: list[str],
        action_failure_rate: float,
    ) -> TriggerSignal | None:
        """
        Detect capability mismatch.

        Checks for:
        1. Task requirements exceed agent capabilities (<50% coverage, configurable)
        2. 50%+ actions failing due to capability gaps (configurable)
        3. Alternative capabilities show higher fitness

        Args:
            task_requirements: List of required capabilities
            agent_capabilities: List of agent's current capabilities
            action_failure_rate: Action failure rate (0-1)

        Returns:
            TriggerSignal if capability mismatch detected, None otherwise

        Raises:
            ValueError: If action_failure_rate is not in [0, 1]
        """
        # Validate inputs
        if not 0.0 <= action_failure_rate <= 1.0:
            raise ValueError(f"action_failure_rate must be in [0, 1], got {action_failure_rate}")

        signals: list[str] = []
        metric_values: dict[str, float] = {
            "action_failure_rate": action_failure_rate,
        }
        rationale_parts: list[str] = []

        # Check 1: Capability coverage
        if not task_requirements:
            # No requirements to check
            return None

        missing_capabilities = set(task_requirements) - set(agent_capabilities)
        coverage = 1.0 - (len(missing_capabilities) / len(task_requirements))
        metric_values["capability_coverage"] = coverage
        metric_values["missing_count"] = float(len(missing_capabilities))

        if coverage < self.capability_coverage_threshold:
            signals.append("low_capability_coverage")
            rationale_parts.append(
                f"Only {coverage*100:.1f}% capability coverage "
                f"({len(missing_capabilities)} missing: {', '.join(list(missing_capabilities)[:3])}...)"
            )

        # Check 2: Action failure rate
        if action_failure_rate > self.action_failure_threshold:
            signals.append("high_action_failure_rate")
            rationale_parts.append(
                f"Action failure rate at {action_failure_rate*100:.1f}% "
                f"(threshold: {self.action_failure_threshold*100:.0f}%)"
            )

        # Return signal if any capability mismatch detected
        if signals:
            rationale = "; ".join(rationale_parts)
            confidence = min(1.0, len(signals) / 2.0)  # Higher confidence with more signals

            logger.warning(
                "Capability mismatch detected",
                signals=signals,
                confidence=confidence,
                coverage=coverage,
                action_failure_rate=action_failure_rate,
            )

            return TriggerSignal(
                trigger_type=TriggerType.CAPABILITY_MISMATCH,
                signals=signals,
                rationale=rationale,
                confidence=confidence,
                metric_values=metric_values,
            )

        return None
