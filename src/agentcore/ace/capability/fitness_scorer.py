"""
Fitness Scoring Algorithms - ACE-026

Multi-factor fitness scoring for capability matching.
Implements advanced scoring algorithms with trend tracking.

Performance target: Part of <100ms evaluation latency
"""

import structlog
from datetime import UTC, datetime
from typing import Any

from ..models.ace_models import FitnessMetrics


logger = structlog.get_logger(__name__)


class FitnessScorer:
    """
    Multi-factor fitness scoring for capabilities.

    Implements three scoring dimensions:
    1. Capability coverage score (task requirements met)
    2. Performance history score (success rate on similar tasks)
    3. Resource efficiency score (time/cost per task)

    Combined with weighted factors for overall fitness.
    """

    def __init__(
        self,
        coverage_weight: float = 0.4,
        performance_weight: float = 0.4,
        efficiency_weight: float = 0.2,
    ):
        """
        Initialize fitness scorer with configurable weights.

        Args:
            coverage_weight: Weight for coverage score (0-1)
            performance_weight: Weight for performance score (0-1)
            efficiency_weight: Weight for efficiency score (0-1)

        Note: Weights should sum to 1.0
        """
        self.logger = logger.bind(component="fitness_scorer")

        # Normalize weights
        total = coverage_weight + performance_weight + efficiency_weight
        self.coverage_weight = coverage_weight / total
        self.performance_weight = performance_weight / total
        self.efficiency_weight = efficiency_weight / total

        self.logger.info(
            "fitness_scorer_initialized",
            coverage_weight=self.coverage_weight,
            performance_weight=self.performance_weight,
            efficiency_weight=self.efficiency_weight,
        )

    def compute_coverage_score(
        self,
        capability_name: str,
        required_capabilities: list[str],
        capability_metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Compute capability coverage score.

        Measures how well a capability matches task requirements.

        Args:
            capability_name: Name of the capability
            required_capabilities: List of required capability names
            capability_metadata: Optional metadata for semantic matching

        Returns:
            Coverage score (0-1)
        """
        if not required_capabilities:
            return 1.0  # Perfect coverage if no requirements

        # Exact match
        if capability_name in required_capabilities:
            return 1.0

        # Partial match (semantic similarity)
        max_similarity = 0.0
        for required in required_capabilities:
            similarity = self._compute_semantic_similarity(
                capability_name, required, capability_metadata
            )
            max_similarity = max(max_similarity, similarity)

        self.logger.debug(
            "coverage_computed",
            capability=capability_name,
            score=max_similarity,
        )

        return max_similarity

    def compute_performance_score(
        self,
        metrics: FitnessMetrics,
        task_history: list[dict[str, Any]] | None = None,
    ) -> float:
        """
        Compute performance history score.

        Analyzes success rate, error patterns, and consistency.

        Args:
            metrics: Fitness metrics with performance data
            task_history: Optional task execution history

        Returns:
            Performance score (0-1)
        """
        # Base score from success rate
        base_score = metrics.success_rate

        # Penalty for high error correlation
        error_penalty = metrics.error_correlation * 0.3
        adjusted_score = base_score * (1.0 - error_penalty)

        # Bonus for consistency (if history available)
        if task_history and len(task_history) > 5:
            consistency_bonus = self._compute_consistency_bonus(task_history)
            adjusted_score = min(1.0, adjusted_score * (1.0 + consistency_bonus))

        self.logger.debug(
            "performance_score_computed",
            success_rate=metrics.success_rate,
            error_correlation=metrics.error_correlation,
            score=adjusted_score,
        )

        return max(0.0, min(1.0, adjusted_score))

    def compute_efficiency_score(
        self,
        metrics: FitnessMetrics,
        time_budget_ms: float | None = None,
        resource_constraints: dict[str, float] | None = None,
    ) -> float:
        """
        Compute resource efficiency score.

        Measures time and resource usage efficiency.

        Args:
            metrics: Fitness metrics with timing and resource data
            time_budget_ms: Optional time budget in milliseconds
            resource_constraints: Optional resource usage constraints

        Returns:
            Efficiency score (0-1)
        """
        # Time efficiency
        if time_budget_ms and time_budget_ms > 0:
            time_efficiency = max(
                0.0,
                1.0 - (metrics.avg_execution_time_ms / time_budget_ms)
            )
        else:
            # Default: penalize slow execution (>5s)
            time_efficiency = max(0.0, 1.0 - (metrics.avg_execution_time_ms / 5000.0))

        # Resource efficiency (from metrics)
        resource_efficiency = metrics.resource_efficiency

        # Combined efficiency (weighted average)
        efficiency_score = (time_efficiency * 0.6 + resource_efficiency * 0.4)

        self.logger.debug(
            "efficiency_score_computed",
            avg_time_ms=metrics.avg_execution_time_ms,
            time_efficiency=time_efficiency,
            resource_efficiency=resource_efficiency,
            score=efficiency_score,
        )

        return max(0.0, min(1.0, efficiency_score))

    def compute_overall_fitness(
        self,
        coverage_score: float,
        performance_score: float,
        efficiency_score: float,
    ) -> float:
        """
        Compute overall fitness score from component scores.

        Args:
            coverage_score: Coverage score (0-1)
            performance_score: Performance score (0-1)
            efficiency_score: Efficiency score (0-1)

        Returns:
            Overall fitness score (0-1)
        """
        overall = (
            coverage_score * self.coverage_weight
            + performance_score * self.performance_weight
            + efficiency_score * self.efficiency_weight
        )

        self.logger.debug(
            "overall_fitness_computed",
            coverage=coverage_score,
            performance=performance_score,
            efficiency=efficiency_score,
            overall=overall,
        )

        return max(0.0, min(1.0, overall))

    def compute_fitness_trend(
        self,
        historical_scores: list[tuple[datetime, float]],
        window_size: int = 10,
    ) -> dict[str, float]:
        """
        Compute fitness trend over time.

        Analyzes whether fitness is improving, stable, or declining.

        Args:
            historical_scores: List of (timestamp, score) tuples
            window_size: Number of recent scores to analyze

        Returns:
            Dictionary with trend metrics:
            - trend_direction: -1 (declining), 0 (stable), 1 (improving)
            - trend_strength: 0-1 indicating strength of trend
            - recent_average: Average of recent scores
            - change_rate: Rate of change per day
        """
        if not historical_scores or len(historical_scores) < 2:
            return {
                "trend_direction": 0,
                "trend_strength": 0.0,
                "recent_average": 0.0,
                "change_rate": 0.0,
            }

        # Sort by timestamp
        sorted_scores = sorted(historical_scores, key=lambda x: x[0])

        # Get recent window
        recent = sorted_scores[-window_size:]
        recent_scores = [score for _, score in recent]

        # Compute metrics
        recent_average = sum(recent_scores) / len(recent_scores)

        # Linear regression for trend
        if len(recent) >= 2:
            timestamps = [(ts - recent[0][0]).total_seconds() for ts, _ in recent]
            scores = recent_scores

            # Simple linear regression
            n = len(timestamps)
            sum_x = sum(timestamps)
            sum_y = sum(scores)
            sum_xy = sum(x * y for x, y in zip(timestamps, scores))
            sum_xx = sum(x * x for x in timestamps)

            if n * sum_xx - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            else:
                slope = 0.0

            # Determine trend direction
            if slope > 0.01:
                trend_direction = 1  # Improving
            elif slope < -0.01:
                trend_direction = -1  # Declining
            else:
                trend_direction = 0  # Stable

            # Trend strength (normalized slope)
            trend_strength = min(1.0, abs(slope) * 100)

            # Change rate (score change per day)
            change_rate = slope * 86400  # seconds per day

        else:
            trend_direction = 0
            trend_strength = 0.0
            change_rate = 0.0

        return {
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "recent_average": recent_average,
            "change_rate": change_rate,
        }

    def _compute_semantic_similarity(
        self,
        capability1: str,
        capability2: str,
        metadata: dict[str, Any] | None,
    ) -> float:
        """
        Compute semantic similarity between capability names.

        Simple implementation using string matching.
        Can be enhanced with word embeddings or LLM in future.

        Args:
            capability1: First capability name
            capability2: Second capability name
            metadata: Optional metadata for enhanced matching

        Returns:
            Similarity score (0-1)
        """
        c1_lower = capability1.lower()
        c2_lower = capability2.lower()

        # Exact match
        if c1_lower == c2_lower:
            return 1.0

        # Contains match
        if c1_lower in c2_lower or c2_lower in c1_lower:
            return 0.8

        # Word overlap
        words1 = set(c1_lower.replace("_", " ").replace("-", " ").split())
        words2 = set(c2_lower.replace("_", " ").replace("-", " ").split())

        if not words1 or not words2:
            return 0.0

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        similarity = overlap / total if total > 0 else 0.0

        return similarity * 0.6  # Scale down for word overlap

    def _compute_consistency_bonus(
        self,
        task_history: list[dict[str, Any]],
    ) -> float:
        """
        Compute consistency bonus from task history.

        Rewards consistent performance over time.

        Args:
            task_history: List of task execution records

        Returns:
            Consistency bonus (0-0.2)
        """
        if len(task_history) < 5:
            return 0.0

        # Extract success/failure from recent history
        recent_results = [
            task.get("success", False) for task in task_history[-10:]
        ]

        # Compute standard deviation of success
        if not recent_results:
            return 0.0

        success_rate = sum(recent_results) / len(recent_results)

        # Low variance = high consistency
        variance = sum(
            (r - success_rate) ** 2 for r in recent_results
        ) / len(recent_results)

        # Convert variance to bonus (inverse relationship)
        consistency_bonus = max(0.0, 0.2 * (1.0 - variance))

        return consistency_bonus
