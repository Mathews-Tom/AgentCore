"""
Capability Recommendation Engine - ACE-027

Provides actionable capability change recommendations based on fitness evaluation.
Implements COMPASS ACE-4 recommendation logic.

Performance target: <500ms for recommendation generation
"""

import structlog
from datetime import UTC, datetime
from uuid import UUID, uuid4

from ..models.ace_models import (
    CapabilityFitness,
    CapabilityGap,
    CapabilityRecommendation,
    CapabilityType,
)


logger = structlog.get_logger(__name__)


class CapabilityRecommender:
    """
    Generates capability change recommendations.

    Provides:
    - Capability addition recommendations
    - Capability removal recommendations
    - Capability upgrade suggestions
    - Rationale and confidence scoring
    """

    def __init__(
        self,
        fitness_threshold: float = 0.5,
        removal_threshold: float = 0.3,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize capability recommender.

        Args:
            fitness_threshold: Below this, recommend changes (default: 0.5)
            removal_threshold: Below this, recommend removal (default: 0.3)
            confidence_threshold: Minimum confidence for recommendations (default: 0.7)
        """
        self.logger = logger.bind(component="capability_recommender")
        self.fitness_threshold = fitness_threshold
        self.removal_threshold = removal_threshold
        self.confidence_threshold = confidence_threshold

        self.logger.info(
            "recommender_initialized",
            fitness_threshold=fitness_threshold,
            removal_threshold=removal_threshold,
        )

    async def recommend_capability_changes(
        self,
        agent_id: str,
        task_id: UUID | None,
        task_type: str | None,
        current_capabilities: list[str],
        fitness_scores: dict[str, CapabilityFitness],
        capability_gaps: list[CapabilityGap],
        alternatives_available: dict[str, list[str]] | None = None,
    ) -> CapabilityRecommendation:
        """
        Generate capability change recommendations.

        Args:
            agent_id: Target agent identifier
            task_id: Related task ID if task-specific
            task_type: Task type for context
            current_capabilities: Agent's current capabilities
            fitness_scores: Computed fitness scores
            capability_gaps: Identified capability gaps
            alternatives_available: Optional mapping of capability type to alternatives

        Returns:
            CapabilityRecommendation with actionable suggestions
        """
        start_time = datetime.now(UTC)

        self.logger.info(
            "generating_recommendations",
            agent_id=agent_id,
            task_type=task_type,
            current_count=len(current_capabilities),
            gap_count=len(capability_gaps),
        )

        # Identify underperforming capabilities
        underperforming = self._identify_underperforming(fitness_scores)

        # Generate additions (from gaps)
        to_add = self._recommend_additions(capability_gaps, current_capabilities)

        # Generate removals (from underperforming)
        to_remove = self._recommend_removals(
            underperforming, fitness_scores, capability_gaps
        )

        # Generate upgrades
        to_upgrade = self._recommend_upgrades(
            underperforming, fitness_scores, alternatives_available
        )

        # Evaluate alternatives
        alternatives_evaluated = self._evaluate_alternatives(
            capability_gaps, alternatives_available
        )

        # Compute confidence and expected improvement
        confidence = self._compute_recommendation_confidence(
            fitness_scores, capability_gaps, to_add, to_remove
        )

        expected_improvement = self._estimate_improvement(
            fitness_scores, to_add, to_remove, capability_gaps
        )

        # Determine risk level
        risk_level = self._assess_risk(to_add, to_remove, to_upgrade)

        # Generate rationale
        rationale = self._generate_rationale(
            fitness_scores, capability_gaps, to_add, to_remove, to_upgrade
        )

        # Create recommendation
        recommendation = CapabilityRecommendation(
            recommendation_id=uuid4(),
            agent_id=agent_id,
            task_id=task_id,
            task_type=task_type,
            current_capabilities=current_capabilities,
            underperforming_capabilities=underperforming,
            capabilities_to_add=to_add,
            capabilities_to_remove=to_remove,
            capabilities_to_upgrade=to_upgrade,
            identified_gaps=capability_gaps,
            fitness_scores={k: v.fitness_score for k, v in fitness_scores.items()},
            alternatives_evaluated=alternatives_evaluated,
            rationale=rationale,
            confidence=confidence,
            expected_improvement=expected_improvement,
            risk_level=risk_level,
            generated_at=datetime.now(UTC),
        )

        # Log performance
        duration_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        self.logger.info(
            "recommendations_generated",
            agent_id=agent_id,
            to_add_count=len(to_add),
            to_remove_count=len(to_remove),
            to_upgrade_count=len(to_upgrade),
            confidence=confidence,
            duration_ms=duration_ms,
        )

        return recommendation

    def _identify_underperforming(
        self, fitness_scores: dict[str, CapabilityFitness]
    ) -> list[str]:
        """
        Identify underperforming capabilities.

        Args:
            fitness_scores: Computed fitness scores

        Returns:
            List of underperforming capability IDs
        """
        underperforming = [
            cap_id
            for cap_id, fitness in fitness_scores.items()
            if fitness.fitness_score < self.fitness_threshold
        ]

        self.logger.debug(
            "underperforming_identified",
            count=len(underperforming),
            capabilities=underperforming,
        )

        return underperforming

    def _recommend_additions(
        self, gaps: list[CapabilityGap], current_capabilities: list[str]
    ) -> list[str]:
        """
        Recommend capabilities to add based on gaps.

        Args:
            gaps: Identified capability gaps
            current_capabilities: Current capabilities

        Returns:
            List of capabilities to add
        """
        to_add = []

        for gap in gaps:
            # Only recommend if not already present
            if gap.required_capability not in current_capabilities:
                # Prioritize critical and high severity gaps
                if gap.gap_severity in ["critical", "high"]:
                    to_add.append(gap.required_capability)

        self.logger.debug(
            "additions_recommended",
            count=len(to_add),
            capabilities=to_add,
        )

        return to_add

    def _recommend_removals(
        self,
        underperforming: list[str],
        fitness_scores: dict[str, CapabilityFitness],
        gaps: list[CapabilityGap],
    ) -> list[str]:
        """
        Recommend capabilities to remove.

        Args:
            underperforming: Underperforming capability IDs
            fitness_scores: Fitness scores
            gaps: Capability gaps

        Returns:
            List of capabilities to remove
        """
        to_remove = []

        # Required capabilities (from gaps)
        required = {gap.required_capability for gap in gaps if gap.gap_severity == "critical"}

        for cap_id in underperforming:
            fitness = fitness_scores.get(cap_id)

            # Only recommend removal if:
            # 1. Below removal threshold
            # 2. Not in required capabilities
            # 3. Low usage frequency
            if (
                fitness
                and fitness.fitness_score < self.removal_threshold
                and cap_id not in required
                and fitness.metrics.usage_frequency < 5
            ):
                to_remove.append(cap_id)

        self.logger.debug(
            "removals_recommended",
            count=len(to_remove),
            capabilities=to_remove,
        )

        return to_remove

    def _recommend_upgrades(
        self,
        underperforming: list[str],
        fitness_scores: dict[str, CapabilityFitness],
        alternatives: dict[str, list[str]] | None,
    ) -> list[dict[str, str]]:
        """
        Recommend capability upgrades.

        Args:
            underperforming: Underperforming capability IDs
            fitness_scores: Fitness scores
            alternatives: Available alternatives

        Returns:
            List of upgrade recommendations with current and new versions
        """
        to_upgrade = []

        if not alternatives:
            return to_upgrade

        for cap_id in underperforming:
            fitness = fitness_scores.get(cap_id)

            # Recommend upgrade if moderately underperforming (not for removal)
            if (
                fitness
                and self.removal_threshold <= fitness.fitness_score < self.fitness_threshold
            ):
                # Check if alternatives exist
                for cap_type, alts in alternatives.items():
                    if cap_id in alts or any(cap_id in alt for alt in alts):
                        # Recommend upgrade to latest version
                        to_upgrade.append({
                            "capability": cap_id,
                            "current_version": "current",
                            "recommended_version": "latest",
                            "reason": f"Low fitness score: {fitness.fitness_score:.2f}",
                        })
                        break

        self.logger.debug(
            "upgrades_recommended",
            count=len(to_upgrade),
        )

        return to_upgrade

    def _evaluate_alternatives(
        self,
        gaps: list[CapabilityGap],
        alternatives: dict[str, list[str]] | None,
    ) -> dict[str, float]:
        """
        Evaluate alternative capabilities for gaps.

        Args:
            gaps: Capability gaps
            alternatives: Available alternatives

        Returns:
            Dictionary of alternative capability to estimated fitness score
        """
        evaluated = {}

        if not alternatives:
            return evaluated

        for gap in gaps:
            gap_type = gap.capability_type.value
            if gap_type in alternatives:
                for alt in alternatives[gap_type]:
                    # Estimate fitness based on gap mitigation potential
                    estimated_fitness = min(1.0, gap.required_fitness + 0.2)
                    evaluated[alt] = estimated_fitness

        self.logger.debug(
            "alternatives_evaluated",
            count=len(evaluated),
        )

        return evaluated

    def _compute_recommendation_confidence(
        self,
        fitness_scores: dict[str, CapabilityFitness],
        gaps: list[CapabilityGap],
        to_add: list[str],
        to_remove: list[str],
    ) -> float:
        """
        Compute confidence in recommendations.

        Args:
            fitness_scores: Fitness scores
            gaps: Capability gaps
            to_add: Capabilities to add
            to_remove: Capabilities to remove

        Returns:
            Confidence score (0-1)
        """
        # Base confidence from sample size
        if fitness_scores:
            avg_sample_size = sum(f.sample_size for f in fitness_scores.values()) / len(
                fitness_scores
            )
            sample_confidence = min(1.0, avg_sample_size / 50.0)  # 50+ samples = high confidence
        else:
            sample_confidence = 0.3

        # Confidence from gap severity clarity
        critical_gaps = sum(1 for g in gaps if g.gap_severity == "critical")
        gap_confidence = min(1.0, critical_gaps / max(1, len(gaps)))

        # Confidence penalty for risky recommendations (many removals)
        risk_penalty = min(0.3, len(to_remove) * 0.1)

        confidence = max(0.0, min(1.0, (sample_confidence * 0.6 + gap_confidence * 0.4) - risk_penalty))

        return confidence

    def _estimate_improvement(
        self,
        fitness_scores: dict[str, CapabilityFitness],
        to_add: list[str],
        to_remove: list[str],
        gaps: list[CapabilityGap],
    ) -> float:
        """
        Estimate expected fitness improvement from recommendations.

        Args:
            fitness_scores: Current fitness scores
            to_add: Capabilities to add
            to_remove: Capabilities to remove
            gaps: Capability gaps

        Returns:
            Expected improvement (0-1)
        """
        # Current average fitness
        if fitness_scores:
            current_avg = sum(f.fitness_score for f in fitness_scores.values()) / len(
                fitness_scores
            )
        else:
            current_avg = 0.5

        # Improvement from additions (address gaps)
        addition_improvement = sum(gap.impact for gap in gaps if gap.required_capability in to_add)
        addition_improvement = min(0.5, addition_improvement / max(1, len(gaps)))

        # Improvement from removals (remove poor performers)
        removal_improvement = len(to_remove) * 0.05  # 5% per removal

        total_improvement = min(1.0, addition_improvement + removal_improvement)

        return total_improvement

    def _assess_risk(
        self,
        to_add: list[str],
        to_remove: list[str],
        to_upgrade: list[dict[str, str]],
    ) -> str:
        """
        Assess risk level of recommendations.

        Args:
            to_add: Capabilities to add
            to_remove: Capabilities to remove
            to_upgrade: Capabilities to upgrade

        Returns:
            Risk level: "low", "medium", "high"
        """
        change_count = len(to_add) + len(to_remove) + len(to_upgrade)

        if change_count == 0:
            return "low"
        elif change_count <= 2 and len(to_remove) == 0:
            return "low"
        elif change_count <= 4 or len(to_remove) <= 1:
            return "medium"
        else:
            return "high"

    def _generate_rationale(
        self,
        fitness_scores: dict[str, CapabilityFitness],
        gaps: list[CapabilityGap],
        to_add: list[str],
        to_remove: list[str],
        to_upgrade: list[dict[str, str]],
    ) -> str:
        """
        Generate detailed rationale for recommendations.

        Args:
            fitness_scores: Fitness scores
            gaps: Capability gaps
            to_add: Capabilities to add
            to_remove: Capabilities to remove
            to_upgrade: Capabilities to upgrade

        Returns:
            Rationale text
        """
        parts = []

        # Summary
        if fitness_scores:
            avg_fitness = sum(f.fitness_score for f in fitness_scores.values()) / len(
                fitness_scores
            )
            parts.append(f"Current average capability fitness: {avg_fitness:.2f}")

        # Gaps
        if gaps:
            critical_count = sum(1 for g in gaps if g.gap_severity == "critical")
            high_count = sum(1 for g in gaps if g.gap_severity == "high")
            if critical_count > 0:
                parts.append(f"Identified {critical_count} critical capability gap(s)")
            if high_count > 0:
                parts.append(f"Identified {high_count} high-priority gap(s)")

        # Additions
        if to_add:
            parts.append(
                f"Recommend adding {len(to_add)} capability(ies) to address gaps: {', '.join(to_add[:3])}"
                + ("..." if len(to_add) > 3 else "")
            )

        # Removals
        if to_remove:
            parts.append(
                f"Recommend removing {len(to_remove)} underperforming capability(ies): {', '.join(to_remove[:2])}"
                + ("..." if len(to_remove) > 2 else "")
            )

        # Upgrades
        if to_upgrade:
            parts.append(f"Recommend upgrading {len(to_upgrade)} capability(ies)")

        # Default message if no changes
        has_changes = bool(gaps or to_add or to_remove or to_upgrade)
        if not has_changes:
            parts.append("No significant capability changes recommended at this time")

        return ". ".join(parts) + "."
