"""
Context-aware recommendation engine for optimization strategies

Provides intelligent recommendations based on historical patterns,
current context, constraints, and objectives.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.analytics.patterns import (
    OptimizationPattern,
    PatternConfidence,
    PatternRecognizer,
)
from agentcore.dspy_optimization.insights.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntryType,
)
from agentcore.dspy_optimization.models import (
    OptimizationConstraints,
    OptimizationObjective,
    OptimizationTarget,
)


class RecommendationPriority(str, Enum):
    """Priority level for recommendations"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationType(str, Enum):
    """Type of recommendation"""

    ALGORITHM = "algorithm"
    PARAMETERS = "parameters"
    STRATEGY = "strategy"
    RESOURCE = "resource"
    WARNING = "warning"


class Recommendation(BaseModel):
    """Single recommendation"""

    title: str
    description: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    suggested_parameters: dict[str, Any] = Field(default_factory=dict)
    expected_improvement: float | None = None
    estimated_time: int | None = None
    estimated_cost: float | None = None
    tags: list[str] = Field(default_factory=list)


class RecommendationContext(BaseModel):
    """Context for generating recommendations"""

    target: OptimizationTarget
    objectives: list[OptimizationObjective]
    constraints: OptimizationConstraints
    previous_attempts: list[str] = Field(default_factory=list)
    available_resources: dict[str, Any] = Field(default_factory=dict)
    time_constraint: str | None = None
    budget_constraint: str | None = None
    preferred_algorithms: list[str] = Field(default_factory=list)
    avoid_algorithms: list[str] = Field(default_factory=list)


class RecommendationEngine:
    """
    Context-aware recommendation engine

    Generates intelligent recommendations for optimization strategies
    based on historical patterns, knowledge base, and current context.

    Key features:
    - Multi-source recommendation generation
    - Context-aware prioritization
    - Evidence-based rationale
    - Resource and time estimation
    - Anti-pattern detection and warnings
    """

    def __init__(
        self,
        pattern_recognizer: PatternRecognizer,
        knowledge_base: KnowledgeBase,
    ) -> None:
        """
        Initialize recommendation engine

        Args:
            pattern_recognizer: Pattern recognizer for analysis
            knowledge_base: Knowledge base for historical insights
        """
        self.pattern_recognizer = pattern_recognizer
        self.knowledge_base = knowledge_base

    async def generate_recommendations(
        self,
        context: RecommendationContext,
        max_recommendations: int = 10,
    ) -> list[Recommendation]:
        """
        Generate recommendations for optimization

        Args:
            context: Context including target, objectives, constraints
            max_recommendations: Maximum recommendations to return

        Returns:
            List of prioritized recommendations
        """
        recommendations: list[Recommendation] = []

        # Get pattern-based recommendations
        pattern_recs = await self._get_pattern_recommendations(context)
        recommendations.extend(pattern_recs)

        # Get knowledge-based recommendations
        knowledge_recs = await self._get_knowledge_recommendations(context)
        recommendations.extend(knowledge_recs)

        # Get constraint-aware recommendations
        constraint_recs = await self._get_constraint_recommendations(context)
        recommendations.extend(constraint_recs)

        # Get anti-pattern warnings
        warnings = await self._get_anti_pattern_warnings(context)
        recommendations.extend(warnings)

        # Deduplicate and prioritize
        recommendations = self._deduplicate_recommendations(recommendations)
        recommendations = self._prioritize_recommendations(recommendations, context)

        return recommendations[:max_recommendations]

    async def _get_pattern_recommendations(
        self,
        context: RecommendationContext,
    ) -> list[Recommendation]:
        """
        Get recommendations from pattern analysis

        Args:
            context: Recommendation context

        Returns:
            List of pattern-based recommendations
        """
        recommendations = []

        # Search for relevant patterns
        pattern_entries = await self.knowledge_base.search_entries(
            entry_type=KnowledgeEntryType.PATTERN,
            target_type=context.target.type,
            min_confidence=0.5,
            limit=5,
        )

        for entry in pattern_entries:
            # Extract pattern info
            pattern_ctx = entry.context
            success_rate = entry.success_rate

            if success_rate < 0.6:
                continue

            # Determine priority
            if success_rate >= 0.8:
                priority = RecommendationPriority.HIGH
            elif success_rate >= 0.7:
                priority = RecommendationPriority.MEDIUM
            else:
                priority = RecommendationPriority.LOW

            # Extract algorithm from pattern
            algorithm = pattern_ctx.get("pattern_key", "").split("_")[0]

            # Skip if in avoid list
            if algorithm in context.avoid_algorithms:
                continue

            recommendation = Recommendation(
                title=f"Use proven algorithm: {algorithm}",
                description=entry.description,
                recommendation_type=RecommendationType.ALGORITHM,
                priority=priority,
                rationale=f"Pattern shows {success_rate:.1%} success rate with "
                f"avg {pattern_ctx.get('avg_improvement', 0):.1%} improvement",
                confidence=entry.confidence_score,
                supporting_evidence=entry.evidence[:3],
                suggested_parameters=pattern_ctx.get("common_parameters", {}),
                expected_improvement=pattern_ctx.get("avg_improvement"),
                estimated_time=pattern_ctx.get("avg_iterations"),
                tags=["pattern-based", algorithm],
            )

            recommendations.append(recommendation)

        return recommendations

    async def _get_knowledge_recommendations(
        self,
        context: RecommendationContext,
    ) -> list[Recommendation]:
        """
        Get recommendations from knowledge base

        Args:
            context: Recommendation context

        Returns:
            List of knowledge-based recommendations
        """
        recommendations = []

        # Get best practices
        best_practices = await self.knowledge_base.get_best_practices(
            target_type=context.target.type,
            min_confidence=0.7,
        )

        for practice in best_practices:
            recommendation = Recommendation(
                title=practice.title,
                description=practice.description,
                recommendation_type=RecommendationType.STRATEGY,
                priority=RecommendationPriority.MEDIUM,
                rationale=f"Best practice with {practice.success_rate:.1%} success rate "
                f"(used {practice.usage_count} times)",
                confidence=practice.confidence_score,
                supporting_evidence=practice.evidence[:3],
                tags=practice.tags + ["best-practice"],
            )

            recommendations.append(recommendation)

        # Get lessons learned
        lessons = await self.knowledge_base.search_entries(
            entry_type=KnowledgeEntryType.LESSON,
            target_type=context.target.type,
            min_confidence=0.6,
            limit=3,
        )

        for lesson in lessons:
            recommendation = Recommendation(
                title=f"Lesson: {lesson.title}",
                description=lesson.description,
                recommendation_type=RecommendationType.STRATEGY,
                priority=RecommendationPriority.LOW,
                rationale="Learned from historical optimization results",
                confidence=lesson.confidence_score,
                supporting_evidence=lesson.evidence[:3],
                tags=lesson.tags + ["lesson"],
            )

            recommendations.append(recommendation)

        return recommendations

    async def _get_constraint_recommendations(
        self,
        context: RecommendationContext,
    ) -> list[Recommendation]:
        """
        Get constraint-aware recommendations

        Args:
            context: Recommendation context

        Returns:
            List of constraint-based recommendations
        """
        recommendations = []

        # Time constraint recommendations
        if context.time_constraint == "low" or context.constraints.max_optimization_time < 1800:
            recommendation = Recommendation(
                title="Use fast-converging algorithm",
                description="Select algorithm optimized for quick convergence due to time constraints",
                recommendation_type=RecommendationType.ALGORITHM,
                priority=RecommendationPriority.HIGH,
                rationale=f"Time constraint: {context.constraints.max_optimization_time}s",
                confidence=0.8,
                suggested_parameters={"max_iterations": 50, "early_stopping": True},
                tags=["time-constraint", "fast"],
            )
            recommendations.append(recommendation)

        # Resource constraint recommendations
        if (
            context.budget_constraint == "low"
            or context.constraints.max_resource_usage < 0.1
        ):
            recommendation = Recommendation(
                title="Optimize for resource efficiency",
                description="Use lightweight algorithms with reduced resource consumption",
                recommendation_type=RecommendationType.RESOURCE,
                priority=RecommendationPriority.HIGH,
                rationale=f"Resource limit: {context.constraints.max_resource_usage:.1%}",
                confidence=0.8,
                suggested_parameters={
                    "batch_size": "small",
                    "parallel_trials": 1,
                },
                tags=["resource-constraint", "efficient"],
            )
            recommendations.append(recommendation)

        # High improvement threshold recommendations
        if context.constraints.min_improvement_threshold > 0.20:
            recommendation = Recommendation(
                title="Use aggressive optimization strategy",
                description="Target high improvement with advanced algorithms and extended search",
                recommendation_type=RecommendationType.STRATEGY,
                priority=RecommendationPriority.MEDIUM,
                rationale=f"High target: {context.constraints.min_improvement_threshold:.1%}",
                confidence=0.7,
                suggested_parameters={
                    "exploration_rate": 0.3,
                    "max_iterations": 200,
                },
                tags=["aggressive", "high-target"],
            )
            recommendations.append(recommendation)

        return recommendations

    async def _get_anti_pattern_warnings(
        self,
        context: RecommendationContext,
    ) -> list[Recommendation]:
        """
        Get warnings about anti-patterns

        Args:
            context: Recommendation context

        Returns:
            List of warning recommendations
        """
        warnings = []

        # Get anti-patterns
        anti_patterns = await self.knowledge_base.get_anti_patterns(
            target_type=context.target.type,
            min_confidence=0.5,
        )

        for anti_pattern in anti_patterns:
            # Extract algorithm from anti-pattern
            algo_tags = [
                tag
                for tag in anti_pattern.tags
                if tag not in ("unreliable", "failure", "anti-pattern")
            ]

            warning = Recommendation(
                title=f"Warning: {anti_pattern.title}",
                description=anti_pattern.description,
                recommendation_type=RecommendationType.WARNING,
                priority=RecommendationPriority.CRITICAL,
                rationale=f"Known failure pattern with {anti_pattern.metadata.get('failure_rate', 0):.1%} failure rate",
                confidence=anti_pattern.confidence_score,
                supporting_evidence=anti_pattern.evidence[:3],
                tags=["warning", "anti-pattern"] + algo_tags,
            )

            warnings.append(warning)

        # Check for problematic previous attempts
        if len(context.previous_attempts) > 3:
            warning = Recommendation(
                title="Multiple optimization attempts detected",
                description="Consider reassessing optimization strategy or target selection",
                recommendation_type=RecommendationType.WARNING,
                priority=RecommendationPriority.HIGH,
                rationale=f"{len(context.previous_attempts)} previous attempts may indicate issues",
                confidence=0.6,
                tags=["warning", "multiple-attempts"],
            )
            warnings.append(warning)

        return warnings

    def _deduplicate_recommendations(
        self,
        recommendations: list[Recommendation],
    ) -> list[Recommendation]:
        """
        Remove duplicate recommendations

        Args:
            recommendations: List of recommendations

        Returns:
            Deduplicated list
        """
        seen_titles = set()
        unique = []

        for rec in recommendations:
            if rec.title not in seen_titles:
                seen_titles.add(rec.title)
                unique.append(rec)

        return unique

    def _prioritize_recommendations(
        self,
        recommendations: list[Recommendation],
        context: RecommendationContext,
    ) -> list[Recommendation]:
        """
        Prioritize recommendations

        Args:
            recommendations: List of recommendations
            context: Recommendation context

        Returns:
            Sorted list by priority and relevance
        """
        priority_scores = {
            RecommendationPriority.CRITICAL: 4,
            RecommendationPriority.HIGH: 3,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 1,
        }

        def score_recommendation(rec: Recommendation) -> tuple[int, float]:
            # Primary sort by priority
            priority_score = priority_scores[rec.priority]

            # Secondary sort by confidence
            confidence = rec.confidence

            # Boost for preferred algorithms
            if context.preferred_algorithms:
                for algo in context.preferred_algorithms:
                    if algo in rec.tags:
                        confidence += 0.1

            return (priority_score, confidence)

        recommendations.sort(key=score_recommendation, reverse=True)

        return recommendations

    async def explain_recommendation(
        self,
        recommendation: Recommendation,
    ) -> dict[str, Any]:
        """
        Provide detailed explanation for recommendation

        Args:
            recommendation: Recommendation to explain

        Returns:
            Detailed explanation dictionary
        """
        explanation = {
            "title": recommendation.title,
            "type": recommendation.recommendation_type.value,
            "priority": recommendation.priority.value,
            "confidence": recommendation.confidence,
            "rationale": recommendation.rationale,
            "evidence_count": len(recommendation.supporting_evidence),
            "evidence_sample": recommendation.supporting_evidence[:3],
            "tags": recommendation.tags,
        }

        # Add expected outcomes
        if recommendation.expected_improvement:
            explanation["expected_improvement"] = f"{recommendation.expected_improvement:.1%}"

        if recommendation.estimated_time:
            explanation["estimated_iterations"] = recommendation.estimated_time

        if recommendation.estimated_cost:
            explanation["estimated_cost"] = recommendation.estimated_cost

        # Add parameter details
        if recommendation.suggested_parameters:
            explanation["suggested_parameters"] = recommendation.suggested_parameters

        return explanation
