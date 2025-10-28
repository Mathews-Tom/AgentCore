"""
Pattern recognition for optimization strategies

Identifies successful optimization patterns, analyzes strategy effectiveness,
and provides recommendations based on historical patterns.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    OptimizationResult,
    OptimizationTarget,
    OptimizationTargetType,
)


class PatternType(str, Enum):
    """Type of optimization pattern"""

    ALGORITHM_EFFECTIVENESS = "algorithm_effectiveness"
    PARAMETER_COMBINATION = "parameter_combination"
    TARGET_TYPE_STRATEGY = "target_type_strategy"
    IMPROVEMENT_TRAJECTORY = "improvement_trajectory"
    COST_EFFICIENCY = "cost_efficiency"
    TIME_TO_CONVERGENCE = "time_to_convergence"


class PatternConfidence(str, Enum):
    """Confidence level for pattern recognition"""

    HIGH = "high"  # >= 80% success rate, >= 10 samples
    MEDIUM = "medium"  # >= 60% success rate, >= 5 samples
    LOW = "low"  # >= 40% success rate, >= 3 samples
    INSUFFICIENT = "insufficient"  # < 3 samples


class OptimizationPattern(BaseModel):
    """Recognized optimization pattern"""

    pattern_type: PatternType
    pattern_key: str
    pattern_description: str
    success_rate: float
    sample_count: int
    confidence: PatternConfidence
    avg_improvement: float
    avg_iterations: int
    avg_duration_seconds: float
    best_results: list[str] = Field(default_factory=list)
    common_parameters: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatternRecognizer:
    """
    Pattern recognition for optimization strategies

    Analyzes historical optimization results to identify successful patterns,
    algorithm effectiveness, parameter combinations, and provides data-driven
    recommendations.

    Key features:
    - Algorithm effectiveness analysis
    - Parameter combination patterns
    - Target-specific strategy identification
    - Success pattern recognition
    - Data-driven recommendations
    """

    def __init__(self) -> None:
        """Initialize pattern recognizer"""
        self._patterns: dict[str, OptimizationPattern] = {}

    async def analyze_patterns(
        self,
        results: list[OptimizationResult],
        min_improvement_threshold: float = 0.20,
    ) -> list[OptimizationPattern]:
        """
        Analyze optimization results for patterns

        Args:
            results: Historical optimization results
            min_improvement_threshold: Minimum improvement to consider success

        Returns:
            List of recognized patterns
        """
        if not results:
            return []

        patterns = []

        # Analyze algorithm effectiveness
        algo_patterns = await self._analyze_algorithm_patterns(
            results, min_improvement_threshold
        )
        patterns.extend(algo_patterns)

        # Analyze parameter combinations
        param_patterns = await self._analyze_parameter_patterns(
            results, min_improvement_threshold
        )
        patterns.extend(param_patterns)

        # Analyze target type strategies
        target_patterns = await self._analyze_target_type_patterns(
            results, min_improvement_threshold
        )
        patterns.extend(target_patterns)

        # Analyze improvement trajectories
        trajectory_patterns = await self._analyze_trajectory_patterns(
            results, min_improvement_threshold
        )
        patterns.extend(trajectory_patterns)

        # Store patterns
        for pattern in patterns:
            self._patterns[pattern.pattern_key] = pattern

        return patterns

    async def get_recommendations(
        self,
        target: OptimizationTarget,
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Get optimization recommendations based on patterns

        Args:
            target: Optimization target
            context: Optional context (budget, time constraints, etc.)

        Returns:
            List of recommendations
        """
        recommendations = []

        # Get relevant patterns for target type
        target_patterns = [
            p
            for p in self._patterns.values()
            if p.pattern_type == PatternType.TARGET_TYPE_STRATEGY
            and target.type.value in p.pattern_key
        ]

        # High confidence patterns
        high_conf_patterns = [
            p for p in target_patterns if p.confidence == PatternConfidence.HIGH
        ]

        if high_conf_patterns:
            # Sort by success rate
            high_conf_patterns.sort(key=lambda p: p.success_rate, reverse=True)
            best_pattern = high_conf_patterns[0]

            recommendations.append(
                f"For {target.type.value} optimization, use {best_pattern.pattern_description} "
                f"(success rate: {best_pattern.success_rate:.1%})"
            )

        # Get algorithm recommendations
        algo_patterns = [
            p
            for p in self._patterns.values()
            if p.pattern_type == PatternType.ALGORITHM_EFFECTIVENESS
            and p.confidence in (PatternConfidence.HIGH, PatternConfidence.MEDIUM)
        ]

        if algo_patterns:
            algo_patterns.sort(key=lambda p: p.avg_improvement, reverse=True)
            top_algo = algo_patterns[0]

            recommendations.append(
                f"Algorithm '{top_algo.pattern_key}' shows best results "
                f"(avg improvement: {top_algo.avg_improvement:.1%})"
            )

        # Get parameter recommendations
        param_patterns = [
            p
            for p in self._patterns.values()
            if p.pattern_type == PatternType.PARAMETER_COMBINATION
            and p.confidence == PatternConfidence.HIGH
        ]

        if param_patterns:
            param_patterns.sort(key=lambda p: p.success_rate, reverse=True)
            best_params = param_patterns[0]

            recommendations.append(
                f"Recommended parameters: {best_params.common_parameters}"
            )

        # Context-based recommendations
        if context:
            if context.get("time_constraint") == "low":
                fast_patterns = sorted(
                    self._patterns.values(),
                    key=lambda p: p.avg_iterations,
                )
                if fast_patterns:
                    recommendations.append(
                        f"For fast optimization, use strategy with avg "
                        f"{fast_patterns[0].avg_iterations} iterations"
                    )

            if context.get("budget_constraint") == "low":
                efficient_patterns = [
                    p
                    for p in self._patterns.values()
                    if p.pattern_type == PatternType.COST_EFFICIENCY
                ]
                if efficient_patterns:
                    efficient_patterns.sort(key=lambda p: p.success_rate, reverse=True)
                    recommendations.append(
                        f"For cost efficiency, use {efficient_patterns[0].pattern_description}"
                    )

        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Start with MIPROv2 algorithm with default parameters "
                "(proven baseline performance)"
            )

        return recommendations

    async def find_similar_patterns(
        self,
        result: OptimizationResult,
        limit: int = 5,
    ) -> list[OptimizationPattern]:
        """
        Find patterns similar to given result

        Args:
            result: Optimization result
            limit: Maximum results to return

        Returns:
            List of similar patterns
        """
        if not result.optimization_details:
            return []

        algorithm = result.optimization_details.algorithm_used

        # Find patterns with same algorithm
        similar = [
            p
            for p in self._patterns.values()
            if algorithm in p.pattern_key
            and p.confidence in (PatternConfidence.HIGH, PatternConfidence.MEDIUM)
        ]

        # Sort by success rate
        similar.sort(key=lambda p: p.success_rate, reverse=True)

        return similar[:limit]

    async def _analyze_algorithm_patterns(
        self,
        results: list[OptimizationResult],
        threshold: float,
    ) -> list[OptimizationPattern]:
        """
        Analyze algorithm effectiveness patterns

        Args:
            results: Optimization results
            threshold: Success threshold

        Returns:
            List of algorithm patterns
        """
        algo_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "successful": 0,
                "improvements": [],
                "iterations": [],
                "durations": [],
                "result_ids": [],
            }
        )

        # Collect algorithm statistics
        for result in results:
            if not result.optimization_details:
                continue

            algo = result.optimization_details.algorithm_used
            stats_entry = algo_stats[algo]

            stats_entry["total"] += 1

            if result.improvement_percentage >= threshold:
                stats_entry["successful"] += 1
                stats_entry["result_ids"].append(result.optimization_id)

            stats_entry["improvements"].append(result.improvement_percentage)
            stats_entry["iterations"].append(result.optimization_details.iterations)

            if result.completed_at and result.created_at:
                duration = (result.completed_at - result.created_at).total_seconds()
                stats_entry["durations"].append(duration)

        # Create patterns
        patterns = []

        for algo, stats_entry in algo_stats.items():
            if stats_entry["total"] == 0:
                continue

            success_rate = stats_entry["successful"] / stats_entry["total"]
            avg_improvement = sum(stats_entry["improvements"]) / len(stats_entry["improvements"])
            avg_iterations = sum(stats_entry["iterations"]) / len(stats_entry["iterations"])
            avg_duration = (
                sum(stats_entry["durations"]) / len(stats_entry["durations"])
                if stats_entry["durations"]
                else 0.0
            )

            confidence = self._determine_confidence(
                success_rate,
                stats_entry["total"],
            )

            recommendations = []
            if success_rate >= 0.8:
                recommendations.append(
                    f"Highly effective algorithm for most optimization tasks"
                )
            elif success_rate >= 0.6:
                recommendations.append(
                    f"Reliable algorithm, consider for general use"
                )
            else:
                recommendations.append(
                    f"Use cautiously or combine with other algorithms"
                )

            pattern = OptimizationPattern(
                pattern_type=PatternType.ALGORITHM_EFFECTIVENESS,
                pattern_key=algo,
                pattern_description=f"{algo} algorithm",
                success_rate=success_rate,
                sample_count=stats_entry["total"],
                confidence=confidence,
                avg_improvement=avg_improvement,
                avg_iterations=int(avg_iterations),
                avg_duration_seconds=avg_duration,
                best_results=stats_entry["result_ids"][:5],
                recommendations=recommendations,
                metadata={
                    "algorithm": algo,
                    "successful_count": stats_entry["successful"],
                },
            )

            patterns.append(pattern)

        return patterns

    async def _analyze_parameter_patterns(
        self,
        results: list[OptimizationResult],
        threshold: float,
    ) -> list[OptimizationPattern]:
        """
        Analyze parameter combination patterns

        Args:
            results: Optimization results
            threshold: Success threshold

        Returns:
            List of parameter patterns
        """
        param_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "successful": 0,
                "improvements": [],
                "iterations": [],
                "parameters": [],
            }
        )

        # Collect parameter statistics
        for result in results:
            if not result.optimization_details:
                continue

            params = result.optimization_details.parameters

            # Create key from significant parameters
            param_key = self._create_parameter_key(params)

            if param_key:
                stats_entry = param_stats[param_key]
                stats_entry["total"] += 1

                if result.improvement_percentage >= threshold:
                    stats_entry["successful"] += 1

                stats_entry["improvements"].append(result.improvement_percentage)
                stats_entry["iterations"].append(result.optimization_details.iterations)
                stats_entry["parameters"].append(params)

        # Create patterns
        patterns = []

        for param_key, stats_entry in param_stats.items():
            if stats_entry["total"] < 3:  # Need at least 3 samples
                continue

            success_rate = stats_entry["successful"] / stats_entry["total"]
            avg_improvement = sum(stats_entry["improvements"]) / len(stats_entry["improvements"])
            avg_iterations = sum(stats_entry["iterations"]) / len(stats_entry["iterations"])

            confidence = self._determine_confidence(success_rate, stats_entry["total"])

            # Extract common parameters
            common_params = self._extract_common_parameters(stats_entry["parameters"])

            pattern = OptimizationPattern(
                pattern_type=PatternType.PARAMETER_COMBINATION,
                pattern_key=param_key,
                pattern_description=f"Parameter combination: {param_key}",
                success_rate=success_rate,
                sample_count=stats_entry["total"],
                confidence=confidence,
                avg_improvement=avg_improvement,
                avg_iterations=int(avg_iterations),
                avg_duration_seconds=0.0,
                common_parameters=common_params,
                recommendations=[
                    f"Use these parameter values for similar optimization tasks"
                ],
            )

            patterns.append(pattern)

        return patterns

    async def _analyze_target_type_patterns(
        self,
        results: list[OptimizationResult],
        threshold: float,
    ) -> list[OptimizationPattern]:
        """
        Analyze target type strategy patterns

        Args:
            results: Optimization results
            threshold: Success threshold

        Returns:
            List of target type patterns
        """
        type_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "successful": 0,
                "improvements": [],
                "algorithms": defaultdict(int),
            }
        )

        # Collect statistics by target type
        for result in results:
            if not result.optimization_details:
                continue

            # Try to extract target type from parameters
            target_type = result.optimization_details.parameters.get("target_type", "unknown")

            stats_entry = type_stats[target_type]
            stats_entry["total"] += 1

            if result.improvement_percentage >= threshold:
                stats_entry["successful"] += 1

            stats_entry["improvements"].append(result.improvement_percentage)
            stats_entry["algorithms"][result.optimization_details.algorithm_used] += 1

        # Create patterns
        patterns = []

        for target_type, stats_entry in type_stats.items():
            if stats_entry["total"] < 3:
                continue

            success_rate = stats_entry["successful"] / stats_entry["total"]
            avg_improvement = sum(stats_entry["improvements"]) / len(stats_entry["improvements"])

            # Find most common algorithm
            best_algo = max(stats_entry["algorithms"].items(), key=lambda x: x[1])[0]

            confidence = self._determine_confidence(success_rate, stats_entry["total"])

            pattern = OptimizationPattern(
                pattern_type=PatternType.TARGET_TYPE_STRATEGY,
                pattern_key=f"{target_type}_{best_algo}",
                pattern_description=f"{best_algo} for {target_type} targets",
                success_rate=success_rate,
                sample_count=stats_entry["total"],
                confidence=confidence,
                avg_improvement=avg_improvement,
                avg_iterations=0,
                avg_duration_seconds=0.0,
                recommendations=[
                    f"For {target_type} optimization, {best_algo} is most effective"
                ],
                metadata={
                    "target_type": target_type,
                    "best_algorithm": best_algo,
                },
            )

            patterns.append(pattern)

        return patterns

    async def _analyze_trajectory_patterns(
        self,
        results: list[OptimizationResult],
        threshold: float,
    ) -> list[OptimizationPattern]:
        """
        Analyze improvement trajectory patterns

        Args:
            results: Optimization results
            threshold: Success threshold

        Returns:
            List of trajectory patterns
        """
        # Categorize by improvement ranges
        categories = {
            "excellent": (0.30, float("inf")),  # > 30%
            "target": (0.20, 0.30),  # 20-30%
            "good": (0.10, 0.20),  # 10-20%
            "moderate": (0.05, 0.10),  # 5-10%
        }

        category_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "algorithms": defaultdict(int),
                "avg_iterations": [],
            }
        )

        # Categorize results
        for result in results:
            if not result.optimization_details:
                continue

            improvement = result.improvement_percentage

            for category, (min_val, max_val) in categories.items():
                if min_val <= improvement < max_val:
                    stats_entry = category_stats[category]
                    stats_entry["count"] += 1
                    stats_entry["algorithms"][result.optimization_details.algorithm_used] += 1
                    stats_entry["avg_iterations"].append(result.optimization_details.iterations)
                    break

        # Create patterns
        patterns = []

        for category, stats_entry in category_stats.items():
            if stats_entry["count"] < 3:
                continue

            best_algo = max(stats_entry["algorithms"].items(), key=lambda x: x[1])[0]
            avg_iterations = sum(stats_entry["avg_iterations"]) / len(stats_entry["avg_iterations"])

            pattern = OptimizationPattern(
                pattern_type=PatternType.IMPROVEMENT_TRAJECTORY,
                pattern_key=f"trajectory_{category}",
                pattern_description=f"{category.capitalize()} improvement trajectory",
                success_rate=1.0,  # By definition these are all in the category
                sample_count=stats_entry["count"],
                confidence=self._determine_confidence(1.0, stats_entry["count"]),
                avg_improvement=categories[category][0],
                avg_iterations=int(avg_iterations),
                avg_duration_seconds=0.0,
                recommendations=[
                    f"To achieve {category} results, use {best_algo} with ~{int(avg_iterations)} iterations"
                ],
                metadata={"category": category, "best_algorithm": best_algo},
            )

            patterns.append(pattern)

        return patterns

    def _determine_confidence(
        self,
        success_rate: float,
        sample_count: int,
    ) -> PatternConfidence:
        """
        Determine pattern confidence

        Args:
            success_rate: Success rate
            sample_count: Number of samples

        Returns:
            Pattern confidence level
        """
        if sample_count < 3:
            return PatternConfidence.INSUFFICIENT
        elif success_rate >= 0.80 and sample_count >= 10:
            return PatternConfidence.HIGH
        elif success_rate >= 0.60 and sample_count >= 5:
            return PatternConfidence.MEDIUM
        else:
            return PatternConfidence.LOW

    def _create_parameter_key(self, params: dict[str, Any]) -> str:
        """
        Create key from parameters

        Args:
            params: Parameter dictionary

        Returns:
            Parameter key string
        """
        # Extract significant parameters (not target info)
        significant_params = {
            k: v
            for k, v in params.items()
            if k not in ("target", "target_type", "target_id")
            and isinstance(v, (int, float, bool, str))
        }

        if not significant_params:
            return ""

        # Create sorted key
        key_parts = [f"{k}={v}" for k, v in sorted(significant_params.items())]
        return "_".join(key_parts[:3])  # Limit to 3 most important params

    def _extract_common_parameters(
        self,
        param_list: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Extract common parameters from list

        Args:
            param_list: List of parameter dictionaries

        Returns:
            Common parameters
        """
        if not param_list:
            return {}

        # Find parameters that appear in all entries
        common = {}
        first_params = param_list[0]

        for key, value in first_params.items():
            if all(p.get(key) == value for p in param_list):
                common[key] = value

        return common
