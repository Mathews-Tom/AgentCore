"""
Best practices extraction and documentation

Automatically extracts best practices from optimization results
and generates structured documentation for team guidance.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.analytics.patterns import (
    OptimizationPattern,
    PatternConfidence,
)
from agentcore.dspy_optimization.models import (
    OptimizationResult,
    OptimizationTargetType,
)


class BestPracticeCategory(str, Enum):
    """Category of best practice"""

    ALGORITHM_SELECTION = "algorithm_selection"
    PARAMETER_TUNING = "parameter_tuning"
    RESOURCE_MANAGEMENT = "resource_management"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    COST_OPTIMIZATION = "cost_optimization"
    QUALITY_ASSURANCE = "quality_assurance"
    WORKFLOW_DESIGN = "workflow_design"


class BestPractice(BaseModel):
    """Single best practice"""

    practice_id: str
    category: BestPracticeCategory
    title: str
    description: str
    rationale: str
    applicability: list[OptimizationTargetType] = Field(default_factory=list)
    do_list: list[str] = Field(default_factory=list)
    dont_list: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    impact: str = Field(default="medium")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class BestPracticeExtractor:
    """
    Best practices extraction engine

    Analyzes optimization patterns and results to automatically
    extract actionable best practices with supporting evidence.

    Key features:
    - Automatic practice extraction from patterns
    - Evidence-based recommendations
    - Category classification
    - Do/Don't list generation
    - Markdown documentation generation
    """

    def __init__(self) -> None:
        """Initialize best practice extractor"""
        self._practices: dict[str, BestPractice] = {}

    async def extract_from_patterns(
        self,
        patterns: list[OptimizationPattern],
        results: list[OptimizationResult],
    ) -> list[BestPractice]:
        """
        Extract best practices from patterns

        Args:
            patterns: Analyzed patterns
            results: Supporting optimization results

        Returns:
            List of extracted best practices
        """
        practices = []

        # Extract algorithm selection practices
        algo_practices = await self._extract_algorithm_practices(patterns, results)
        practices.extend(algo_practices)

        # Extract parameter tuning practices
        param_practices = await self._extract_parameter_practices(patterns, results)
        practices.extend(param_practices)

        # Extract performance practices
        perf_practices = await self._extract_performance_practices(patterns, results)
        practices.extend(perf_practices)

        # Extract cost optimization practices
        cost_practices = await self._extract_cost_practices(patterns, results)
        practices.extend(cost_practices)

        # Store practices
        for practice in practices:
            self._practices[practice.practice_id] = practice

        return practices

    async def _extract_algorithm_practices(
        self,
        patterns: list[OptimizationPattern],
        results: list[OptimizationResult],
    ) -> list[BestPractice]:
        """Extract algorithm selection best practices"""
        practices = []

        # Find most successful algorithms
        algo_success: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "success": 0, "avg_improvement": 0.0, "evidence": []}
        )

        for result in results:
            if not result.optimization_details:
                continue

            algo = result.optimization_details.algorithm_used
            algo_success[algo]["count"] += 1

            if result.improvement_percentage >= 0.20:
                algo_success[algo]["success"] += 1
                algo_success[algo]["evidence"].append(result.optimization_id)

            algo_success[algo]["avg_improvement"] += result.improvement_percentage

        # Create practices for successful algorithms
        for algo, stats in algo_success.items():
            if stats["count"] < 3:
                continue

            success_rate = stats["success"] / stats["count"]
            avg_improvement = stats["avg_improvement"] / stats["count"]

            if success_rate >= 0.7:
                practice = BestPractice(
                    practice_id=f"algo_{algo}_{int(datetime.now(UTC).timestamp())}",
                    category=BestPracticeCategory.ALGORITHM_SELECTION,
                    title=f"Prefer {algo} for reliable optimization",
                    description=f"Algorithm {algo} demonstrates consistent success "
                    f"with {success_rate:.1%} success rate and "
                    f"{avg_improvement:.1%} average improvement",
                    rationale=f"Based on {stats['count']} optimization runs, "
                    f"{algo} consistently achieves target improvements",
                    do_list=[
                        f"Use {algo} as primary optimization algorithm",
                        "Monitor convergence and adjust iterations as needed",
                        "Start with default parameters and tune based on results",
                    ],
                    dont_list=[
                        "Don't skip baseline measurement",
                        "Don't terminate optimization early without validation",
                    ],
                    examples=[f"Result ID: {eid}" for eid in stats["evidence"][:3]],
                    supporting_evidence=stats["evidence"],
                    confidence=min(success_rate, 0.95),
                    impact="high",
                    metadata={
                        "algorithm": algo,
                        "success_rate": success_rate,
                        "sample_count": stats["count"],
                    },
                )

                practices.append(practice)

        return practices

    async def _extract_parameter_practices(
        self,
        patterns: list[OptimizationPattern],
        results: list[OptimizationResult],
    ) -> list[BestPractice]:
        """Extract parameter tuning best practices"""
        practices = []

        # Find successful parameter combinations
        param_patterns = [p for p in patterns if "parameter" in p.pattern_type.value]

        for pattern in param_patterns:
            if pattern.confidence not in (
                PatternConfidence.HIGH,
                PatternConfidence.MEDIUM,
            ):
                continue

            if not pattern.common_parameters:
                continue

            practice = BestPractice(
                practice_id=f"param_{pattern.pattern_key}_{int(datetime.now(UTC).timestamp())}",
                category=BestPracticeCategory.PARAMETER_TUNING,
                title=f"Optimize with proven parameter combination",
                description=f"Parameter set achieving {pattern.success_rate:.1%} success rate",
                rationale=f"Based on {pattern.sample_count} runs, this parameter "
                "combination consistently produces improvements",
                do_list=[
                    "Start with these proven parameter values",
                    "Make small incremental adjustments",
                    "Test changes with A/B validation",
                ],
                dont_list=[
                    "Don't make multiple parameter changes simultaneously",
                    "Don't skip validation after parameter changes",
                ],
                examples=pattern.best_results[:3],
                supporting_evidence=pattern.best_results,
                confidence=pattern.success_rate,
                impact="medium",
                metadata={
                    "parameters": pattern.common_parameters,
                    "pattern_key": pattern.pattern_key,
                },
            )

            practices.append(practice)

        return practices

    async def _extract_performance_practices(
        self,
        patterns: list[OptimizationPattern],
        results: list[OptimizationResult],
    ) -> list[BestPractice]:
        """Extract performance optimization best practices"""
        practices = []

        # Identify fast-converging patterns
        fast_patterns = [p for p in patterns if p.avg_iterations < 50]

        if len(fast_patterns) >= 2:
            # Find common characteristics
            common_algos = [p.pattern_key for p in fast_patterns]
            most_common_algo = max(set(common_algos), key=common_algos.count)

            avg_iterations = sum(p.avg_iterations for p in fast_patterns) / len(
                fast_patterns
            )

            practice = BestPractice(
                practice_id=f"perf_fast_converge_{int(datetime.now(UTC).timestamp())}",
                category=BestPracticeCategory.PERFORMANCE_OPTIMIZATION,
                title="Use fast-converging algorithms for quick iteration",
                description=f"Achieve results in ~{int(avg_iterations)} iterations "
                f"using {most_common_algo}",
                rationale="Fast convergence enables rapid experimentation and feedback",
                do_list=[
                    "Use early stopping to prevent over-optimization",
                    "Set reasonable iteration limits",
                    "Monitor convergence metrics",
                ],
                dont_list=[
                    "Don't sacrifice quality for speed without validation",
                    "Don't skip statistical significance testing",
                ],
                examples=[p.pattern_key for p in fast_patterns[:3]],
                supporting_evidence=[e for p in fast_patterns for e in p.best_results],
                confidence=0.8,
                impact="medium",
                metadata={"avg_iterations": avg_iterations},
            )

            practices.append(practice)

        # Identify high-performance patterns
        high_perf = [p for p in patterns if p.avg_improvement >= 0.30]

        if len(high_perf) >= 2:
            best_algos = [p.pattern_key for p in high_perf]
            most_effective = max(set(best_algos), key=best_algos.count)

            avg_improvement = sum(p.avg_improvement for p in high_perf) / len(high_perf)

            practice = BestPractice(
                practice_id=f"perf_high_impact_{int(datetime.now(UTC).timestamp())}",
                category=BestPracticeCategory.PERFORMANCE_OPTIMIZATION,
                title="Target aggressive improvement with proven strategies",
                description=f"Achieve {avg_improvement:.1%}+ improvement "
                f"using {most_effective}",
                rationale="High-impact optimizations justify additional resource investment",
                do_list=[
                    "Allocate sufficient time for thorough optimization",
                    "Use larger exploration budgets",
                    "Validate improvements with statistical testing",
                ],
                dont_list=[
                    "Don't over-optimize beyond practical benefits",
                    "Don't ignore resource costs of aggressive optimization",
                ],
                examples=[p.pattern_key for p in high_perf[:3]],
                supporting_evidence=[e for p in high_perf for e in p.best_results],
                confidence=0.85,
                impact="high",
                metadata={"avg_improvement": avg_improvement},
            )

            practices.append(practice)

        return practices

    async def _extract_cost_practices(
        self,
        patterns: list[OptimizationPattern],
        results: list[OptimizationResult],
    ) -> list[BestPractice]:
        """Extract cost optimization best practices"""
        practices = []

        # Identify efficient patterns (good results with low iterations)
        efficient = [
            p for p in patterns if p.avg_improvement >= 0.20 and p.avg_iterations < 100
        ]

        if len(efficient) >= 2:
            avg_iterations = sum(p.avg_iterations for p in efficient) / len(efficient)
            avg_improvement = sum(p.avg_improvement for p in efficient) / len(efficient)

            practice = BestPractice(
                practice_id=f"cost_efficient_{int(datetime.now(UTC).timestamp())}",
                category=BestPracticeCategory.COST_OPTIMIZATION,
                title="Balance performance and cost efficiency",
                description=f"Achieve {avg_improvement:.1%} improvement "
                f"in ~{int(avg_iterations)} iterations",
                rationale="Cost-effective optimization maximizes ROI",
                do_list=[
                    "Set iteration limits based on improvement goals",
                    "Use early stopping when target is reached",
                    "Monitor resource usage continuously",
                ],
                dont_list=[
                    "Don't run excessive iterations for diminishing returns",
                    "Don't ignore resource consumption metrics",
                ],
                examples=[p.pattern_key for p in efficient[:3]],
                supporting_evidence=[e for p in efficient for e in p.best_results],
                confidence=0.75,
                impact="medium",
                metadata={
                    "avg_iterations": avg_iterations,
                    "avg_improvement": avg_improvement,
                },
            )

            practices.append(practice)

        return practices

    def generate_markdown_documentation(
        self,
        title: str = "DSPy Optimization Best Practices",
    ) -> str:
        """
        Generate markdown documentation of best practices

        Args:
            title: Document title

        Returns:
            Markdown formatted documentation
        """
        lines = [
            f"# {title}",
            "",
            f"*Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
            "",
            "This document contains automatically extracted best practices from historical optimization data.",
            "",
            "## Table of Contents",
            "",
        ]

        # Group practices by category
        by_category: dict[BestPracticeCategory, list[BestPractice]] = defaultdict(list)
        for practice in self._practices.values():
            by_category[practice.category].append(practice)

        # Add TOC
        for category in BestPracticeCategory:
            if category in by_category:
                practices = by_category[category]
                category_title = category.value.replace("_", " ").title()
                lines.append(f"- [{category_title}](#{category.value})")

        lines.extend(["", "---", ""])

        # Add practices by category
        for category in BestPracticeCategory:
            if category not in by_category:
                continue

            practices = by_category[category]
            category_title = category.value.replace("_", " ").title()

            lines.extend(
                [f"## {category_title}", "", f'<a name="{category.value}"></a>', ""]
            )

            for practice in practices:
                lines.extend(self._format_practice_markdown(practice))

        # Add metadata section
        lines.extend(
            [
                "---",
                "",
                "## Metadata",
                "",
                f"- **Total Practices**: {len(self._practices)}",
                f"- **Categories**: {len(by_category)}",
                f"- **Average Confidence**: {sum(p.confidence for p in self._practices.values()) / len(self._practices):.2f}"
                if self._practices
                else "- **Average Confidence**: N/A",
                "",
            ]
        )

        return "\n".join(lines)

    def _format_practice_markdown(self, practice: BestPractice) -> list[str]:
        """Format single practice as markdown"""
        lines = [
            f"### {practice.title}",
            "",
            f"**Impact**: {practice.impact.upper()} | **Confidence**: {practice.confidence:.0%}",
            "",
            practice.description,
            "",
            f"**Rationale**: {practice.rationale}",
            "",
        ]

        # Add applicability
        if practice.applicability:
            targets = ", ".join(t.value for t in practice.applicability)
            lines.extend([f"**Applicable to**: {targets}", ""])

        # Add do/don't lists
        if practice.do_list:
            lines.extend(["**Do:**", ""])
            for item in practice.do_list:
                lines.append(f"- ✅ {item}")
            lines.append("")

        if practice.dont_list:
            lines.extend(["**Don't:**", ""])
            for item in practice.dont_list:
                lines.append(f"- ❌ {item}")
            lines.append("")

        # Add examples
        if practice.examples:
            lines.extend(["**Examples:**", ""])
            for example in practice.examples[:3]:
                lines.append(f"- `{example}`")
            lines.append("")

        # Add evidence count
        if practice.supporting_evidence:
            lines.append(
                f"*Based on {len(practice.supporting_evidence)} optimization runs*"
            )
            lines.append("")

        lines.append("---")
        lines.append("")

        return lines

    def get_practices_by_category(
        self,
        category: BestPracticeCategory,
    ) -> list[BestPractice]:
        """
        Get practices by category

        Args:
            category: Best practice category

        Returns:
            List of practices in category
        """
        return [p for p in self._practices.values() if p.category == category]

    def get_high_impact_practices(
        self,
        min_confidence: float = 0.7,
    ) -> list[BestPractice]:
        """
        Get high-impact practices

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of high-impact practices
        """
        return [
            p
            for p in self._practices.values()
            if p.impact == "high" and p.confidence >= min_confidence
        ]

    def export_practices(self) -> list[dict[str, Any]]:
        """
        Export practices as JSON-serializable dicts

        Returns:
            List of practice dictionaries
        """
        return [p.model_dump(mode="json") for p in self._practices.values()]
