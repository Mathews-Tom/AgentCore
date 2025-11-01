"""
Algorithm validator

Comprehensive validation of optimization algorithms combining benchmarks,
baseline comparisons, statistical testing, and reproducibility validation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.algorithms.base import BaseOptimizer
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.monitoring.statistics import (
    SignificanceResult,
    StatisticalTester,
)
from agentcore.dspy_optimization.validation.baselines import (
    BaselineComparison,
    GridSearchOptimizer,
    RandomSearchOptimizer,
)
from agentcore.dspy_optimization.validation.benchmarks import (
    BenchmarkResult,
    BenchmarkSuite,
)
from agentcore.dspy_optimization.validation.reproducibility import (
    ReproducibilityResult,
    ReproducibilityValidator,
)


class ValidationResult(BaseModel):
    """Result of algorithm validation"""

    algorithm_name: str
    overall_score: float
    passes_validation: bool
    benchmark_results: list[BenchmarkResult] = Field(default_factory=list)
    baseline_comparison: BaselineComparison | None = None
    statistical_significance: SignificanceResult | None = None
    reproducibility: ReproducibilityResult | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendation: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Comprehensive validation report"""

    validation_results: list[ValidationResult] = Field(default_factory=list)
    algorithm_rankings: list[dict[str, Any]] = Field(default_factory=list)
    summary: str
    recommendations: list[str] = Field(default_factory=list)
    timestamp: str


class AlgorithmValidator:
    """
    Comprehensive algorithm validator

    Validates optimization algorithms through:
    1. Benchmark suite against research claims
    2. Baseline comparisons (random/grid search)
    3. Statistical significance testing
    4. Reproducibility validation
    """

    def __init__(
        self,
        benchmark_suite: BenchmarkSuite | None = None,
        statistical_tester: StatisticalTester | None = None,
        reproducibility_validator: ReproducibilityValidator | None = None,
    ) -> None:
        """
        Initialize algorithm validator

        Args:
            benchmark_suite: Benchmark suite (uses default if None)
            statistical_tester: Statistical tester (uses default if None)
            reproducibility_validator: Reproducibility validator (uses default if None)
        """
        self.benchmark_suite = benchmark_suite or BenchmarkSuite()
        self.statistical_tester = statistical_tester or StatisticalTester()
        self.reproducibility_validator = (
            reproducibility_validator or ReproducibilityValidator()
        )

        # Add standard benchmarks if suite is empty
        if not self.benchmark_suite.benchmarks:
            self.benchmark_suite.add_standard_benchmarks()

    async def validate_algorithm(
        self,
        optimizer: BaseOptimizer,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate algorithm comprehensively

        Args:
            optimizer: Optimizer to validate
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data

        Returns:
            Validation result
        """
        algorithm_name = optimizer.get_algorithm_name()

        # 1. Run benchmarks
        benchmark_results = await self.benchmark_suite.run_all(
            optimizers=[optimizer],
            request=request,
        )

        # 2. Compare against baselines
        baseline_comparison = await self._compare_baselines(
            optimizer=optimizer,
            request=request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )

        # 3. Statistical significance testing
        statistical_significance = await self._test_statistical_significance(
            optimizer=optimizer,
            request=request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )

        # 4. Reproducibility validation
        reproducibility = await self.reproducibility_validator.validate_algorithm(
            optimizer=optimizer,
            request=request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            benchmark_results=benchmark_results,
            baseline_comparison=baseline_comparison,
            statistical_significance=statistical_significance,
            reproducibility=reproducibility,
        )

        # Determine if passes validation
        passes_validation = self._check_validation_passing(
            overall_score=overall_score,
            benchmark_results=benchmark_results,
            baseline_comparison=baseline_comparison,
            statistical_significance=statistical_significance,
            reproducibility=reproducibility,
        )

        # Identify strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(
            benchmark_results=benchmark_results,
            baseline_comparison=baseline_comparison,
            statistical_significance=statistical_significance,
            reproducibility=reproducibility,
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            algorithm_name=algorithm_name,
            passes_validation=passes_validation,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        return ValidationResult(
            algorithm_name=algorithm_name,
            overall_score=overall_score,
            passes_validation=passes_validation,
            benchmark_results=benchmark_results,
            baseline_comparison=baseline_comparison,
            statistical_significance=statistical_significance,
            reproducibility=reproducibility,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=recommendation,
        )

    async def validate_multiple_algorithms(
        self,
        optimizers: list[BaseOptimizer],
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> ValidationReport:
        """
        Validate multiple algorithms and generate report

        Args:
            optimizers: List of optimizers to validate
            request: Optimization request
            baseline_metrics: Baseline metrics
            training_data: Training data

        Returns:
            Validation report
        """
        from datetime import UTC, datetime

        validation_results: list[ValidationResult] = []

        # Validate each algorithm
        for optimizer in optimizers:
            result = await self.validate_algorithm(
                optimizer=optimizer,
                request=request,
                baseline_metrics=baseline_metrics,
                training_data=training_data,
            )
            validation_results.append(result)

        # Rank algorithms by overall score
        algorithm_rankings = sorted(
            [
                {
                    "algorithm_name": r.algorithm_name,
                    "overall_score": r.overall_score,
                    "passes_validation": r.passes_validation,
                }
                for r in validation_results
            ],
            key=lambda x: x["overall_score"],
            reverse=True,
        )

        # Generate summary and recommendations
        summary = self._generate_summary(validation_results, algorithm_rankings)
        recommendations = self._generate_recommendations(
            validation_results, algorithm_rankings
        )

        return ValidationReport(
            validation_results=validation_results,
            algorithm_rankings=algorithm_rankings,
            summary=summary,
            recommendations=recommendations,
            timestamp=datetime.now(UTC).isoformat(),
        )

    async def _compare_baselines(
        self,
        optimizer: BaseOptimizer,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> BaselineComparison:
        """Compare algorithm against baseline optimizers"""
        # Run algorithm
        algorithm_result = await optimizer.optimize(
            request=request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )

        # Run random search
        random_search = RandomSearchOptimizer(llm=optimizer.llm, num_trials=10, seed=42)
        random_result = await random_search.optimize(
            request=request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )

        # Run grid search
        grid_search = GridSearchOptimizer(llm=optimizer.llm, grid_size=3)
        grid_result = await grid_search.optimize(
            request=request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )

        # Compare improvements
        algorithm_improvement = algorithm_result.improvement_percentage
        random_improvement = random_result.improvement_percentage
        grid_improvement = grid_result.improvement_percentage

        return BaselineComparison(
            algorithm_name=optimizer.get_algorithm_name(),
            algorithm_improvement=algorithm_improvement,
            random_search_improvement=random_improvement,
            grid_search_improvement=grid_improvement,
            beats_random_search=algorithm_improvement > random_improvement,
            beats_grid_search=algorithm_improvement > grid_improvement,
            improvement_over_random=algorithm_improvement - random_improvement,
            improvement_over_grid=algorithm_improvement - grid_improvement,
        )

    async def _test_statistical_significance(
        self,
        optimizer: BaseOptimizer,
        request: OptimizationRequest,
        baseline_metrics: PerformanceMetrics,
        training_data: list[dict[str, Any]],
    ) -> SignificanceResult:
        """Test statistical significance of improvements"""
        # Generate baseline samples
        baseline_samples = [
            {
                "success_rate": baseline_metrics.success_rate,
                "cost": baseline_metrics.avg_cost_per_task,
                "latency": baseline_metrics.avg_latency_ms,
            }
            for _ in range(30)
        ]

        # Run optimization and generate optimized samples
        result = await optimizer.optimize(
            request=request,
            baseline_metrics=baseline_metrics,
            training_data=training_data,
        )

        optimized_metrics = result.optimized_performance or baseline_metrics
        optimized_samples = [
            {
                "success_rate": optimized_metrics.success_rate,
                "cost": optimized_metrics.avg_cost_per_task,
                "latency": optimized_metrics.avg_latency_ms,
            }
            for _ in range(30)
        ]

        # Perform statistical test
        return await self.statistical_tester.compare_metrics(
            baseline_samples=baseline_samples,
            optimized_samples=optimized_samples,
        )

    def _calculate_overall_score(
        self,
        benchmark_results: list[BenchmarkResult],
        baseline_comparison: BaselineComparison,
        statistical_significance: SignificanceResult,
        reproducibility: ReproducibilityResult,
    ) -> float:
        """Calculate overall validation score (0-100)"""
        score = 0.0

        # Benchmark score (40 points)
        if benchmark_results:
            benchmark_score = sum(
                25 if r.meets_research_claims else 0 for r in benchmark_results
            ) / len(benchmark_results)
            score += benchmark_score

        # Baseline comparison score (25 points)
        if baseline_comparison.beats_random_search:
            score += 10
        if baseline_comparison.beats_grid_search:
            score += 15

        # Statistical significance score (20 points)
        if statistical_significance.is_significant:
            score += 20

        # Reproducibility score (15 points)
        if reproducibility.is_reproducible:
            score += 15

        return min(score, 100.0)

    def _check_validation_passing(
        self,
        overall_score: float,
        benchmark_results: list[BenchmarkResult],
        baseline_comparison: BaselineComparison,
        statistical_significance: SignificanceResult,
        reproducibility: ReproducibilityResult,
    ) -> bool:
        """Check if algorithm passes validation"""
        # Must meet minimum criteria
        return (
            overall_score >= 60.0
            and any(r.meets_research_claims for r in benchmark_results)
            and baseline_comparison.beats_random_search
            and statistical_significance.is_significant
            and reproducibility.is_reproducible
        )

    def _analyze_strengths_weaknesses(
        self,
        benchmark_results: list[BenchmarkResult],
        baseline_comparison: BaselineComparison,
        statistical_significance: SignificanceResult,
        reproducibility: ReproducibilityResult,
    ) -> tuple[list[str], list[str]]:
        """Identify algorithm strengths and weaknesses"""
        strengths: list[str] = []
        weaknesses: list[str] = []

        # Benchmark analysis
        meets_claims = sum(1 for r in benchmark_results if r.meets_research_claims)
        if meets_claims == len(benchmark_results):
            strengths.append("Meets all research paper claims")
        elif meets_claims > 0:
            strengths.append(f"Meets {meets_claims}/{len(benchmark_results)} research claims")
        else:
            weaknesses.append("Does not meet research paper claims")

        # Baseline comparison
        if baseline_comparison.beats_both_baselines:
            strengths.append("Outperforms both baseline methods")
        elif baseline_comparison.beats_random_search:
            strengths.append("Outperforms random search")
            weaknesses.append("Does not outperform grid search")
        else:
            weaknesses.append("Does not consistently beat baseline methods")

        # Statistical significance
        if statistical_significance.is_significant:
            p_value = statistical_significance.p_value
            strengths.append(f"Statistically significant improvements (p={p_value:.4f})")
        else:
            weaknesses.append("Improvements not statistically significant")

        # Reproducibility
        if reproducibility.is_reproducible:
            strengths.append(f"Reproducible results (variance={reproducibility.variance:.4f})")
        else:
            weaknesses.append(f"Non-reproducible results (variance={reproducibility.variance:.4f})")

        return strengths, weaknesses

    def _generate_recommendation(
        self,
        algorithm_name: str,
        passes_validation: bool,
        overall_score: float,
        strengths: list[str],
        weaknesses: list[str],
    ) -> str:
        """Generate validation recommendation"""
        if passes_validation:
            return (
                f"{algorithm_name} passes validation with score {overall_score:.1f}/100. "
                f"Recommended for production use. Key strengths: {', '.join(strengths[:2])}"
            )
        else:
            return (
                f"{algorithm_name} does not pass validation (score {overall_score:.1f}/100). "
                f"Requires improvements. Key issues: {', '.join(weaknesses[:2])}"
            )

    def _generate_summary(
        self,
        validation_results: list[ValidationResult],
        algorithm_rankings: list[dict[str, Any]],
    ) -> str:
        """Generate validation summary"""
        total = len(validation_results)
        passing = sum(1 for r in validation_results if r.passes_validation)

        summary_lines = [
            f"Algorithm Validation Summary",
            f"Total Algorithms: {total}",
            f"Passing Validation: {passing}/{total}",
            "",
            "Algorithm Rankings:",
        ]

        for idx, ranking in enumerate(algorithm_rankings, 1):
            status = "✓ Pass" if ranking["passes_validation"] else "✗ Fail"
            summary_lines.append(
                f"{idx}. {ranking['algorithm_name']}: {ranking['overall_score']:.1f}/100 {status}"
            )

        return "\n".join(summary_lines)

    def _generate_recommendations(
        self,
        validation_results: list[ValidationResult],
        algorithm_rankings: list[dict[str, Any]],
    ) -> list[str]:
        """Generate recommendations"""
        recommendations = []

        # Recommend best algorithm
        if algorithm_rankings:
            best = algorithm_rankings[0]
            if best["passes_validation"]:
                recommendations.append(
                    f"Recommended algorithm: {best['algorithm_name']} (score: {best['overall_score']:.1f})"
                )

        # Highlight common issues
        all_weaknesses = [
            weakness
            for result in validation_results
            for weakness in result.weaknesses
        ]

        if "Does not meet research paper claims" in all_weaknesses:
            recommendations.append(
                "Multiple algorithms not meeting research claims - review implementation"
            )

        if any("Non-reproducible" in w for w in all_weaknesses):
            recommendations.append(
                "Reproducibility issues detected - ensure proper seed handling"
            )

        return recommendations
