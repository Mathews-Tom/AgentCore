"""
ROI (Return on Investment) calculation

Calculates optimization ROI by comparing costs vs performance gains,
including infrastructure costs, compute time, and business value.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    OptimizationResult,
    OptimizationTarget,
    PerformanceMetrics,
)


class CostModel(str, Enum):
    """Cost model for ROI calculation"""

    TOKEN_BASED = "token_based"  # Based on LLM token usage
    TIME_BASED = "time_based"  # Based on compute time
    REQUEST_BASED = "request_based"  # Based on number of requests
    HYBRID = "hybrid"  # Combination of multiple factors


class ROIMetrics(BaseModel):
    """Detailed ROI metrics"""

    optimization_cost: float = Field(
        description="Total cost of optimization process"
    )
    baseline_operational_cost: float = Field(
        description="Baseline operational cost per period"
    )
    optimized_operational_cost: float = Field(
        description="Optimized operational cost per period"
    )
    cost_savings_per_period: float = Field(
        description="Cost savings per period"
    )
    performance_gain_value: float = Field(
        description="Business value of performance gain"
    )
    total_benefit: float = Field(
        description="Total benefit (savings + value)"
    )
    roi_percentage: float = Field(
        description="ROI as percentage"
    )
    payback_period_days: float = Field(
        description="Days to recoup optimization cost"
    )
    net_present_value: float = Field(
        description="NPV over forecast period"
    )


class CostBreakdown(BaseModel):
    """Breakdown of costs"""

    compute_cost: float = 0.0
    token_cost: float = 0.0
    infrastructure_cost: float = 0.0
    human_time_cost: float = 0.0
    total_cost: float = 0.0


class ROIReport(BaseModel):
    """Comprehensive ROI report"""

    target: OptimizationTarget
    optimization_id: str
    metrics: ROIMetrics
    optimization_costs: CostBreakdown
    baseline_costs: CostBreakdown
    optimized_costs: CostBreakdown
    performance_improvements: dict[str, float] = Field(default_factory=dict)
    assumptions: dict[str, Any] = Field(default_factory=dict)
    forecast_period_days: int
    is_profitable: bool
    break_even_date: datetime | None = None
    recommendations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ROICalculator:
    """
    ROI calculator for optimization investments

    Calculates return on investment by comparing optimization costs
    against operational cost savings and performance value gains.

    Key features:
    - Multi-factor cost modeling
    - Performance gain valuation
    - Payback period calculation
    - NPV calculation
    - Break-even analysis
    - Cost-benefit recommendations
    """

    def __init__(
        self,
        cost_model: CostModel = CostModel.HYBRID,
        discount_rate: float = 0.10,
    ) -> None:
        """
        Initialize ROI calculator

        Args:
            cost_model: Cost model to use
            discount_rate: Annual discount rate for NPV (default: 10%)
        """
        self.cost_model = cost_model
        self.discount_rate = discount_rate

        # Default cost factors (can be overridden)
        self.token_cost_per_1k = 0.002  # $0.002 per 1k tokens
        self.compute_hour_cost = 0.50  # $0.50 per compute hour
        self.engineer_hour_cost = 100.0  # $100 per engineer hour

    async def calculate_roi(
        self,
        result: OptimizationResult,
        optimization_costs: dict[str, float],
        baseline_volume: int,
        forecast_period_days: int = 365,
        business_value_factors: dict[str, float] | None = None,
    ) -> ROIReport:
        """
        Calculate ROI for optimization

        Args:
            result: Optimization result
            optimization_costs: Dict of optimization cost components
            baseline_volume: Expected request volume per day
            forecast_period_days: Forecast period for ROI calculation
            business_value_factors: Optional business value multipliers

        Returns:
            Comprehensive ROI report

        Raises:
            ValueError: If missing required metrics
        """
        if not result.baseline_performance or not result.optimized_performance:
            raise ValueError("Missing performance metrics in optimization result")

        # Calculate cost breakdowns
        opt_cost_breakdown = self._calculate_optimization_costs(optimization_costs)

        baseline_cost_breakdown = self._calculate_operational_costs(
            result.baseline_performance,
            baseline_volume,
        )

        optimized_cost_breakdown = self._calculate_operational_costs(
            result.optimized_performance,
            baseline_volume,
        )

        # Calculate cost savings
        daily_savings = (
            baseline_cost_breakdown.total_cost - optimized_cost_breakdown.total_cost
        )

        # Calculate performance gain value
        performance_value = self._calculate_performance_value(
            result.baseline_performance,
            result.optimized_performance,
            baseline_volume,
            business_value_factors or {},
        )

        # Total benefit
        total_daily_benefit = daily_savings + performance_value

        # Calculate ROI metrics
        total_benefit_forecast = total_daily_benefit * forecast_period_days

        roi_percentage = (
            (total_benefit_forecast - opt_cost_breakdown.total_cost)
            / opt_cost_breakdown.total_cost
            * 100
            if opt_cost_breakdown.total_cost > 0
            else 0.0
        )

        # Payback period
        payback_days = (
            opt_cost_breakdown.total_cost / total_daily_benefit
            if total_daily_benefit > 0
            else float("inf")
        )

        # NPV calculation
        npv = self._calculate_npv(
            opt_cost_breakdown.total_cost,
            total_daily_benefit,
            forecast_period_days,
        )

        # Break-even date
        break_even_date = None
        if payback_days < float("inf"):
            break_even_date = datetime.utcnow() + timedelta(days=payback_days)

        # Performance improvements
        perf_improvements = {
            "success_rate": (
                result.optimized_performance.success_rate
                - result.baseline_performance.success_rate
            ),
            "cost_reduction": (
                result.baseline_performance.avg_cost_per_task
                - result.optimized_performance.avg_cost_per_task
            ),
            "latency_reduction": (
                result.baseline_performance.avg_latency_ms
                - result.optimized_performance.avg_latency_ms
            ),
            "quality_improvement": (
                result.optimized_performance.quality_score
                - result.baseline_performance.quality_score
            ),
        }

        # Determine if profitable
        is_profitable = roi_percentage > 0 and payback_days <= forecast_period_days

        # Generate recommendations
        recommendations = self._generate_recommendations(
            roi_percentage,
            payback_days,
            daily_savings,
            performance_value,
            is_profitable,
        )

        return ROIReport(
            target=OptimizationTarget(
                type=result.optimization_details.parameters.get("target_type", "agent")
                if result.optimization_details
                else "agent",
                id=result.optimization_details.parameters.get("target_id", "unknown")
                if result.optimization_details
                else "unknown",
            ),
            optimization_id=result.optimization_id,
            metrics=ROIMetrics(
                optimization_cost=opt_cost_breakdown.total_cost,
                baseline_operational_cost=baseline_cost_breakdown.total_cost,
                optimized_operational_cost=optimized_cost_breakdown.total_cost,
                cost_savings_per_period=daily_savings,
                performance_gain_value=performance_value,
                total_benefit=total_daily_benefit,
                roi_percentage=roi_percentage,
                payback_period_days=payback_days,
                net_present_value=npv,
            ),
            optimization_costs=opt_cost_breakdown,
            baseline_costs=baseline_cost_breakdown,
            optimized_costs=optimized_cost_breakdown,
            performance_improvements=perf_improvements,
            assumptions={
                "baseline_volume_per_day": baseline_volume,
                "forecast_period_days": forecast_period_days,
                "discount_rate": self.discount_rate,
                "cost_model": self.cost_model.value,
            },
            forecast_period_days=forecast_period_days,
            is_profitable=is_profitable,
            break_even_date=break_even_date,
            recommendations=recommendations,
            metadata={
                "algorithm": result.optimization_details.algorithm_used
                if result.optimization_details
                else "unknown",
                "improvement_percentage": result.improvement_percentage,
            },
        )

    async def compare_roi(
        self,
        reports: list[ROIReport],
    ) -> dict[str, Any]:
        """
        Compare ROI across multiple optimizations

        Args:
            reports: List of ROI reports to compare

        Returns:
            Comparison summary
        """
        if not reports:
            return {"total_reports": 0}

        # Sort by ROI
        sorted_reports = sorted(
            reports,
            key=lambda r: r.metrics.roi_percentage,
            reverse=True,
        )

        # Calculate aggregates
        avg_roi = sum(r.metrics.roi_percentage for r in reports) / len(reports)
        avg_payback = sum(r.metrics.payback_period_days for r in reports) / len(reports)
        total_cost = sum(r.metrics.optimization_cost for r in reports)
        total_savings = sum(
            r.metrics.cost_savings_per_period * r.forecast_period_days for r in reports
        )
        profitable_count = sum(1 for r in reports if r.is_profitable)

        return {
            "total_reports": len(reports),
            "avg_roi_percentage": avg_roi,
            "avg_payback_days": avg_payback,
            "total_optimization_cost": total_cost,
            "total_forecasted_savings": total_savings,
            "profitable_count": profitable_count,
            "profitable_percentage": profitable_count / len(reports) * 100,
            "best_roi": {
                "optimization_id": sorted_reports[0].optimization_id,
                "roi_percentage": sorted_reports[0].metrics.roi_percentage,
                "payback_days": sorted_reports[0].metrics.payback_period_days,
            },
            "worst_roi": {
                "optimization_id": sorted_reports[-1].optimization_id,
                "roi_percentage": sorted_reports[-1].metrics.roi_percentage,
                "payback_days": sorted_reports[-1].metrics.payback_period_days,
            },
        }

    def _calculate_optimization_costs(
        self,
        costs: dict[str, float],
    ) -> CostBreakdown:
        """
        Calculate optimization cost breakdown

        Args:
            costs: Dictionary of cost components

        Returns:
            Cost breakdown
        """
        compute = costs.get("compute_hours", 0.0) * self.compute_hour_cost
        tokens = costs.get("tokens", 0.0) / 1000 * self.token_cost_per_1k
        infrastructure = costs.get("infrastructure", 0.0)
        human_time = costs.get("engineer_hours", 0.0) * self.engineer_hour_cost

        return CostBreakdown(
            compute_cost=compute,
            token_cost=tokens,
            infrastructure_cost=infrastructure,
            human_time_cost=human_time,
            total_cost=compute + tokens + infrastructure + human_time,
        )

    def _calculate_operational_costs(
        self,
        metrics: PerformanceMetrics,
        volume: int,
    ) -> CostBreakdown:
        """
        Calculate daily operational costs

        Args:
            metrics: Performance metrics
            volume: Daily request volume

        Returns:
            Cost breakdown
        """
        # Calculate per-request costs
        per_request_cost = metrics.avg_cost_per_task
        daily_total = per_request_cost * volume

        # Estimate breakdown (tokens typically 60-70% of cost)
        token_cost = daily_total * 0.65
        compute_cost = daily_total * 0.25
        infrastructure_cost = daily_total * 0.10

        return CostBreakdown(
            compute_cost=compute_cost,
            token_cost=token_cost,
            infrastructure_cost=infrastructure_cost,
            human_time_cost=0.0,  # Not included in operational costs
            total_cost=daily_total,
        )

    def _calculate_performance_value(
        self,
        baseline: PerformanceMetrics,
        optimized: PerformanceMetrics,
        volume: int,
        value_factors: dict[str, float],
    ) -> float:
        """
        Calculate business value of performance improvements

        Args:
            baseline: Baseline metrics
            optimized: Optimized metrics
            volume: Daily volume
            value_factors: Value multipliers per metric

        Returns:
            Daily performance value
        """
        # Default value factors ($ per unit improvement per request)
        default_factors = {
            "success_rate": 0.10,  # $0.10 per 1% success rate improvement
            "quality": 0.05,  # $0.05 per 1% quality improvement
            "latency": 0.001,  # $0.001 per ms latency reduction
        }

        factors = {**default_factors, **value_factors}

        # Calculate improvements
        success_improvement = (
            optimized.success_rate - baseline.success_rate
        ) * 100  # Convert to percentage

        quality_improvement = (
            optimized.quality_score - baseline.quality_score
        ) * 100

        latency_improvement = baseline.avg_latency_ms - optimized.avg_latency_ms

        # Calculate daily value
        success_value = success_improvement * factors["success_rate"] * volume
        quality_value = quality_improvement * factors["quality"] * volume
        latency_value = latency_improvement * factors["latency"] * volume

        return max(0.0, success_value + quality_value + latency_value)

    def _calculate_npv(
        self,
        initial_cost: float,
        daily_benefit: float,
        forecast_days: int,
    ) -> float:
        """
        Calculate Net Present Value

        Args:
            initial_cost: Upfront optimization cost
            daily_benefit: Daily benefit amount
            forecast_days: Forecast period

        Returns:
            NPV value
        """
        # Convert annual discount rate to daily
        daily_rate = self.discount_rate / 365

        # Calculate NPV of future benefits
        npv = -initial_cost

        for day in range(1, forecast_days + 1):
            discounted_benefit = daily_benefit / ((1 + daily_rate) ** day)
            npv += discounted_benefit

        return npv

    def _generate_recommendations(
        self,
        roi_percentage: float,
        payback_days: float,
        daily_savings: float,
        performance_value: float,
        is_profitable: bool,
    ) -> list[str]:
        """
        Generate ROI-based recommendations

        Args:
            roi_percentage: ROI percentage
            payback_days: Payback period in days
            daily_savings: Daily cost savings
            performance_value: Daily performance value
            is_profitable: Whether investment is profitable

        Returns:
            List of recommendations
        """
        recommendations = []

        # Profitability
        if is_profitable:
            if roi_percentage > 200:
                recommendations.append(
                    f"Excellent ROI ({roi_percentage:.1f}%) - strongly recommend deployment"
                )
            elif roi_percentage > 100:
                recommendations.append(
                    f"Strong ROI ({roi_percentage:.1f}%) - recommend deployment"
                )
            else:
                recommendations.append(
                    f"Positive ROI ({roi_percentage:.1f}%) - consider deployment"
                )
        else:
            recommendations.append(
                f"Negative ROI ({roi_percentage:.1f}%) - reconsider optimization strategy"
            )

        # Payback period
        if payback_days < 30:
            recommendations.append(
                f"Fast payback ({payback_days:.1f} days) - minimal risk"
            )
        elif payback_days < 90:
            recommendations.append(
                f"Reasonable payback ({payback_days:.1f} days) - acceptable risk"
            )
        elif payback_days < 365:
            recommendations.append(
                f"Long payback ({payback_days:.1f} days) - evaluate risk tolerance"
            )
        else:
            recommendations.append(
                f"Very long payback ({payback_days:.1f} days) - high risk"
            )

        # Value drivers
        if daily_savings > performance_value * 2:
            recommendations.append(
                "Primary value from cost savings - focus on cost optimization"
            )
        elif performance_value > daily_savings * 2:
            recommendations.append(
                "Primary value from performance gains - emphasize quality improvements"
            )
        else:
            recommendations.append(
                "Balanced value from costs and performance - optimize both dimensions"
            )

        # Scaling recommendations
        if is_profitable and daily_savings > 0:
            annual_savings = daily_savings * 365
            recommendations.append(
                f"Scale optimization to maximize {annual_savings:.2f} annual savings"
            )

        return recommendations
