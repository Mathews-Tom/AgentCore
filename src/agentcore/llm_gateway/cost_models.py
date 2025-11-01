"""Cost optimization models for LLM provider management.

Data models for cost tracking, budget management, and cost optimization
with support for real-time analysis and reporting.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CostPeriod(str, Enum):
    """Time periods for cost aggregation and reporting."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class BudgetAlertSeverity(str, Enum):
    """Severity levels for budget alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class OptimizationStrategy(str, Enum):
    """Cost optimization strategies."""

    COST_ONLY = "cost_only"  # Minimize cost only
    BALANCED = "balanced"  # Balance cost and performance
    PERFORMANCE_FIRST = "performance_first"  # Prioritize performance
    ADAPTIVE = "adaptive"  # Dynamically adjust based on usage


class CostMetrics(BaseModel):
    """Real-time cost metrics for a request or time period.

    Tracks detailed cost breakdown, token usage, and performance metrics
    for cost optimization and reporting.
    """

    total_cost: float = Field(
        description="Total cost in USD",
        ge=0.0,
    )
    input_cost: float = Field(
        description="Cost for input tokens in USD",
        ge=0.0,
    )
    output_cost: float = Field(
        description="Cost for output tokens in USD",
        ge=0.0,
    )
    input_tokens: int = Field(
        description="Number of input tokens processed",
        ge=0,
    )
    output_tokens: int = Field(
        description="Number of output tokens generated",
        ge=0,
    )
    provider_id: str = Field(
        description="Provider that handled the request",
    )
    model: str = Field(
        description="Model used for the request",
    )
    timestamp: datetime = Field(
        description="When the cost was recorded",
    )
    latency_ms: int | None = Field(
        default=None,
        description="Request latency in milliseconds",
        ge=0,
    )
    request_id: str | None = Field(
        default=None,
        description="Unique request identifier",
    )
    tenant_id: str | None = Field(
        default=None,
        description="Tenant identifier for cost allocation",
    )
    workflow_id: str | None = Field(
        default=None,
        description="Workflow identifier for tracking",
    )
    agent_id: str | None = Field(
        default=None,
        description="Agent identifier for tracking",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Custom tags for cost categorization",
    )


class CostSummary(BaseModel):
    """Aggregated cost summary for a time period.

    Provides summary statistics for cost analysis and reporting.
    """

    period_start: datetime = Field(
        description="Start of the time period",
    )
    period_end: datetime = Field(
        description="End of the time period",
    )
    total_cost: float = Field(
        description="Total cost in USD for the period",
        ge=0.0,
    )
    total_requests: int = Field(
        description="Number of requests in the period",
        ge=0,
    )
    total_input_tokens: int = Field(
        description="Total input tokens processed",
        ge=0,
    )
    total_output_tokens: int = Field(
        description="Total output tokens generated",
        ge=0,
    )
    average_cost_per_request: float = Field(
        description="Average cost per request in USD",
        ge=0.0,
    )
    average_cost_per_1k_tokens: float = Field(
        description="Average cost per 1K tokens in USD",
        ge=0.0,
    )
    average_latency_ms: int | None = Field(
        default=None,
        description="Average request latency in milliseconds",
        ge=0,
    )
    provider_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by provider (provider_id -> cost)",
    )
    model_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by model (model -> cost)",
    )
    tenant_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by tenant (tenant_id -> cost)",
    )


class BudgetThreshold(BaseModel):
    """Budget threshold configuration for alerts.

    Defines when to trigger alerts based on budget consumption.
    """

    threshold_percent: float = Field(
        description="Budget consumption percentage that triggers alert (0-100)",
        ge=0.0,
        le=100.0,
    )
    severity: BudgetAlertSeverity = Field(
        description="Alert severity level",
    )
    notify_emails: list[str] = Field(
        default_factory=list,
        description="Email addresses to notify",
    )
    notify_webhooks: list[str] = Field(
        default_factory=list,
        description="Webhook URLs to notify",
    )
    auto_throttle: bool = Field(
        default=False,
        description="Automatically throttle requests when threshold reached",
    )
    throttle_percent: float = Field(
        default=50.0,
        description="Percentage to reduce request rate (0-100)",
        ge=0.0,
        le=100.0,
    )


class BudgetConfig(BaseModel):
    """Budget configuration for cost control.

    Defines spending limits, alerts, and enforcement policies for
    a tenant or time period.
    """

    tenant_id: str = Field(
        description="Tenant identifier for this budget",
    )
    limit_amount: float = Field(
        description="Budget limit in USD",
        gt=0.0,
    )
    period: CostPeriod = Field(
        description="Time period for budget enforcement",
    )
    period_start: datetime = Field(
        description="Start of the budget period",
    )
    period_end: datetime = Field(
        description="End of the budget period",
    )
    current_spend: float = Field(
        default=0.0,
        description="Current spending in USD for this period",
        ge=0.0,
    )
    thresholds: list[BudgetThreshold] = Field(
        default_factory=list,
        description="Alert thresholds for this budget",
    )
    hard_limit: bool = Field(
        default=False,
        description="Enforce hard limit (reject requests when exceeded)",
    )
    rollover_enabled: bool = Field(
        default=False,
        description="Allow unused budget to roll over to next period",
    )
    rollover_amount: float = Field(
        default=0.0,
        description="Amount rolled over from previous period in USD",
        ge=0.0,
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Custom tags for budget categorization",
    )


class BudgetAlert(BaseModel):
    """Budget alert notification.

    Generated when budget thresholds are exceeded.
    """

    alert_id: str = Field(
        description="Unique alert identifier",
    )
    budget_config: BudgetConfig = Field(
        description="Budget configuration that triggered the alert",
    )
    threshold: BudgetThreshold = Field(
        description="Threshold that was exceeded",
    )
    current_spend: float = Field(
        description="Current spending in USD",
        ge=0.0,
    )
    percent_consumed: float = Field(
        description="Percentage of budget consumed",
        ge=0.0,
    )
    timestamp: datetime = Field(
        description="When the alert was triggered",
    )
    message: str = Field(
        description="Alert message",
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether alert has been acknowledged",
    )


class CostOptimizationRecommendation(BaseModel):
    """Recommendation for cost optimization.

    Provides actionable suggestions for reducing costs based on
    usage patterns and provider pricing.
    """

    recommendation_id: str = Field(
        description="Unique recommendation identifier",
    )
    type: str = Field(
        description="Recommendation type (e.g., 'provider_switch', 'caching', 'batching')",
    )
    title: str = Field(
        description="Short recommendation title",
    )
    description: str = Field(
        description="Detailed recommendation description",
    )
    potential_savings: float = Field(
        description="Estimated cost savings in USD",
        ge=0.0,
    )
    potential_savings_percent: float = Field(
        description="Estimated percentage cost reduction",
        ge=0.0,
    )
    impact: str = Field(
        description="Impact level (low, medium, high)",
    )
    effort: str = Field(
        description="Implementation effort (low, medium, high)",
    )
    confidence: float = Field(
        description="Confidence in the recommendation (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Specific action items to implement recommendation",
    )
    affected_providers: list[str] = Field(
        default_factory=list,
        description="Providers affected by this recommendation",
    )
    affected_models: list[str] = Field(
        default_factory=list,
        description="Models affected by this recommendation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional recommendation metadata",
    )
    timestamp: datetime = Field(
        description="When recommendation was generated",
    )


class CostReport(BaseModel):
    """Comprehensive cost report with analytics and recommendations.

    Provides detailed cost analysis, trends, and optimization opportunities.
    """

    report_id: str = Field(
        description="Unique report identifier",
    )
    generated_at: datetime = Field(
        description="When report was generated",
    )
    period_start: datetime = Field(
        description="Start of reporting period",
    )
    period_end: datetime = Field(
        description="End of reporting period",
    )
    summary: CostSummary = Field(
        description="Cost summary for the period",
    )
    budget_status: dict[str, Any] = Field(
        default_factory=dict,
        description="Budget consumption status by tenant",
    )
    trends: dict[str, Any] = Field(
        default_factory=dict,
        description="Cost trends and patterns",
    )
    recommendations: list[CostOptimizationRecommendation] = Field(
        default_factory=list,
        description="Cost optimization recommendations",
    )
    top_providers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top providers by cost and usage",
    )
    top_models: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top models by cost and usage",
    )
    top_tenants: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top tenants by spending",
    )
    cost_efficiency_score: float = Field(
        description="Overall cost efficiency score (0.0-100.0)",
        ge=0.0,
        le=100.0,
    )
    optimization_opportunities: dict[str, Any] = Field(
        default_factory=dict,
        description="Identified optimization opportunities",
    )


class ProviderCostComparison(BaseModel):
    """Cost comparison between providers for a specific request profile.

    Used for intelligent provider selection based on estimated costs.
    """

    provider_id: str = Field(
        description="Provider identifier",
    )
    estimated_cost: float = Field(
        description="Estimated cost for request in USD",
        ge=0.0,
    )
    estimated_latency_ms: int = Field(
        description="Estimated latency in milliseconds",
        ge=0,
    )
    cost_per_1k_tokens: float = Field(
        description="Average cost per 1K tokens in USD",
        ge=0.0,
    )
    quality_score: float = Field(
        description="Quality/capability score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    availability_score: float = Field(
        description="Availability/reliability score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    total_score: float = Field(
        description="Combined optimization score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    selection_reason: str = Field(
        description="Reason for ranking",
    )


class OptimizationContext(BaseModel):
    """Context for cost optimization decisions.

    Provides request characteristics and constraints for intelligent
    provider selection.
    """

    estimated_input_tokens: int = Field(
        description="Estimated input token count",
        ge=0,
    )
    estimated_output_tokens: int = Field(
        description="Estimated output token count",
        ge=0,
    )
    max_acceptable_cost: float | None = Field(
        default=None,
        description="Maximum acceptable cost in USD",
        ge=0.0,
    )
    max_acceptable_latency_ms: int | None = Field(
        default=None,
        description="Maximum acceptable latency in milliseconds",
        ge=0,
    )
    required_quality_level: float = Field(
        default=0.8,
        description="Required quality level (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    optimization_strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.BALANCED,
        description="Optimization strategy to use",
    )
    tenant_id: str | None = Field(
        default=None,
        description="Tenant identifier for budget checking",
    )
    priority: int = Field(
        default=5,
        description="Request priority (1-10, higher = more important)",
        ge=1,
        le=10,
    )
    allow_degraded_providers: bool = Field(
        default=True,
        description="Allow selection of degraded but operational providers",
    )
