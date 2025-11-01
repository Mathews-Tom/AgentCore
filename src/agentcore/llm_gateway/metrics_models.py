"""Performance monitoring models for LLM provider management.

Data models for tracking 50+ metrics per request, SLA monitoring, performance
analysis, and real-time analytics dashboards.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Types of metrics tracked."""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COST = "cost"
    QUALITY = "quality"
    RESOURCE = "resource"
    ERROR = "error"


class SLAStatus(str, Enum):
    """SLA compliance status."""

    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class PerformanceLevel(str, Enum):
    """Performance level classification."""

    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RequestMetrics(BaseModel):
    """Comprehensive metrics for a single LLM request.

    Tracks 50+ metrics including latency, tokens, cost, quality,
    and resource usage for detailed performance analysis.
    """

    # Request Identification
    request_id: str = Field(description="Unique request identifier")
    trace_id: str | None = Field(default=None, description="Distributed trace ID")
    timestamp: datetime = Field(description="Request timestamp")

    # Provider and Model Metrics
    provider_id: str = Field(description="Provider that handled request")
    provider_name: str = Field(description="Human-readable provider name")
    model: str = Field(description="Model used for request")
    model_version: str | None = Field(default=None, description="Model version")

    # Latency Metrics (ms)
    total_latency_ms: int = Field(description="Total end-to-end latency", ge=0)
    routing_latency_ms: int = Field(description="Provider selection latency", ge=0)
    queue_latency_ms: int = Field(description="Queue wait time", ge=0)
    provider_latency_ms: int = Field(description="Provider processing time", ge=0)
    network_latency_ms: int = Field(description="Network round-trip time", ge=0)
    ttft_ms: int | None = Field(
        default=None,
        description="Time to first token (streaming)",
        ge=0,
    )

    # Token Metrics
    input_tokens: int = Field(description="Input tokens processed", ge=0)
    output_tokens: int = Field(description="Output tokens generated", ge=0)
    total_tokens: int = Field(description="Total tokens", ge=0)
    cached_tokens: int = Field(default=0, description="Tokens served from cache", ge=0)
    prompt_tokens: int = Field(default=0, description="Prompt tokens only", ge=0)
    completion_tokens: int = Field(
        default=0, description="Completion tokens only", ge=0
    )

    # Cost Metrics (USD)
    total_cost: float = Field(description="Total request cost", ge=0.0)
    input_cost: float = Field(description="Input tokens cost", ge=0.0)
    output_cost: float = Field(description="Output tokens cost", ge=0.0)
    cache_cost_saved: float = Field(
        default=0.0, description="Cost saved by caching", ge=0.0
    )

    # Throughput Metrics
    tokens_per_second: float = Field(
        description="Token generation rate", ge=0.0
    )
    requests_per_minute: float | None = Field(
        default=None, description="Request rate", ge=0.0
    )

    # Quality Metrics
    success: bool = Field(description="Request succeeded")
    response_complete: bool = Field(
        default=True,
        description="Response was complete (not truncated)",
    )
    finish_reason: str | None = Field(
        default=None,
        description="Completion finish reason",
    )
    quality_score: float | None = Field(
        default=None,
        description="Response quality score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    # Error Metrics
    error_occurred: bool = Field(default=False, description="Error occurred")
    error_type: str | None = Field(default=None, description="Error type")
    error_message: str | None = Field(default=None, description="Error message")
    retry_count: int = Field(default=0, description="Number of retries", ge=0)
    fallback_used: bool = Field(
        default=False,
        description="Fallback provider used",
    )

    # Resource Usage Metrics
    memory_used_mb: float | None = Field(
        default=None,
        description="Memory used (MB)",
        ge=0.0,
    )
    cpu_percent: float | None = Field(
        default=None,
        description="CPU usage percentage",
        ge=0.0,
        le=100.0,
    )
    network_bytes_sent: int | None = Field(
        default=None,
        description="Network bytes sent",
        ge=0,
    )
    network_bytes_received: int | None = Field(
        default=None,
        description="Network bytes received",
        ge=0,
    )

    # Cache Metrics
    cache_hit: bool = Field(default=False, description="Cache hit occurred")
    cache_level: str | None = Field(
        default=None,
        description="Cache level (l1, l2)",
    )
    cache_lookup_ms: int | None = Field(
        default=None,
        description="Cache lookup latency",
        ge=0,
    )

    # Context Metadata
    tenant_id: str | None = Field(default=None, description="Tenant identifier")
    workflow_id: str | None = Field(default=None, description="Workflow identifier")
    agent_id: str | None = Field(default=None, description="Agent identifier")
    session_id: str | None = Field(default=None, description="Session identifier")

    # Request Configuration
    temperature: float | None = Field(
        default=None,
        description="Temperature parameter",
        ge=0.0,
    )
    max_tokens: int | None = Field(
        default=None,
        description="Max tokens parameter",
        ge=0,
    )
    stream: bool = Field(default=False, description="Streaming mode")

    # Additional Metrics
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Custom metric tags",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class PerformanceMetrics(BaseModel):
    """Aggregated performance metrics for time period or provider.

    Summary statistics for performance monitoring and analysis.
    """

    period_start: datetime = Field(description="Period start time")
    period_end: datetime = Field(description="Period end time")

    # Request Statistics
    total_requests: int = Field(description="Total requests", ge=0)
    successful_requests: int = Field(description="Successful requests", ge=0)
    failed_requests: int = Field(description="Failed requests", ge=0)
    success_rate: float = Field(description="Success rate percentage", ge=0.0, le=100.0)

    # Latency Statistics (ms)
    avg_latency_ms: float = Field(description="Average latency", ge=0.0)
    p50_latency_ms: float = Field(description="P50 latency", ge=0.0)
    p95_latency_ms: float = Field(description="P95 latency", ge=0.0)
    p99_latency_ms: float = Field(description="P99 latency", ge=0.0)
    max_latency_ms: int = Field(description="Maximum latency", ge=0)
    min_latency_ms: int = Field(description="Minimum latency", ge=0)

    # Throughput Statistics
    requests_per_second: float = Field(description="Requests per second", ge=0.0)
    tokens_per_second: float = Field(description="Tokens per second", ge=0.0)
    avg_tokens_per_request: float = Field(
        description="Average tokens per request", ge=0.0
    )

    # Cost Statistics (USD)
    total_cost: float = Field(description="Total cost", ge=0.0)
    avg_cost_per_request: float = Field(description="Average cost per request", ge=0.0)
    avg_cost_per_1k_tokens: float = Field(
        description="Average cost per 1K tokens", ge=0.0
    )

    # Error Statistics
    error_rate: float = Field(description="Error rate percentage", ge=0.0, le=100.0)
    timeout_count: int = Field(description="Timeout errors", ge=0)
    rate_limit_count: int = Field(description="Rate limit errors", ge=0)
    provider_error_count: int = Field(description="Provider errors", ge=0)

    # Provider Breakdown
    provider_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Request count by provider",
    )
    model_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Request count by model",
    )


class SLAMetrics(BaseModel):
    """SLA compliance metrics and thresholds.

    Tracks service level agreement compliance for availability,
    response time, and success rate targets.
    """

    # SLA Targets
    availability_target: float = Field(
        description="Target availability percentage",
        ge=0.0,
        le=100.0,
    )
    response_time_target_ms: int = Field(
        description="Target response time (ms)",
        ge=0,
    )
    success_rate_target: float = Field(
        description="Target success rate percentage",
        ge=0.0,
        le=100.0,
    )

    # Actual Performance
    actual_availability: float = Field(
        description="Actual availability percentage",
        ge=0.0,
        le=100.0,
    )
    actual_response_time_ms: float = Field(
        description="Actual average response time (ms)",
        ge=0.0,
    )
    actual_success_rate: float = Field(
        description="Actual success rate percentage",
        ge=0.0,
        le=100.0,
    )

    # Compliance Status
    availability_status: SLAStatus = Field(
        description="Availability SLA compliance"
    )
    response_time_status: SLAStatus = Field(
        description="Response time SLA compliance"
    )
    success_rate_status: SLAStatus = Field(
        description="Success rate SLA compliance"
    )
    overall_status: SLAStatus = Field(
        description="Overall SLA compliance"
    )

    # Time Period
    period_start: datetime = Field(description="Measurement period start")
    period_end: datetime = Field(description="Measurement period end")
    measurement_window_hours: int = Field(
        description="Measurement window (hours)",
        ge=1,
    )

    # Violations
    availability_violations: int = Field(
        default=0,
        description="Availability violations count",
        ge=0,
    )
    response_time_violations: int = Field(
        default=0,
        description="Response time violations count",
        ge=0,
    )
    success_rate_violations: int = Field(
        default=0,
        description="Success rate violations count",
        ge=0,
    )


class ProviderPerformanceMetrics(BaseModel):
    """Performance metrics for a specific provider.

    Per-provider performance tracking for comparison and optimization.
    """

    provider_id: str = Field(description="Provider identifier")
    provider_name: str = Field(description="Provider display name")

    # Performance Metrics
    performance_metrics: PerformanceMetrics = Field(
        description="Aggregated performance metrics"
    )

    # Provider-Specific Metrics
    availability_score: float = Field(
        description="Provider availability score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    reliability_score: float = Field(
        description="Provider reliability score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    cost_efficiency_score: float = Field(
        description="Cost efficiency score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    overall_score: float = Field(
        description="Overall provider score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    # Health Status
    health_status: str = Field(
        description="Provider health status",
    )
    performance_level: PerformanceLevel = Field(
        description="Performance level classification"
    )

    # Last Updated
    last_updated: datetime = Field(description="Metrics last updated")


class PerformanceAlert(BaseModel):
    """Performance alert for threshold violations.

    Generated when performance metrics violate configured thresholds.
    """

    alert_id: str = Field(description="Unique alert identifier")
    severity: AlertSeverity = Field(description="Alert severity")
    metric_type: MetricType = Field(description="Type of metric violated")

    # Threshold Information
    threshold_name: str = Field(description="Threshold name")
    threshold_value: float = Field(description="Threshold value")
    actual_value: float = Field(description="Actual measured value")
    violation_percent: float = Field(
        description="Percentage violation",
        ge=0.0,
    )

    # Context
    provider_id: str | None = Field(default=None, description="Affected provider")
    model: str | None = Field(default=None, description="Affected model")
    tenant_id: str | None = Field(default=None, description="Affected tenant")

    # Timing
    timestamp: datetime = Field(description="Alert timestamp")
    duration_seconds: int | None = Field(
        default=None,
        description="Violation duration (seconds)",
        ge=0,
    )

    # Alert Details
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message")
    recommendation: str | None = Field(
        default=None,
        description="Recommended action",
    )

    # Status
    acknowledged: bool = Field(
        default=False,
        description="Alert acknowledged",
    )
    resolved: bool = Field(
        default=False,
        description="Alert resolved",
    )
    resolved_at: datetime | None = Field(
        default=None,
        description="Resolution timestamp",
    )


class PerformanceInsight(BaseModel):
    """Performance optimization insight.

    Actionable recommendations for performance improvements based on
    historical data and trend analysis.
    """

    insight_id: str = Field(description="Unique insight identifier")
    insight_type: str = Field(
        description="Insight type (e.g., 'latency_spike', 'cost_optimization')"
    )
    title: str = Field(description="Insight title")
    description: str = Field(description="Detailed description")

    # Impact Assessment
    impact_level: str = Field(
        description="Impact level (low, medium, high)",
    )
    confidence: float = Field(
        description="Confidence in insight (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )

    # Metrics
    affected_requests: int = Field(
        description="Number of affected requests",
        ge=0,
    )
    potential_improvement: dict[str, float] = Field(
        default_factory=dict,
        description="Potential improvements by metric",
    )

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations",
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Specific action items",
    )

    # Context
    affected_providers: list[str] = Field(
        default_factory=list,
        description="Affected providers",
    )
    affected_models: list[str] = Field(
        default_factory=list,
        description="Affected models",
    )
    time_period: dict[str, datetime] = Field(
        default_factory=dict,
        description="Analysis time period",
    )

    # Metadata
    generated_at: datetime = Field(description="Insight generation time")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class PrometheusMetrics(BaseModel):
    """Prometheus-compatible metrics export format.

    Formatted metrics for Prometheus/Grafana integration.
    """

    # Counter Metrics
    total_requests: int = Field(description="Total requests counter", ge=0)
    successful_requests: int = Field(description="Successful requests counter", ge=0)
    failed_requests: int = Field(description="Failed requests counter", ge=0)

    # Gauge Metrics
    current_latency_ms: float = Field(
        description="Current latency gauge",
        ge=0.0,
    )
    current_throughput: float = Field(
        description="Current throughput gauge",
        ge=0.0,
    )
    current_error_rate: float = Field(
        description="Current error rate gauge",
        ge=0.0,
        le=100.0,
    )

    # Histogram Metrics
    latency_histogram: dict[str, int] = Field(
        default_factory=dict,
        description="Latency distribution histogram",
    )
    token_count_histogram: dict[str, int] = Field(
        default_factory=dict,
        description="Token count distribution",
    )

    # Summary Metrics
    latency_summary: dict[str, float] = Field(
        default_factory=dict,
        description="Latency percentile summary (p50, p95, p99)",
    )

    # Labels
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Metric labels (provider, model, etc.)",
    )

    # Timestamp
    timestamp: datetime = Field(description="Metrics timestamp")


class DashboardData(BaseModel):
    """Real-time analytics dashboard data.

    Aggregated data for real-time performance dashboards.
    """

    # Overview Metrics
    total_requests_24h: int = Field(
        description="Total requests in last 24h",
        ge=0,
    )
    success_rate_24h: float = Field(
        description="Success rate in last 24h",
        ge=0.0,
        le=100.0,
    )
    avg_latency_24h: float = Field(
        description="Average latency in last 24h",
        ge=0.0,
    )
    total_cost_24h: float = Field(
        description="Total cost in last 24h",
        ge=0.0,
    )

    # Current Performance
    current_throughput: float = Field(
        description="Current requests per second",
        ge=0.0,
    )
    current_latency_ms: float = Field(
        description="Current average latency",
        ge=0.0,
    )
    current_error_rate: float = Field(
        description="Current error rate percentage",
        ge=0.0,
        le=100.0,
    )

    # SLA Status
    sla_compliance: SLAMetrics | None = Field(
        default=None,
        description="Current SLA compliance",
    )

    # Provider Performance
    top_providers: list[ProviderPerformanceMetrics] = Field(
        default_factory=list,
        description="Top performing providers",
    )

    # Active Alerts
    active_alerts: list[PerformanceAlert] = Field(
        default_factory=list,
        description="Active performance alerts",
    )

    # Recent Insights
    recent_insights: list[PerformanceInsight] = Field(
        default_factory=list,
        description="Recent performance insights",
    )

    # Time Series Data
    latency_timeseries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Latency time series data",
    )
    throughput_timeseries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Throughput time series data",
    )
    error_rate_timeseries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Error rate time series data",
    )

    # Last Updated
    last_updated: datetime = Field(description="Dashboard last updated")
