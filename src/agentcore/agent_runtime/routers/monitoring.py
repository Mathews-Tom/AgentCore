"""
Monitoring and observability API endpoints.

This module provides REST API endpoints for accessing metrics, traces,
alerts, and performance dashboards.
"""

from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ..services.alerting_service import (
    AlertSeverity,
    NotificationChannel,
    get_alerting_service,
)
from ..services.distributed_tracing import get_distributed_tracer
from ..services.metrics_collector import get_metrics_collector
from ..services.resource_manager import get_resource_manager

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


# Request/Response Models
class MetricSnapshot(BaseModel):
    """Metric snapshot response."""

    timestamp: str
    metrics: dict[str, Any]


class TraceSpanResponse(BaseModel):
    """Trace span response."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    kind: str
    start_time: float
    end_time: float | None
    duration_ms: float
    status: str
    status_message: str | None
    attributes: dict[str, Any]
    events: list[dict[str, Any]]


class TraceSummaryResponse(BaseModel):
    """Trace summary response."""

    trace_id: str
    span_count: int
    total_duration_ms: float
    error_count: int
    start_time: float
    end_time: float
    operations: list[str]


class AlertResponse(BaseModel):
    """Alert response."""

    alert_id: str
    rule_name: str
    severity: str
    title: str
    description: str
    labels: dict[str, str]
    state: str
    created_at: str
    updated_at: str
    acknowledged_at: str | None
    resolved_at: str | None
    acknowledged_by: str | None
    resolution_note: str | None


class AlertCreateRequest(BaseModel):
    """Alert creation request."""

    rule_name: str = Field(description="Name of the alert rule")
    title: str = Field(description="Alert title")
    description: str = Field(description="Alert description")
    labels: dict[str, str] | None = Field(default=None, description="Alert labels")
    channels: list[str] | None = Field(
        default=None, description="Notification channels"
    )


class AlertAcknowledgeRequest(BaseModel):
    """Alert acknowledgment request."""

    acknowledged_by: str = Field(description="User/system acknowledging")


class AlertResolveRequest(BaseModel):
    """Alert resolution request."""

    resolution_note: str | None = Field(default=None, description="Resolution note")


class SystemMetricsResponse(BaseModel):
    """System metrics response."""

    current_usage: dict[str, float]
    limits: dict[str, float]
    active_agents: int
    active_alerts: int | None


class DashboardResponse(BaseModel):
    """Performance dashboard response."""

    system_metrics: dict[str, Any]
    agent_metrics: dict[str, Any]
    performance_metrics: dict[str, Any]
    alert_summary: dict[str, Any]
    trace_summary: dict[str, Any]


# Prometheus Metrics Endpoint
@router.get("/metrics", summary="Prometheus metrics endpoint")
async def get_prometheus_metrics() -> Any:
    """
    Get Prometheus-formatted metrics.

    Returns Prometheus text format metrics for scraping.
    """
    collector = get_metrics_collector()
    registry = collector.get_registry()

    metrics_output = generate_latest(registry)

    return {
        "content": metrics_output.decode("utf-8"),
        "content_type": CONTENT_TYPE_LATEST,
    }


# Metrics Endpoints
@router.get("/metrics/snapshot", response_model=MetricSnapshot)
async def get_metrics_snapshot() -> MetricSnapshot:
    """
    Get current metrics snapshot.

    Returns:
        Current metric values snapshot
    """
    collector = get_metrics_collector()
    snapshot = collector.snapshot_metrics()

    return MetricSnapshot(**snapshot)


@router.get("/metrics/history", response_model=list[MetricSnapshot])
async def get_metrics_history(
    metric_name: str | None = Query(default=None, description="Specific metric name"),
    limit: int = Query(default=100, ge=1, le=1000, description="Number of snapshots"),
) -> list[MetricSnapshot]:
    """
    Get metrics history.

    Args:
        metric_name: Optional metric name filter
        limit: Number of recent snapshots to return

    Returns:
        List of metric snapshots
    """
    collector = get_metrics_collector()
    history = collector.get_metric_history(metric_name=metric_name, limit=limit)

    return [MetricSnapshot(**h) for h in history]


# Distributed Tracing Endpoints
@router.get("/traces/{trace_id}", response_model=TraceSummaryResponse)
async def get_trace_summary(trace_id: str) -> TraceSummaryResponse:
    """
    Get trace summary.

    Args:
        trace_id: Trace identifier

    Returns:
        Trace summary with span statistics
    """
    tracer = get_distributed_tracer()
    summary = tracer.get_trace_summary(trace_id)

    if not summary:
        raise HTTPException(status_code=404, detail="Trace not found")

    return TraceSummaryResponse(**summary)


@router.get("/traces/{trace_id}/spans", response_model=list[TraceSpanResponse])
async def get_trace_spans(trace_id: str) -> list[TraceSpanResponse]:
    """
    Get all spans for a trace.

    Args:
        trace_id: Trace identifier

    Returns:
        List of spans in the trace
    """
    tracer = get_distributed_tracer()
    spans = tracer.get_trace_spans(trace_id)

    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")

    return [TraceSpanResponse(**span.to_dict()) for span in spans]


@router.get("/traces/metrics", response_model=dict[str, Any])
async def get_tracing_metrics() -> dict[str, Any]:
    """
    Get tracing metrics.

    Returns:
        Tracing statistics
    """
    tracer = get_distributed_tracer()
    return tracer.get_metrics()


# Alerting Endpoints
@router.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    severity: AlertSeverity | None = Query(
        default=None, description="Filter by severity"
    ),
    label_key: str | None = Query(default=None, description="Label key filter"),
    label_value: str | None = Query(default=None, description="Label value filter"),
) -> list[AlertResponse]:
    """
    Get active alerts.

    Args:
        severity: Optional severity filter
        label_key: Optional label key filter
        label_value: Optional label value filter

    Returns:
        List of active alerts
    """
    alerting = get_alerting_service()

    label_filters = None
    if label_key and label_value:
        label_filters = {label_key: label_value}

    alerts = alerting.get_active_alerts(severity=severity, label_filters=label_filters)

    return [AlertResponse(**alert.to_dict()) for alert in alerts]


@router.post("/alerts", response_model=AlertResponse, status_code=201)
async def create_alert(request: AlertCreateRequest) -> AlertResponse:
    """
    Manually trigger an alert.

    Args:
        request: Alert creation request

    Returns:
        Created alert
    """
    alerting = get_alerting_service()

    # Convert string channels to enum
    channels = None
    if request.channels:
        channels = [NotificationChannel(c) for c in request.channels]

    alert = await alerting.trigger_alert(
        rule_name=request.rule_name,
        title=request.title,
        description=request.description,
        labels=request.labels,
        channels=channels,
    )

    return AlertResponse(**alert.to_dict())


@router.post("/alerts/{alert_id}/acknowledge", response_model=dict[str, bool])
async def acknowledge_alert(
    alert_id: str,
    request: AlertAcknowledgeRequest,
) -> dict[str, bool]:
    """
    Acknowledge an alert.

    Args:
        alert_id: Alert identifier
        request: Acknowledgment request

    Returns:
        Success status
    """
    alerting = get_alerting_service()
    success = await alerting.acknowledge_alert(alert_id, request.acknowledged_by)

    if not success:
        raise HTTPException(
            status_code=404, detail="Alert not found or already handled"
        )

    return {"success": True}


@router.post("/alerts/{alert_id}/resolve", response_model=dict[str, bool])
async def resolve_alert(
    alert_id: str,
    request: AlertResolveRequest | None = None,
) -> dict[str, bool]:
    """
    Resolve an alert.

    Args:
        alert_id: Alert identifier
        request: Resolution request

    Returns:
        Success status
    """
    alerting = get_alerting_service()
    resolution_note = request.resolution_note if request else None

    success = await alerting.resolve_alert(alert_id, resolution_note)

    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")

    return {"success": True}


@router.get("/alerts/history", response_model=list[AlertResponse])
async def get_alert_history(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to look back"),
    severity: AlertSeverity | None = Query(
        default=None, description="Filter by severity"
    ),
) -> list[AlertResponse]:
    """
    Get alert history.

    Args:
        hours: Number of hours to look back
        severity: Optional severity filter

    Returns:
        List of historical alerts
    """
    alerting = get_alerting_service()
    alerts = alerting.get_alert_history(hours=hours, severity=severity)

    return [AlertResponse(**alert.to_dict()) for alert in alerts]


@router.get("/alerts/statistics", response_model=dict[str, Any])
async def get_alert_statistics() -> dict[str, Any]:
    """
    Get alerting statistics.

    Returns:
        Alert statistics
    """
    alerting = get_alerting_service()
    return alerting.get_statistics()


# System Resources Endpoints
@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics() -> SystemMetricsResponse:
    """
    Get system resource metrics.

    Returns:
        System metrics including CPU, memory, and agent counts
    """
    resource_manager = get_resource_manager()
    metrics = resource_manager.get_system_metrics()

    return SystemMetricsResponse(**metrics)


@router.get("/system/alerts", response_model=list[dict[str, Any]])
async def get_system_alerts(
    severity: str | None = Query(default=None, description="Filter by severity"),
) -> list[dict[str, Any]]:
    """
    Get system resource alerts.

    Args:
        severity: Optional severity filter

    Returns:
        List of resource alerts
    """
    from ..services.resource_manager import AlertSeverity as ResourceAlertSeverity

    resource_manager = get_resource_manager()

    severity_filter = None
    if severity:
        severity_filter = ResourceAlertSeverity(severity)

    alerts = resource_manager.get_alerts(severity=severity_filter)

    return [
        {
            "alert_id": alert.alert_id,
            "resource_type": alert.resource_type.value,
            "severity": alert.severity.value,
            "message": alert.message,
            "threshold": alert.threshold,
            "current_value": alert.current_value,
            "timestamp": alert.timestamp.isoformat(),
            "acknowledged": alert.acknowledged,
        }
        for alert in alerts
    ]


# Performance Dashboard
@router.get("/dashboard", response_model=DashboardResponse)
async def get_performance_dashboard() -> DashboardResponse:
    """
    Get comprehensive performance dashboard.

    Returns:
        Dashboard with system metrics, traces, alerts, and performance data
    """
    # Collect data from all monitoring services
    resource_manager = get_resource_manager()
    collector = get_metrics_collector()
    tracer = get_distributed_tracer()
    alerting = get_alerting_service()

    # System metrics
    system_metrics = resource_manager.get_system_metrics()

    # Agent metrics
    agent_metrics = {
        "total_agents": sum(
            collector.agents_total._metrics.get((p.value, "completed"), {})._value.get()
            for p in ["react", "cot", "multi_agent", "autonomous"]
        ),
    }

    # Performance metrics (from optimizer would go here)
    performance_metrics = {
        "cache_stats": {},
        "pool_stats": {},
    }

    # Alert summary
    alert_summary = alerting.get_statistics()

    # Trace summary
    trace_summary = tracer.get_metrics()

    return DashboardResponse(
        system_metrics=system_metrics,
        agent_metrics=agent_metrics,
        performance_metrics=performance_metrics,
        alert_summary=alert_summary,
        trace_summary=trace_summary,
    )


# Health Check
@router.get("/health", response_model=dict[str, str])
async def monitoring_health_check() -> dict[str, str]:
    """
    Monitoring service health check.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "monitoring",
    }
