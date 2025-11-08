"""
ACE Performance Monitoring JSON-RPC Methods (COMPASS ACE-1)

JSON-RPC 2.0 methods for ACE performance monitoring, baseline tracking,
and metrics summary. Integrates with PerformanceMonitor, BaselineTracker,
and ErrorAccumulator components.

Implements spec.md Section 7.1: Performance Monitoring Methods
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, Field, ValidationError

from agentcore.a2a_protocol.database import get_session
from agentcore.a2a_protocol.models.jsonrpc import JsonRpcRequest
from agentcore.a2a_protocol.services.jsonrpc_handler import register_jsonrpc_method
from agentcore.ace.models.ace_models import PerformanceMetrics
from agentcore.ace.monitors.baseline_tracker import BaselineTracker
from agentcore.ace.monitors.error_accumulator import ErrorAccumulator
from agentcore.ace.monitors.performance_monitor import PerformanceMonitor

logger = structlog.get_logger()

# Initialize components (singleton pattern)
performance_monitor = PerformanceMonitor(get_session=get_session)
baseline_tracker = BaselineTracker(get_session=get_session)
error_accumulator = ErrorAccumulator()


# Request/Response Models for API Validation


class TrackPerformanceRequest(BaseModel):
    """Request to track performance metrics."""

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    task_id: str = Field(..., description="Task identifier")
    stage: str = Field(
        ..., description="Reasoning stage (planning, execution, reflection, verification)"
    )
    metrics: dict[str, Any] = Field(..., description="Performance metrics")


class GetBaselineRequest(BaseModel):
    """Request to get performance baseline."""

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    stage: str = Field(
        ..., description="Reasoning stage (planning, execution, reflection, verification)"
    )
    task_type: str | None = Field(None, description="Optional task type")


class GetMetricsRequest(BaseModel):
    """Request to get current performance metrics."""

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    task_id: str = Field(..., description="Task identifier")


class GetMetricsSummaryRequest(BaseModel):
    """Request to get metrics summary."""

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    task_id: str = Field(..., description="Task identifier")


class GetMetricHistoryRequest(BaseModel):
    """Request to get metric history."""

    agent_id: str = Field(..., description="Agent identifier", min_length=1, max_length=255)
    task_id: str = Field(..., description="Task identifier")
    start_time: datetime = Field(..., description="Start time for history query")
    end_time: datetime = Field(..., description="End time for history query")


# JSON-RPC Method Handlers


@register_jsonrpc_method("ace.track_performance")
async def handle_track_performance(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Track performance metrics for an agent task.

    Method: ace.track_performance
    Params:
        - agent_id: string
        - task_id: string (UUID)
        - stage: string (planning, execution, reflection, verification)
        - metrics: object with performance metric values

    Returns:
        - success: boolean
        - message: string
        - recorded_at: string (ISO8601)
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError(
                "Parameters required: agent_id, task_id, stage, metrics"
            )

        # Validate request params
        req = TrackPerformanceRequest(**request.params)

        # Convert task_id to UUID
        task_id = UUID(req.task_id)

        # Create PerformanceMetrics from metrics dict
        metrics = PerformanceMetrics(
            task_id=task_id,
            agent_id=req.agent_id,
            stage=req.stage,
            **req.metrics
        )

        # Record metrics via PerformanceMonitor
        await performance_monitor.record_metrics(
            task_id=task_id,
            agent_id=req.agent_id,
            stage=req.stage,
            metrics=metrics,
        )

        logger.info(
            "Performance metrics tracked via JSON-RPC",
            agent_id=req.agent_id,
            task_id=str(task_id),
            stage=req.stage,
            method="ace.track_performance",
        )

        return {
            "success": True,
            "message": "Performance metrics recorded successfully",
            "recorded_at": datetime.now(UTC).isoformat(),
        }

    except ValidationError as e:
        logger.error("Performance tracking validation failed", error=str(e))
        raise ValueError(f"Request validation failed: {e}")
    except ValueError as e:
        logger.error("Performance tracking failed", error=str(e))
        raise
    except Exception as e:
        logger.error("Performance tracking failed", error=str(e))
        raise


@register_jsonrpc_method("ace.get_baseline")
async def handle_get_baseline(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get performance baseline for agent and stage.

    Method: ace.get_baseline
    Params:
        - agent_id: string
        - stage: string (planning, execution, reflection, verification)
        - task_type: string (optional)

    Returns:
        - baseline: PerformanceBaseline object or null if not available
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id, stage")

        # Validate request params
        req = GetBaselineRequest(**request.params)

        # Get baseline from BaselineTracker
        baseline = await baseline_tracker.get_baseline(
            agent_id=req.agent_id,
            stage=req.stage,
            task_type=req.task_type,
        )

        logger.info(
            "Baseline retrieved via JSON-RPC",
            agent_id=req.agent_id,
            stage=req.stage,
            task_type=req.task_type,
            baseline_found=baseline is not None,
            method="ace.get_baseline",
        )

        if baseline:
            return {"baseline": baseline.model_dump(mode="json")}
        return {"baseline": None}

    except ValidationError as e:
        logger.error("Get baseline validation failed", error=str(e))
        raise ValueError(f"Request validation failed: {e}")
    except ValueError as e:
        logger.error("Get baseline failed", error=str(e))
        raise
    except Exception as e:
        logger.error("Get baseline failed", error=str(e))
        raise


@register_jsonrpc_method("ace.get_metrics")
async def handle_get_metrics(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get current performance metrics for task.

    Method: ace.get_metrics
    Params:
        - agent_id: string
        - task_id: string (UUID)

    Returns:
        - metrics: PerformanceMetrics object or null if not found
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id, task_id")

        # Validate request params
        req = GetMetricsRequest(**request.params)

        # Convert task_id to UUID
        task_id = UUID(req.task_id)

        # Get metrics from PerformanceMonitor
        metrics = await performance_monitor.get_current_metrics(
            task_id=task_id,
            agent_id=req.agent_id,
        )

        logger.info(
            "Metrics retrieved via JSON-RPC",
            agent_id=req.agent_id,
            task_id=str(task_id),
            metrics_found=metrics is not None,
            method="ace.get_metrics",
        )

        if metrics:
            return {"metrics": metrics.model_dump(mode="json")}
        return {"metrics": None}

    except ValidationError as e:
        logger.error("Get metrics validation failed", error=str(e))
        raise ValueError(f"Request validation failed: {e}")
    except ValueError as e:
        logger.error("Get metrics failed", error=str(e))
        raise
    except Exception as e:
        logger.error("Get metrics failed", error=str(e))
        raise


@register_jsonrpc_method("ace.get_metrics_summary")
async def handle_get_metrics_summary(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get comprehensive metrics summary for task.

    Method: ace.get_metrics_summary
    Params:
        - agent_id: string
        - task_id: string (UUID)

    Returns:
        - summary: object containing:
            - latest_metrics: PerformanceMetrics or null
            - baseline: PerformanceBaseline or null (for latest stage)
            - error_trends: error analysis from ErrorAccumulator
            - total_errors: total error count
            - critical_errors: critical error count
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id, task_id")

        # Validate request params
        req = GetMetricsSummaryRequest(**request.params)

        # Convert task_id to UUID
        task_id = UUID(req.task_id)

        # Get latest metrics
        latest_metrics = await performance_monitor.get_current_metrics(
            task_id=task_id,
            agent_id=req.agent_id,
        )

        # Get baseline (if we have metrics)
        baseline = None
        if latest_metrics:
            baseline = await baseline_tracker.get_baseline(
                agent_id=req.agent_id,
                stage=latest_metrics.stage,
                task_type=None,  # No task_type in current implementation
            )

        # Get error trends from ErrorAccumulator
        error_trends = error_accumulator.get_error_trends(
            agent_id=req.agent_id,
            task_id=task_id,
        )

        # Build summary
        summary = {
            "agent_id": req.agent_id,
            "task_id": str(task_id),
            "latest_metrics": (
                latest_metrics.model_dump(mode="json") if latest_metrics else None
            ),
            "baseline": baseline.model_dump(mode="json") if baseline else None,
            "error_trends": error_trends,
            "total_errors": error_trends["total_errors"],
            "critical_errors": error_trends["critical_error_count"],
            "generated_at": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "Metrics summary generated via JSON-RPC",
            agent_id=req.agent_id,
            task_id=str(task_id),
            has_metrics=latest_metrics is not None,
            has_baseline=baseline is not None,
            total_errors=error_trends["total_errors"],
            method="ace.get_metrics_summary",
        )

        return {"summary": summary}

    except ValidationError as e:
        logger.error("Get metrics summary validation failed", error=str(e))
        raise ValueError(f"Request validation failed: {e}")
    except ValueError as e:
        logger.error("Get metrics summary failed", error=str(e))
        raise
    except Exception as e:
        logger.error("Get metrics summary failed", error=str(e))
        raise


@register_jsonrpc_method("ace.get_metric_history")
async def handle_get_metric_history(request: JsonRpcRequest) -> dict[str, Any]:
    """
    Get metric history for task within time range.

    Method: ace.get_metric_history
    Params:
        - agent_id: string
        - task_id: string (UUID)
        - start_time: string (ISO8601 datetime)
        - end_time: string (ISO8601 datetime)

    Returns:
        - metrics: array of PerformanceMetrics objects
        - count: integer (number of metrics returned)
        - start_time: string (requested start time)
        - end_time: string (requested end time)
    """
    try:
        if not request.params or not isinstance(request.params, dict):
            raise ValueError("Parameters required: agent_id, task_id, start_time, end_time")

        # Validate request params
        req = GetMetricHistoryRequest(**request.params)

        # Convert task_id to UUID
        task_id = UUID(req.task_id)

        # Query database for metrics in time range
        from agentcore.ace.database.repositories import MetricsRepository

        async with get_session() as session:
            # Get all metrics for task (limit 100 for performance)
            db_metrics = await MetricsRepository.list_by_task(
                session, task_id, limit=100
            )

            # Filter by time range
            filtered_metrics = [
                m for m in db_metrics
                if req.start_time <= m.recorded_at <= req.end_time
            ]

            # Convert to Pydantic models
            metrics = [
                PerformanceMetrics(
                    metric_id=m.metric_id,
                    task_id=m.task_id,
                    agent_id=m.agent_id,
                    stage=m.stage,
                    stage_success_rate=m.stage_success_rate,
                    stage_error_rate=m.stage_error_rate,
                    stage_duration_ms=m.stage_duration_ms,
                    stage_action_count=m.stage_action_count,
                    overall_progress_velocity=m.overall_progress_velocity,
                    error_accumulation_rate=m.error_accumulation_rate,
                    context_staleness_score=m.context_staleness_score,
                    intervention_effectiveness=m.intervention_effectiveness,
                    baseline_delta=m.baseline_delta,
                    recorded_at=m.recorded_at,
                )
                for m in filtered_metrics
            ]

        logger.info(
            "Metric history retrieved via JSON-RPC",
            agent_id=req.agent_id,
            task_id=str(task_id),
            count=len(metrics),
            start_time=req.start_time.isoformat(),
            end_time=req.end_time.isoformat(),
            method="ace.get_metric_history",
        )

        return {
            "metrics": [m.model_dump(mode="json") for m in metrics],
            "count": len(metrics),
            "start_time": req.start_time.isoformat(),
            "end_time": req.end_time.isoformat(),
        }

    except ValidationError as e:
        logger.error("Get metric history validation failed", error=str(e))
        raise ValueError(f"Request validation failed: {e}")
    except ValueError as e:
        logger.error("Get metric history failed", error=str(e))
        raise
    except Exception as e:
        logger.error("Get metric history failed", error=str(e))
        raise
