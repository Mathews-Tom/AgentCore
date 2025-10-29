"""
Real-time Optimization Monitoring Hooks

Provides hooks for monitoring optimization progress and agent performance
in real-time during optimization runs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Callable

import structlog
from pydantic import BaseModel, Field

from agentcore.dspy_optimization.models import (
    OptimizationResult,
    OptimizationStatus,
    PerformanceMetrics,
)

logger = structlog.get_logger()


class MonitoringEventType(str, Enum):
    """Types of monitoring events."""

    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_ITERATION = "optimization_iteration"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    OPTIMIZATION_FAILED = "optimization_failed"
    PERFORMANCE_IMPROVED = "performance_improved"
    PERFORMANCE_DEGRADED = "performance_degraded"
    BASELINE_CAPTURED = "baseline_captured"
    METRICS_UPDATED = "metrics_updated"


class MonitoringEvent(BaseModel):
    """Monitoring event data."""

    event_type: MonitoringEventType = Field(..., description="Event type")
    agent_id: str = Field(..., description="Agent identifier")
    optimization_id: str = Field(..., description="Optimization run identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    data: dict[str, Any] = Field(default_factory=dict, description="Event data")
    metrics: PerformanceMetrics | None = Field(None, description="Current metrics")


MonitoringCallback = Callable[[MonitoringEvent], None]


class OptimizationMonitor:
    """
    Real-time optimization monitoring.

    Tracks optimization progress and emits events for external monitoring,
    alerting, and dashboard updates.
    """

    def __init__(self) -> None:
        """Initialize optimization monitor."""
        self._callbacks: dict[MonitoringEventType, list[MonitoringCallback]] = {}
        self._active_optimizations: dict[str, dict[str, Any]] = {}
        self._event_history: list[MonitoringEvent] = []
        self._max_history_size = 1000

        logger.info("Optimization monitor initialized")

    def register_callback(
        self, event_type: MonitoringEventType, callback: MonitoringCallback
    ) -> None:
        """
        Register callback for monitoring events.

        Args:
            event_type: Event type to listen for
            callback: Callback function to invoke
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []

        self._callbacks[event_type].append(callback)

        logger.debug("Registered monitoring callback", event_type=event_type)

    def unregister_callback(
        self, event_type: MonitoringEventType, callback: MonitoringCallback
    ) -> None:
        """
        Unregister callback for monitoring events.

        Args:
            event_type: Event type
            callback: Callback function to remove
        """
        if event_type in self._callbacks:
            try:
                self._callbacks[event_type].remove(callback)
                logger.debug("Unregistered monitoring callback", event_type=event_type)
            except ValueError:
                logger.warning("Callback not found", event_type=event_type)

    def emit_event(self, event: MonitoringEvent) -> None:
        """
        Emit monitoring event to registered callbacks.

        Args:
            event: Monitoring event to emit
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history.pop(0)

        # Invoke callbacks
        callbacks = self._callbacks.get(event.event_type, [])
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(
                    "Monitoring callback failed",
                    event_type=event.event_type,
                    error=str(e),
                )

        logger.debug(
            "Emitted monitoring event",
            event_type=event.event_type,
            agent_id=event.agent_id,
            optimization_id=event.optimization_id,
        )

    def on_optimization_started(
        self,
        agent_id: str,
        optimization_id: str,
        baseline_metrics: PerformanceMetrics,
        objectives: dict[str, Any],
    ) -> None:
        """
        Handle optimization started event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            baseline_metrics: Baseline performance metrics
            objectives: Optimization objectives
        """
        self._active_optimizations[optimization_id] = {
            "agent_id": agent_id,
            "started_at": datetime.utcnow(),
            "baseline_metrics": baseline_metrics,
            "objectives": objectives,
            "iteration_count": 0,
        }

        event = MonitoringEvent(
            event_type=MonitoringEventType.OPTIMIZATION_STARTED,
            agent_id=agent_id,
            optimization_id=optimization_id,
            metrics=baseline_metrics,
            data={
                "objectives": objectives,
                "baseline": baseline_metrics.model_dump(),
            },
        )

        self.emit_event(event)

        logger.info(
            "Optimization started",
            agent_id=agent_id,
            optimization_id=optimization_id,
        )

    def on_optimization_iteration(
        self,
        agent_id: str,
        optimization_id: str,
        iteration: int,
        current_metrics: PerformanceMetrics,
        iteration_data: dict[str, Any],
    ) -> None:
        """
        Handle optimization iteration event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            iteration: Current iteration number
            current_metrics: Current performance metrics
            iteration_data: Iteration-specific data
        """
        if optimization_id in self._active_optimizations:
            self._active_optimizations[optimization_id]["iteration_count"] = iteration
            self._active_optimizations[optimization_id]["current_metrics"] = current_metrics

        event = MonitoringEvent(
            event_type=MonitoringEventType.OPTIMIZATION_ITERATION,
            agent_id=agent_id,
            optimization_id=optimization_id,
            metrics=current_metrics,
            data={
                "iteration": iteration,
                "iteration_data": iteration_data,
            },
        )

        self.emit_event(event)

        logger.debug(
            "Optimization iteration",
            agent_id=agent_id,
            optimization_id=optimization_id,
            iteration=iteration,
        )

    def on_optimization_completed(
        self,
        agent_id: str,
        optimization_id: str,
        result: OptimizationResult,
    ) -> None:
        """
        Handle optimization completed event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            result: Optimization result
        """
        optimization_data = self._active_optimizations.pop(optimization_id, {})

        event = MonitoringEvent(
            event_type=MonitoringEventType.OPTIMIZATION_COMPLETED,
            agent_id=agent_id,
            optimization_id=optimization_id,
            metrics=result.optimized_performance,
            data={
                "result": result.model_dump(),
                "duration_seconds": (
                    (datetime.utcnow() - optimization_data.get("started_at", datetime.utcnow())).total_seconds()
                    if "started_at" in optimization_data
                    else 0
                ),
            },
        )

        self.emit_event(event)

        logger.info(
            "Optimization completed",
            agent_id=agent_id,
            optimization_id=optimization_id,
            improvement=result.improvement_percentage,
        )

    def on_optimization_failed(
        self,
        agent_id: str,
        optimization_id: str,
        error: str,
    ) -> None:
        """
        Handle optimization failed event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            error: Error message
        """
        self._active_optimizations.pop(optimization_id, None)

        event = MonitoringEvent(
            event_type=MonitoringEventType.OPTIMIZATION_FAILED,
            agent_id=agent_id,
            optimization_id=optimization_id,
            data={"error": error},
        )

        self.emit_event(event)

        logger.error(
            "Optimization failed",
            agent_id=agent_id,
            optimization_id=optimization_id,
            error=error,
        )

    def on_performance_improved(
        self,
        agent_id: str,
        optimization_id: str,
        baseline: PerformanceMetrics,
        improved: PerformanceMetrics,
        improvement_percentage: float,
    ) -> None:
        """
        Handle performance improvement event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            baseline: Baseline metrics
            improved: Improved metrics
            improvement_percentage: Improvement percentage
        """
        event = MonitoringEvent(
            event_type=MonitoringEventType.PERFORMANCE_IMPROVED,
            agent_id=agent_id,
            optimization_id=optimization_id,
            metrics=improved,
            data={
                "baseline": baseline.model_dump(),
                "improved": improved.model_dump(),
                "improvement_percentage": improvement_percentage,
            },
        )

        self.emit_event(event)

        logger.info(
            "Performance improved",
            agent_id=agent_id,
            improvement_percentage=improvement_percentage,
        )

    def on_performance_degraded(
        self,
        agent_id: str,
        optimization_id: str,
        baseline: PerformanceMetrics,
        degraded: PerformanceMetrics,
        degradation_percentage: float,
    ) -> None:
        """
        Handle performance degradation event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            baseline: Baseline metrics
            degraded: Degraded metrics
            degradation_percentage: Degradation percentage
        """
        event = MonitoringEvent(
            event_type=MonitoringEventType.PERFORMANCE_DEGRADED,
            agent_id=agent_id,
            optimization_id=optimization_id,
            metrics=degraded,
            data={
                "baseline": baseline.model_dump(),
                "degraded": degraded.model_dump(),
                "degradation_percentage": degradation_percentage,
            },
        )

        self.emit_event(event)

        logger.warning(
            "Performance degraded",
            agent_id=agent_id,
            degradation_percentage=degradation_percentage,
        )

    def on_baseline_captured(
        self,
        agent_id: str,
        optimization_id: str,
        baseline_metrics: PerformanceMetrics,
    ) -> None:
        """
        Handle baseline captured event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            baseline_metrics: Captured baseline metrics
        """
        event = MonitoringEvent(
            event_type=MonitoringEventType.BASELINE_CAPTURED,
            agent_id=agent_id,
            optimization_id=optimization_id,
            metrics=baseline_metrics,
            data={"baseline": baseline_metrics.model_dump()},
        )

        self.emit_event(event)

        logger.info(
            "Baseline captured",
            agent_id=agent_id,
            optimization_id=optimization_id,
        )

    def on_metrics_updated(
        self,
        agent_id: str,
        optimization_id: str,
        metrics: PerformanceMetrics,
    ) -> None:
        """
        Handle metrics updated event.

        Args:
            agent_id: Agent identifier
            optimization_id: Optimization run ID
            metrics: Updated metrics
        """
        event = MonitoringEvent(
            event_type=MonitoringEventType.METRICS_UPDATED,
            agent_id=agent_id,
            optimization_id=optimization_id,
            metrics=metrics,
            data={"metrics": metrics.model_dump()},
        )

        self.emit_event(event)

        logger.debug(
            "Metrics updated",
            agent_id=agent_id,
            optimization_id=optimization_id,
        )

    def get_active_optimizations(self) -> list[dict[str, Any]]:
        """
        Get list of currently active optimizations.

        Returns:
            List of active optimization data
        """
        return [
            {
                "optimization_id": opt_id,
                **opt_data,
            }
            for opt_id, opt_data in self._active_optimizations.items()
        ]

    def get_optimization_status(self, optimization_id: str) -> dict[str, Any] | None:
        """
        Get status of specific optimization.

        Args:
            optimization_id: Optimization run ID

        Returns:
            Optimization status data or None if not found
        """
        return self._active_optimizations.get(optimization_id)

    def get_event_history(
        self,
        agent_id: str | None = None,
        event_type: MonitoringEventType | None = None,
        limit: int = 100,
    ) -> list[MonitoringEvent]:
        """
        Get event history with optional filters.

        Args:
            agent_id: Filter by agent ID
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of monitoring events
        """
        filtered_events = self._event_history

        if agent_id:
            filtered_events = [e for e in filtered_events if e.agent_id == agent_id]

        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]

        return filtered_events[-limit:]

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Cleared monitoring event history")


# Global monitor instance
optimization_monitor = OptimizationMonitor()
