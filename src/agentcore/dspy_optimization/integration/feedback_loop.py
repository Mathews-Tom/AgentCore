"""
Agent Performance Feedback Loop

Implements closed-loop optimization by continuously monitoring agent
performance and triggering re-optimization when performance degrades.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import structlog

from agentcore.dspy_optimization.integration.agent_connector import AgentRuntimeConnector
from agentcore.dspy_optimization.integration.monitoring_hooks import (
    OptimizationMonitor,
    optimization_monitor,
)
from agentcore.dspy_optimization.integration.target_spec import (
    AgentOptimizationProfile,
    AgentOptimizationTarget,
)
from agentcore.dspy_optimization.models import (
    OptimizationRequest,
    OptimizationResult,
    PerformanceMetrics,
)
from agentcore.dspy_optimization.pipeline import OptimizationPipeline

logger = structlog.get_logger()


class FeedbackLoopConfig:
    """Configuration for feedback loop."""

    def __init__(
        self,
        check_interval_seconds: int = 300,
        performance_degradation_threshold: float = 0.1,
        min_data_points: int = 10,
        cooldown_period_seconds: int = 3600,
        enable_auto_optimization: bool = True,
    ) -> None:
        """
        Initialize feedback loop configuration.

        Args:
            check_interval_seconds: How often to check agent performance
            performance_degradation_threshold: Threshold for triggering re-optimization
            min_data_points: Minimum data points before evaluating performance
            cooldown_period_seconds: Minimum time between optimizations
            enable_auto_optimization: Enable automatic re-optimization
        """
        self.check_interval_seconds = check_interval_seconds
        self.performance_degradation_threshold = performance_degradation_threshold
        self.min_data_points = min_data_points
        self.cooldown_period_seconds = cooldown_period_seconds
        self.enable_auto_optimization = enable_auto_optimization


class AgentPerformanceTracker:
    """Tracks agent performance metrics over time."""

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self._performance_history: dict[str, list[tuple[datetime, PerformanceMetrics]]] = {}
        self._baseline_metrics: dict[str, PerformanceMetrics] = {}

    def record_performance(self, agent_id: str, metrics: PerformanceMetrics) -> None:
        """
        Record agent performance metrics.

        Args:
            agent_id: Agent identifier
            metrics: Performance metrics
        """
        if agent_id not in self._performance_history:
            self._performance_history[agent_id] = []

        self._performance_history[agent_id].append((datetime.utcnow(), metrics))

        # Keep only recent history (last 1000 data points)
        if len(self._performance_history[agent_id]) > 1000:
            self._performance_history[agent_id] = self._performance_history[agent_id][-1000:]

        logger.debug("Recorded performance", agent_id=agent_id)

    def set_baseline(self, agent_id: str, metrics: PerformanceMetrics) -> None:
        """
        Set baseline metrics for agent.

        Args:
            agent_id: Agent identifier
            metrics: Baseline metrics
        """
        self._baseline_metrics[agent_id] = metrics
        logger.info("Set baseline metrics", agent_id=agent_id)

    def get_baseline(self, agent_id: str) -> PerformanceMetrics | None:
        """
        Get baseline metrics for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Baseline metrics or None if not set
        """
        return self._baseline_metrics.get(agent_id)

    def get_recent_performance(
        self, agent_id: str, window_seconds: int = 3600
    ) -> list[PerformanceMetrics]:
        """
        Get recent performance metrics within time window.

        Args:
            agent_id: Agent identifier
            window_seconds: Time window in seconds

        Returns:
            List of recent performance metrics
        """
        if agent_id not in self._performance_history:
            return []

        cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent = [
            metrics
            for timestamp, metrics in self._performance_history[agent_id]
            if timestamp >= cutoff_time
        ]

        return recent

    def calculate_average_performance(
        self, agent_id: str, window_seconds: int = 3600
    ) -> PerformanceMetrics | None:
        """
        Calculate average performance over time window.

        Args:
            agent_id: Agent identifier
            window_seconds: Time window in seconds

        Returns:
            Average performance metrics or None if insufficient data
        """
        recent_metrics = self.get_recent_performance(agent_id, window_seconds)

        if not recent_metrics:
            return None

        # Calculate averages
        avg_success_rate = sum(m.success_rate for m in recent_metrics) / len(recent_metrics)
        avg_cost = sum(m.avg_cost_per_task for m in recent_metrics) / len(recent_metrics)
        avg_latency = sum(m.avg_latency_ms for m in recent_metrics) / len(recent_metrics)
        avg_quality = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)

        return PerformanceMetrics(
            success_rate=avg_success_rate,
            avg_cost_per_task=avg_cost,
            avg_latency_ms=int(avg_latency),
            quality_score=avg_quality,
        )

    def detect_performance_degradation(
        self, agent_id: str, threshold: float = 0.1
    ) -> tuple[bool, float]:
        """
        Detect if agent performance has degraded.

        Args:
            agent_id: Agent identifier
            threshold: Degradation threshold (0.0-1.0)

        Returns:
            Tuple of (has_degraded, degradation_percentage)
        """
        baseline = self.get_baseline(agent_id)
        current = self.calculate_average_performance(agent_id)

        if not baseline or not current:
            return False, 0.0

        # Calculate degradation across all metrics
        success_degradation = max(0.0, baseline.success_rate - current.success_rate)
        cost_degradation = max(0.0, current.avg_cost_per_task - baseline.avg_cost_per_task) / max(baseline.avg_cost_per_task, 0.001)
        latency_degradation = max(0.0, current.avg_latency_ms - baseline.avg_latency_ms) / max(baseline.avg_latency_ms, 1)
        quality_degradation = max(0.0, baseline.quality_score - current.quality_score)

        # Weighted average degradation
        total_degradation = (
            success_degradation * 0.4
            + cost_degradation * 0.2
            + latency_degradation * 0.2
            + quality_degradation * 0.2
        )

        has_degraded = total_degradation > threshold

        logger.debug(
            "Performance degradation check",
            agent_id=agent_id,
            degradation=total_degradation,
            threshold=threshold,
            has_degraded=has_degraded,
        )

        return has_degraded, total_degradation


class AgentPerformanceFeedbackLoop:
    """
    Closed-loop agent performance optimization.

    Monitors agent performance in real-time and triggers re-optimization
    when performance degrades below acceptable thresholds.
    """

    def __init__(
        self,
        connector: AgentRuntimeConnector,
        pipeline: OptimizationPipeline,
        config: FeedbackLoopConfig | None = None,
        monitor: OptimizationMonitor | None = None,
    ) -> None:
        """
        Initialize feedback loop.

        Args:
            connector: Agent runtime connector
            pipeline: Optimization pipeline
            config: Feedback loop configuration
            monitor: Optimization monitor
        """
        self.connector = connector
        self.pipeline = pipeline
        self.config = config or FeedbackLoopConfig()
        self.monitor = monitor or optimization_monitor

        self.tracker = AgentPerformanceTracker()
        self._running = False
        self._monitored_agents: set[str] = set()
        self._last_optimization: dict[str, datetime] = {}

        logger.info("Agent performance feedback loop initialized")

    def add_agent(self, agent_id: str, baseline_metrics: PerformanceMetrics | None = None) -> None:
        """
        Add agent to feedback loop monitoring.

        Args:
            agent_id: Agent identifier
            baseline_metrics: Optional baseline metrics (will be fetched if not provided)
        """
        self._monitored_agents.add(agent_id)

        if baseline_metrics:
            self.tracker.set_baseline(agent_id, baseline_metrics)
        else:
            # Will fetch baseline on first check
            logger.debug("Agent added without baseline, will fetch", agent_id=agent_id)

        logger.info("Added agent to feedback loop", agent_id=agent_id)

    def remove_agent(self, agent_id: str) -> None:
        """
        Remove agent from feedback loop monitoring.

        Args:
            agent_id: Agent identifier
        """
        self._monitored_agents.discard(agent_id)
        logger.info("Removed agent from feedback loop", agent_id=agent_id)

    async def start(self) -> None:
        """Start feedback loop monitoring."""
        if self._running:
            logger.warning("Feedback loop already running")
            return

        self._running = True
        logger.info("Starting feedback loop")

        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())

    async def stop(self) -> None:
        """Stop feedback loop monitoring."""
        self._running = False
        logger.info("Stopping feedback loop")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_agents()
                await asyncio.sleep(self.config.check_interval_seconds)
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    async def _check_all_agents(self) -> None:
        """Check performance of all monitored agents."""
        for agent_id in list(self._monitored_agents):
            try:
                await self._check_agent_performance(agent_id)
            except Exception as e:
                logger.error("Error checking agent performance", agent_id=agent_id, error=str(e))

    async def _check_agent_performance(self, agent_id: str) -> None:
        """
        Check individual agent performance.

        Args:
            agent_id: Agent identifier
        """
        # Fetch current metrics
        current_metrics = await self.connector.get_agent_performance_metrics(agent_id)
        if not current_metrics:
            logger.debug("No metrics available for agent", agent_id=agent_id)
            return

        # Record current performance
        self.tracker.record_performance(agent_id, current_metrics)

        # Set baseline if not set
        baseline = self.tracker.get_baseline(agent_id)
        if not baseline:
            self.tracker.set_baseline(agent_id, current_metrics)
            logger.info("Set initial baseline for agent", agent_id=agent_id)
            return

        # Check for performance degradation
        has_degraded, degradation = self.tracker.detect_performance_degradation(
            agent_id, self.config.performance_degradation_threshold
        )

        if has_degraded:
            logger.warning(
                "Performance degradation detected",
                agent_id=agent_id,
                degradation=degradation,
            )

            # Trigger re-optimization if enabled and not in cooldown
            if self.config.enable_auto_optimization and self._can_optimize(agent_id):
                await self._trigger_optimization(agent_id, current_metrics)

    def _can_optimize(self, agent_id: str) -> bool:
        """
        Check if agent can be optimized (not in cooldown).

        Args:
            agent_id: Agent identifier

        Returns:
            True if optimization can be triggered
        """
        if agent_id not in self._last_optimization:
            return True

        last_opt = self._last_optimization[agent_id]
        cooldown = timedelta(seconds=self.config.cooldown_period_seconds)
        can_optimize = datetime.utcnow() - last_opt > cooldown

        if not can_optimize:
            logger.debug("Agent in optimization cooldown", agent_id=agent_id)

        return can_optimize

    async def _trigger_optimization(
        self, agent_id: str, current_metrics: PerformanceMetrics
    ) -> None:
        """
        Trigger re-optimization for agent.

        Args:
            agent_id: Agent identifier
            current_metrics: Current performance metrics
        """
        logger.info("Triggering re-optimization", agent_id=agent_id)

        try:
            # Create optimization request
            request = AgentOptimizationTarget.create_optimization_request(
                agent_id=agent_id,
                profile=AgentOptimizationProfile.BALANCED,
            )

            # Validate request
            is_valid, error = await self.connector.validate_optimization_request(request)
            if not is_valid:
                logger.error("Invalid optimization request", agent_id=agent_id, error=error)
                return

            # Get baseline metrics
            baseline = self.tracker.get_baseline(agent_id)
            if not baseline:
                logger.error("No baseline metrics available", agent_id=agent_id)
                return

            # Notify monitoring
            self.monitor.on_optimization_started(
                agent_id=agent_id,
                optimization_id=request.target.id,
                baseline_metrics=baseline,
                objectives={},
            )

            # Run optimization (would need training data in production)
            training_data: list[dict[str, Any]] = []  # Placeholder
            result = await self.pipeline.run_optimization(
                request=request,
                baseline_metrics=baseline,
                training_data=training_data,
            )

            # Update last optimization timestamp
            self._last_optimization[agent_id] = datetime.utcnow()

            # Update agent metrics if optimization succeeded
            if result.optimized_performance:
                await self.connector.update_agent_performance_metrics(
                    agent_id, result.optimized_performance
                )
                self.tracker.set_baseline(agent_id, result.optimized_performance)

                logger.info(
                    "Re-optimization completed",
                    agent_id=agent_id,
                    improvement=result.improvement_percentage,
                )

        except Exception as e:
            logger.error("Re-optimization failed", agent_id=agent_id, error=str(e))

    async def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        """
        Get feedback loop status for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Status information
        """
        baseline = self.tracker.get_baseline(agent_id)
        current = self.tracker.calculate_average_performance(agent_id)
        has_degraded, degradation = self.tracker.detect_performance_degradation(agent_id)
        last_opt = self._last_optimization.get(agent_id)

        return {
            "agent_id": agent_id,
            "is_monitored": agent_id in self._monitored_agents,
            "has_baseline": baseline is not None,
            "baseline_metrics": baseline.model_dump() if baseline else None,
            "current_metrics": current.model_dump() if current else None,
            "performance_degraded": has_degraded,
            "degradation_percentage": degradation,
            "last_optimization": last_opt.isoformat() if last_opt else None,
            "can_optimize": self._can_optimize(agent_id),
        }

    def get_monitored_agents(self) -> list[str]:
        """
        Get list of monitored agents.

        Returns:
            List of agent IDs
        """
        return list(self._monitored_agents)

    def is_running(self) -> bool:
        """
        Check if feedback loop is running.

        Returns:
            True if running
        """
        return self._running
