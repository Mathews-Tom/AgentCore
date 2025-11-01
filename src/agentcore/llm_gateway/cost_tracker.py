"""Real-time cost tracking and budget enforcement.

Tracks LLM request costs, enforces budgets, and generates alerts when
thresholds are exceeded. Provides cost aggregation and reporting.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import structlog

from agentcore.integration.portkey.cost_models import (
    BudgetAlert,
    BudgetAlertSeverity,
    BudgetConfig,
    CostMetrics,
    CostPeriod,
    CostSummary,
)
from agentcore.integration.portkey.exceptions import PortkeyBudgetExceededError

logger = structlog.get_logger(__name__)


class CostTracker:
    """Real-time cost tracking and budget enforcement.

    Maintains cost history, enforces budget limits, and generates alerts
    when spending thresholds are exceeded.
    """

    def __init__(
        self,
        history_retention_days: int = 90,
        alert_debounce_seconds: int = 300,
    ) -> None:
        """Initialize the cost tracker.

        Args:
            history_retention_days: Number of days to retain cost history
            alert_debounce_seconds: Seconds between duplicate alerts
        """
        self._cost_history: list[CostMetrics] = []
        self._budgets: dict[str, BudgetConfig] = {}
        self._alerts: list[BudgetAlert] = []
        self._last_alert_time: dict[str, datetime] = {}

        self.history_retention = timedelta(days=history_retention_days)
        self.alert_debounce = timedelta(seconds=alert_debounce_seconds)

        logger.info(
            "cost_tracker_initialized",
            history_retention_days=history_retention_days,
            alert_debounce_seconds=alert_debounce_seconds,
        )

    def track_request(
        self,
        provider: str,
        model: str,
        cost: float,
        tokens: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        request_id: str | None = None,
        tenant_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Track a request with simplified parameters (backward compatibility).

        Args:
            provider: Provider identifier
            model: Model used
            cost: Total cost in USD
            tokens: Total tokens (input + output)
            input_tokens: Optional input tokens (defaults to tokens/2)
            output_tokens: Optional output tokens (defaults to tokens/2)
            latency_ms: Request latency in milliseconds
            request_id: Unique request identifier
            tenant_id: Tenant identifier
            workflow_id: Workflow identifier
            agent_id: Agent identifier
            tags: Custom tags
        """
        # If input/output tokens not provided, split total evenly
        if input_tokens is None and output_tokens is None:
            input_tokens = tokens // 2
            output_tokens = tokens - input_tokens
        elif input_tokens is None:
            input_tokens = tokens - (output_tokens or 0)
        elif output_tokens is None:
            output_tokens = tokens - input_tokens

        metrics = CostMetrics(
            total_cost=cost,
            input_cost=cost / 2,  # Simple split assumption
            output_cost=cost / 2,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_id=provider,
            model=model,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            request_id=request_id,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            tags=tags or {},
        )

        self.record_cost(metrics)

    def record_cost(self, metrics: CostMetrics) -> None:
        """Record cost metrics for a request.

        Args:
            metrics: Cost metrics to record

        Raises:
            PortkeyBudgetExceededError: If hard budget limit exceeded
        """
        # Add to history
        self._cost_history.append(metrics)

        # Update budget spending if applicable
        if metrics.tenant_id:
            budget = self._budgets.get(metrics.tenant_id)
            if budget:
                # Check if budget period is still active
                if budget.period_start <= metrics.timestamp <= budget.period_end:
                    budget.current_spend += metrics.total_cost

                    # Check for threshold violations
                    self._check_budget_thresholds(budget, metrics)

                    # Enforce hard limit if enabled
                    if budget.hard_limit and budget.current_spend > budget.limit_amount:
                        error = PortkeyBudgetExceededError(
                            f"Budget exceeded for tenant {budget.tenant_id}: "
                            f"${budget.current_spend:.2f} / ${budget.limit_amount:.2f}"
                        )
                        error.details = {  # type: ignore[attr-defined]
                            "tenant_id": budget.tenant_id,
                            "current_spend": budget.current_spend,
                            "limit_amount": budget.limit_amount,
                            "budget_period": budget.period,
                        }
                        raise error

        # Cleanup old history periodically
        self._cleanup_old_history()

        logger.debug(
            "cost_recorded",
            total_cost=metrics.total_cost,
            provider_id=metrics.provider_id,
            tenant_id=metrics.tenant_id,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
        )

    def calculate_request_cost(
        self,
        provider_id: str,
        input_tokens: int,
        output_tokens: int,
        input_token_price: float,
        output_token_price: float,
        model: str,
        latency_ms: int | None = None,
        request_id: str | None = None,
        tenant_id: str | None = None,
        workflow_id: str | None = None,
        agent_id: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> CostMetrics:
        """Calculate cost metrics for a request.

        Args:
            provider_id: Provider identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            input_token_price: Price per 1K input tokens in USD
            output_token_price: Price per 1K output tokens in USD
            model: Model used
            latency_ms: Request latency in milliseconds
            request_id: Unique request identifier
            tenant_id: Tenant identifier
            workflow_id: Workflow identifier
            agent_id: Agent identifier
            tags: Custom tags

        Returns:
            Calculated cost metrics
        """
        # Calculate costs (prices are per 1K tokens)
        input_cost = (input_tokens / 1000) * input_token_price
        output_cost = (output_tokens / 1000) * output_token_price
        total_cost = input_cost + output_cost

        return CostMetrics(
            total_cost=total_cost,
            input_cost=input_cost,
            output_cost=output_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_id=provider_id,
            model=model,
            timestamp=datetime.now(),
            latency_ms=latency_ms,
            request_id=request_id,
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            agent_id=agent_id,
            tags=tags or {},
        )

    def get_summary(
        self,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
        tenant_id: str | None = None,
        provider_id: str | None = None,
    ) -> CostSummary:
        """Get cost summary for a time period.

        Args:
            period_start: Start of time period (defaults to 30 days ago)
            period_end: End of time period (defaults to now)
            tenant_id: Optional tenant filter
            provider_id: Optional provider filter

        Returns:
            Cost summary for the period
        """
        now = datetime.now()
        period_start = period_start or (now - timedelta(days=30))
        period_end = period_end or now

        # Filter metrics by criteria
        filtered = [
            m
            for m in self._cost_history
            if period_start <= m.timestamp <= period_end
            and (tenant_id is None or m.tenant_id == tenant_id)
            and (provider_id is None or m.provider_id == provider_id)
        ]

        if not filtered:
            # Return empty summary
            return CostSummary(
                period_start=period_start,
                period_end=period_end,
                total_cost=0.0,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                average_cost_per_request=0.0,
                average_cost_per_1k_tokens=0.0,
                average_latency_ms=None,
                provider_breakdown={},
                model_breakdown={},
                tenant_breakdown={},
            )

        # Calculate aggregates
        total_cost = sum(m.total_cost for m in filtered)
        total_requests = len(filtered)
        total_input_tokens = sum(m.input_tokens for m in filtered)
        total_output_tokens = sum(m.output_tokens for m in filtered)

        average_cost_per_request = total_cost / total_requests

        total_tokens = total_input_tokens + total_output_tokens
        average_cost_per_1k_tokens = (
            (total_cost / total_tokens) * 1000 if total_tokens > 0 else 0.0
        )

        # Calculate average latency (only for metrics with latency data)
        latencies = [m.latency_ms for m in filtered if m.latency_ms is not None]
        average_latency_ms = int(sum(latencies) / len(latencies)) if latencies else None

        # Provider breakdown
        provider_breakdown: dict[str, float] = defaultdict(float)
        for m in filtered:
            provider_breakdown[m.provider_id] += m.total_cost

        # Model breakdown
        model_breakdown: dict[str, float] = defaultdict(float)
        for m in filtered:
            model_breakdown[m.model] += m.total_cost

        # Tenant breakdown
        tenant_breakdown: dict[str, float] = defaultdict(float)
        for m in filtered:
            if m.tenant_id:
                tenant_breakdown[m.tenant_id] += m.total_cost

        return CostSummary(
            period_start=period_start,
            period_end=period_end,
            total_cost=total_cost,
            total_requests=total_requests,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            average_cost_per_request=average_cost_per_request,
            average_cost_per_1k_tokens=average_cost_per_1k_tokens,
            average_latency_ms=average_latency_ms,
            provider_breakdown=dict(provider_breakdown),
            model_breakdown=dict(model_breakdown),
            tenant_breakdown=dict(tenant_breakdown),
        )

    def set_budget(self, budget: BudgetConfig) -> None:
        """Configure a budget for a tenant.

        Args:
            budget: Budget configuration
        """
        self._budgets[budget.tenant_id] = budget

        logger.info(
            "budget_configured",
            tenant_id=budget.tenant_id,
            limit_amount=budget.limit_amount,
            period=budget.period,
            hard_limit=budget.hard_limit,
            thresholds=len(budget.thresholds),
        )

    def get_budget(self, tenant_id: str) -> BudgetConfig | None:
        """Get budget configuration for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Budget configuration or None if not found
        """
        return self._budgets.get(tenant_id)

    def get_all_budgets(self) -> list[BudgetConfig]:
        """Get all budget configurations.

        Returns:
            List of all budget configurations
        """
        return list(self._budgets.values())

    def check_budget_available(
        self,
        tenant_id: str,
        estimated_cost: float,
    ) -> tuple[bool, str | None]:
        """Check if budget is available for estimated cost.

        Args:
            tenant_id: Tenant identifier
            estimated_cost: Estimated cost of request in USD

        Returns:
            Tuple of (is_available, reason_if_not_available)
        """
        budget = self._budgets.get(tenant_id)
        if not budget:
            # No budget configured - allow request
            return True, None

        # Check if budget period is active
        now = datetime.now()
        if not (budget.period_start <= now <= budget.period_end):
            return True, "Budget period not active"

        # Check if adding estimated cost would exceed limit
        projected_spend = budget.current_spend + estimated_cost

        if budget.hard_limit and projected_spend > budget.limit_amount:
            remaining = budget.limit_amount - budget.current_spend
            return (
                False,
                f"Budget limit would be exceeded: ${projected_spend:.2f} > ${budget.limit_amount:.2f} "
                f"(${remaining:.2f} remaining)",
            )

        return True, None

    def get_alerts(
        self,
        tenant_id: str | None = None,
        acknowledged: bool | None = None,
    ) -> list[BudgetAlert]:
        """Get budget alerts.

        Args:
            tenant_id: Optional tenant filter
            acknowledged: Optional acknowledged status filter

        Returns:
            List of budget alerts matching criteria
        """
        alerts = self._alerts

        if tenant_id is not None:
            alerts = [a for a in alerts if a.budget_config.tenant_id == tenant_id]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a budget alert.

        Args:
            alert_id: Alert identifier

        Returns:
            True if alert was found and acknowledged
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(
                    "alert_acknowledged",
                    alert_id=alert_id,
                    tenant_id=alert.budget_config.tenant_id,
                )
                return True
        return False

    def get_cost_history(
        self,
        limit: int | None = None,
        tenant_id: str | None = None,
        provider_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[CostMetrics]:
        """Get cost history matching criteria.

        Args:
            limit: Maximum number of records to return
            tenant_id: Optional tenant filter
            provider_id: Optional provider filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of cost metrics matching criteria
        """
        filtered = self._cost_history

        # Apply filters
        if tenant_id is not None:
            filtered = [m for m in filtered if m.tenant_id == tenant_id]

        if provider_id is not None:
            filtered = [m for m in filtered if m.provider_id == provider_id]

        if start_time is not None:
            filtered = [m for m in filtered if m.timestamp >= start_time]

        if end_time is not None:
            filtered = [m for m in filtered if m.timestamp <= end_time]

        # Sort by timestamp (most recent first)
        filtered = sorted(filtered, key=lambda m: m.timestamp, reverse=True)

        # Apply limit
        if limit is not None:
            filtered = filtered[:limit]

        return filtered

    def get_stats(self) -> dict[str, Any]:
        """Get cost tracker statistics.

        Returns:
            Dictionary with statistics
        """
        now = datetime.now()
        last_24h = [m for m in self._cost_history if m.timestamp >= now - timedelta(days=1)]
        last_7d = [m for m in self._cost_history if m.timestamp >= now - timedelta(days=7)]

        total_records = len(self._cost_history)
        total_cost = sum(m.total_cost for m in self._cost_history)

        return {
            "total_records": total_records,
            "total_requests": total_records,  # Alias for backward compatibility
            "total_cost": total_cost,
            "active_budgets": len(self._budgets),
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts if not a.acknowledged]),
            "last_24h_requests": len(last_24h),
            "last_24h_cost": sum(m.total_cost for m in last_24h),
            "last_7d_requests": len(last_7d),
            "last_7d_cost": sum(m.total_cost for m in last_7d),
        }

    def _check_budget_thresholds(
        self,
        budget: BudgetConfig,
        metrics: CostMetrics,
    ) -> None:
        """Check if budget thresholds have been exceeded.

        Args:
            budget: Budget configuration
            metrics: Current cost metrics
        """
        if not budget.thresholds:
            return

        # Calculate percentage consumed
        percent_consumed = (budget.current_spend / budget.limit_amount) * 100

        # Check each threshold
        for threshold in budget.thresholds:
            # Check if threshold exceeded
            if percent_consumed >= threshold.threshold_percent:
                # Check if we should send alert (debounce)
                alert_key = f"{budget.tenant_id}_{threshold.threshold_percent}"
                last_alert = self._last_alert_time.get(alert_key)

                if last_alert is None or (
                    datetime.now() - last_alert >= self.alert_debounce
                ):
                    # Generate alert
                    alert = self._create_alert(
                        budget=budget,
                        threshold=threshold,
                        percent_consumed=percent_consumed,
                    )

                    self._alerts.append(alert)
                    self._last_alert_time[alert_key] = datetime.now()

                    logger.warning(
                        "budget_alert_triggered",
                        alert_id=alert.alert_id,
                        tenant_id=budget.tenant_id,
                        severity=threshold.severity,
                        percent_consumed=percent_consumed,
                        threshold_percent=threshold.threshold_percent,
                    )

    def _create_alert(
        self,
        budget: BudgetConfig,
        threshold: BudgetThreshold,
        percent_consumed: float,
    ) -> BudgetAlert:
        """Create a budget alert.

        Args:
            budget: Budget configuration
            threshold: Threshold that was exceeded
            percent_consumed: Current percentage consumed

        Returns:
            Budget alert
        """
        alert_id = str(uuid.uuid4())

        # Generate message based on severity
        if threshold.severity == BudgetAlertSeverity.EMERGENCY:
            message = (
                f"EMERGENCY: Budget critically exceeded for tenant {budget.tenant_id}. "
                f"Current spend: ${budget.current_spend:.2f} / ${budget.limit_amount:.2f} "
                f"({percent_consumed:.1f}%)"
            )
        elif threshold.severity == BudgetAlertSeverity.CRITICAL:
            message = (
                f"CRITICAL: Budget threshold exceeded for tenant {budget.tenant_id}. "
                f"Current spend: ${budget.current_spend:.2f} / ${budget.limit_amount:.2f} "
                f"({percent_consumed:.1f}%)"
            )
        elif threshold.severity == BudgetAlertSeverity.WARNING:
            message = (
                f"WARNING: Budget threshold reached for tenant {budget.tenant_id}. "
                f"Current spend: ${budget.current_spend:.2f} / ${budget.limit_amount:.2f} "
                f"({percent_consumed:.1f}%)"
            )
        else:
            message = (
                f"INFO: Budget threshold reached for tenant {budget.tenant_id}. "
                f"Current spend: ${budget.current_spend:.2f} / ${budget.limit_amount:.2f} "
                f"({percent_consumed:.1f}%)"
            )

        return BudgetAlert(
            alert_id=alert_id,
            budget_config=budget,
            threshold=threshold,
            current_spend=budget.current_spend,
            percent_consumed=percent_consumed,
            timestamp=datetime.now(),
            message=message,
            acknowledged=False,
        )

    def _cleanup_old_history(self) -> None:
        """Remove cost history older than retention period."""
        cutoff_time = datetime.now() - self.history_retention
        initial_count = len(self._cost_history)

        self._cost_history = [
            m for m in self._cost_history if m.timestamp >= cutoff_time
        ]

        removed = initial_count - len(self._cost_history)
        if removed > 0:
            logger.debug(
                "cost_history_cleaned",
                removed_records=removed,
                remaining_records=len(self._cost_history),
            )


# Global cost tracker instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance.

    Returns:
        Global CostTracker instance
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
