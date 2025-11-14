"""
Cost Tracker for Memory Service Compression Operations

Implements comprehensive cost tracking, budget monitoring, and alerting
for compression operations using the compression_metrics table.

Component ID: MEM-014
Ticket: MEM-014 (Implement Cost Tracking for Compression)

Features:
- Record compression costs to database (compression_metrics table)
- Monthly budget tracking with configurable limits
- Alert at 75% budget consumption threshold
- Cost breakdown queries by agent, task, operation type, date range
- Dashboard query support for cost visualization
- Integration with ContextCompressor for automatic cost tracking
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from agentcore.a2a_protocol.config import settings
from agentcore.a2a_protocol.database.connection import get_session

logger = structlog.get_logger()


class CompressionMetricsModel:
    """
    Database model for compression_metrics table.

    Maps to compression_metrics table created by MEM-004 migration.
    Uses PostgreSQL-specific types (UUID, Numeric, JSONB).
    """

    def __init__(
        self,
        metric_id: UUID,
        stage_id: UUID | None,
        task_id: UUID | None,
        compression_type: str,
        input_tokens: int,
        output_tokens: int,
        compression_ratio: float,
        critical_fact_retention_rate: float | None,
        coherence_score: float | None,
        cost_usd: Decimal,
        model_used: str,
        recorded_at: datetime,
        agent_id: UUID | None = None,
    ):
        """Initialize compression metrics model."""
        self.metric_id = metric_id
        self.stage_id = stage_id
        self.task_id = task_id
        self.compression_type = compression_type
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.compression_ratio = compression_ratio
        self.critical_fact_retention_rate = critical_fact_retention_rate
        self.coherence_score = coherence_score
        self.cost_usd = cost_usd
        self.model_used = model_used
        self.recorded_at = recorded_at
        self.agent_id = agent_id


class BudgetAlert:
    """Budget consumption alert model."""

    def __init__(
        self,
        alert_id: str,
        current_spend: float,
        budget_limit: float,
        consumption_percentage: float,
        threshold_percentage: float,
        period_start: datetime,
        period_end: datetime,
        message: str,
    ):
        """Initialize budget alert."""
        self.alert_id = alert_id
        self.current_spend = current_spend
        self.budget_limit = budget_limit
        self.consumption_percentage = consumption_percentage
        self.threshold_percentage = threshold_percentage
        self.period_start = period_start
        self.period_end = period_end
        self.message = message


class CostTracker:
    """
    Cost tracking service for compression operations.

    Tracks token usage and costs for all compression operations,
    monitors monthly budget consumption, and triggers alerts when
    spending exceeds configured thresholds.

    Integrates with ContextCompressor to automatically record costs
    and provides dashboard query methods for cost analysis.
    """

    # Budget alert threshold (75% of monthly budget)
    ALERT_THRESHOLD_PERCENTAGE = 75.0

    def __init__(self, trace_id: str | None = None):
        """
        Initialize CostTracker.

        Args:
            trace_id: Optional trace ID for request tracking
        """
        self._logger = logger.bind(component="cost_tracker")
        self._trace_id = trace_id

    async def record_compression_cost(
        self,
        compression_type: str,
        input_tokens: int,
        output_tokens: int,
        compression_ratio: float,
        cost_usd: float,
        model_used: str,
        stage_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        critical_fact_retention_rate: float | None = None,
        coherence_score: float | None = None,
    ) -> str:
        """
        Record compression cost to compression_metrics table.

        Args:
            compression_type: Type of compression (stage, task)
            input_tokens: Number of input tokens processed
            output_tokens: Number of output tokens generated
            compression_ratio: Achieved compression ratio
            cost_usd: Cost in USD for this operation
            model_used: Model used for compression
            stage_id: Optional stage ID
            task_id: Optional task ID
            agent_id: Optional agent ID
            critical_fact_retention_rate: Optional fact retention rate
            coherence_score: Optional coherence score

        Returns:
            Metric ID (UUID string)
        """
        metric_id = uuid4()

        # Convert string IDs to UUIDs if provided
        stage_uuid = UUID(stage_id) if stage_id else None
        task_uuid = UUID(task_id) if task_id else None
        agent_uuid = UUID(agent_id) if agent_id else None

        async with get_session() as session:
            # Insert directly using raw SQL for better control
            query = """
                INSERT INTO compression_metrics (
                    metric_id, stage_id, task_id, compression_type,
                    input_tokens, output_tokens, compression_ratio,
                    critical_fact_retention_rate, coherence_score,
                    cost_usd, model_used, recorded_at
                ) VALUES (
                    :metric_id, :stage_id, :task_id, :compression_type,
                    :input_tokens, :output_tokens, :compression_ratio,
                    :critical_fact_retention_rate, :coherence_score,
                    :cost_usd, :model_used, :recorded_at
                )
            """

            await session.execute(
                query,
                {
                    "metric_id": metric_id,
                    "stage_id": stage_uuid,
                    "task_id": task_uuid,
                    "compression_type": compression_type,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "compression_ratio": compression_ratio,
                    "critical_fact_retention_rate": critical_fact_retention_rate,
                    "coherence_score": coherence_score,
                    "cost_usd": Decimal(str(cost_usd)),
                    "model_used": model_used,
                    "recorded_at": datetime.now(UTC),
                },
            )
            await session.commit()

        self._logger.info(
            "compression_cost_recorded",
            metric_id=str(metric_id),
            compression_type=compression_type,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model_used,
        )

        # Check budget threshold after recording
        await self.check_budget_threshold()

        return str(metric_id)

    async def get_monthly_usage(
        self,
        year: int | None = None,
        month: int | None = None,
    ) -> dict[str, Any]:
        """
        Get monthly cost usage statistics.

        Args:
            year: Year (defaults to current year)
            month: Month (1-12, defaults to current month)

        Returns:
            Dictionary with monthly usage stats:
            - total_cost: Total cost in USD
            - total_operations: Number of compression operations
            - total_input_tokens: Total input tokens
            - total_output_tokens: Total output tokens
            - avg_cost_per_operation: Average cost per operation
            - period_start: Start of period
            - period_end: End of period
        """
        now = datetime.now(UTC)
        year = year or now.year
        month = month or now.month

        # Calculate period boundaries
        period_start = datetime(year, month, 1, tzinfo=UTC)
        if month == 12:
            period_end = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            period_end = datetime(year, month + 1, 1, tzinfo=UTC)

        async with get_session() as session:
            query = """
                SELECT
                    COALESCE(SUM(cost_usd), 0) as total_cost,
                    COUNT(*) as total_operations,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens
                FROM compression_metrics
                WHERE recorded_at >= :period_start
                  AND recorded_at < :period_end
            """

            result = await session.execute(
                query,
                {
                    "period_start": period_start,
                    "period_end": period_end,
                },
            )
            row = result.fetchone()

        total_cost = float(row[0]) if row[0] else 0.0
        total_operations = int(row[1]) if row[1] else 0
        total_input_tokens = int(row[2]) if row[2] else 0
        total_output_tokens = int(row[3]) if row[3] else 0

        avg_cost = total_cost / total_operations if total_operations > 0 else 0.0

        return {
            "total_cost": total_cost,
            "total_operations": total_operations,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_cost_per_operation": avg_cost,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
        }

    async def check_budget_threshold(self) -> BudgetAlert | None:
        """
        Check if current month spending exceeds budget threshold.

        Triggers alert if spending >= 75% of monthly budget.

        Returns:
            BudgetAlert if threshold exceeded, None otherwise
        """
        monthly_budget = settings.MONTHLY_TOKEN_BUDGET_USD
        usage = await self.get_monthly_usage()

        current_spend = usage["total_cost"]
        consumption_pct = (current_spend / monthly_budget) * 100.0

        if consumption_pct >= self.ALERT_THRESHOLD_PERCENTAGE:
            alert = BudgetAlert(
                alert_id=f"alert-{uuid4()}",
                current_spend=current_spend,
                budget_limit=monthly_budget,
                consumption_percentage=consumption_pct,
                threshold_percentage=self.ALERT_THRESHOLD_PERCENTAGE,
                period_start=datetime.fromisoformat(usage["period_start"]),
                period_end=datetime.fromisoformat(usage["period_end"]),
                message=(
                    f"Budget alert: {consumption_pct:.1f}% of monthly budget consumed. "
                    f"Current spend: ${current_spend:.2f} / ${monthly_budget:.2f}"
                ),
            )

            self._logger.warning(
                "budget_threshold_exceeded",
                alert_id=alert.alert_id,
                current_spend=current_spend,
                budget_limit=monthly_budget,
                consumption_percentage=consumption_pct,
                threshold=self.ALERT_THRESHOLD_PERCENTAGE,
            )

            return alert

        return None

    async def get_cost_breakdown(
        self,
        by: str = "operation_type",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get cost breakdown by specified dimension.

        Args:
            by: Breakdown dimension (operation_type, agent, task, model)
            start_date: Start date filter (optional)
            end_date: End date filter (optional)

        Returns:
            Dictionary with breakdown data:
            - breakdown: List of (key, cost, operations, tokens) tuples
            - total_cost: Total cost across all entries
            - total_operations: Total operations
            - period_start: Start of period
            - period_end: End of period
        """
        # Default to current month if dates not provided
        now = datetime.now(UTC)
        start_date = start_date or datetime(now.year, now.month, 1, tzinfo=UTC)
        end_date = end_date or now

        # Map dimension to column
        dimension_map = {
            "operation_type": "compression_type",
            "agent": "agent_id",
            "task": "task_id",
            "model": "model_used",
        }

        if by not in dimension_map:
            raise ValueError(
                f"Invalid breakdown dimension: {by}. "
                f"Must be one of: {list(dimension_map.keys())}"
            )

        column = dimension_map[by]

        async with get_session() as session:
            query = f"""
                SELECT
                    {column} as dimension_key,
                    SUM(cost_usd) as total_cost,
                    COUNT(*) as operations,
                    SUM(input_tokens + output_tokens) as total_tokens
                FROM compression_metrics
                WHERE recorded_at >= :start_date
                  AND recorded_at <= :end_date
                GROUP BY {column}
                ORDER BY total_cost DESC
            """

            result = await session.execute(
                query,
                {
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            rows = result.fetchall()

        breakdown = [
            {
                "key": str(row[0]) if row[0] else "unknown",
                "cost": float(row[1]) if row[1] else 0.0,
                "operations": int(row[2]) if row[2] else 0,
                "tokens": int(row[3]) if row[3] else 0,
            }
            for row in rows
        ]

        total_cost = sum(item["cost"] for item in breakdown)
        total_operations = sum(item["operations"] for item in breakdown)

        return {
            "breakdown": breakdown,
            "total_cost": total_cost,
            "total_operations": total_operations,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
        }

    async def get_costs_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day",
    ) -> dict[str, Any]:
        """
        Get costs grouped by time period.

        Args:
            start_date: Start date
            end_date: End date
            group_by: Grouping period (hour, day, week, month)

        Returns:
            Dictionary with time-series cost data:
            - costs: List of (period, cost, operations) tuples
            - total_cost: Total cost
            - total_operations: Total operations
        """
        # Map group_by to PostgreSQL date_trunc argument
        trunc_map = {
            "hour": "hour",
            "day": "day",
            "week": "week",
            "month": "month",
        }

        if group_by not in trunc_map:
            raise ValueError(
                f"Invalid group_by: {group_by}. "
                f"Must be one of: {list(trunc_map.keys())}"
            )

        trunc_arg = trunc_map[group_by]

        async with get_session() as session:
            query = f"""
                SELECT
                    DATE_TRUNC(:trunc_arg, recorded_at) as period,
                    SUM(cost_usd) as total_cost,
                    COUNT(*) as operations
                FROM compression_metrics
                WHERE recorded_at >= :start_date
                  AND recorded_at <= :end_date
                GROUP BY period
                ORDER BY period ASC
            """

            result = await session.execute(
                query,
                {
                    "trunc_arg": trunc_arg,
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
            rows = result.fetchall()

        costs = [
            {
                "period": row[0].isoformat() if row[0] else None,
                "cost": float(row[1]) if row[1] else 0.0,
                "operations": int(row[2]) if row[2] else 0,
            }
            for row in rows
        ]

        total_cost = sum(item["cost"] for item in costs)
        total_operations = sum(item["operations"] for item in costs)

        return {
            "costs": costs,
            "total_cost": total_cost,
            "total_operations": total_operations,
            "group_by": group_by,
        }

    async def get_costs_by_agent(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get costs grouped by agent.

        Args:
            start_date: Start date filter (optional, defaults to current month)
            end_date: End date filter (optional, defaults to now)

        Returns:
            Dictionary with agent cost breakdown
        """
        return await self.get_cost_breakdown(
            by="agent",
            start_date=start_date,
            end_date=end_date,
        )

    async def get_costs_by_operation_type(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get costs grouped by operation type.

        Args:
            start_date: Start date filter (optional, defaults to current month)
            end_date: End date filter (optional, defaults to now)

        Returns:
            Dictionary with operation type cost breakdown
        """
        return await self.get_cost_breakdown(
            by="operation_type",
            start_date=start_date,
            end_date=end_date,
        )


__all__ = ["CostTracker", "BudgetAlert", "CompressionMetricsModel"]
