"""
Budget enforcement for training jobs.

Provides budget tracking, pre-flight checks, and enforcement with configurable alert thresholds.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import structlog

logger = structlog.get_logger()


class BudgetStatus(str, Enum):
    """Budget status enumeration."""

    OK = "ok"
    WARNING_75 = "warning_75"
    WARNING_90 = "warning_90"
    EXCEEDED = "exceeded"


class BudgetEnforcer:
    """
    Budget enforcement for training operations.

    Tracks cost against budget, performs pre-flight checks,
    and enforces budget limits with configurable alert thresholds.
    """

    def __init__(
        self,
        max_budget_usd: Decimal,
        warning_threshold_75: float = 0.75,
        warning_threshold_90: float = 0.90,
    ) -> None:
        """
        Initialize budget enforcer.

        Args:
            max_budget_usd: Maximum budget in USD
            warning_threshold_75: Warning threshold at 75% (default: 0.75)
            warning_threshold_90: Warning threshold at 90% (default: 0.90)
        """
        self.max_budget_usd = max_budget_usd
        self.current_cost_usd = Decimal("0.00")
        self.warning_threshold_75 = Decimal(str(warning_threshold_75))
        self.warning_threshold_90 = Decimal(str(warning_threshold_90))

        logger.info(
            "budget_enforcer_initialized",
            max_budget_usd=str(max_budget_usd),
            warning_75_pct=warning_threshold_75,
            warning_90_pct=warning_threshold_90,
        )

    def check_budget_available(self, required_amount: Decimal) -> tuple[bool, BudgetStatus]:
        """
        Check if budget is available for operation.

        Args:
            required_amount: Amount required for operation (USD)

        Returns:
            Tuple of (is_available, budget_status)
        """
        projected_cost = self.current_cost_usd + required_amount

        # Check if budget would be exceeded
        if projected_cost > self.max_budget_usd:
            logger.warning(
                "budget_exceeded_check",
                current_cost=str(self.current_cost_usd),
                required_amount=str(required_amount),
                projected_cost=str(projected_cost),
                max_budget=str(self.max_budget_usd),
            )
            return False, BudgetStatus.EXCEEDED

        # Calculate utilization percentage
        utilization = projected_cost / self.max_budget_usd

        # Determine status based on thresholds
        if utilization >= self.warning_threshold_90:
            status = BudgetStatus.WARNING_90
            logger.warning(
                "budget_warning_90",
                utilization=float(utilization),
                current_cost=str(self.current_cost_usd),
                max_budget=str(self.max_budget_usd),
            )
        elif utilization >= self.warning_threshold_75:
            status = BudgetStatus.WARNING_75
            logger.warning(
                "budget_warning_75",
                utilization=float(utilization),
                current_cost=str(self.current_cost_usd),
                max_budget=str(self.max_budget_usd),
            )
        else:
            status = BudgetStatus.OK

        return True, status

    def add_cost(self, amount: Decimal) -> None:
        """
        Add cost to current total.

        Args:
            amount: Cost amount to add (USD)
        """
        self.current_cost_usd += amount

        logger.debug(
            "cost_added",
            amount=str(amount),
            current_cost=str(self.current_cost_usd),
            max_budget=str(self.max_budget_usd),
            remaining=str(self.max_budget_usd - self.current_cost_usd),
        )

    def get_remaining_budget(self) -> Decimal:
        """
        Get remaining budget.

        Returns:
            Remaining budget in USD
        """
        return self.max_budget_usd - self.current_cost_usd

    def get_utilization_percentage(self) -> float:
        """
        Get budget utilization percentage.

        Returns:
            Utilization as percentage (0-100)
        """
        if self.max_budget_usd == Decimal("0"):
            return 0.0

        utilization = (self.current_cost_usd / self.max_budget_usd) * 100
        return float(utilization)

    def is_budget_exceeded(self) -> bool:
        """
        Check if budget is currently exceeded.

        Returns:
            True if budget exceeded
        """
        return self.current_cost_usd > self.max_budget_usd

    def get_status(self) -> dict[str, Any]:
        """
        Get current budget status.

        Returns:
            Budget status information
        """
        remaining = self.get_remaining_budget()
        utilization = self.get_utilization_percentage()

        # Determine status
        if self.is_budget_exceeded():
            status = BudgetStatus.EXCEEDED
        elif utilization >= float(self.warning_threshold_90) * 100:
            status = BudgetStatus.WARNING_90
        elif utilization >= float(self.warning_threshold_75) * 100:
            status = BudgetStatus.WARNING_75
        else:
            status = BudgetStatus.OK

        return {
            "status": status.value,
            "current_cost_usd": float(self.current_cost_usd),
            "max_budget_usd": float(self.max_budget_usd),
            "remaining_usd": float(remaining),
            "utilization_percentage": utilization,
            "is_exceeded": self.is_budget_exceeded(),
        }

    def reset(self) -> None:
        """Reset current cost to zero."""
        self.current_cost_usd = Decimal("0.00")
        logger.info("budget_enforcer_reset")


def check_budget(
    current_cost: Decimal,
    max_budget: Decimal,
    required_amount: Decimal = Decimal("0.00"),
) -> tuple[bool, str]:
    """
    Standalone budget check function.

    Args:
        current_cost: Current cost (USD)
        max_budget: Maximum budget (USD)
        required_amount: Additional amount required (USD)

    Returns:
        Tuple of (is_ok, message)
    """
    projected_cost = current_cost + required_amount

    if projected_cost > max_budget:
        return False, (
            f"Budget would be exceeded: "
            f"${projected_cost:.2f} > ${max_budget:.2f} "
            f"(current: ${current_cost:.2f}, required: ${required_amount:.2f})"
        )

    remaining = max_budget - projected_cost
    return True, f"Budget OK (remaining: ${remaining:.2f})"
